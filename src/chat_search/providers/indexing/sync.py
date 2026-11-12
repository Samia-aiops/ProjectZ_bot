"""Provider sync: backend API -> Elasticsearch.

Runs outside the chat service (``python -m chat_search sync-providers``),
scheduled weekly by Airflow. Exit code 0 = fine, 1 = failed or removal refused.

The sync is idempotent (documents are upserted by provider id) and stale
removal only happens after a complete, healthy pull, so retries are safe.
"""

from __future__ import annotations

import base64
import logging
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
from elasticsearch.helpers import async_bulk

from ... import config
from ...logging_setup import log_failure, sanitize_url
from ..client import make_es_client
from .normalize import build_provider_doc
from .schema import ensure_providers_index

log = logging.getLogger(__name__)


def safe_api_key_id(value: str | None) -> str:
    if not value:
        return "missing"
    try:
        decoded = base64.b64decode(value).decode("utf-8")
        return decoded.split(":", 1)[0]
    except Exception:
        return "unrecognized-format"


@dataclass
class SyncStats:
    fetched: int = 0
    skipped: int = 0
    indexed: int = 0
    removed: int = 0
    bulk_errors: Counter[str] = field(default_factory=Counter)
    rejection_reasons: Counter[str] = field(default_factory=Counter)
    rejection_samples: int = 0


def _rejection_reason(item: Any) -> str:
    if not isinstance(item, dict):
        return "invalid_record"
    if not str(item.get("user_id") or item.get("id") or item.get("provider_id") or "").strip():
        return "missing_provider_id"
    full_name = str(item.get("name") or f"{item.get('firstname') or ''} {item.get('lastname') or ''}").strip()
    if not full_name:
        return "missing_name"
    return "normalizer_rejected_other"


def _log_rejection_sample(stats: SyncStats, item: Any, reason: str) -> None:
    if stats.rejection_samples >= 5:
        return
    stats.rejection_samples += 1
    log.warning(
        "Provider rejected: record_number=%d reason=%s keys=%s has_user_id=%s has_id=%s has_name=%s has_firstname=%s",
        stats.fetched,
        reason,
        sorted(item.keys()) if isinstance(item, dict) else [],
        bool(item.get("user_id")) if isinstance(item, dict) else False,
        bool(item.get("id")) if isinstance(item, dict) else False,
        bool(item.get("name")) if isinstance(item, dict) else False,
        bool(item.get("firstname")) if isinstance(item, dict) else False,
    )


async def fetch_provider_pages(http: httpx.AsyncClient) -> AsyncIterator[list[dict[str, Any]]]:
    """Yield pages from the providers API, following Laravel-style pagination safely."""
    base = urlsplit(config.PROVIDERS_API_URL)
    headers = {
        "Authorization": f"Bearer {config.PROVIDERS_API_TOKEN}",
        "Accept": "application/json",
    }

    url = config.PROVIDERS_API_URL
    params: dict[str, Any] | None = {
        **config.PROVIDERS_API_PARAMS,
        "per_page": config.PROVIDERS_API_PAGE_SIZE,
        "page": 1,
    }

    for _ in range(config.PROVIDERS_API_MAX_PAGES):
        response = await http.get(url, params=params, headers=headers)
        response.raise_for_status()
        body = response.json()
        pagination: dict[str, Any] = {}

        if isinstance(body, list):
            items = body
        elif isinstance(body, dict):
            data = body.get(config.PROVIDERS_API_ITEMS_KEY)
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                providers = data.get("providers")
                if isinstance(providers, dict):
                    pagination = providers
                    items = providers.get("data")
                elif isinstance(data.get("data"), list):
                    # Standard Laravel paginator under data.
                    pagination = data
                    items = data["data"]
                else:
                    # Fallback for APIs using other nested collection names.
                    items = None
                    for key in ("items", "results", "providers"):
                        candidate = data.get(key)
                        if isinstance(candidate, list) and all(isinstance(item, dict) for item in candidate):
                            items = candidate
                            break
                    if items is None:
                        log.error(
                            "Provider response structure unexpected; top-level keys=%s, data keys=%s",
                            list(body.keys()),
                            list(data.keys()),
                        )
                        raise RuntimeError("Could not locate provider list in API response")
            else:
                log.error(
                    "Configured items key %r has type %s; response keys=%s",
                    config.PROVIDERS_API_ITEMS_KEY,
                    type(data).__name__,
                    list(body.keys()),
                )
                raise RuntimeError("Providers API items key not found")

            if not isinstance(items, list):
                log.error("Provider collection has unexpected type: %s", type(items).__name__)
                raise RuntimeError("Provider collection is not a list")
        else:
            raise RuntimeError(f"Unexpected providers API response type: {type(body).__name__}")

        if not all(isinstance(item, dict) for item in items):
            raise RuntimeError("Provider list contains unexpected item types")
        if not items:
            return

        yield items

        next_url = pagination.get("next_page_url") or (body.get("next_page_url") if isinstance(body, dict) else None)
        if next_url:
            next_url = urljoin(url, next_url)
            parts = urlsplit(next_url)
            if parts.scheme != base.scheme or parts.netloc != base.netloc:
                raise RuntimeError("Refusing to follow a next-page link to another host")
            url = next_url
            params = None
            continue

        current_page = pagination.get("current_page")
        last_page = pagination.get("last_page")
        if current_page is not None and last_page is not None:
            if int(current_page) >= int(last_page):
                return
            params = {
                **config.PROVIDERS_API_PARAMS,
                "per_page": config.PROVIDERS_API_PAGE_SIZE,
                "page": int(current_page) + 1,
            }
            url = config.PROVIDERS_API_URL
            continue

        # Fallback for APIs without pagination metadata.
        if params is None or len(items) < config.PROVIDERS_API_PAGE_SIZE:
            return
        params = {**params, "page": params["page"] + 1}

    raise RuntimeError("Providers API pagination did not finish (page limit reached)")


async def _index_page(es: Any, page: list[dict[str, Any]], run_id: str, stats: SyncStats) -> None:
    actions: list[dict[str, Any]] = []
    for item in page:
        stats.fetched += 1
        try:
            doc = build_provider_doc(item)
        except Exception as exc:
            stats.skipped += 1
            stats.rejection_reasons["normalization_exception"] += 1
            if stats.rejection_samples < 5:
                stats.rejection_samples += 1
                log_failure(
                    log,
                    "provider normalization raised an exception",
                    exc,
                    record_number=stats.fetched,
                    level=logging.WARNING,
                )
            continue

        if doc is None:
            stats.skipped += 1
            reason = _rejection_reason(item)
            stats.rejection_reasons[reason] += 1
            _log_rejection_sample(stats, item, reason)
            continue

        provider_id = str(doc.get("provider_id") or "").strip()
        if not provider_id:
            stats.skipped += 1
            stats.rejection_reasons["missing_provider_id"] += 1
            continue
        doc["provider_id"] = provider_id

        actions.append(
            {
                "_op_type": "index",
                "_index": config.ES_PROVIDERS_INDEX,
                "_id": provider_id,
                "_source": {**doc, "sync_run": run_id},
            }
        )

    if not actions:
        return

    ok, errors = await async_bulk(
        es,
        actions,
        chunk_size=25,
        request_timeout=60,
        max_retries=3,
        raise_on_error=False,
        raise_on_exception=False,
    )
    stats.indexed += ok

    # Record error types only; never log provider documents.
    for error_entry in errors:
        detail = next(iter(error_entry.values()), {}) if isinstance(error_entry, dict) else {}
        error = detail.get("error") if isinstance(detail, dict) else None
        if isinstance(error, dict):
            error_type = error.get("type", "unknown")
        elif isinstance(error, str):
            error_type = error
        else:
            error_type = "unknown"
        stats.bulk_errors[error_type] += 1


async def _remove_stale(es: Any, run_id: str, indexed: int) -> tuple[int, bool]:
    """Delete documents this complete run did not touch, with a safety ratio."""
    stale_query = {"bool": {"must_not": [{"term": {"sync_run": run_id}}]}}
    stale_response = await es.count(index=config.ES_PROVIDERS_INDEX, query=stale_query)
    stale = int(stale_response.get("count", 0))
    total = indexed + stale

    if stale and total > 0 and indexed / total < config.PROVIDER_SYNC_MIN_RATIO:
        log.error(
            "Sync would remove %d of %d provider documents; removal skipped because minimum sync ratio was not met",
            stale,
            total,
        )
        return 0, False
    if not stale:
        return 0, True

    result = await es.delete_by_query(
        index=config.ES_PROVIDERS_INDEX,
        query=stale_query,
        conflicts="proceed",
        refresh=True,
    )
    removed = int(result.get("deleted", 0))
    log.info("Removed %d stale provider documents", removed)
    if result.get("failures"):
        log.error("Stale-provider deletion reported failures; inspect Elasticsearch task results")
        return removed, False
    return removed, True


async def sync_providers(http: httpx.AsyncClient | None = None, es: Any = None) -> bool:
    """Sync backend providers into Elasticsearch safely.

    Returns True when the sync completes successfully; False when indexing is
    incomplete or stale deletion is unsafe.
    """
    if not (config.ES_URL and config.PROVIDERS_API_URL and config.PROVIDERS_API_TOKEN):
        raise RuntimeError("Set ES_URL, PROVIDERS_API_URL and PROVIDERS_API_TOKEN")

    run_id = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
    own_http = http is None
    own_es = es is None

    http = http or httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5, read=config.PROVIDERS_API_TIMEOUT, write=10, pool=5)
    )
    es = es or make_es_client(write=True)
    stats = SyncStats()
    complete = False

    try:
        log.info("Provider sync starting: run_id=%s index=%s", run_id, config.ES_PROVIDERS_INDEX)
        log.info("Sync API key ID from environment=%s", safe_api_key_id(config.ES_SYNC_API_KEY))
        log.info("ES_URL=%s", sanitize_url(config.ES_URL))
        log.info("ES_SYNC_API_KEY configured=%s", bool(config.ES_SYNC_API_KEY))
        log.info("ES_API_KEY configured=%s", bool(config.ES_API_KEY))

        # Confirm the existing Elasticsearch credentials and permissions.
        identity = await es.security.authenticate()
        log.info(
            "Elasticsearch identity: username=%s api_key=%s",
            identity.get("username"),
            identity.get("api_key", {}).get("name"),
        )

        await ensure_providers_index(es)
        await es.indices.put_mapping(
            index=config.ES_PROVIDERS_INDEX,
            body={"properties": {"sync_run": {"type": "keyword"}}},
        )

        async for page in fetch_provider_pages(http):
            log.info("Provider sync received page with %d records", len(page))
            await _index_page(es, page, run_id, stats)

        log.info(
            "Provider processing totals before refresh: fetched=%d indexed=%d skipped=%d bulk_errors=%d",
            stats.fetched,
            stats.indexed,
            stats.skipped,
            sum(stats.bulk_errors.values()),
        )
        if stats.rejection_reasons:
            log.warning("Provider rejection summary: %s", dict(stats.rejection_reasons))

        await es.indices.refresh(index=config.ES_PROVIDERS_INDEX)

        # Never delete existing documents if indexing failed or produced no usable providers.
        if stats.indexed == 0:
            log.error(
                "Provider sync produced no usable indexed providers; nothing was removed. "
                "fetched=%d skipped=%d rejection_reasons=%s",
                stats.fetched,
                stats.skipped,
                dict(stats.rejection_reasons),
            )
        elif stats.bulk_errors:
            log.error("Bulk indexing errors by type=%s; stale-provider removal skipped", dict(stats.bulk_errors))
        else:
            stats.removed, complete = await _remove_stale(es, run_id, stats.indexed)

        log.info(
            "Provider sync run=%s fetched=%d indexed=%d skipped=%d bulk_errors=%d removed=%d complete=%s",
            run_id,
            stats.fetched,
            stats.indexed,
            stats.skipped,
            sum(stats.bulk_errors.values()),
            stats.removed,
            complete,
        )
        return complete
    finally:
        if own_http:
            await http.aclose()
        if own_es:
            await es.close()
