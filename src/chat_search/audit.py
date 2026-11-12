"""One audit record per chat request: who asked, what was asked, what was answered.

This is the only log line that carries the caller's words and the model's
output, so it goes to the dedicated ``chat_search.audit`` logger. Route that
logger to a sink with its own access control, encryption and retention: the
text is patient-facing content and may be PHI. The shared formatter still
redacts configured secrets and credential-shaped text, and both text fields are
bounded by ``AUDIT_MAX_CHARS``.

The request line in ``observability`` keeps carrying metadata only.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

from . import config
from .auth import Principal
from .logging_setup import redact
from .observability import request_ctx

log = logging.getLogger("chat_search.audit")
# A quieter root level must not silently drop the audit trail.
log.setLevel(logging.INFO)

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_MAX_SOURCES = 20
_MAX_PROVIDERS = 10


def _text(value: str) -> str | None:
    """Redact first, then trim: a truncated secret could otherwise survive."""
    if not config.AUDIT_LOG_CONTENT:
        return None
    limit = max(config.AUDIT_MAX_CHARS, 0)
    return _CONTROL_RE.sub(" ", redact(value))[:limit]


def _documents(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents = []
    for source in sources[:_MAX_SOURCES]:
        if not isinstance(source, dict) or not isinstance(source.get("document"), str):
            continue
        documents.append(
            {
                "document": _CONTROL_RE.sub(" ", source["document"])[:160],
                "page": source.get("page"),
            }
        )
    return documents


def _providers(providers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Directory answers cite providers, not documents; keep only public labels."""
    listed = []
    for provider in providers[:_MAX_PROVIDERS]:
        if not isinstance(provider, dict):
            continue
        entry = {key: provider[key] for key in ("name", "specialty", "city") if provider.get(key)}
        if entry:
            listed.append(entry)
    return listed


def audit_chat(
    principal: Principal,
    *,
    question: str,
    answer: str,
    node: str,
    session_id: str | None,
    mode: str,
    sources: list[dict[str, Any]] | None = None,
    providers: list[dict[str, Any]] | None = None,
    cached: bool = False,
    outcome: str = "ok",
    error: str | None = None,
    latency_ms: int | None = None,
    history_turns: int | None = None,
) -> None:
    """Emit one structured audit record; a logging failure never affects the response."""
    if not config.AUDIT_LOG_ENABLED:
        return
    ctx = request_ctx.get() or {}
    record: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "event": "chat",
        "request_id": ctx.get("request_id"),
        "tenant_id": principal.tenant_id,
        "role": principal.role,
        "subject": principal.subject,
        # Exactly one of these identifies the caller for the role.
        "patient_id": principal.patient_id,
        "provider_id": principal.provider_id,
        "session_id": session_id,
        "mode": mode,
        "node": node,
        "input": _text(question),
        "output": _text(answer),
        "sources": _documents(sources or []),
        "cached": cached,
        "outcome": outcome,
    }
    if listed := _providers(providers or []):
        record["providers"] = listed
    if error:
        record["error"] = error
    if history_turns is not None:
        record["history_turns"] = history_turns
    if latency_ms is not None:
        record["latency_ms"] = latency_ms
    try:
        log.info("%s", json.dumps(record, ensure_ascii=False, default=str, separators=(",", ":")))
    except Exception:
        pass  # audit must never break a response


def audit_provider_sql(
    principal: Principal,
    *,
    question: str,
    sql: str | None = None,
    result: dict[str, Any] | None = None,
    outcome: str = "ok",
    error: str | None = None,
    latency_ms: int | None = None,
) -> None:
    """One audit record per provider Text-to-SQL query.

    Gated by ``AUDIT_LOG_ENABLED`` and ``PROVIDER_SQL_AUDIT``. The question and
    the rewritten SQL are content (PHI-adjacent) and only logged when
    ``AUDIT_LOG_CONTENT`` is on; result rows are never logged, only their count
    and backend. A logging failure never affects the response.
    """
    if not config.AUDIT_LOG_ENABLED or not config.PROVIDER_SQL_AUDIT:
        return
    ctx = request_ctx.get() or {}
    record: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "event": "provider_sql",
        "request_id": ctx.get("request_id"),
        "tenant_id": principal.tenant_id,
        "role": principal.role,
        "subject": principal.subject,
        "provider_id": principal.provider_id,
        "input": _text(question),
        "sql": _text(sql) if sql is not None else None,
        "source": (result or {}).get("source"),
        "row_count": (result or {}).get("row_count"),
        "outcome": outcome,
    }
    if error:
        record["error"] = error
    if latency_ms is not None:
        record["latency_ms"] = latency_ms
    try:
        log.info("%s", json.dumps(record, ensure_ascii=False, default=str, separators=(",", ":")))
    except Exception:
        pass  # audit must never break a response
