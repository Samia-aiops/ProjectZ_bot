"""Stage 4: turn the structured query into a deterministic Elasticsearch query.

The tenant filter comes exclusively from the verified JWT, and the response is
restricted to ``PROVIDER_PUBLIC_FIELDS``. Nothing on this path calls an LLM.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

from .. import config
from ..auth import Principal
from ..errors import AppError
from ..messages import BUSY_MSG, SEARCH_UNAVAILABLE_MSG
from ..observability import metric
from ..providers.fields import PROVIDER_PUBLIC_FIELDS
from ..runtime import get_runtime, limit
from .configure import retrieval_settings
from .languages import language_labels
from .types import ProviderQuery


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def language_filters(languages: tuple[str, ...] | list[str]) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    for language in languages:
        labels = language_labels(language)
        if len(labels) == 1:
            filters.append({"match_phrase": {"languages": labels[0]}})
        else:
            filters.append(
                {
                    "bool": {
                        "should": [{"match_phrase": {"languages": label}} for label in labels],
                        "minimum_should_match": 1,
                    }
                }
            )
    return filters


def build_query(query: ProviderQuery, *, tenant_id: str, strict: bool) -> dict[str, Any]:
    """Build the ES body; only the QUE output and the verified tenant id go in."""
    settings = retrieval_settings()
    fuzzy: dict[str, Any] = {}
    if not strict and settings.fuzziness:
        fuzzy = {
            "fuzziness": settings.fuzziness,
            "prefix_length": settings.fuzzy_prefix_length,
            "max_expansions": settings.fuzzy_max_expansions,
        }

    must: list[dict[str, Any]] = []
    if query.terms:
        must.append(
            {
                "match": {
                    "search_text": {
                        "query": query.terms,
                        "operator": "or",
                        # The search analyzer collapses a synonym phrase to one
                        # canonical term, so two concepts must both match; longer
                        # queries keep the configured tolerance for a stray word.
                        "minimum_should_match": settings.strict_msm if strict else settings.relaxed_msm,
                        **fuzzy,
                    }
                }
            }
        )
    if query.cities:
        # The first location is the city constraint; a second GPE in the same
        # request is normally a state/country suffix ("Boston, MA").
        must.append({"match": {"city": {"query": query.cities[0]}}})
    if not must:
        must = [{"match_all": {}}]

    should: list[dict[str, Any]] = []
    if query.terms:
        fields = [
            f"specialty^{_number(settings.specialty_boost)}",
            f"name^{_number(settings.name_boost)}",
            f"conditions_treated^{_number(settings.conditions_boost)}",
        ]
        should.append({"multi_match": {"query": query.terms, "fields": fields, **fuzzy}})
    should.append({"term": {"accepting_new_patients": {"value": True, "boost": settings.accepting_boost}}})

    return {
        "function_score": {
            "query": {
                "bool": {
                    # The language stays required in both passes: a relaxed pass
                    # may loosen the specialty words, never the spoken language.
                    "filter": [
                        {"term": {"active": True}},
                        {"term": {"tenant_id": tenant_id}},
                        *language_filters(query.languages),
                    ],
                    "must": must,
                    "should": should,
                }
            },
            "field_value_factor": {
                "field": "rating",
                "modifier": settings.rating_modifier,
                "missing": settings.rating_missing,
            },
            "boost_mode": "multiply",
        }
    }


async def search_providers(
    principal: Principal, query: ProviderQuery, *, results: int | None = None
) -> tuple[list[dict[str, Any]], bool, str | None]:
    """Strict search first (most terms must match); if empty, a relaxed one.

    -> (hits, relaxed, language_fallback). The tenant comes only from the
    principal; ``query`` never carries access fields.
    """
    runtime = get_runtime()
    if runtime.es is None:
        raise AppError(502, "search_unavailable", SEARCH_UNAVAILABLE_MSG)

    size = config.PROVIDER_RESULTS if results is None else max(1, results)

    async def run(strict: bool, current: ProviderQuery) -> list[dict[str, Any]]:
        async with limit(runtime.sem_rag):
            try:
                result = await asyncio.wait_for(
                    runtime.es.search(
                        index=config.ES_PROVIDERS_INDEX,
                        query=build_query(current, tenant_id=principal.tenant_id, strict=strict),
                        size=size,
                        source=PROVIDER_PUBLIC_FIELDS,
                        track_total_hits=False,
                    ),
                    config.ES_TIMEOUT + 1,
                )
            except TimeoutError:
                raise AppError(504, "timeout", BUSY_MSG) from None
            except Exception as exc:
                metric(es_error=type(exc).__name__)
                raise AppError(502, "search_unavailable", SEARCH_UNAVAILABLE_MSG) from None
        hits: list[dict[str, Any]] = []
        for hit in result["hits"]["hits"]:
            source = hit.get("_source") or {}
            if source.get("name"):
                hits.append({key: source[key] for key in PROVIDER_PUBLIC_FIELDS if key in source})
        return hits

    hits = await run(True, query)
    if hits:
        return hits, False, None
    if query.terms and query.languages:
        # The language is a hard requirement, but the other words are not: when
        # "spanish doctor in newyork" matches nobody because the directory has no
        # Spanish speaker there, list the Spanish speakers it does have and say
        # so, instead of a bare "no match" that hides the reason.
        speakers = await run(True, replace(query, terms="", cities=()))
        if speakers:
            return speakers, False, query.languages[0]
    if not query.terms or query.languages:
        # A language query with no strict hit is answered as "no match": the
        # relaxed pass would otherwise return providers who speak another language.
        return hits, False, None
    return await run(False, query), True, None
