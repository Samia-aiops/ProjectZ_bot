"""Provider directory pipeline (Elasticsearch, no LLM)."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from .. import config
from ..auth import Principal
from ..guardrails import normalize, rule_guard
from ..messages import PROVIDER_DENIED_MSG, PROVIDER_PERMISSION_MSG, REFUSAL, SEARCH_UNAVAILABLE_MSG
from ..observability import metric
from ..providers.search import format_providers
from ..que import provider_languages, search_providers, understand_provider_query
from ..runtime import get_runtime
from ..state import Turn
from ..understanding import current_understanding, understand
from .events import Event, done_event, single_reply, token_event


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _has_topic(text: str) -> bool:
    found = current_understanding()
    if found is None or found.original != text:
        found = understand(text, medical=False)
    return bool(found.specialties or found.provider_nouns or found.canonical_terms)


def _with_history_topic(text: str, history: list[Turn]) -> str:
    """A follow-up ("finish the search first") names its subject only through history."""
    if _has_topic(text):
        return text
    prior = next(
        (turn.text for turn in reversed(history) if turn.role == "user" and _has_topic(turn.text)),
        None,
    )
    if prior is None:
        return text
    languages = provider_languages(text)
    metric(provider_topic="history")
    return f"{prior} who speaks {languages[0]}" if languages else prior


async def provider_search_pipeline(
    principal: Principal,
    message: str,
    history: list[Turn] | None = None,
) -> AsyncIterator[Event]:
    metric(cache="skip")
    # The directory is a patient feature, so no other role may reach it. The
    # gate runs before the ES check: a provider asking explicitly must get
    # "not available", never a hint that the directory itself is down.
    if principal.role != "patient" or (
        config.PROVIDER_SEARCH_PERMISSION and config.PROVIDER_SEARCH_PERMISSION not in principal.permissions
    ):
        metric(provider_search="denied")
        message = PROVIDER_DENIED_MSG if principal.role != "patient" else PROVIDER_PERMISSION_MSG
        async for event in single_reply(message):
            yield event
        return

    runtime = get_runtime()
    if runtime.es is None:
        async for event in single_reply(SEARCH_UNAVAILABLE_MSG):
            yield event
        return

    started = time.perf_counter()
    question = normalize(message)
    # Every rule applies here as well: only public directory fields are returned,
    # but a message asking for someone else's records must still be refused
    # rather than turned into a search.
    category = rule_guard(question)
    if category:
        metric(guard=category)
        async for event in single_reply(REFUSAL[category]):
            yield event
        return

    # A follow-up ("finish the search first") is guarded on its own words but
    # searched with the earlier question that named the specialty.
    text = _with_history_topic(question, history or [])
    # QUE stages 1-2: extract specialties/languages/cities and normalize them
    # against the catalogue. Qualifications never come from this structure.
    structured = understand_provider_query(text)
    metric(expand_ms=_ms(started))
    # No LLM prompt is involved here, so stage 3 (the injection classifier,
    # hundreds of ms of CPU) is skipped; the deterministic rules above still
    # apply. PII detection runs in the chat layer only when a turn may be stored.
    es_started = time.perf_counter()
    try:
        hits, relaxed, language = await search_providers(principal, structured)
    except Exception as exc:
        # A directory outage is a tool failure, not an answer: report search as
        # unavailable instead of letting it become a generic "something went wrong".
        metric(provider_search="error", es_error=type(exc).__name__, es_ms=_ms(es_started))
        async for event in single_reply(SEARCH_UNAVAILABLE_MSG):
            yield event
        return
    metric(
        es_ms=_ms(es_started),
        guard="ok",
        es_hits=len(hits),
        es_relaxed=relaxed,
        es_language_fallback=bool(language),
    )
    # A language-only query ("who speaks Urdu") is a search, not a top-rated list.
    searched = bool(structured.terms or structured.languages or structured.cities)
    answer = format_providers(hits, relaxed, searched, language)
    # Stream line by line: the browser paints the first providers while the rest arrive.
    for chunk in answer.splitlines(keepends=True):
        yield token_event(chunk)
    data: dict = {"providers": hits, "structured_query": structured.as_dict()}
    understanding = current_understanding()
    if understanding is not None:
        data["understanding"] = understanding.as_dict()
    yield done_event([], False, data)
