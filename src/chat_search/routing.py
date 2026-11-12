"""Intent routing between document RAG and provider search.

Routing and the structured understanding share one vocabulary: the regexes and
the appointment/booking rules live in :mod:`chat_search.understanding`, so a
change there cannot silently drift from what the response reports.
"""

from __future__ import annotations

from .que import language_provider_request, provider_languages
from .understanding import (
    FAQ_HINT_RE,
    LANGUAGE_REQUEST_RE,
    PROVIDER_INTENT_RE,
    understand,
)

__all__ = [
    "FAQ_HINT_RE",
    "LANGUAGE_REQUEST_RE",
    "PROVIDER_INTENT_RE",
    "route",
]


def route(text: str, mode: str) -> str:
    """Return 'faq' (document RAG) or 'providers' (Elasticsearch). An explicit mode wins.

    Directory intent is detected the same way for every role: the provider
    pipeline refuses non-patients, so a provider asking for a doctor gets the
    feature-unavailable reply instead of falling into a documents lookup.
    """
    if mode in ("faq", "providers"):
        return mode

    understanding = understand(text, medical=False)
    if understanding.appointment is not None:
        return "faq"
    if understanding.intent == "provider_search":
        return "providers"

    language_query = language_provider_request(text) or bool(
        LANGUAGE_REQUEST_RE.search(text) and provider_languages(text)
    )
    if not FAQ_HINT_RE.search(text) and (PROVIDER_INTENT_RE.search(text) or language_query):
        return "providers"
    return "faq"
