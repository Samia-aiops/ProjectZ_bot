"""Query Understanding Engine for the provider directory.

Four stages, in order:

1. :mod:`chat_search.que.extract` — spaCy phrase/NER extraction of specialty,
   language, location and intent.
2. :mod:`chat_search.que.languages` — normalization to the specialty and
   language catalogues.
3. :mod:`chat_search.que.configure` — stopwords, boosts and pass thresholds.
4. :mod:`chat_search.que.retrieve` — deterministic Elasticsearch query with the
   JWT tenant filter and the public-field allowlist.
"""

from .extract import (
    language_provider_request,
    provider_languages,
    reset_que,
    understand_provider_query,
)
from .retrieve import build_query, language_filters, search_providers
from .types import Entity, ProviderQuery

__all__ = [
    "Entity",
    "ProviderQuery",
    "build_query",
    "language_filters",
    "language_provider_request",
    "provider_languages",
    "reset_que",
    "search_providers",
    "understand_provider_query",
]
