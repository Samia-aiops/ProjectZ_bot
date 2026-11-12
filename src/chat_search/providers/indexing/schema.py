"""Elasticsearch index schema for provider search (write side).

The document lives in the ``providers`` index:

    tenant_id (keyword), provider_id (keyword), active (bool), name, specialty,
    subspecialties[], conditions_treated[], languages[], city, bio, rating (float),
    review_count (int), accepting_new_patients (bool), next_available (date)

Only ``PROVIDER_PUBLIC_FIELDS`` are ever returned to a patient.

Synonyms are applied by Elasticsearch at search time. ``provider_text`` only
normalizes and stems; ``provider_search`` adds a ``synonym_graph`` filter built
from the same catalogue the rest of query understanding uses. The graph is what
turns "heart doctor in Boston" into "cardiology boston" before matching, so a
multi-word patient phrase collapses into one concept instead of diluting
``minimum_should_match``. Rules live in the index settings, which the sync job
updates when the catalogue changes; no application-side expansion is involved.
"""

from __future__ import annotations

import logging
from typing import Any

from elasticsearch import AsyncElasticsearch

from ... import config
from ...concepts import get_catalogue
from ..fields import SPECIALTY_LABELS

log = logging.getLogger(__name__)


def provider_synonym_rules() -> list[str]:
    """One-way ``synonym_graph`` rules: patient wording -> directory wording.

    Every phrasing in a specialty group is rewritten to the canonical specialty
    and, when the directory uses a different label, that label too. The
    canonical term is repeated in the output so a document storing either form
    still matches; ``OB-GYN`` and ``eye doctor`` are the known label cases.
    Only ``specialty`` concepts participate; FAQ topics never become rules.
    """
    catalogue = get_catalogue()
    variants: dict[str, list[str]] = {}
    for phrase, concept in catalogue.phrase_to_concept.items():
        if concept.category != "specialty" or phrase == concept.canonical:
            continue
        variants.setdefault(concept.canonical, []).append(phrase)

    rules: list[str] = []
    for canonical in sorted(catalogue.specialties):
        label = SPECIALTY_LABELS.get(canonical)
        outputs = [canonical] + ([label] if label else [])
        phrases = [canonical, *variants.get(canonical, [])]
        if label:
            phrases.append(label)
        for phrase in dict.fromkeys(phrases):
            if not phrase or "," in phrase or "=>" in phrase:
                continue
            if phrase == canonical and not label:
                continue
            rules.append(f"{phrase} => {', '.join(outputs)}")
    return rules


def _text_field(*, copy_to: bool = True) -> dict[str, Any]:
    field: dict[str, Any] = {
        "type": "text",
        "analyzer": "provider_text",
        "search_analyzer": "provider_search",
    }
    if copy_to:
        field["copy_to"] = "search_text"
    return field


# Fields that carry searchable text and therefore need the search analyzer.
_TEXT_FIELDS = (
    "name",
    "specialty",
    "subspecialties",
    "conditions_treated",
    "languages",
    "city",
    "bio",
    "search_text",
)

PROVIDERS_MAPPING = {
    "dynamic": False,
    "properties": {
        "tenant_id": {"type": "keyword"},
        "provider_id": {"type": "keyword"},
        "active": {"type": "boolean"},
        "name": _text_field(),
        "specialty": _text_field(),
        "subspecialties": _text_field(),
        "conditions_treated": _text_field(),
        "languages": _text_field(),
        "city": _text_field(),
        "bio": _text_field(copy_to=False),  # noisy: searchable, not in search_text
        "search_text": _text_field(copy_to=False),
        "rating": {"type": "float"},
        "review_count": {"type": "integer"},
        "accepting_new_patients": {"type": "boolean"},
        "next_available": {"type": "date"},
        "sync_run": {"type": "keyword"},  # which sync run last wrote the document
    },
}


def providers_settings(rules: list[str] | None = None) -> dict[str, Any]:
    """Index settings with synonyms only on the search-time analyzer."""
    rules = provider_synonym_rules() if rules is None else rules
    search_filters = ["lowercase", "asciifolding"]
    if rules:
        search_filters.append("provider_synonyms")
    search_filters += ["stop", "porter_stem"]

    analysis: dict[str, Any] = {
        # The catalogue normalizes "children's" to "childrens"; stripping the
        # apostrophe before tokenizing makes both spellings hit the same rule.
        "char_filter": {
            "strip_apostrophes": {
                "type": "pattern_replace",
                "pattern": "['\u2019]",
                "replacement": "",
            }
        },
        "analyzer": {
            "provider_text": {
                "type": "custom",
                "char_filter": ["strip_apostrophes"],
                "tokenizer": "standard",
                "filter": ["lowercase", "asciifolding", "stop", "porter_stem"],
            },
            "provider_search": {
                "type": "custom",
                "char_filter": ["strip_apostrophes"],
                "tokenizer": "standard",
                "filter": search_filters,
            },
        },
    }
    if rules:
        analysis["filter"] = {
            "provider_synonyms": {
                "type": "synonym_graph",
                "synonyms": rules,
            }
        }
    return {"analysis": analysis}


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)] if value else []


async def _upgrade_providers_index(client: AsyncElasticsearch, index: str, rules: list[str]) -> None:
    """Move an existing index from index-time synonyms to search-time graph."""
    settings = (await client.indices.get_settings(index=index, flat_settings=True))[index]["settings"]
    search_filters = _as_list(settings.get("index.analysis.analyzer.provider_search.filter"))
    synonyms_active = "provider_synonyms" in search_filters
    installed_rules = _as_list(settings.get("index.analysis.filter.provider_synonyms.synonyms"))
    mapping = (await client.indices.get_mapping(index=index))[index]["mappings"].get("properties", {})
    mapping_ready = all(
        mapping.get(field, {}).get("search_analyzer") == "provider_search" for field in _TEXT_FIELDS
    )

    analysis_ready = ("provider_synonyms" in search_filters) == bool(rules)
    if analysis_ready and synonyms_active and rules:
        analysis_ready = (
            settings.get("index.analysis.filter.provider_synonyms.type") == "synonym_graph"
            and installed_rules == rules
        )
    if analysis_ready and mapping_ready:
        return

    log.info(
        "Updating provider index analysis: search-time synonym_graph rules=%d (had %d)",
        len(rules),
        len(installed_rules),
    )
    await client.indices.close(index=index)
    try:
        await client.indices.put_settings(index=index, settings=providers_settings(rules))
    finally:
        await client.indices.open(index=index)
    await client.indices.put_mapping(index=index, properties=PROVIDERS_MAPPING["properties"])


async def ensure_providers_index(client: AsyncElasticsearch) -> None:
    """Create the index, or upgrade an existing one to the current analysis."""
    index = config.ES_PROVIDERS_INDEX
    rules = provider_synonym_rules()
    if not await client.indices.exists(index=index):
        await client.indices.create(
            index=index,
            settings=providers_settings(rules),
            mappings=PROVIDERS_MAPPING,
        )
        log.info("Created Elasticsearch index %s (provider synonyms=%d)", index, len(rules))
        return
    await _upgrade_providers_index(client, index, rules)
