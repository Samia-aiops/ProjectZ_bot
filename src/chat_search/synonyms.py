"""Backward-compatible view over the structured concept catalogue.

Callers that only need the old "canonical specialty -> patient wording" view
keep importing this module. The data itself lives in ``concepts.json`` and is
loaded by :mod:`chat_search.concepts`; this module compiles the subset that
routing, retrieval expansion, spelling and the provider analyzer expect.

The legacy JSON shape is still accepted by the loader, and ``SYNONYMS_JSON`` /
``SYNONYMS_FILE`` still override the source, so existing deployments do not
need to migrate in lockstep with the service.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any

from . import concepts, config

DEFAULT_SYNONYMS: dict[str, Any] = concepts.DEFAULT_PAYLOAD


@dataclass(frozen=True)
class Catalog:
    groups: dict[str, tuple[str, ...]]
    phrase_to_group: dict[str, str]
    protected: frozenset[str]
    domain: frozenset[str]
    signature: str
    pattern: re.Pattern[str] | None
    rewrites: dict[str, str]
    rewrite_pattern: re.Pattern[str] | None


def _from_catalogue(catalogue: concepts.ConceptCatalogue) -> Catalog:
    phrase_to_group = {
        phrase: concept.canonical
        for phrase, concept in catalogue.phrase_to_concept.items()
        if concept.category in concepts.SEARCH_CATEGORIES
    }
    if phrase_to_group:
        alternation = "|".join(re.escape(phrase) for phrase in sorted(phrase_to_group, key=len, reverse=True))
        pattern: re.Pattern[str] | None = re.compile(rf"(?<![a-z0-9])(?:{alternation})(?![a-z0-9])")
    else:
        pattern = None
    return Catalog(
        groups=dict(catalogue.groups),
        phrase_to_group=phrase_to_group,
        protected=catalogue.protected,
        domain=catalogue.domain,
        signature=catalogue.signature,
        pattern=pattern,
        rewrites=dict(catalogue.rewrites),
        rewrite_pattern=catalogue.rewrite_pattern,
    )


def _build(payload: Any, signature: str) -> Catalog:
    return _from_catalogue(concepts.build_catalogue(payload, signature))


DISABLED = _from_catalogue(concepts.DISABLED)

_lock = threading.Lock()
_catalog: Catalog | None = None


def get_catalog(force: bool = False) -> Catalog:
    """Return the current catalogue, reloading the file when it has changed."""
    global _catalog
    if not config.SYNONYMS_ENABLED:
        return DISABLED
    catalogue = concepts.get_catalogue(force=force)
    current = _catalog
    if current is not None and current.signature == catalogue.signature:
        return current
    with _lock:
        if _catalog is None or _catalog.signature != catalogue.signature:
            _catalog = _from_catalogue(catalogue)
        return _catalog


def synonym_terms(text: str, catalog: Catalog | None = None) -> list[str]:
    """Canonical terms found in ``text`` (at most ``SYNONYM_MAX_EXPANSIONS``)."""
    catalogue = catalog or get_catalog()
    if catalogue.pattern is None or not text:
        return []

    normalized = concepts.normalize_text(text)
    found: list[str] = []
    seen: set[str] = set()
    for match in catalogue.pattern.finditer(normalized):
        canonical = catalogue.phrase_to_group[match.group(0)]
        if canonical in seen:
            continue
        seen.add(canonical)
        if match.group(0) != canonical:
            found.append(canonical)
        if len(found) >= max(config.SYNONYM_MAX_EXPANSIONS, 1):
            break
    return found


def phrase_words(text: str, catalog: Catalog | None = None) -> frozenset[str]:
    """Words belonging to a catalogue phrase found in ``text``.

    Provider search removes request words ("best", "near"), but a generic
    provider noun is part of a phrase when the catalogue has one ("heart
    doctor", "child specialist"). Keeping those words lets Elasticsearch apply
    the synonym graph instead of receiving a broken phrase.
    """
    catalogue = catalog or get_catalog()
    if catalogue.pattern is None or not text:
        return frozenset()
    normalized = concepts.normalize_text(text)
    return frozenset(
        word
        for match in catalogue.pattern.finditer(normalized)
        for word in match.group(0).split()
    )


def is_specialty_phrase(text: str) -> bool:
    """True when ``text`` names a catalogue specialty, canonical or variant.

    Unlike ``synonym_terms`` (which returns only what still needs adding), this
    answers "is this a specialty at all", e.g. to keep a provider noun from
    being read as a language in "urdu speaking dermatologist".
    """
    catalogue = get_catalog()
    if catalogue.pattern is None or not text:
        return False
    normalized = concepts.normalize_text(text)
    known = catalogue.groups.keys() | catalogue.phrase_to_group.keys()
    return normalized in known or any(word in known for word in normalized.split())


def rewrite_text(text: str, catalog: Catalog | None = None) -> str:
    """Replace patient wording that the ranker scores badly with the portal's term.

    Only for search text: the caller's message still reaches the LLM unchanged.
    """
    catalogue = catalog or get_catalog()
    if catalogue.rewrite_pattern is None or not text:
        return text
    normalized = concepts.normalize_text(text)
    rewritten = catalogue.rewrite_pattern.sub(lambda match: catalogue.rewrites[match.group(0)], normalized)
    return rewritten if rewritten != normalized else text


def expand_synonyms(text: str) -> str:
    """Prefix canonical terms so local retrieval sees the document wording."""
    catalogue = get_catalog()
    text = rewrite_text(text, catalogue)
    terms = synonym_terms(text, catalogue)
    return f"{' '.join(terms)} {text}" if terms else text


def domain_words() -> frozenset[str]:
    return get_catalog().domain


def protected_words() -> frozenset[str]:
    return get_catalog().protected


def reset_synonyms() -> None:
    global _catalog
    with _lock:
        _catalog = None
    concepts.reset_concepts()


def warm_synonyms() -> Catalog:
    return get_catalog(force=True)
