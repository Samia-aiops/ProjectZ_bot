"""Stages 1-2: NLP extraction and catalogue normalization for the directory.

A provider request is analyzed once with spaCy: catalogue phrases supply the
specialty and language mentions, the model's NER supplies locations, and token
lemmas decide the intent. Every mention is then mapped to the catalogue
canonical (``cardiology``, ``urdu``, ``Islamabad``) before retrieval sees it.

Without spaCy (or without the English model) the extractor degrades to the
concept catalogue's own matcher; locations are then left in the free-text
terms, which is how the directory searched before the QUE existed.
"""

from __future__ import annotations

import logging
import re
import threading
from functools import lru_cache
from typing import Any

from .. import config
from ..concepts import ConceptCatalogue, get_catalogue, iter_matches, normalize_text
from .configure import retrieval_settings
from .languages import language_aliases, language_key
from .types import Entity, ProviderQuery

log = logging.getLogger(__name__)

_LOCATION_LABELS = frozenset({"GPE", "LOC", "FAC"})
_LANGUAGE_CUES = frozenset({"speak", "fluent"})
_LANGUAGE_NOUN = "language"
_COPULAS = frozenset({"be"})
_REQUEST_CUES = frozenset(
    {"recommend", "suggest", "find", "search", "need", "want", "look", "show", "list", "see", "get", "help", "have", "any"}
)
_RATING_CUES = frozenset({"best", "top", "good", "highest", "rated", "rating"})
_SKIP_POS = frozenset({"DET", "ADP", "SCONJ", "PART", "CCONJ", "AUX", "PUNCT", "SPACE", "NUM"})
_ACCEPT_POS = frozenset({"PROPN", "NOUN", "ADJ"})

_lock = threading.Lock()
_analysis_lock = threading.Lock()
_nlp_state: tuple[Any, Any, str] | None = None
_nlp_failed = False


def _load_nlp(catalogue: ConceptCatalogue) -> tuple[Any, Any, str] | None:
    global _nlp_state, _nlp_failed
    signature = f"{catalogue.signature}|languages:{len(language_aliases())}"
    if _nlp_state is not None and _nlp_state[2] == signature:
        return _nlp_state
    if _nlp_failed:
        return None
    with _lock:
        if _nlp_state is not None and _nlp_state[2] == signature:
            return _nlp_state
        if _nlp_failed:
            return None
        try:
            import spacy
            from spacy.matcher import PhraseMatcher

            try:
                nlp = spacy.load(config.QUE_SPACY_MODEL)
            except Exception as exc:
                log.warning("QUE model %s unavailable (%s); using catalogue matching", config.QUE_SPACY_MODEL, type(exc).__name__)
                nlp = spacy.blank("en")
            matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
            for concept in catalogue.concepts:
                if concept.category != "specialty":
                    continue
                patterns = [nlp.make_doc(phrase) for phrase in concept.phrases if phrase]
                if patterns:
                    matcher.add(f"spec:{concept.id}", patterns)
            for alias, key in language_aliases():
                pattern = nlp.make_doc(alias)
                if len(pattern):
                    matcher.add(f"lang:{key}", [pattern])
            _nlp_state = (nlp, matcher, signature)
            log.info("QUE ready (model=%s, concepts=%d, languages=%d)", config.QUE_SPACY_MODEL, len(catalogue.concepts), len(language_aliases()))
        except Exception as exc:
            _nlp_failed = True
            log.warning("spaCy unavailable (%s); QUE uses catalogue matching", type(exc).__name__)
        return _nlp_state


def _dedupe(entities: list[Entity]) -> list[Entity]:
    """Prefer the longest mention at each position and drop overlaps."""
    entities.sort(key=lambda entity: (entity.start, -(entity.end - entity.start)))
    chosen: list[Entity] = []
    last_end = -1
    for entity in entities:
        if entity.start < last_end:
            continue
        chosen.append(entity)
        last_end = entity.end
    return chosen


def _catalogue_entities(doc: Any, matcher: Any, catalogue: ConceptCatalogue) -> list[Entity]:
    found: list[Entity] = []
    for match_id, start, end in matcher(doc):
        key = doc.vocab.strings[match_id]
        text = doc[start:end].text
        span_start = doc[start].idx
        span_end = doc[end - 1].idx + len(doc[end - 1].text)
        if key.startswith("spec:"):
            concept = catalogue.by_id.get(key[5:])
            if concept is not None:
                found.append(
                    Entity(text=text, canonical=concept.canonical, label="specialty", source="catalogue", start=span_start, end=span_end)
                )
        else:
            found.append(
                Entity(
                    text=text,
                    canonical=language_key(text) or normalize_text(text),
                    label="language",
                    source="catalogue",
                    start=span_start,
                    end=span_end,
                )
            )
    return found


def _is_known_word(text: str, catalogue: ConceptCatalogue) -> bool:
    normalized = normalize_text(text)
    return (
        not normalized
        or normalized in catalogue.phrase_to_concept
        or language_key(normalized) is not None
        or normalized in retrieval_settings().stopwords
        or normalized.split()[0] in catalogue.provider_nouns
    )


def _location_entities(doc: Any, known: list[Entity], catalogue: ConceptCatalogue) -> list[Entity]:
    spans = [(entity.start, entity.end) for entity in known]
    found: list[Entity] = []
    for ent in doc.ents:
        if ent.label_ not in _LOCATION_LABELS:
            continue
        if any(ent.start_char < end and start < ent.end_char for start, end in spans):
            continue
        if _is_known_word(ent.text, catalogue):
            continue
        city = " ".join(word[:1].upper() + word[1:] for word in ent.text.split())
        found.append(
            Entity(text=ent.text, canonical=city, label="location", source="ner", start=ent.start_char, end=ent.end_char)
        )
    return found


def _candidate(token: Any, catalogue: ConceptCatalogue) -> Any | None:
    if token.is_space or token.is_punct or token.pos_ not in _ACCEPT_POS:
        return None
    lowered = token.text.lower()
    lemma = (token.lemma_ or lowered).lower()
    settings = retrieval_settings()
    if lowered in settings.stopwords or lemma in settings.stopwords:
        return None
    if lowered in catalogue.provider_nouns or lemma in catalogue.provider_nouns:
        return None
    if lowered in catalogue.phrase_to_concept:
        return None
    return token


def _next_candidate(doc: Any, index: int, catalogue: ConceptCatalogue) -> Any | None:
    for token in doc[index + 1 : index + 4]:
        if token.is_space:
            continue
        if token.is_punct:
            if token.text == ":":
                continue
            return None
        if token.pos_ in _SKIP_POS:
            continue
        return _candidate(token, catalogue)
    return None


def _previous_candidate(doc: Any, index: int, catalogue: ConceptCatalogue) -> Any | None:
    for token in reversed(doc[max(0, index - 2) : index]):
        if token.is_space or token.is_punct:
            continue
        if token.pos_ in _SKIP_POS:
            continue
        return _candidate(token, catalogue)
    return None


def _cued_languages(doc: Any, catalogue: ConceptCatalogue) -> list[Entity]:
    """Languages named around a cue, including ones the catalogue does not have."""
    found: list[Entity] = []
    for index, token in enumerate(doc):
        lemma = (token.lemma_ or token.text).lower()
        candidate: Any | None = None
        if lemma in _LANGUAGE_CUES:
            if token.tag_ == "VBG" or token.text.lower().endswith("speaking"):
                candidate = _previous_candidate(doc, index, catalogue) or _next_candidate(doc, index, catalogue)
            else:
                candidate = _next_candidate(doc, index, catalogue)
        elif lemma == _LANGUAGE_NOUN:
            following = doc[index + 1] if index + 1 < len(doc) else None
            if following is not None and (following.text == ":" or (following.lemma_ or "").lower() in _COPULAS):
                candidate = _next_candidate(doc, index, catalogue)
        if candidate is None:
            continue
        text = candidate.text
        canonical = language_key(text) or normalize_text(text)
        found.append(
            Entity(text=text, canonical=canonical, label="language", source="cue", start=candidate.idx, end=candidate.idx + len(text))
        )
    return found


def _inside(index: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= index < end for start, end in spans)


def _terms(doc: Any, entities: list[Entity], catalogue: ConceptCatalogue) -> str:
    settings = retrieval_settings()
    protected = [(entity.start, entity.end) for entity in entities if entity.label == "specialty"]
    excluded = [(entity.start, entity.end) for entity in entities if entity.label in {"language", "location"}]
    words: list[str] = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        if _inside(token.idx, excluded):
            continue
        lemma = (token.lemma_ or token.text).lower()
        if (lemma in settings.stopwords or token.text.lower() in settings.stopwords) and not _inside(token.idx, protected):
            continue
        words.append(token.text.lower())
    return " ".join(words)[:200]


def _intent(forms: set[str], entities: list[Entity], catalogue: ConceptCatalogue) -> str:
    if forms & catalogue.appointment_actions and forms & catalogue.appointment_objects:
        return "appointment"
    if any(entity.label == "specialty" for entity in entities):
        return "provider_search"
    provider = bool(forms & catalogue.provider_nouns)
    request = bool(forms & _REQUEST_CUES)
    rating = bool(forms & _RATING_CUES)
    city = any(entity.label == "location" for entity in entities)
    if provider and (request or rating or city):
        return "provider_search"
    languages = any(entity.label == "language" for entity in entities)
    cued = bool(forms & (_LANGUAGE_CUES | {_LANGUAGE_NOUN}))
    if languages and (request or cued or provider):
        return "provider_search"
    return "faq"


def _language_request(doc: Any, entities: list[Entity], catalogue: ConceptCatalogue) -> bool:
    for entity in entities:
        if entity.label != "language":
            continue
        after = [token for token in doc if token.idx >= entity.end]
        for token in after[:3]:
            if token.is_space or token.is_punct:
                continue
            lemma = (token.lemma_ or token.text).lower()
            if lemma in _LANGUAGE_CUES or lemma in _COPULAS or token.tag_ == "VBG":
                continue
            return token.text.lower() in catalogue.provider_nouns or lemma in catalogue.provider_nouns
    return False


def _forms(doc: Any) -> set[str]:
    forms: set[str] = set()
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        forms.add(token.text.lower())
        lemma = (token.lemma_ or "").lower()
        if lemma:
            forms.add(lemma)
    return forms


def _analyse(text: str, catalogue: ConceptCatalogue) -> ProviderQuery:
    state = _load_nlp(catalogue)
    if state is None:
        return _fallback(text, catalogue)
    nlp, matcher, _ = state
    with _analysis_lock:
        doc = nlp(text)
        entities = _catalogue_entities(doc, matcher, catalogue)
    entities = _dedupe([*entities, *_location_entities(doc, entities, catalogue), *_cued_languages(doc, catalogue)])
    specialties = tuple(dict.fromkeys(entity.canonical for entity in entities if entity.label == "specialty"))
    languages = tuple(dict.fromkeys(entity.canonical for entity in entities if entity.label == "language"))
    cities = tuple(dict.fromkeys(entity.canonical for entity in entities if entity.label == "location"))
    return ProviderQuery(
        original=text,
        normalized=normalize_text(text),
        intent=_intent(_forms(doc), entities, catalogue),
        specialties=specialties,
        languages=languages,
        cities=cities,
        terms=_terms(doc, entities, catalogue),
        entities=tuple(entities),
        language_request=_language_request(doc, entities, catalogue),
    )


def _fallback(text: str, catalogue: ConceptCatalogue) -> ProviderQuery:
    """No spaCy: catalogue matching and token membership, no location NER."""
    normalized = normalize_text(text)
    entities = [
        Entity(text=normalized[start:end], canonical=concept.canonical, label="specialty", start=start, end=end)
        for start, end, concept in iter_matches(normalized, catalogue)
        if concept.category == "specialty"
    ]
    for alias, key in language_aliases():
        for match in re.finditer(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized):
            entities.append(
                Entity(text=match.group(0), canonical=key, label="language", start=match.start(), end=match.end())
            )
    entities = _dedupe(entities)
    entities.sort(key=lambda entity: entity.start)
    settings = retrieval_settings()
    protected = [(entity.start, entity.end) for entity in entities if entity.label == "specialty"]
    excluded = [(entity.start, entity.end) for entity in entities if entity.label in {"language", "location"}]
    words: list[str] = []
    for match in re.finditer(r"[a-z0-9][a-z0-9'+-]*", normalized):
        if _inside(match.start(), excluded):
            continue
        if match.group(0) in settings.stopwords and not _inside(match.start(), protected):
            continue
        words.append(match.group(0))
    forms = set(normalized.split())
    return ProviderQuery(
        original=text,
        normalized=normalized,
        intent=_intent(forms, entities, catalogue),
        specialties=tuple(dict.fromkeys(entity.canonical for entity in entities if entity.label == "specialty")),
        languages=tuple(dict.fromkeys(entity.canonical for entity in entities if entity.label == "language")),
        terms=" ".join(words)[:200],
        entities=tuple(entities),
        language_request=False,
    )


@lru_cache(maxsize=256)
def _understand_cached(text: str, signature: str) -> ProviderQuery:
    catalogue = get_catalogue()
    try:
        return _analyse(text, catalogue)
    except Exception as exc:
        log.warning("QUE extraction failed (%s); using catalogue fallback", type(exc).__name__)
        return _fallback(text, catalogue)


def understand_provider_query(text: str) -> ProviderQuery:
    """Run stages 1-2 over one request and return the structured query."""
    catalogue = get_catalogue()
    return _understand_cached(text, catalogue.signature)


def provider_languages(text: str) -> list[str]:
    """Canonical languages named in ``text`` (used by routing and understanding)."""
    return list(understand_provider_query(text).languages)


def language_provider_request(text: str) -> bool:
    """True when a language mention is attached to a provider noun."""
    return understand_provider_query(text).language_request


def reset_que() -> None:
    """Drop the NLP cache (tests and a catalogue swap)."""
    global _nlp_state, _nlp_failed
    with _lock:
        _nlp_state = None
        _nlp_failed = False
    _understand_cached.cache_clear()
