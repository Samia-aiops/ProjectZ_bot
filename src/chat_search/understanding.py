"""Structured query understanding: normalize, classify intent, map concepts.

The pipeline is deliberately layered so a failure in the most expensive stage
never blocks an answer:

1. :func:`normalize_query` lowercases, removes punctuation and normalizes
   whitespace; optionally it applies the conservative spell corrector.
2. A spaCy ``PhraseMatcher`` over the structured concept catalogue finds
   specialties, FAQ topics and clinical vocabulary. If spaCy is unavailable the
   catalogue's compiled regex is used instead.
3. Explicit appointment rules combine an action ("book", "cancel", "move",
   "view", ...) with an appointment object ("appointment", "visit",
   "consultation", ...). A specialty alone never becomes an appointment.
4. scispaCy (``en_core_sci_sm``) adds clinical entities that are not in the
   catalogue; when a local UMLS database is configured, entities also carry a
   CUI. Missing scispaCy/UMLS data degrades to step 2 with one warning.

The result is :class:`QueryUnderstanding`, a JSON-ready structured object. The
chat pipeline attaches it to the final SSE/JSON ``data`` payload, so callers see
what was understood without parsing the answer.
"""

from __future__ import annotations

import contextvars
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config
from .concepts import (
    MEDICAL_CATEGORIES,
    SEARCH_CATEGORIES,
    Concept,
    ConceptCatalogue,
    get_catalogue,
    iter_matches,
    normalize_text,
    rewrite_text,
)
from .que import language_provider_request, provider_languages

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent vocabulary (kept here so routing and the structured output agree)
# ---------------------------------------------------------------------------

FAQ_HINT_RE = re.compile(
    r"^\s*(how (do|does|can|to|should)|where (do|can)|can i|may i|is there a way)\b|"
    r"\b(policy|policies|guide|manual|onboarding|faq|frequently asked)\b|"
    r"\b(reset|change|update|forgot|recover)\b.{0,24}\b(password|email|profile|account|preferences?)\b|"
    r"\b(insurance|coverage|claims?|billing|invoices?|payments?|refunds?)\b.{0,40}"
    r"\b(policy|policies|covered|cover|work|works|process)\b|"
    # Product and account vocabulary. A question or statement about the product
    # itself ("does DocHyve provide care", "settings lab uploads", "my parent as
    # a member") is answered from the documents, never from the directory.
    r"\b(dochyve|platform|portal|settings?|uploads?|attachments?|health check|"
    r"members?|caregivers?|authori[sz]ed|prox(?:y|ies)|dependents?)\b|"
    r"\bwhat (?:is|are) the (?:limit|maximum|max|size|steps?)\b|"
    # Insurance plans: "plan A", "which plan should I choose" and plan costs are
    # answered from the documents. A directory query names a doctor or specialty,
    # not a plan ("find a doctor on my plan" stays a search).
    r"\b(?:which|what)\s+plans?\b|\binsurance\s+plans?\b|\bplans?\s+(?:[a-d]|[0-9])\b|"
    r"\b(?:choose|pick|select|change|switch)\w*\s+(?:a\s+|my\s+|the\s+)?plan\b|"
    r"\b(deductibles?|co-?pays?|premiums?)\b",
    re.I,
)
_PROVIDER_WORDS = (
    r"doctors?|providers?|physicians?|specialists?|clinicians?|dr\.?|cardiolog\w+|dermatolog\w+|"
    r"pediatric\w*|paediatric\w*|orthop\w+|gyn\w+|neurolog\w+|psychiatr\w+|psycholog\w+|dentist\w*|"
    r"ophthalmolog\w+|urolog\w+|oncolog\w+|endocrin\w+|gastro\w+|ent|general practitioner|gp|therapist\w*"
)
_SPECIALTY_WORDS = (
    r"cardiolog\w+|dermatolog\w+|pediatric\w*|paediatric\w*|orthop\w+|gyn\w+|neurolog\w+|psychiatr\w+|"
    r"psycholog\w+|dentist\w*|ophthalmolog\w+|optometr\w+|urolog\w+|oncolog\w+|endocrin\w+|gastro\w+|"
    r"otolaryngolog\w+|pulmonolog\w+|rheumatolog\w+|nephrolog\w+|hematolog\w+|allerg\w+|"
    r"general practitioner|general medicine|family medicine|internal medicine|geriatric\w+|podiatr\w+|"
    r"chiropract\w+|physical therapy|physiotherap\w+|occupational therapy|speech therapy|"
    r"audiolog\w+|dietetic\w+|fertility|midwif\w+|sports medicine|plastic surgery|neurosurger\w+|"
    r"general surgery|proctolog\w+|vascular surgery|pain management|sleep medicine|urgent care|"
    r"emergency medicine|gp"
)
PROVIDER_INTENT_RE = re.compile(
    rf"\b(recommend\w*|suggest\w*|find\w*|looking for|search\w*|need|want|who (can|should) i see|"
    rf"best|top|good|do you have|is there|are there|any)\b.{{0,60}}\b({_PROVIDER_WORDS})\b|"
    rf"\b(doctors?|specialists?|providers?)\b.{{0,40}}\b(near|in|who speaks?|speaking|available)\b|"
    rf"\b(?:best|top|highest|good)[\s-]*(?:rated|ratings?)\b.{{0,60}}\b({_PROVIDER_WORDS})\b|"
    rf"\b({_PROVIDER_WORDS})\b.{{0,40}}\b(?:best|top|highest|good)[\s-]*(?:rated|ratings?)\b|"
    rf"\b({_SPECIALTY_WORDS})\b",
    re.I,
)
LANGUAGE_REQUEST_RE = re.compile(r"\b(?:speaks?|speaking|fluent|language)\b", re.I)
_REQUEST_CUE_RE = re.compile(
    r"\b(?:recommend\w*|suggest\w*|find\w*|looking for|search\w*|need|want|show\w*|list\w*|"
    r"who (?:can|should) i see|do you have|is there|are there|any)\b",
    re.I,
)
_RATING_CUE_RE = re.compile(
    r"\b(?:best|top|highest|good)[\s-]*(?:rated|ratings?)\b|"
    r"\b(?:rated|ratings?)\b.{0,20}\b(?:best|top|highest|good)\b",
    re.I,
)

# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MedicalConcept:
    text: str
    canonical: str
    category: str
    cui: str | None = None
    source: str = "catalogue"

    def as_dict(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "text": self.text,
            "canonical": self.canonical,
            "category": self.category,
            "source": self.source,
        }
        if self.cui:
            entry["cui"] = self.cui
        return entry


@dataclass(frozen=True)
class AppointmentIntent:
    action: str
    object: str

    def as_dict(self) -> dict[str, str]:
        return {"action": self.action, "object": self.object}


@dataclass(frozen=True)
class QueryUnderstanding:
    original: str
    text: str
    intent: str
    specialties: tuple[str, ...] = ()
    faq_topics: tuple[str, ...] = ()
    medical_concepts: tuple[MedicalConcept, ...] = ()
    provider_nouns: tuple[str, ...] = ()
    appointment: AppointmentIntent | None = None
    languages: tuple[str, ...] = ()
    canonical_terms: tuple[str, ...] = ()
    rewritten: str = ""

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready view attached to the final response."""
        return {
            "original": self.original,
            "normalized": self.text,
            "intent": self.intent,
            "specialties": list(self.specialties),
            "faq_topics": list(self.faq_topics),
            "medical_concepts": [concept.as_dict() for concept in self.medical_concepts],
            "provider_nouns": list(self.provider_nouns),
            "appointment": self.appointment.as_dict() if self.appointment else None,
            "languages": list(self.languages),
            "canonical_terms": list(self.canonical_terms),
        }

    def search_text(self) -> str:
        """Text for local retrieval: canonical concepts first, wording preserved."""
        return f"{' '.join(self.canonical_terms)} {self.rewritten}" if self.canonical_terms else self.rewritten


_understanding_ctx: contextvars.ContextVar[QueryUnderstanding | None] = contextvars.ContextVar(
    "query_understanding", default=None
)


def current_understanding() -> QueryUnderstanding | None:
    return _understanding_ctx.get()


def bind_understanding(understanding: QueryUnderstanding) -> contextvars.Token[QueryUnderstanding | None]:
    return _understanding_ctx.set(understanding)


def unbind_understanding(token: contextvars.Token[QueryUnderstanding | None]) -> None:
    _understanding_ctx.reset(token)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Hit:
    start: int
    end: int
    concept: Concept
    phrase: str


_matcher_lock = threading.Lock()
_matcher: Any = None
_matcher_nlp: Any = None
_matcher_signature: str | None = None
_matcher_failed = False

_sci_lock = threading.Lock()
_sci_nlp: Any = None
_sci_failed = False


def _ensure_matcher(catalogue: ConceptCatalogue) -> tuple[Any, Any] | tuple[None, None]:
    global _matcher, _matcher_nlp, _matcher_signature, _matcher_failed

    if _matcher is not None and _matcher_signature == catalogue.signature:
        return _matcher, _matcher_nlp
    if _matcher_failed:
        return None, None

    with _matcher_lock:
        if _matcher is not None and _matcher_signature == catalogue.signature:
            return _matcher, _matcher_nlp
        if _matcher_failed:
            return None, None
        try:
            import spacy
            from spacy.matcher import PhraseMatcher

            nlp = spacy.blank("en")
            matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
            for concept in catalogue.concepts:
                patterns = [nlp.make_doc(phrase) for phrase in concept.phrases if phrase]
                if patterns:
                    matcher.add(concept.id, patterns)
            _matcher, _matcher_nlp = matcher, nlp
            _matcher_signature = catalogue.signature
            log.info("Phrase matcher ready (concepts=%d)", len(catalogue.concepts))
        except Exception as exc:
            _matcher_failed = True
            log.warning("spaCy PhraseMatcher unavailable (%s); using catalogue regex", type(exc).__name__)
        return _matcher, _matcher_nlp


def _dedupe(hits: list[_Hit]) -> list[_Hit]:
    """Prefer the longest phrase at each position; drop overlapping matches."""
    hits.sort(key=lambda hit: (hit.start, -(hit.end - hit.start)))
    chosen: list[_Hit] = []
    last_end = 0
    for hit in hits:
        if hit.start < last_end:
            continue
        chosen.append(hit)
        last_end = hit.end
    return chosen


def _catalogue_hits(text: str, catalogue: ConceptCatalogue) -> list[_Hit]:
    if not text or not catalogue.phrase_to_concept:
        return []
    matcher, nlp = _ensure_matcher(catalogue)
    if matcher is None:
        return _dedupe(
            [
                _Hit(start=start, end=end, concept=concept, phrase=text[start:end])
                for start, end, concept in iter_matches(text, catalogue)
            ]
        )

    doc = nlp.make_doc(text)
    found: list[_Hit] = []
    for match_id, start, end in matcher(doc):
        concept = catalogue.by_id.get(nlp.vocab.strings[match_id])
        if concept is not None:
            found.append(_Hit(start=doc[start].idx, end=doc[end - 1].idx + len(doc[end - 1]), concept=concept, phrase=doc[start:end].text))
    return _dedupe(found)


def _add_linker(nlp: Any) -> Any:
    """Attach UMLS entity linking, preferring a local data directory.

    scispaCy 0.6 re-exports the linker paths from ``candidate_generation`` and
    builds a linker from a ``CandidateGenerator``. A local directory must hold
    the four linker files ``LinkerPaths.from_directory`` expects; a ``*.jsonl``
    knowledge base and ``umls_semantic_type_tree.tsv`` are used when present.
    Anything incomplete leaves linking off (catalogue CUIs still work) and logs
    a warning, so startup never reaches for a multi-gigabyte download.
    """
    from scispacy.linking import EntityLinker

    directory = Path(config.UMLS_DATA_PATH).expanduser() if config.UMLS_DATA_PATH else None
    if directory is None:
        log.warning("UMLS_ENABLED is set but UMLS_DATA_PATH is empty; entity linking stays off")
        return nlp
    if not directory.is_dir():
        log.warning("UMLS_DATA_PATH %s is not a directory; entity linking stays off", directory)
        return nlp

    try:
        from scispacy.candidate_generation import CandidateGenerator, LinkerPaths
        from scispacy.linking_utils import UmlsKnowledgeBase

        paths = LinkerPaths.from_directory(directory)
        kb_files = sorted(path for path in directory.iterdir() if path.suffix in {".json", ".jsonl"})
        types_path = directory / "umls_semantic_type_tree.tsv"
        kb = (
            UmlsKnowledgeBase(kb_files[0], types_file_path=str(types_path))
            if kb_files and types_path.exists()
            else None
        )
        options: dict[str, Any] = {
            "ann_index": paths.get_ann_index(),
            "tfidf_vectorizer": paths.get_tfidf_vectorizer(),
            "ann_concept_aliases_list": paths.get_concept_aliases(),
        }
        if kb is not None:
            options["kb"] = kb
        nlp.add_pipe(
            EntityLinker(candidate_generator=CandidateGenerator(**options), resolve_abbreviations=True),
            name="scispacy_linker",
        )
    except Exception as exc:
        # A custom linker is optional: the catalogue CUIs still come through.
        log.warning("Local UMLS linker unusable (%s); entity linking stays off", type(exc).__name__)
    return nlp


def _ensure_sci() -> Any:
    global _sci_nlp, _sci_failed

    if not config.SCISPACY_ENABLED:
        return None
    if _sci_nlp is not None or _sci_failed:
        return _sci_nlp
    with _sci_lock:
        if _sci_nlp is not None or _sci_failed:
            return _sci_nlp
        try:
            import spacy

            nlp = spacy.load(config.SCISPACY_MODEL)
            if config.UMLS_ENABLED:
                nlp = _add_linker(nlp)
            _sci_nlp = nlp
            log.info("Clinical NER ready (model=%s)", config.SCISPACY_MODEL)
        except Exception as exc:
            _sci_failed = True
            log.warning(
                "Clinical NER unavailable (%s); catalogue concepts only. Install scispacy and %s.",
                type(exc).__name__,
                config.SCISPACY_MODEL,
            )
        return _sci_nlp


def _cui(entity: Any) -> str | None:
    try:
        candidates = entity._.kb_ents
    except Exception:
        return None
    if not candidates:
        return None
    try:
        return str(candidates[0][0])
    except (IndexError, TypeError):
        return None


def _sci_concepts(text: str, spans: list[tuple[int, int]], known: set[str]) -> list[MedicalConcept]:
    nlp = _ensure_sci()
    if nlp is None or not text:
        return []
    try:
        doc = nlp(text)
    except Exception as exc:
        log.warning("Clinical NER failed (%s); catalogue concepts only", type(exc).__name__)
        return []

    found: list[MedicalConcept] = []
    for entity in doc.ents:
        canonical = normalize_text(entity.text)
        if not canonical or canonical in known:
            continue
        if any(start < entity.end_char and entity.start_char < end for start, end in spans):
            continue
        cui = _cui(entity)
        found.append(
            MedicalConcept(
                text=entity.text,
                canonical=canonical,
                category="medical",
                cui=cui,
                source="umls" if cui else config.SCISPACY_MODEL,
            )
        )
        if len(found) >= 20:
            break
    return found


# ---------------------------------------------------------------------------
# Understanding
# ---------------------------------------------------------------------------


def normalize_query(text: str, *, spelling: bool | None = None) -> str:
    """Step 1: lowercase/punctuation/whitespace normalization, optional spelling."""
    normalized = normalize_text(text)
    if spelling is None:
        spelling = config.UNDERSTANDING_SPELLING
    if spelling and config.SPELLCHECK_ENABLED and normalized:
        try:
            from .spelling import spell_correct

            normalized = normalize_text(spell_correct(normalized))
        except Exception:
            pass
    return normalized


def _appointment_intent(
    tokens: set[str], hits: list[_Hit], catalogue: ConceptCatalogue
) -> AppointmentIntent | None:
    actions = sorted(tokens & catalogue.appointment_actions)
    objects = sorted(tokens & catalogue.appointment_objects)
    if not actions:
        return None
    if objects:
        return AppointmentIntent(action=actions[0], object=objects[0])
    # "book a doctor": the booking topic is explicit but the object is a person.
    booking_topic = any(
        hit.concept.category == "faq_topic"
        and hit.concept.canonical in ("appointment booking", "appointment booking reschedule")
        for hit in hits
    )
    return AppointmentIntent(action=actions[0], object="appointment") if booking_topic else None


def _provider_signal(
    text: str,
    hits: list[_Hit],
    provider_nouns: tuple[str, ...],
    language_query: bool,
) -> bool:
    if any(hit.concept.category == "specialty" for hit in hits):
        return True
    if language_query:
        return True
    if provider_nouns and (_REQUEST_CUE_RE.search(text) or _RATING_CUE_RE.search(text)):
        return True
    return bool(PROVIDER_INTENT_RE.search(text))


def understand(
    message: str,
    *,
    spelling: bool | None = None,
    medical: bool = True,
) -> QueryUnderstanding:
    """Run the full understanding pipeline over one message."""
    text = normalize_query(message, spelling=spelling)
    catalogue = get_catalogue()
    if not config.UNDERSTANDING_ENABLED:
        return QueryUnderstanding(
            original=message,
            text=text,
            intent="faq",
            rewritten=message,
        )

    hits = _catalogue_hits(text, catalogue)
    tokens = set(re.findall(r"[a-z0-9]+", text))
    provider_nouns = tuple(sorted(tokens & catalogue.provider_nouns))
    appointment = _appointment_intent(tokens, hits, catalogue)
    languages = tuple(provider_languages(text))
    language_query = language_provider_request(text) or bool(LANGUAGE_REQUEST_RE.search(text) and languages)

    specialties = tuple(
        dict.fromkeys(hit.concept.canonical for hit in hits if hit.concept.category == "specialty")
    )
    faq_topics = tuple(
        dict.fromkeys(hit.concept.canonical for hit in hits if hit.concept.category == "faq_topic")
    )
    canonical_terms = tuple(
        dict.fromkeys(
            hit.concept.canonical
            for hit in hits
            if hit.concept.category in SEARCH_CATEGORIES and hit.phrase != hit.concept.canonical
        )
    )[: max(config.SYNONYM_MAX_EXPANSIONS, 1)]

    medical_concepts: list[MedicalConcept] = []
    seen: set[tuple[str, str]] = set()
    for hit in hits:
        if hit.concept.category not in MEDICAL_CATEGORIES:
            continue
        key = (hit.phrase, hit.concept.canonical)
        if key in seen:
            continue
        seen.add(key)
        medical_concepts.append(
            MedicalConcept(
                text=hit.phrase,
                canonical=hit.concept.canonical,
                category=hit.concept.category,
                cui=hit.concept.cui,
                source="catalogue",
            )
        )

    if medical:
        medical_concepts.extend(
            _sci_concepts(text, [(hit.start, hit.end) for hit in hits], {concept.canonical for concept in medical_concepts})
        )

    if appointment is not None:
        intent = "appointment"
    elif FAQ_HINT_RE.search(text):
        intent = "faq"
    elif _provider_signal(text, hits, provider_nouns, language_query):
        intent = "provider_search"
    else:
        intent = "faq"

    return QueryUnderstanding(
        original=message,
        text=text,
        intent=intent,
        specialties=specialties,
        faq_topics=faq_topics,
        medical_concepts=tuple(medical_concepts),
        provider_nouns=provider_nouns,
        appointment=appointment,
        languages=languages,
        canonical_terms=canonical_terms,
        rewritten=rewrite_text(message, catalogue),
    )


def expand_query(text: str, *, spelling: bool = False) -> str:
    """Canonical concept terms + rewritten wording, for local retrieval."""
    return understand(text, spelling=spelling, medical=False).search_text()


def reset_understanding() -> None:
    """Drop the matcher and clinical model caches (tests and a catalogue swap)."""
    global _matcher, _matcher_nlp, _matcher_signature, _matcher_failed, _sci_nlp, _sci_failed
    with _matcher_lock:
        _matcher = None
        _matcher_nlp = None
        _matcher_signature = None
        _matcher_failed = False
    with _sci_lock:
        _sci_nlp = None
        _sci_failed = False
    _understanding_ctx.set(None)


def warm_understanding() -> str:
    """Build the matcher and load the clinical model once at startup."""
    catalogue = get_catalogue(force=True)
    matcher, _ = _ensure_matcher(catalogue)
    sci = _ensure_sci() if config.SCISPACY_ENABLED else None
    if matcher is None:
        return "regex"
    return "spacy+scispacy" if sci is not None else "spacy"
