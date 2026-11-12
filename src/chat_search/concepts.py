"""Structured concept catalogue for query understanding.

``concepts.json`` describes what a patient's words mean, not just which
specialty they name::

    {
      "concepts": [
        {"id": "cardiology", "canonical": "cardiology", "category": "specialty",
         "terms": ["cardiologist", "heart doctor"], "cui": "C0007222"},
        {"id": "hypertension", "canonical": "hypertension", "category": "condition",
         "terms": ["high blood pressure"], "cui": "C0020538"}
      ],
      "provider_nouns": ["doctor", "physician"],
      "appointment_actions": ["book", "cancel", "reschedule"],
      "appointment_objects": ["appointment", "visit", "consultation"],
      "rewrites": {"face to face": "in person"}
    }

Every entry carries a category so each consumer can take only what it needs:
provider search uses ``specialty``, FAQ retrieval uses ``specialty`` and
``faq_topic``, spelling protection uses ``condition``/``medication``/
``procedure``/``identifier``, and intent detection uses ``provider_noun`` plus
the appointment action/object vocabulary. The optional ``cui`` links an entry
to UMLS.

The legacy ``synonyms.json`` shape (``specialties``, ``protected_terms``,
``rewrites``) is still accepted, so an existing deployment or test can keep its
file while it migrates. The file is re-read at most once every
``CONCEPTS_RELOAD_SECONDS``; invalid JSON keeps the last good catalogue.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config

log = logging.getLogger(__name__)

SEARCH_CATEGORIES = frozenset({"specialty", "faq_topic"})
PROTECTED_CATEGORIES = frozenset({"condition", "medication", "procedure", "identifier"})
MEDICAL_CATEGORIES = frozenset({"specialty", "condition", "medication", "procedure", "identifier"})

_LEGACY_FAQ_TOPICS = frozenset(
    {
        "file attachment",
        "appointment booking",
        "appointment booking reschedule",
        "book for someone else",
    }
)
_RESERVED_KEYS = frozenset(
    {
        "version",
        "concepts",
        "specialties",
        "protected_terms",
        "term_categories",
        "rewrites",
        "provider_nouns",
        "appointment_actions",
        "appointment_objects",
    }
)

_DEFAULT_PROVIDER_NOUNS = (
    "doctor doctors physician physicians provider providers specialist specialists clinician clinicians "
    "therapist therapists gp pcp"
)
_DEFAULT_APPOINTMENT_ACTIONS = (
    "book books booked booking cancel cancels cancelled canceled cancelling canceling reschedule "
    "reschedules rescheduled rescheduling schedule schedules scheduled scheduling move moves moved moving "
    "change changes changed changing modify modifies modified modifying update updates updated updating "
    "manage manages managed managing confirm confirms confirmed confirming view views viewed viewing"
)
_DEFAULT_APPOINTMENT_OBJECTS = (
    "appointment appointments visit visits consultation consultations consult consults booking bookings"
)

# Used when no file is configured or it cannot be read. Keeping a small default
# means routing and retrieval still work before an operator installs a file.
DEFAULT_PAYLOAD: dict[str, Any] = {
    "specialties": {
        "file attachment": [
            "attachment",
            "attachments",
            "attach a file",
            "attach file",
            "attached file",
            "upload a file",
        ],
        "appointment booking": [
            "appointment",
            "appointments",
            "book an appointment",
            "book a doctor",
            "book a physician",
            "book a visit",
            "video appointment",
            "video visit",
            "telehealth visit",
            "reschedule",
            "rescheduling",
        ],
        "appointment booking reschedule": [
            "reschedule my visit",
            "move my visit",
            "move my appointment",
            "move my booking",
            "change my visit",
            "change my appointment",
            "change my booking",
            "modify appointment",
            "modify visit",
        ],
        "book for someone else": [
            "book for someone else",
            "book for a family member",
            "book for my family",
            "family member",
            "family members",
            "dependent",
            "dependents",
            "rather than me",
            "instead of me",
            "on her behalf",
            "on his behalf",
        ],
        "pediatrics": [
            "pediatrician",
            "pediatric doctor",
            "paediatrician",
            "kids doctor",
            "kid doctor",
            "child doctor",
            "child specialist",
            "childrens doctor",
            "baby doctor",
        ],
        "cardiology": ["cardiologist", "heart doctor", "heart specialist"],
        "dermatology": ["dermatologist", "skin doctor", "skin specialist"],
        "orthopedics": ["orthopedist", "orthopedic doctor", "bone doctor", "bone specialist"],
        "gynecology": ["gynecologist", "gynaecologist", "obgyn", "ob gyn", "womens health doctor"],
        "neurology": ["neurologist", "brain doctor", "nerve doctor"],
        "psychiatry": ["psychiatrist", "mental health doctor"],
        "dentistry": ["dentist", "dental doctor", "teeth doctor", "tooth doctor"],
        "ophthalmology": ["ophthalmologist", "eye doctor", "eye specialist"],
        "oncology": ["oncologist", "cancer doctor", "cancer specialist"],
        "general medicine": [
            "family doctor",
            "family physician",
            "family medicine",
            "primary care doctor",
            "primary care physician",
            "general practitioner",
            "gp",
            "pcp",
        ],
    },
    "protected_terms": [
        "hypertension",
        "arrhythmia",
        "atrial fibrillation",
        "diabetes",
        "asthma",
        "eczema",
        "psoriasis",
        "migraine",
        "epilepsy",
        "glaucoma",
        "cataract",
        "arthritis",
        "osteoporosis",
        "endometriosis",
        "fibromyalgia",
        "cholesterol",
        "immunization",
        "vaccination",
        "ibuprofen",
        "acetaminophen",
        "metformin",
        "insulin",
        "statin",
        "npi",
    ],
}


@dataclass(frozen=True)
class Concept:
    id: str
    canonical: str
    category: str
    terms: tuple[str, ...]
    phrases: tuple[str, ...]
    cui: str | None = None

    @property
    def searchable(self) -> bool:
        return self.category in SEARCH_CATEGORIES


@dataclass(frozen=True)
class ConceptCatalogue:
    concepts: tuple[Concept, ...]
    by_id: dict[str, Concept]
    phrase_to_concept: dict[str, Concept]
    groups: dict[str, tuple[str, ...]]
    protected: frozenset[str]
    domain: frozenset[str]
    rewrites: dict[str, str]
    rewrite_pattern: re.Pattern[str] | None
    pattern: re.Pattern[str] | None
    provider_nouns: frozenset[str]
    appointment_actions: frozenset[str]
    appointment_objects: frozenset[str]
    faq_topics: frozenset[str]
    specialties: frozenset[str]
    signature: str


def _normalize(text: str) -> str:
    """Lowercase, drop apostrophes, turn punctuation/hyphens into spaces."""
    text = text.lower().replace("\u2019", "'").replace("'", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


normalize_text = _normalize


def _pluralize(phrase: str) -> str | None:
    """Plural form of a patient phrase (``child specialist`` -> ``child specialists``)."""
    head, _, last = phrase.rpartition(" ")
    if not last or last[-1] in "s0123456789":
        return None
    if last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        last = last[:-1] + "ies"
    elif last.endswith(("ch", "sh", "x", "z")):
        last += "es"
    else:
        last += "s"
    return f"{head} {last}" if head else last


def _as_phrases(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _phrases(canonical: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    phrases = [canonical]
    for term in terms:
        phrases.append(term)
        plural = _pluralize(term)
        if plural:
            phrases.append(plural)
    return tuple(dict.fromkeys(phrase for phrase in phrases if phrase))


def _concept(
    *,
    identifier: str,
    canonical: str,
    category: str,
    terms: list[str] | tuple[str, ...],
    cui: str | None = None,
) -> Concept | None:
    canonical = _normalize(canonical)
    if not canonical:
        return None
    normalized_terms = tuple(dict.fromkeys(_normalize(term) for term in terms if _normalize(term)))
    normalized_category = re.sub(r"[^a-z0-9]+", "_", category.strip().lower()).strip("_") or "concept"
    return Concept(
        id=_normalize(identifier) or canonical,
        canonical=canonical,
        category=normalized_category,
        terms=normalized_terms,
        phrases=_phrases(canonical, normalized_terms),
        cui=cui,
    )


def _from_concepts(payload: list[Any]) -> list[Concept]:
    concepts: list[Concept] = []
    seen: dict[tuple[str, str], int] = {}
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        category = str(raw.get("category") or "concept")
        canonical = str(raw.get("canonical") or raw.get("name") or raw.get("id") or "")
        terms = _as_phrases(raw.get("terms") if "terms" in raw else raw.get("variants", []))
        cui = str(raw["cui"]).strip() if raw.get("cui") else None
        identifier = str(raw.get("id") or canonical)
        concept = _concept(
            identifier=identifier,
            canonical=canonical,
            category=category,
            terms=terms,
            cui=cui,
        )
        if concept is None:
            continue
        key = (concept.category, concept.canonical)
        if key in seen:
            # A later duplicate may add aliases; keep the first id and order.
            existing = concepts[seen[key]]
            merged_terms = tuple(dict.fromkeys((*existing.terms, *concept.terms)))
            concepts[seen[key]] = Concept(
                id=existing.id,
                canonical=existing.canonical,
                category=existing.category,
                terms=merged_terms,
                phrases=_phrases(existing.canonical, merged_terms),
                cui=existing.cui or concept.cui,
            )
            continue
        seen[key] = len(concepts)
        concepts.append(concept)
    return concepts


def _from_legacy(payload: dict[str, Any]) -> list[Concept]:
    specialties = payload.get("specialties")
    if not isinstance(specialties, dict):
        specialties = {key: value for key, value in payload.items() if key not in _RESERVED_KEYS}

    concepts: list[Concept] = []
    for raw_canonical, raw_variants in specialties.items():
        canonical = _normalize(str(raw_canonical))
        category = "faq_topic" if canonical in _LEGACY_FAQ_TOPICS else "specialty"
        concept = _concept(
            identifier=canonical,
            canonical=canonical,
            category=category,
            terms=_as_phrases(raw_variants),
        )
        if concept is not None:
            concepts.append(concept)

    term_categories = payload.get("term_categories")
    term_categories = term_categories if isinstance(term_categories, dict) else {}
    for index, raw_term in enumerate(_as_phrases(payload.get("protected_terms", []))):
        term = _normalize(raw_term)
        if not term:
            continue
        category = _normalize(str(term_categories.get(raw_term, "condition"))) or "condition"
        concept = _concept(
            identifier=f"{category}-{index}-{term}",
            canonical=term,
            category=category,
            terms=[],
        )
        if concept is not None:
            concepts.append(concept)
    return concepts


def _vocabulary(payload: dict[str, Any], key: str, default: str) -> frozenset[str]:
    entries = [*_as_phrases(payload.get(key, [])), *default.split()]
    tokens = {token for entry in entries for token in _normalize(entry).split() if token}
    return frozenset(tokens)


def build_catalogue(payload: Any, signature: str) -> ConceptCatalogue:
    if not isinstance(payload, dict):
        raise ValueError("concept catalogue must be a JSON object")

    raw_concepts = payload.get("concepts")
    if isinstance(raw_concepts, list):
        concepts = _from_concepts(raw_concepts)
    else:
        concepts = _from_legacy(payload)

    phrase_to_concept: dict[str, Concept] = {}
    groups: dict[str, tuple[str, ...]] = {}
    faq_topics: set[str] = set()
    specialties: set[str] = set()
    protected_tokens: set[str] = set()
    for concept in concepts:
        if concept.searchable:
            groups[concept.canonical] = concept.terms
            (faq_topics if concept.category == "faq_topic" else specialties).add(concept.canonical)
        if concept.category in PROTECTED_CATEGORIES:
            protected_tokens.update(
                token
                for phrase in (concept.canonical, *concept.terms)
                for token in phrase.split()
                if len(token) >= 2
            )
        for phrase in concept.phrases:
            existing = phrase_to_concept.get(phrase)
            if existing is not None and existing.id != concept.id:
                log.warning(
                    "Concept phrase %r maps to both %r and %r; keeping %r",
                    phrase,
                    existing.canonical,
                    concept.canonical,
                    existing.canonical,
                )
                continue
            phrase_to_concept[phrase] = concept

    protected = frozenset(protected_tokens)
    domain = frozenset(
        token
        for concept in concepts
        if concept.searchable
        for phrase in concept.phrases
        for token in phrase.split()
        if len(token) >= 2
    ) | protected

    if phrase_to_concept:
        alternation = "|".join(re.escape(phrase) for phrase in sorted(phrase_to_concept, key=len, reverse=True))
        pattern: re.Pattern[str] | None = re.compile(rf"(?<![a-z0-9])(?:{alternation})(?![a-z0-9])")
    else:
        pattern = None

    raw_rewrites = payload.get("rewrites")
    rewrites = (
        {
            _normalize(str(source)): _normalize(str(target))
            for source, target in raw_rewrites.items()
            if _normalize(str(source)) and _normalize(str(target))
        }
        if isinstance(raw_rewrites, dict)
        else {}
    )
    rewrite_pattern: re.Pattern[str] | None = None
    if rewrites:
        alternatives = "|".join(re.escape(phrase) for phrase in sorted(rewrites, key=len, reverse=True))
        rewrite_pattern = re.compile(rf"(?<![a-z0-9])(?:{alternatives})(?![a-z0-9])")

    return ConceptCatalogue(
        concepts=tuple(concepts),
        by_id={concept.id: concept for concept in concepts},
        phrase_to_concept=phrase_to_concept,
        groups=groups,
        protected=protected,
        domain=domain,
        rewrites=rewrites,
        rewrite_pattern=rewrite_pattern,
        pattern=pattern,
        provider_nouns=_vocabulary(payload, "provider_nouns", _DEFAULT_PROVIDER_NOUNS),
        appointment_actions=_vocabulary(payload, "appointment_actions", _DEFAULT_APPOINTMENT_ACTIONS),
        appointment_objects=_vocabulary(payload, "appointment_objects", _DEFAULT_APPOINTMENT_OBJECTS),
        faq_topics=frozenset(faq_topics),
        specialties=frozenset(specialties),
        signature=signature,
    )


def rewrite_text(text: str, catalogue: ConceptCatalogue) -> str:
    """Replace wording the ranker scores badly with the portal's term (search text only)."""
    if catalogue.rewrite_pattern is None or not text:
        return text
    normalized = _normalize(text)
    rewritten = catalogue.rewrite_pattern.sub(lambda match: catalogue.rewrites[match.group(0)], normalized)
    return rewritten if rewritten != normalized else text


def iter_matches(text: str, catalogue: ConceptCatalogue) -> list[tuple[int, int, Concept]]:
    """Regex fallback when spaCy is unavailable; returns non-overlapping matches."""
    if catalogue.pattern is None or not text:
        return []
    found: list[tuple[int, int, Concept]] = []
    for match in catalogue.pattern.finditer(text):
        concept = catalogue.phrase_to_concept.get(match.group(0))
        if concept is not None:
            found.append((match.start(), match.end(), concept))
    return found


def _resolve_path() -> Path | None:
    if Path(config.SYNONYMS_FILE).expanduser() != Path("synonyms.json"):
        # An explicit legacy file wins; a missing one means defaults, never a
        # silent switch to concepts.json (tests and small deployments rely on it).
        path = Path(config.SYNONYMS_FILE).expanduser()
        resolved = path if path.is_absolute() else Path.cwd() / path
        if not resolved.exists():
            resolved = Path(__file__).resolve().parents[2] / path
        return resolved if resolved.exists() else None

    candidates = [Path(config.CONCEPTS_FILE).expanduser()]
    if config.SYNONYMS_FILE:
        candidates.append(Path(config.SYNONYMS_FILE).expanduser())
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else Path.cwd() / candidate
        if not resolved.exists():
            resolved = Path(__file__).resolve().parents[2] / candidate
        if resolved.exists():
            return resolved
    return None


def _snapshot() -> tuple[str, Path | None]:
    inline = config.CONCEPTS_JSON.strip() or config.SYNONYMS_JSON.strip()
    if inline:
        digest = hashlib.sha256(inline.encode()).hexdigest()[:16]
        return f"inline:{digest}", None
    path = _resolve_path()
    if path is None:
        return "defaults", None
    try:
        stat = path.stat()
    except OSError:
        return "defaults", None
    return f"file:{path}:{stat.st_mtime_ns}:{stat.st_size}", path


def _read_payload(path: Path | None) -> Any:
    inline = config.CONCEPTS_JSON.strip() or config.SYNONYMS_JSON.strip()
    if inline:
        return json.loads(inline)
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    return DEFAULT_PAYLOAD


_lock = threading.Lock()
_catalogue: ConceptCatalogue | None = None
_signature: str | None = None
_checked_at = 0.0
_last_warning: str | None = None


def _enabled() -> bool:
    return config.CONCEPTS_ENABLED and config.SYNONYMS_ENABLED


def get_catalogue(force: bool = False) -> ConceptCatalogue:
    """Return the current catalogue, reloading the file when it has changed."""
    global _catalogue, _signature, _checked_at, _last_warning

    if not _enabled():
        return DISABLED

    ttl = max(min(config.CONCEPTS_RELOAD_SECONDS, config.SYNONYMS_RELOAD_SECONDS), 0.0)
    if not force and _catalogue is not None and time.monotonic() - _checked_at < ttl:
        return _catalogue

    with _lock:
        if not force and _catalogue is not None and time.monotonic() - _checked_at < ttl:
            return _catalogue

        signature, path = _snapshot()
        if force or _catalogue is None or signature != _signature:
            try:
                _catalogue = build_catalogue(_read_payload(path), signature)
                _last_warning = None
            except Exception as exc:
                if signature != _last_warning:
                    log.warning("Concept catalogue ignored (%s); using the last good copy", type(exc).__name__)
                    _last_warning = signature
                if _catalogue is None:
                    _catalogue = build_catalogue(DEFAULT_PAYLOAD, "defaults")
            else:
                log.info(
                    "Concept catalogue loaded concepts=%d source=%s",
                    len(_catalogue.concepts),
                    signature.split(":", 1)[0],
                )
            _signature = signature
        _checked_at = time.monotonic()
        return _catalogue


DISABLED = build_catalogue({"concepts": []}, "disabled")
DEFAULT_CATALOGUE = build_catalogue(DEFAULT_PAYLOAD, "defaults")


def reset_concepts() -> None:
    global _catalogue, _signature, _checked_at, _last_warning
    with _lock:
        _catalogue = None
        _signature = None
        _checked_at = 0.0
        _last_warning = None


def warm_concepts() -> ConceptCatalogue:
    return get_catalogue(force=True)
