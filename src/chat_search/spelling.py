"""Conservative spelling correction for incoming questions.

Uses SymSpell with the packaged English frequency dictionary plus domain terms
from the synonym catalogue, so ``cardiologst`` becomes ``cardiologist`` but
names, acronyms, numbers and known medical vocabulary are left alone.

Rules that keep corrections safe:

* only unknown words are touched (a word with distance 0 is returned as-is);
* at most one edit for words shorter than eight characters, two for longer ones;
* long words that are not in the English dictionary may take one extra edit
  when the candidate is a clinical term (``paditricin`` -> ``pediatrician``);
* rare candidates are rejected by a frequency floor;
* ALL-CAPS tokens, mixed-case tokens, possessives and numbers are skipped;
* capitalised words are only corrected at the start of the query, never in the
  middle (where they are usually names such as ``Aetna``);
* at most ``SPELL_MAX_CORRECTIONS`` words change per query.

If ``symspellpy`` or its dictionary is unavailable, correction degrades to a
no-op with a single warning; queries are never blocked by this feature.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any

from . import config
from .synonyms import get_catalog

log = logging.getLogger(__name__)

_DICTIONARY_FILE = "frequency_dictionary_en_82_765.txt"
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']*")
# Never rewrite parts of an address or URL: they are PII and must stay verbatim.
_OPAQUE_RE = re.compile(r"[^\s@]+@[^\s@]+|\bhttps?://\S+|\bwww\.\S+", re.I)
_MIN_TOKEN_LEN = 3
_MAX_EDIT_DISTANCE_CAP = 2
# Long clinical words are rare enough that one extra edit is still safe: the
# fallback runs only for words missing from the English dictionary.
_DOMAIN_MAX_EDIT_DISTANCE = 3
_DOMAIN_MIN_TOKEN_LEN = 8
_MAX_LENGTH_DELTA = 2

# Very common clinical words that patients type and general dictionaries may not
# rank highly. They are never treated as misspellings and become correction
# targets for near-miss spellings.
MEDICAL_TERMS = frozenset(
    """allergy allergist anesthesiology anesthesia arrhythmia arthritis asthma atrial fibrillation
    cardiology cardiologist cataract chemotherapy chiropractor cholesterol colitis crohn dementia
    dentistry dermatology dermatologist diabetes diagnosis dialysis endocrinology endometriosis
    epilepsy gastroenterology glaucoma gynecology gynecologist hematology hepatitis hypertension
    immunization infectious migraine nephrology neurology neurologist oncology oncologist ophthalmology
    orthopedics osteoarthritis osteoporosis otolaryngology pathology pediatrics pediatrician
    physiotherapy physiotherapist podiatry psoriasis psychiatry psychiatrist psychology pulmonology
    radiology rheumatology rheumatologist vaccination""".split()
)

_lock = threading.Lock()
_symspell: Any = None
_domain_symspell: Any = None
_verbosity: Any = None
_symspell_signature: str | None = None
_failed = False


def _edit_limit(word: str) -> int:
    if len(word) < 8:
        return 1
    return min(_MAX_EDIT_DISTANCE_CAP, max(config.SPELL_MAX_EDIT_DISTANCE, 1))


def _ensure_symspell() -> Any:
    """Build the corrector once per catalogue version; ``None`` means disabled."""
    global _symspell, _domain_symspell, _verbosity, _symspell_signature, _failed

    catalog = get_catalog()
    if _symspell is not None and _symspell_signature == catalog.signature:
        return _symspell
    if _failed:
        return None

    with _lock:
        catalog = get_catalog()
        if _symspell is not None and _symspell_signature == catalog.signature:
            return _symspell
        if _failed:
            return None
        try:
            import symspellpy
            from symspellpy import SymSpell, Verbosity

            dictionary = Path(symspellpy.__file__).parent / _DICTIONARY_FILE
            max_distance = min(_MAX_EDIT_DISTANCE_CAP, max(config.SPELL_MAX_EDIT_DISTANCE, 1))
            corrector = SymSpell(max_dictionary_edit_distance=max_distance, prefix_length=7)
            if not corrector.load_dictionary(str(dictionary), term_index=0, count_index=1, encoding="utf-8"):
                raise RuntimeError(f"could not load {dictionary.name}")

            words = {word for word in MEDICAL_TERMS if word.isalpha() and len(word) > 1}
            words |= {word for word in catalog.domain if word.isalpha() and len(word) > 1}
            for word in sorted(words):
                corrector.create_dictionary_entry(word, config.SPELL_DOMAIN_FREQUENCY)

            # Clinical-only index for long typos that need a third edit. It has
            # no English vocabulary, so it can never rewrite an ordinary word.
            domain_corrector = SymSpell(
                max_dictionary_edit_distance=_DOMAIN_MAX_EDIT_DISTANCE,
                prefix_length=7,
            )
            for word in sorted(words):
                domain_corrector.create_dictionary_entry(word, config.SPELL_DOMAIN_FREQUENCY)

            _symspell = corrector
            _domain_symspell = domain_corrector
            _verbosity = Verbosity
            _symspell_signature = catalog.signature
            log.info("Spell checker ready (terms=%d, max_edit_distance=%d)", len(words), max_distance)
        except Exception as exc:
            _failed = True
            log.warning("Spell correction disabled (%s)", type(exc).__name__)
        return _symspell


def _accepted(term: str, distance: int, count: int, token: str) -> str | None:
    """Return the replacement for an accepted candidate, or ``None``."""
    if distance == 0 or term == token.lower():
        return None
    required = config.SPELL_MIN_FREQUENCY * (5 if token[0].isupper() else 1)
    if count < required:
        return None
    return term.capitalize() if token[0].isupper() else term


def _domain_suggestion(lower: str, token: str, limit: int) -> str | None:
    """Correct a long clinical word the English dictionary does not know."""
    if _domain_symspell is None or len(lower) < _DOMAIN_MIN_TOKEN_LEN:
        return None
    if lower in getattr(_symspell, "words", {}):
        return None  # a real word: the normal lookup already had its chance
    domain_limit = min(_DOMAIN_MAX_EDIT_DISTANCE, limit + 1)
    if domain_limit <= limit:
        return None
    suggestions = _domain_symspell.lookup(
        lower,
        _verbosity.CLOSEST,
        max_edit_distance=domain_limit,
        include_unknown=False,
        transfer_casing=False,
    )
    for best in suggestions:
        if best.term[:1] != lower[:1]:
            continue
        if abs(len(best.term) - len(lower)) > _MAX_LENGTH_DELTA:
            continue
        fixed = _accepted(best.term, best.distance, best.count, token)
        if fixed is not None:
            return fixed
    return None


def _correct_token(
    token: str,
    start: int,
    *,
    protected: frozenset[str],
    domain: frozenset[str],
) -> str:
    if len(token) < _MIN_TOKEN_LEN or "'" in token or token.isupper():
        return token
    if any(char.isupper() for char in token[1:]):
        return token
    if token[0].isupper() and start > 0:
        return token  # mid-sentence capital: a name, city or plan

    lower = token.lower()
    if lower in protected or lower in domain or lower in MEDICAL_TERMS:
        return token

    limit = _edit_limit(lower)
    if limit > 0:
        suggestions = _symspell.lookup(
            lower,
            _verbosity.CLOSEST,
            max_edit_distance=limit,
            include_unknown=True,
            transfer_casing=False,
        )
        if suggestions:
            best = suggestions[0]
            fixed = _accepted(best.term, best.distance, best.count, token)
            if fixed is not None:
                return fixed

    return _domain_suggestion(lower, token, limit) or token


def spell_correct(text: str) -> str:
    """Return ``text`` with conservative spelling fixes; never raises."""
    if not config.SPELLCHECK_ENABLED or not text:
        return text
    corrector = _ensure_symspell()
    if corrector is None:
        return text

    catalog = get_catalog()
    protected = catalog.protected
    domain = catalog.domain
    opaque = [match.span() for match in _OPAQUE_RE.finditer(text)]
    pieces: list[str] = []
    last = 0
    changed = 0
    for match in _TOKEN_RE.finditer(text):
        pieces.append(text[last : match.start()])
        last = match.end()
        token = match.group(0)
        overlaps = any(start < last and match.start() < end for start, end in opaque)
        if changed < config.SPELL_MAX_CORRECTIONS and not overlaps:
            fixed = _correct_token(token, match.start(), protected=protected, domain=domain)
            if fixed != token:
                changed += 1
            token = fixed
        pieces.append(token)
    pieces.append(text[last:])
    return "".join(pieces)


def reset_spelling() -> None:
    global _symspell, _domain_symspell, _verbosity, _symspell_signature, _failed
    with _lock:
        _symspell = None
        _domain_symspell = None
        _verbosity = None
        _symspell_signature = None
        _failed = False


def warm_spelling() -> bool:
    if not config.SPELLCHECK_ENABLED:
        return False
    return _ensure_symspell() is not None
