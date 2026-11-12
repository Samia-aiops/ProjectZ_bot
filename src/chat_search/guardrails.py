"""Input guardrails, in four stages.

1. ``normalize`` — Unicode/zero-width cleanup and whitespace collapsing.
2. ``rule_guard`` — emergency first, then the business rules (injection,
   identity, privacy, restricted, medical).
3. ``ml_guard`` — ProtectAI's LLM Guard prompt-injection classifier.
4. ``pii_guard`` — Presidio PII detection, used to keep personal data out of
   the shared cache and conversation history.

These are heuristics. They reduce risk, they are not a guarantee.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

from . import config
from .messages import GREETING_REPLY, REFUSAL
from .observability import metric
from .runtime import get_runtime, limit, run_cpu

log = logging.getLogger(__name__)

_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")


def normalize(text: str) -> str:
    text = _ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", text))
    return re.sub(r"\s+", " ", text).strip()


GREETING_RE = re.compile(
    r"^(hi+|hello+|hey+|hiya|howdy|yo|greetings|good (morning|afternoon|evening)|"
    r"salam|assalam[ -]?o[ -]?alaikum|assalamu alaikum|as-salamu alaykum|"
    r"how are you( doing| today)?|what'?s up|sup)"
    r"( (there|team|assistant|bot|everyone|all))?[\s!.,?]*$",
    re.I,
)


def greeting_reply(message: str) -> str | None:
    text = normalize(message)
    return GREETING_REPLY if len(text) <= 60 and GREETING_RE.match(text) else None


# Strong injection phrases. Also used to drop retrieved chunks that try to give orders.
INJECTION_RE = re.compile(
    r"(ignore|disregard|forget|override|bypass)\s+(all\s+|any\s+|the\s+|your\s+|my\s+)?"
    r"(previous|prior|above|earlier|system|safety|security)?\s*"
    r"(instructions?|prompts?|rules|guidelines|policies|guardrails|filters)|"
    r"\b(new|updated)\s+instructions?\s*:|"
    r"\b(system|developer|hidden|internal)\s+(prompt|message|instructions?)\b|"
    r"\b(jailbreak|developer mode|do anything now)\b|\bDAN\b|"
    r"<\|?\s*(system|im_start|im_end|assistant)\s*\|?>|\[/?INST\]|"
    # Bracketed role tags and "<role> override" directives attempt to replace
    # the assistant's instructions outright, even without an "ignore" verb.
    r"\[\s*(?:system|developer|admin|root|assistant|internal|hidden)\b[^\]]{0,40}\]|"
    r"\b(?:system|developer|admin|root)\s+override\b|"
    r"^\s*#{2,}\s*(system|instructions?)\b|"
    # Prompt exfiltration and role overrides. "Your" separates the assistant's
    # own instructions from the product steps a caller legitimately asks for.
    r"\b(repeat|print|show|reveal|leak|output)\b.{0,30}\b(words?|text|prompt)\b.{0,20}\b(above|earlier)\b|"
    r"\b(repeat|print|show|reveal|leak|output)\b.{0,20}\byour (system |hidden |internal )?"
    r"(prompt|instructions?|rules)\b|"
    r"\bwhat (is|are) your (system |hidden |internal )?(prompt|instructions?)\b|"
    r"\bwhat (is|are) your (system|hidden|internal) rules\b|"
    r"\b(pretend|act|behave|respond) (to be|as if|as (a|an|the|my))\b|"
    r"\byou are now\b|\bfrom now on,? you\b",
    re.I | re.M,
)
EMERGENCY_RE = re.compile(
    r"\b(suicid\w*|kill myself|end my life|self[- ]?harm|hurt myself|cut myself|"
    r"want to die|don'?t want to live|no reason to live|better off dead|"
    r"overdos\w*|took too many (pills|tablets)|"
    r"chest (pain|pressure|tightness)|(?:pain|pressure|tightness) in (?:my|the) chest|"
    r"crushing chest|heart attack|cardiac arrest|"
    # Colloquial cardiac phrasing: people describe it as the heart "going down",
    # racing or stopping rather than naming a symptom.
    r"heart\s+(?:is\s+)?(?:racing|pounding|stopping|slowing|going\s+down|"
    r"beating\s+(?:fast|slow|irregularly))|"
    r"heart\s*rate\s+(?:is\s+)?(?:dropping|falling|too\s+(?:low|high)|very\s+(?:low|high))|"
    r"heart\s+(?:stopped|has\s+stopped|hurts)|"
    r"(?:about|going)\s+to\s+(?:faint|pass\s+out)|"
    r"can(?:'?t|\s?not)\s+breathe|not breathing|stopped breathing|"
    r"severe bleeding|bleeding heavily|heavy bleeding|coughing (up )?blood|vomiting blood|"
    r"stroke|seizure|choking|anaphylaxis|severe allergic reaction|"
    # Airway swelling after a medication or sting is anaphylaxis, even when the
    # caller frames it as a mild symptom ("can I ignore it?").
    r"throat\s+(?:is\s+|feels?\s+|got\s+)?(?:swelling|swollen|closing|tight(?:ening)?)|"
    r"swollen\s+(?:throat|tongue|lips?|face)|"
    r"(?:tongue|lips?|face)\s+(?:is\s+|are\s+|feels?\s+)?(?:swelling|swollen|puffy)|"
    # FAST signs. People describe the symptom, not the diagnosis.
    r"droop\w*\s+(?:face|mouth|lip)|face\s+(?:is\s+|looks\s+|feels\s+)?droop\w*|"
    r"slurr\w*\s+(?:(?:my|the|his|her|their)\s+)?(?:speech|words|voice)|"
    r"speech\s+(?:is\s+|sounds\s+)?slurr\w*|"
    r"(?:face|arm|leg|hand)\s+(?:is\s+|feels\s+|went\s+|got\s+|has\s+gone\s+)?(?:numb|weak|limp)|"
    r"(?:numbness|weakness)\s+(?:on\s+|in\s+)?(?:one|my|the)\s+(?:left\s+|right\s+)?side|"
    r"can(?:'?t|\s?not)\s+(?:move|lift|feel)\s+(?:my\s+|one\s+|the\s+)?(?:arm|leg|face)|"
    r"unconscious|passed out|unresponsive|blue lips)\b",
    re.I,
)
PRIVACY_RE = re.compile(
    # A family member, dependent or named patient plus their records. "my adult
    # daughter" and "my teenager" are the same request as "my daughter's labs".
    r"\bmy\s+(?:adult\s+|teenage\s+|teen\s+|grown\s+|elderly\s+|young\s+|little\s+|older\s+)?"
    r"(?:daughter|son|child|children|kid|kids|mother|father|mom|mum|dad|wife|husband|spouse|"
    r"partner|sister|brother|grandmother|grandfather|grandma|grandpa|grandson|granddaughter|"
    r"teenager|friend|neighbou?r|colleague|caregiver|dependent)s?'?s?\b"
    r".{0,60}\b(records?|charts?|labs?|lab\s+reports?|"
    # A bare "reports" is also a verb ("my son reports an error"), so the noun
    # reading needs a word that cannot start the verb's object.
    r"reports?(?!\s+(?:that|an?|the|this|these|those|some|any|no|back|to|about|on)\b|\s+\w+ing\b)|"
    r"results?|diagnos\w+|prescriptions?|"
    r"history|appointments?|notes?|visit\s+reasons?|confidential\w*)\b|"
    r"\b(another|other|someone else'?s?|somebody else'?s?)\b"
    r".{0,40}\b(records?|charts?|labs?|reports?|results?|diagnos\w+|prescriptions?|history|appointments?|notes?)\b|"
    # "her lab reports", "their diagnoses": a possessive pointing at someone else.
    r"\b(show|give|list|export|download|dump|find|look ?up|get|read|access|see)\b.{0,40}"
    r"\b(?:their|his|her)\b.{0,30}"
    r"\b(diagnos\w+|records?|labs?|lab\s+reports?|reports?|results?|prescriptions?|history)\b|"
    # "a patient's email address", "another patient's chart".
    r"\b(?:a|another|other|any)\s+patient'?s?\b.{0,40}"
    r"\b(email|contact|records?|labs?|results?|diagnos\w+|prescriptions?|history|visit\s+reasons?)\b|"
    r"\b(show|give|list|export|download|dump|find|look ?up|get)\b.{0,40}\b(all|every|other)\b.{0,25}"
    r"\b(patients?|users?|records?|accounts?|providers?)\b|"
    # Asking the assistant to go read a person's row out of the database is an
    # access request, not a product question, even for one named person.
    r"\b(retrieve|fetch|pull|query|read|access|look ?up|get)\b.{0,40}"
    r"\b(info|information|details|records?|data|profile)\b.{0,30}\b(from|in|off)\b.{0,10}"
    r"\b(?:the\s+)?(?:db|database|tables?|backend)\b|"
    # A named doctor or provider plus a personal identifier or detail — an NPI,
    # contact details, a profile, or someone else's appointments — is a lookup,
    # not a directory search. The name is required, so "find a doctor near me"
    # and "update my provider profile" are not caught.
    r"\b(?:dr\.?\s*[a-z][\w'-]*|doctor\s+(?!(?:near|in|who|with|for|that|this|available|"
    r"accepting|at|on|from|here|now|please|and|or|is|was|said|will|can|should)\b)[a-z][\w'-]*|"
    r"provider\s+(?!(?:near|in|who|with|for|that|this|available|accepting|at|on|from|here|"
    r"now|please|and|or|is|was|said|will|can|should)\b)[a-z][\w'-]*)'?s?\b.{0,40}"
    r"\b(npi|national provider identifier|info|information|details|profile|contact|email|"
    r"phone|address|license|records?|appointments?|patients?)\b|"
    r"\b(?:npi|national provider identifier)\b.{0,25}\b(?:of|for)\b.{0,20}\b(?:dr\.?|doctor)\b|"
    # Reassigning every appointment onto the caller's profile has no product flow.
    r"\b(?:move|transfer|assign|merge|copy|make)\b.{0,50}\b(?:all|every)\b.{0,20}"
    r"\bappointments?\b.{0,30}\b(?:mine|my\s+(?:profile|account)|to\s+me)\b",
    re.I,
)
RESTRICTED_RE = re.compile(
    r"\b(reveal|show|print|repeat|leak|dump)\b.{0,40}\b(prompt|instructions?|rules|config\w*|claims|tokens?|jwt|secrets?|api[ _-]?keys?|passwords?|env\w*)\b|"
    r"\b(other|another|different)\s+(tenants?|organi[sz]ations?|hospitals?|clinics?|companies)\b|"
    r"\b(disable|turn off|remove)\b.{0,20}\b(security|authentication|guardrails?|filters?|rbac)\b",
    re.I,
)
IDENTITY_RE = re.compile(
    # "start as Patient B", "log in as another patient", "act as someone else":
    # becoming a different person is what signing in does, not what chat does.
    r"\b(?:start|begin|continue|resume|act|respond|answer|sign\s*in|log\s*in|log\s*on)\b"
    r".{0,40}\bas\s+(?:(?:a|an|the)\s+)?"
    r"(?:patient\s+[a-z0-9]+\b|user\s+[a-z0-9]+\b|"
    r"another\s+(?:patient|user|person|account)\b|"
    r"(?:a\s+)?different\s+(?:patient|user|person|account)\b|"
    r"someone\s+else\b|somebody\s+else\b)|"
    r"\bimpersonat\w*\b.{0,30}\b(?:patient|user|doctor|provider|staff)\b|"
    # "switch to Patient B without login", "change accounts with no sign-in".
    r"\b(?:switch|change|swap|take\s+over)\b.{0,30}"
    r"\b(?:accounts?|users?|patients?|sessions?|identit(?:y|ies))\b.{0,50}"
    r"\b(?:without|no)\b.{0,20}\b(?:log(?:ging)?\s*in|login|sign(?:ing)?\s*in|sign[- ]?in|"
    r"authentication|password|credentials|verification)\b",
    re.I,
)
MEDICAL_RE = re.compile(
    r"\b(diagnos(e|is) (me|my|for)|do i have|should i (take|stop|start|skip|increase|decrease)|"
    r"(what|which) (medicine|medication|drug|antibiotic|painkiller)s?\b|"
    r"how (much|many)\b.{0,30}\b(mg|ml|tablets?|pills?|doses?|units)\b|"
    r"safe to (take|mix|combine)|is it (cancer|serious|dangerous))\b|"
    # A stated diagnosis or organ damage is a health question: "have liver
    # damage" has nothing to do with the product, even without a leading "I".
    r"\b(?:have|has|got|suffers?|suffering|diagnosed\s+with)\b.{0,30}"
    r"\b(?:damage|disease|failure|infection|cancer|injury|disorder)\b|"
    r"\b(?:liver|kidney|heart|lung|brain|nerve|organ)s?\s+"
    r"(?:damage|disease|failure|problems?|issues?)\b",
    re.I,
)
PII_RE = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.-]+|"
    r"(?<!\d)(?!(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/.]\d{1,2}[/.]\d{2,4})(?!\d))\+?\d[\d\s().-]{8,}\d|"
    r"\b\d{3}-\d{2}-\d{4}\b|\b\d{6,}\b|"
    r"\b(my name is|i am called|date of birth|dob)\b",
    re.I,
)
# Presidio has no recognizer for either pattern: a self-introduction and a bare
# long number (an NPI or member id) are the cases the old regex caught and that
# must keep their cache/history protection.
_SELF_IDENTIFIER_RE = r"(?i)\b(my name is|i am called|date of birth|dob)\b"
_LONG_NUMBER_RE = r"\b\d{6,}\b"
# Presidio validates phone numbers, so a fictional or test number (a 555
# prefix, for example) is not recognised. This shape-only recognizer keeps the
# coverage of the previous regex for anything that looks like a contact number.
# A calendar date ("next available 2026-11-04") is not a contact number: treating
# it as one hid every provider answer from the conversation history.
_CONTACT_NUMBER_RE = (
    r"(?<!\d)(?!(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/.]\d{1,2}[/.]\d{2,4})(?!\d))"
    r"\+?\d[\d\s().-]{8,}\d"
)
# A first-person request about the caller's own account or records is the
# product's core flow, but the injection model reads the imperative and the
# possessive ("show me my appointments") as an override and blocks it. Such a
# request skips the model only after the deterministic rules above cleared it,
# and only when it does not name the assistant's own behaviour.
SELF_SERVICE_RE = re.compile(
    r"\bmy\s+(?:own\s+)?(?:appointments?|bookings?|visits?|schedules?|calendars?|records?|"
    r"reports?|lab\s+reports?|lab\s+results?|results?|prescriptions?|medications?|documents?|"
    r"files?|messages?|notes?|profile|account|settings|billing|invoices?|payments?|insurance|"
    r"family|dependents?|children|providers?|doctors?)\b|"
    r"\b(?:how|where|what|can|could|do|does|should)\b.{0,80}"
    r"\b(?:delete|remove|download|export|upload|update|change|cancel|book|reschedule|view|see|"
    r"find|access|manage|add|turn\s+on|turn\s+off)\b.{0,40}"
    r"\b(?:appointments?|bookings?|visits?|records?|reports?|results?|prescriptions?|"
    r"documents?|files?|profile|account|settings|billing|invoices?|messages?|providers?|doctors?)\b",
    re.I,
)
_ASSISTANT_RE = re.compile(
    r"\b(?:your|yours|yourself|assistant|system|prompt|instructions?|rules|guidelines|"
    r"guardrails?|filters?|config\w*|secrets?)\b",
    re.I,
)

_RULE_ORDER = (
    ("emergency", EMERGENCY_RE),
    ("injection", INJECTION_RE),
    ("identity", IDENTITY_RE),
    ("privacy", PRIVACY_RE),
    ("restricted", RESTRICTED_RE),
    ("medical", MEDICAL_RE),
)
# Scheduling and workload questions ("how many appointments do I have today")
# contain diagnosis-shaped phrasing ("do I have") but ask about the caller's
# own records, not medical advice. Only the medical rule is exempted.
_OPERATIONAL_RE = re.compile(
    r"\b(?:appointments?|bookings?|visits?|schedules?|availability|openings?|slots?|"
    r"no[-\s]?shows?|patients?|capacity)\b",
    re.I,
)


def rule_guard(text: str, skip: frozenset[str] = frozenset()) -> str | None:
    """Return the blocked category, or None. Emergency first so the user gets help."""
    for category, pattern in _RULE_ORDER:
        if category in skip:
            continue
        if category == "medical" and _OPERATIONAL_RE.search(text):
            continue
        if pattern.search(text):
            return category
    return None


def self_service_request(text: str) -> bool:
    """True for a first-person question about the caller's own account or records."""
    return bool(SELF_SERVICE_RE.search(text)) and not _ASSISTANT_RE.search(text)


def _guard_sync(text: str) -> tuple[str, float]:
    # DeBERTa-v3 does not use token type embeddings; asking for them only
    # produces a warning and a wasted tensor.
    out = get_runtime().guard(
        text[:2000],
        truncation=True,
        max_length=512,
        return_token_type_ids=False,
    )[0]
    return str(out["label"]).upper(), float(out["score"])


async def ml_guard(text: str) -> bool:
    """Stage 3: True = block. Fails closed (503) if the injection model cannot run."""
    started = time.perf_counter()
    async with limit(get_runtime().sem_guard):
        label, score = await run_cpu(_guard_sync, text)
    metric(
        guard_ms=int((time.perf_counter() - started) * 1000),
        guard_label=label,
        guard_score=round(score, 3),
    )
    return label in config.GUARD_BLOCK_LABELS and score >= config.GUARD_THRESHOLD


@dataclass(frozen=True)
class PIIResult:
    """What Presidio found. An empty entity list means no personal data."""

    entities: tuple[str, ...] = ()
    score: float = 0.0

    @property
    def found(self) -> bool:
        return bool(self.entities)


def load_pii_engine() -> Any:
    """Build the Presidio analyzer, or raise if Presidio or its spaCy model is missing."""
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": config.PII_SPACY_MODEL}],
        }
    )
    engine = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
    for entity, regex, score in (
        ("SELF_IDENTIFIER", _SELF_IDENTIFIER_RE, 0.9),
        ("LONG_NUMBER", _LONG_NUMBER_RE, 0.7),
        ("PHONE_NUMBER", _CONTACT_NUMBER_RE, 0.6),
    ):
        engine.registry.add_recognizer(
            PatternRecognizer(
                supported_entity=entity,
                patterns=[Pattern(name=entity.lower(), regex=regex, score=score)],
                supported_language="en",
            )
        )
    engine.analyze("warmup", language="en")  # load spaCy before the first caller
    return engine


def _pii_engine() -> Any:
    try:
        return getattr(get_runtime(), "pii", None)
    except RuntimeError:
        return None


def _regex_pii(text: str) -> PIIResult:
    """Fallback for a deployment without Presidio, matching the previous detector."""
    return PIIResult(("REGEX",), 1.0) if PII_RE.search(text) else PIIResult()


def _analyze_pii(engine: Any, text: str) -> PIIResult:
    spans = engine.analyze(
        text=text,
        language="en",
        entities=list(config.PII_ENTITIES) or None,
        score_threshold=config.PII_SCORE_THRESHOLD,
    )
    entities = tuple(dict.fromkeys(span.entity_type for span in spans))
    return PIIResult(entities=entities, score=max((span.score for span in spans), default=0.0))


async def pii_guard(text: str) -> PIIResult:
    """Stage 4: detect personal data. Never refuses; callers use the result to
    keep the turn out of the shared cache and out of conversation history."""
    if not config.PII_ENABLED:
        return PIIResult()
    text = normalize(text)[:4000]
    if not text:
        return PIIResult()
    engine = _pii_engine()
    if engine is None:
        metric(pii="fallback")
        return _regex_pii(text)

    started = time.perf_counter()
    try:
        result = await run_cpu(_analyze_pii, engine, text)
    except Exception as exc:
        # A failed scan must not let personal data into the shared cache or the
        # history; treat the text as personal and keep answering.
        log.warning("PII analysis failed (%s); treating the text as personal", type(exc).__name__)
        metric(pii="error", pii_error=type(exc).__name__, pii_ms=int((time.perf_counter() - started) * 1000))
        return PIIResult(("ERROR",), 1.0)
    metric(
        pii="found" if result.found else "none",
        pii_entities=",".join(result.entities),
        pii_ms=int((time.perf_counter() - started) * 1000),
    )
    return result


async def screen(text: str, skip: frozenset[str] = frozenset()) -> str | None:
    """Stages 1–3: normalize, apply the rules, then run the injection model."""
    text = normalize(text)
    category = rule_guard(text, skip)
    if category:
        metric(guard=category)
        return REFUSAL[category]
    if get_runtime().guard and await ml_guard(text):
        metric(guard="injection_model")
        return REFUSAL["injection"]
    metric(guard="ok")
    return None
