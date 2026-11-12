"""Stage 3: retrieval knobs that operators can change without touching code."""

from __future__ import annotations

from dataclasses import dataclass

from .. import config

# Words that describe the request, not the provider. Operators can replace the
# list with PROVIDER_STOPWORDS; the language cues are always dropped because the
# extractor has already turned them into a filter.
DEFAULT_STOPWORDS = frozenset(
    """recommend recommended recommendation recommendations suggest suggestion find finding search searching
    need needs give want looking look please pls me i my we us you your can could would should will who whom which whose
    that speaks speak speaking near nearby around good best top great doctor doctors dr physician physicians
    provider providers specialist specialists clinician clinicians help get see show give list any some
    available accepting new patient patients do have has is are there a an the to for in of with where what how
    and or rating ratings rated highest language languages fluent fluently primary""".split()
)
_LANGUAGE_CUE_WORDS = frozenset({"speak", "speaks", "speaking", "fluent", "fluently", "language", "languages"})


@dataclass(frozen=True)
class RetrievalSettings:
    stopwords: frozenset[str]
    strict_msm: str
    relaxed_msm: str
    fuzziness: str | None
    fuzzy_prefix_length: int
    fuzzy_max_expansions: int
    specialty_boost: float
    name_boost: float
    conditions_boost: float
    accepting_boost: float
    rating_modifier: str
    rating_missing: float


def stopwords() -> frozenset[str]:
    override = frozenset(config.PROVIDER_STOPWORDS.split())
    return (override | _LANGUAGE_CUE_WORDS) if override else DEFAULT_STOPWORDS


def retrieval_settings() -> RetrievalSettings:
    """Read the live configuration so a test or operator can override it."""
    return RetrievalSettings(
        stopwords=stopwords(),
        strict_msm=config.PROVIDER_STRICT_MSM,
        relaxed_msm=config.PROVIDER_RELAXED_MSM,
        fuzziness=config.PROVIDER_FUZZINESS or None,
        fuzzy_prefix_length=config.PROVIDER_FUZZY_PREFIX_LENGTH,
        fuzzy_max_expansions=config.PROVIDER_FUZZY_MAX_EXPANSIONS,
        specialty_boost=config.PROVIDER_BOOST_SPECIALTY,
        name_boost=config.PROVIDER_BOOST_NAME,
        conditions_boost=config.PROVIDER_BOOST_CONDITIONS,
        accepting_boost=config.PROVIDER_BOOST_ACCEPTING,
        rating_modifier=config.PROVIDER_RATING_MODIFIER,
        rating_missing=config.PROVIDER_RATING_MISSING,
    )
