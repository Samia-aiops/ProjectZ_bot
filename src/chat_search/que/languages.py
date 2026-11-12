"""Language catalogue used by the QUE normalize stage and the ES filters.

The keys are the canonical language names the rest of the service uses. The
values are the ways the directory writes the same language, so one filter can
match every spelling ("farsi" also reaches "persian").
"""

from __future__ import annotations

from ..concepts import normalize_text

LANGUAGE_CATALOGUE: dict[str, tuple[str, ...]] = {
    "albanian": ("albanian",),
    "amharic": ("amharic",),
    "arabic": ("arabic",),
    "armenian": ("armenian",),
    "asl": ("american sign language", "asl"),
    "bengali": ("bengali", "bangla"),
    "bangla": ("bangla", "bengali"),
    "bosnian": ("bosnian",),
    "bulgarian": ("bulgarian",),
    "burmese": ("burmese", "myanmar"),
    "cantonese": ("cantonese",),
    "chinese": ("chinese", "mandarin", "cantonese"),
    "creole": ("creole", "haitian creole"),
    "croatian": ("croatian",),
    "czech": ("czech",),
    "dari": ("dari",),
    "dutch": ("dutch",),
    "english": ("english",),
    "farsi": ("farsi", "persian"),
    "filipino": ("filipino", "tagalog"),
    "french": ("french",),
    "georgian": ("georgian",),
    "german": ("german",),
    "greek": ("greek",),
    "gujarati": ("gujarati",),
    "haitian creole": ("haitian creole", "creole"),
    "hebrew": ("hebrew",),
    "hindi": ("hindi",),
    "hungarian": ("hungarian",),
    "igbo": ("igbo",),
    "indonesian": ("indonesian", "bahasa"),
    "italian": ("italian",),
    "japanese": ("japanese",),
    "kannada": ("kannada",),
    "khmer": ("khmer", "cambodian"),
    "korean": ("korean",),
    "kurdish": ("kurdish",),
    "lao": ("lao",),
    "malay": ("malay",),
    "malayalam": ("malayalam",),
    "mandarin": ("mandarin", "chinese"),
    "marathi": ("marathi",),
    "nepali": ("nepali",),
    "pashto": ("pashto",),
    "persian": ("persian", "farsi"),
    "polish": ("polish",),
    "portuguese": ("portuguese",),
    "punjabi": ("punjabi",),
    "romanian": ("romanian",),
    "russian": ("russian",),
    "serbian": ("serbian",),
    "sign language": ("sign language", "american sign language", "asl"),
    "sinhala": ("sinhala", "sinhalese"),
    "slovak": ("slovak",),
    "somali": ("somali",),
    "spanish": ("spanish",),
    "swahili": ("swahili",),
    "tagalog": ("tagalog", "filipino"),
    "tamil": ("tamil",),
    "telugu": ("telugu",),
    "thai": ("thai",),
    "turkish": ("turkish",),
    "ukrainian": ("ukrainian",),
    "urdu": ("urdu",),
    "vietnamese": ("vietnamese",),
    "yoruba": ("yoruba",),
}

# Canonical keys win over an alias that happens to be another key ("persian").
_ALIAS_TO_KEY: dict[str, str] = {}
for _key in LANGUAGE_CATALOGUE:
    _ALIAS_TO_KEY[normalize_text(_key)] = _key
for _key, _labels in LANGUAGE_CATALOGUE.items():
    for _label in _labels:
        _ALIAS_TO_KEY.setdefault(normalize_text(_label), _key)


def language_key(text: str) -> str | None:
    """Canonical language for a mention, or None when it is not in the catalogue."""
    return _ALIAS_TO_KEY.get(normalize_text(text))


def language_labels(language: str) -> tuple[str, ...]:
    """Directory spellings for a canonical language (itself when unknown)."""
    return LANGUAGE_CATALOGUE.get(language, (language,))


def language_aliases() -> tuple[tuple[str, str], ...]:
    """All catalogue spellings mapped to their canonical key, longest first."""
    return tuple(sorted(_ALIAS_TO_KEY.items(), key=lambda item: len(item[0]), reverse=True))
