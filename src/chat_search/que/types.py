"""Structured provider query produced by the Query Understanding Engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Entity:
    """One normalized mention: specialty, language or location."""

    text: str
    canonical: str
    label: str
    source: str = "catalogue"
    start: int = 0
    end: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "canonical": self.canonical,
            "label": self.label,
            "source": self.source,
        }


@dataclass(frozen=True)
class ProviderQuery:
    """The QUE output handed to Elasticsearch; access fields never live here."""

    original: str
    normalized: str
    intent: str
    specialties: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    cities: tuple[str, ...] = ()
    terms: str = ""
    entities: tuple[Entity, ...] = ()
    language_request: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "specialty": self.specialties[0] if self.specialties else None,
            "language": self.languages[0] if self.languages else None,
            "city": self.cities[0] if self.cities else None,
            "specialties": list(self.specialties),
            "languages": list(self.languages),
            "cities": list(self.cities),
            "terms": self.terms,
            "entities": [entity.as_dict() for entity in self.entities],
        }
