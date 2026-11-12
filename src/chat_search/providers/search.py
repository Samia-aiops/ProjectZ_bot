"""Presentation of provider directory hits.

Query understanding and retrieval live in :mod:`chat_search.que`; this module
only turns the returned public fields into the answer text.
"""

from __future__ import annotations

from typing import Any


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)] if value else []


def _language_name(language: str) -> str:
    return "ASL" if language == "asl" else language.title()


def format_providers(hits: list[dict[str, Any]], relaxed: bool, searched: bool, language: str | None = None) -> str:
    if not hits:
        return (
            "I couldn't find any providers matching that. Try a specialty (for example cardiology), "
            "a language or a city."
        )
    if language:
        name = _language_name(language)
        # The fallback drops the other words (specialty, city) to keep the
        # language requirement, so say which constraint was kept.
        head = (
            f"I couldn't find a provider matching all of those details who speaks {name}. "
            f"These providers speak {name}:"
        )
    elif not searched:
        head = "Here are some top-rated providers:"
    elif relaxed:
        head = "I couldn't find an exact match, so these are the closest providers:"
    else:
        head = "Here are providers that match your search:"

    lines = []
    for index, hit in enumerate(hits, start=1):
        line = f"{index}. {hit['name']}"
        if hit.get("specialty"):
            subspecialties = _as_list(hit.get("subspecialties"))[:2]
            line += f" \u2014 {hit['specialty']}"
            if subspecialties:
                line += f" ({', '.join(subspecialties)})"
        extra = []
        if hit.get("city"):
            extra.append(str(hit["city"]))
        if languages := _as_list(hit.get("languages")):
            extra.append("Speaks " + ", ".join(languages))
        try:
            rating = f"Rating {float(hit['rating']):.1f}"
            extra.append(rating + (f" ({int(hit['review_count'])} reviews)" if hit.get("review_count") else ""))
        except (KeyError, TypeError, ValueError):
            pass
        if hit.get("accepting_new_patients"):
            extra.append("Accepting new patients")
        if hit.get("next_available"):
            extra.append("Next available " + str(hit["next_available"])[:10])
        lines.append(line + (" \u00b7 " + " \u00b7 ".join(extra) if extra else ""))
    return head + "\n\n" + "\n".join(lines) + "\n\nThis is a directory search, not medical advice."
