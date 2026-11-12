"""Text helpers shared by retrieval and indexing.

They live outside ``indexing`` so serving never imports write-side modules.
"""

from __future__ import annotations

from typing import Any


def est_tokens(text: str) -> int:
    return int(max(len(text.split()) * 1.3, len(text) / 4)) + 1


def embed_text(title: str, chunk: dict[str, Any]) -> str:
    """What gets embedded / reranked: the chunk with its title and section path in front."""
    head = f"{title} > {chunk['section']}" if chunk.get("section") else title
    return f"{head}\n{chunk['text']}"
