"""HTTP API request/response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .. import config
from ..state import SESSION_RE


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=config.MAX_MESSAGE_CHARS)
    mode: Literal["auto", "faq", "providers"] = "auto"  # auto = route by intent
    # Omit to continue the caller's rolling conversation; send one to keep
    # parallel threads apart.
    session_id: str | None = Field(default=None, max_length=64, pattern=SESSION_RE.pattern)


class Source(BaseModel):
    document: str
    page: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    cached: bool = False
    data: dict | None = None  # structured results, e.g. {"providers": [...]}
