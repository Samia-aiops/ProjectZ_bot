"""Chat orchestration: greeting shortcut, rate limit, agent, per-user slots."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from .. import config
from ..audit import audit_chat
from ..auth import Principal
from ..errors import AppError
from ..guardrails import REFUSAL, greeting_reply, normalize, pii_guard
from ..messages import (
    FALLBACK_TEXTS,
    FRAGMENT_REPLY,
    NO_HISTORY_MSG,
    PROVIDER_DENIED_MSG,
    PROVIDER_PERMISSION_MSG,
    SEARCH_UNAVAILABLE_MSG,
)
from ..observability import elapsed_ms, metric
from ..rate_limit import enforce_rate_limit
from ..runtime import get_runtime, user_key
from ..spelling import spell_correct
from ..state import append_turns, load_history
from ..understanding import bind_understanding, unbind_understanding, understand
from .agent import agent_events
from .events import Event, safe_events, single_reply

# Canned replies (a fragment prompt, a directory outage, a role refusal) say
# nothing about the caller's topic, so they never become context for the next turn.
CANNED_REPLIES = FALLBACK_TEXTS | {
    FRAGMENT_REPLY,
    NO_HISTORY_MSG,
    PROVIDER_DENIED_MSG,
    PROVIDER_PERMISSION_MSG,
    SEARCH_UNAVAILABLE_MSG,
}


async def rag_events(
    principal: Principal,
    message: str,
    mode: str = "auto",
    session_id: str | None = None,
) -> AsyncIterator[Event]:
    """Correct the query, route it and hold the user's slot for the full answer."""
    normalized = normalize(message)
    spell_started = time.perf_counter()
    text = spell_correct(normalized)
    metric(spell_ms=int((time.perf_counter() - spell_started) * 1000))
    if text != normalized:
        metric(spelling="corrected")
    if reply := greeting_reply(text):  # "helllo" still gets the cheap greeting
        metric(guard="greeting", cache="skip")
        audit_chat(
            principal,
            question=normalized,
            answer=reply,
            node="greeting",
            session_id=session_id,
            mode=mode,
            latency_ms=elapsed_ms(),
        )
        async for event in single_reply(reply):
            yield event
        return

    parts: list[str] = []
    sources: list[dict[str, Any]] = []
    providers: list[dict[str, Any]] = []
    cached = False
    node = "faq"
    question_pii: bool | None = None
    history_turns = 0
    outcome = "ok"
    error: str | None = None
    started = time.perf_counter()
    understanding = understand(text, spelling=False)
    understanding_token = bind_understanding(understanding)
    try:
        async with get_runtime().user_slots.hold(user_key(principal), config.USER_WAIT_TIMEOUT):
            # Read inside the slot: this request cannot start from a history that a
            # previous answer of the same caller is still writing.
            history = await load_history(principal, session_id)
            history_turns = len(history)
            metric(history_turns=history_turns)

            pipeline_message = text
            # The agent picks the pipeline (LLM router node, conditional edge to
            # the RAG or provider node) and streams it back; meta events carry the
            # routing decision and the PII verdict without reaching the client.
            async for event in agent_events(principal, pipeline_message, history, mode):
                kind = event["type"]
                if kind == "meta":
                    if "route" in event:
                        node = event["route"]
                        metric(node=node)
                    if "pii" in event:
                        question_pii = bool(event["pii"])
                    continue
                if kind == "token":
                    parts.append(event["text"])
                elif kind == "done":
                    sources = event.get("sources") or []
                    cached = bool(event.get("cached"))
                    providers = (event.get("data") or {}).get("providers") or []
                    data = {**(event.get("data") or {}), "understanding": understanding.as_dict()}
                    event = {**event, "data": data}
                yield event

            # Store the turn before releasing the slot: a follow-up sent the instant
            # the answer lands must see this question in its history.
            answer = "".join(parts).strip()
            # Both sides are checked: the model can echo identifying data that
            # the question itself did not contain. A provider question has no
            # shape, so the question is scanned here too.
            if question_pii is None:
                question_pii = (await pii_guard(pipeline_message)).found
            if (
                answer
                and answer not in CANNED_REPLIES
                and answer not in REFUSAL.values()
                and not question_pii
                and not (await pii_guard(answer)).found
            ):
                await append_turns(principal, session_id, pipeline_message, answer)
    except (GeneratorExit, asyncio.CancelledError):
        outcome = "cancelled"
        raise
    except Exception as exc:
        outcome = "error"
        error = getattr(exc, "code", None) or type(exc).__name__
        raise
    finally:
        unbind_understanding(understanding_token)
        audit_chat(
            principal,
            question=normalized,
            answer="".join(parts),
            sources=sources,
            providers=providers,
            node=node,
            session_id=session_id,
            mode=mode,
            cached=cached,
            outcome=outcome,
            error=error,
            latency_ms=int((time.perf_counter() - started) * 1000),
            history_turns=history_turns,
        )


async def open_events(
    principal: Principal,
    message: str,
    mode: str = "auto",
    session_id: str | None = None,
) -> AsyncIterator[Event]:
    """Pre-stream checks raise real HTTP errors (e.g. 429); everything after is an event."""
    normalized = normalize(message)
    reply = greeting_reply(normalized)
    if reply is None and len(normalized) <= 60:  # a greeting is never longer
        # "helllo" must take the same cheap path as "hello": it is corrected
        # before the rate limiter so the exemption does not depend on spelling.
        reply = greeting_reply(spell_correct(normalized))
    if reply:  # first: no rate limit, models, Redis or LLM
        metric(guard="greeting", cache="skip", ttft_ms=elapsed_ms())
        audit_chat(
            principal,
            question=normalized,
            answer=reply,
            node="greeting",
            session_id=session_id,
            mode=mode,
            latency_ms=elapsed_ms(),
        )
        return single_reply(reply)
    try:
        await enforce_rate_limit(principal)
    except AppError as exc:
        audit_chat(
            principal,
            question=normalized,
            answer="",
            node="rate_limit",
            session_id=session_id,
            mode=mode,
            outcome="rate_limited",
            error=exc.code,
            latency_ms=elapsed_ms(),
        )
        raise
    return safe_events(rag_events(principal, message, mode, session_id))
