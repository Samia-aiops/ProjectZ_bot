"""LangGraph agent: one routing decision, then document RAG or provider search.

The graph is the only place that picks a pipeline. The router node asks a
System 1 model for ``faq`` or ``providers``: by default Laya, a small
non-autoregressive checkpoint that decides in one forward pass instead of
spending the rate-limited completion budget on a routing prompt. The conditional
edge dispatches to the matching node, and each node forwards the existing
pipeline's events through LangGraph's custom stream so tokens still reach the
client as they arrive.

Deterministic rules stay in front of the model (an explicit ``mode``, an
appointment/booking question, a correction that follows the previous turn) and
behind it: when Laya is unloaded, slow or unsure the LLM router is asked, and
when that is disabled or unavailable the rules decide.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any, TypedDict

import httpx
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from .. import config
from ..auth import Principal
from ..cache import await_peer, claim_inflight, release_inflight
from ..errors import AppError
from ..guardrails import REFUSAL, normalize, rule_guard
from ..logging_setup import log_failure
from ..observability import elapsed_ms, metric
from ..routing import FAQ_HINT_RE, route
from ..runtime import get_runtime, limit, run_cpu
from ..state import Turn, is_contextual_followup, last_question
from ..understanding import understand
from .events import Event, done_event, single_reply, token_event
from .faq import query_shape, rag_pipeline
from .provider import provider_search_pipeline
from .provider_sql import provider_sql_pipeline

log = logging.getLogger(__name__)

ROUTER_PROMPT = """
You route a patient's or provider's message for a healthcare software platform's support
assistant. Choose exactly one label.

faq        - questions about the platform: policies, onboarding, how to use a
             screen, files and documents, billing, appointments and booking
             steps, and any clinical or medical question.
providers  - the user wants to find, list or compare doctors or clinics:
             a specialty ("heart doctor"), a doctor's name, a city, a language a
             doctor speaks, availability or ratings.
analytics  - provider-only: questions about their own appointments, availability,
             schedules, patient summaries under their care, or performance data.

Rules:
1. "How do I book/move/cancel an appointment?" is faq, even when it names a
   specialty.
2. "Find a doctor who speaks Urdu / works in Boston / treats children" is
   providers.
3. "Show me my appointments for today/week/month" or "How is my performance?" is analytics.
4. A follow-up without its own subject ("and who speaks Urdu?") uses the
   earlier turns to decide.
5. Never answer the message.

Reply with the label only: faq, providers or analytics.
""".strip()

_PROVIDER_LABELS = {"providers", "provider", "search", "directory"}
_FAQ_LABELS = {"faq", "rag", "documents"}
_ANALYTICS_LABELS = {"analytics", "appointments", "data", "schedule", "performance"}


class AgentState(TypedDict, total=False):
    principal: Principal
    question: str
    history: list[Turn]
    mode: str
    route: str
    routed_by: str


def _parse_label(content: str) -> str | None:
    line = next((line.strip() for line in content.splitlines() if line.strip()), "")
    label = line.lower().strip(".\"'`*_ ").split(" ")[0] if line else ""
    if label in _FAQ_LABELS:
        return "faq"
    if label in _PROVIDER_LABELS:
        return "providers"
    if label in _ANALYTICS_LABELS:
        return "analytics"
    return None


def _router_messages(question: str, history: list[Turn]) -> list[dict[str, str]]:
    transcript = "\n".join(f"{turn.role}: {turn.text}" for turn in history)
    conversation = f"<conversation>\n{transcript}\n</conversation>\n\n" if transcript else ""
    return [
        {"role": "system", "content": ROUTER_PROMPT},
        {"role": "user", "content": f"{conversation}MESSAGE:\n{question}"},
    ]


async def _ask_router(question: str, history: list[Turn]) -> str | None:
    """The model's label, or None so the caller falls back to the rules."""
    if not config.AGENT_ROUTER_ENABLED or not config.LLM_API_KEY:
        return None
    payload: dict[str, Any] = {
        "model": config.LLM_MODEL,
        "temperature": 0,
        # gpt-oss counts reasoning tokens here; a small cap can spend them all
        # before the label and return empty content.
        "max_tokens": 300,
        "messages": _router_messages(question, history),
    }
    if config.LLM_REASONING_EFFORT:
        payload["reasoning_effort"] = config.LLM_REASONING_EFFORT
    runtime = get_runtime()
    try:
        async with limit(runtime.sem_llm):
            response = await runtime.http.post(
                f"{config.LLM_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
                json=payload,
                timeout=config.AGENT_ROUTER_TIMEOUT,
            )
    except (httpx.HTTPError, AppError):  # unavailable or no slot: the rules still answer
        metric(router="unavailable")
        return None
    if response.status_code >= 400:
        metric(router=str(response.status_code))
        return None
    try:
        choices = response.json().get("choices") or []
        content = (choices[0].get("message") or {}).get("content") or ""
    except (ValueError, AttributeError, IndexError):
        metric(router="bad_payload")
        return None
    label = _parse_label(content)
    if label is None:
        metric(router="unparsed")
    return label


_LAYA_QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": (
            "Choose where a healthcare platform's support assistant should answer the message. "
            "A follow-up without a subject of its own takes its topic from the earlier turns."
        ),
        "criteria": {
            "faq": (
                "Questions about the platform itself: policies, onboarding, how to use a screen, "
                "files and documents, billing, appointments and booking steps, and clinical or "
                "medical questions."
            ),
            "providers": (
                "The user wants to find, list or compare doctors or clinics: by specialty, "
                "doctor's name, city, a language a doctor speaks, availability or ratings."
            ),
            "analytics": (
                "Provider-only: questions about their own appointments, availability, schedules, "
                "patient summaries under their care, or performance data."
            ),
        },
    }
}

# Kept short enough that the checkpoint's 512-token window holds the message and
# the instructions without truncating either.
_LAYA_HISTORY_CHARS = 200
_LAYA_MESSAGE_CHARS = 600

_laya_agent: Any = None
_laya_failed = False


def _laya_state(question: str, history: list[Turn]) -> str:
    turns = [f"{turn.role}: {turn.text[:_LAYA_HISTORY_CHARS]}" for turn in history[-2:]]
    turns.append(f"user: {question[:_LAYA_MESSAGE_CHARS]}")
    return "\n".join(turns)


def _load_laya() -> Any:
    import laya

    return laya.load(
        config.AGENT_ROUTER_MODEL,
        device=config.AGENT_ROUTER_DEVICE,
        subfolder=config.AGENT_ROUTER_SUBFOLDER or None,
    )


async def warm_router() -> bool:
    """Load Laya off the event loop; until it is ready the LLM router answers.

    A failed load is not retried: a missing checkpoint would otherwise stall a
    request's routing budget on every message instead of falling back once.
    """
    global _laya_agent, _laya_failed
    if config.AGENT_ROUTER_BACKEND != "laya":
        return False
    if _laya_agent is not None:
        return True
    if _laya_failed:
        return False
    try:
        _laya_agent = await asyncio.to_thread(_load_laya)
    except Exception as exc:
        _laya_failed = True
        log_failure(log, "Laya router unavailable; the LLM routes instead", exc)
        return False
    metric(router="laya_ready")
    return True


async def _ask_laya(question: str, history: list[Turn]) -> str | None:
    """Laya's route, or None when it is not loaded, slow, unsure or unexpected."""
    agent = _laya_agent
    if agent is None:
        return None
    state = _laya_state(question, history)
    try:
        result = await run_cpu(
            lambda: agent.predict(state, _LAYA_QUESTIONS),
            timeout=config.AGENT_ROUTER_LAYA_TIMEOUT,
        )
    except Exception:
        # Timeout or model error: the rules answer faster than asking the LLM now.
        metric(laya="error")
        return None
    answer = ((result or {}).get("answers") or {}).get("route") or {}
    raw = answer.get("answer_confidence")
    if raw is None:
        raw = answer.get("confidence")
    confidence = float(raw) if isinstance(raw, (int, float)) else 0.0
    metric(laya_confidence=round(confidence, 3))
    choice = answer.get("choice")
    if choice not in ("faq", "providers", "analytics"):
        metric(laya="unparsed")
        return None
    if confidence < config.AGENT_ROUTER_MIN_CONFIDENCE:
        metric(laya="low_confidence")
        return None
    return choice


def _guard_label(question: str, mode: str, label: str, source: str) -> tuple[str, str]:
    """The deterministic rules keep a model label precise in both directions.

    A directory phrase stays providers when the model misses the language filter
    ("who speaks Urdu?"), and a product question stays documents even when the
    model reads a medical word as a specialty ("does DocHyve provide medical
    care?"). The FAQ hint is checked here, not only on the Laya branch, so a
    label that came from the LLM router gets the same guard.
    """
    deterministic = route(question, mode)
    if deterministic == "providers" and label != "providers":
        metric(laya="overruled_by_rules")
        return "providers", "rules"
    if label == "providers" and FAQ_HINT_RE.search(question):
        metric(laya="faq_hint")
        return deterministic, "rules"
    return label, source


async def decide_route(question: str, history: list[Turn], mode: str) -> tuple[str, str]:
    """Return ``(route, routed_by)``; ``routed_by`` is mode, rules, history, laya or llm."""
    if mode in ("faq", "providers"):
        return mode, "mode"
    # Booking steps name a specialty ("book an appointment with a heart doctor")
    # but are answered from the documents, so the model is never asked.
    if understand(question, medical=False).appointment is not None:
        return "faq", "rules"
    # A correction or "continue" borrows the route of the question it refers to;
    # asking the model would re-derive that from the same history and spend
    # tokens that the answer needs on a rate-limited deployment.
    if is_contextual_followup(question, history):
        previous = last_question(history)
        if previous:
            metric(router="history")
            return route(previous, mode), "history"

    started = time.perf_counter()
    backend = config.AGENT_ROUTER_BACKEND
    label: str | None = None
    source = "rules"
    if backend == "laya":
        label = await _ask_laya(question, history)
        source = "laya"
        if label is None and _laya_agent is None:
            # Still warming up (or the checkpoint failed to load): the LLM is
            # the better answer than the rules while Laya cannot decide.
            label = await _ask_router(question, history)
            source = "llm"
    elif backend == "llm":
        label = await _ask_router(question, history)
        source = "llm"
    metric(router_ms=int((time.perf_counter() - started) * 1000))
    if label is None:
        metric(router="rules")
        return route(question, mode), "rules"
    label, source = _guard_label(question, mode, label, source)
    metric(router=source)
    return label, source


async def _router_node(state: AgentState) -> dict[str, str]:
    question = state["question"]
    # Screen before routing: a refusal must not depend on which pipeline the
    # router picks, and neither pipeline should pay for retrieval or a model call
    # to answer a message that is blocked outright.
    category = rule_guard(normalize(question))
    if category:
        metric(guard=category, agent_route="blocked")
        writer = get_stream_writer()
        writer({"type": "meta", "route": "blocked", "routed_by": "guard"})
        async for event in single_reply(REFUSAL[category]):
            writer(event)
        return {"route": "blocked", "routed_by": "guard"}

    selected, source = await decide_route(question, state.get("history") or [], state.get("mode", "auto"))
    metric(agent_route=selected)
    get_stream_writer()({"type": "meta", "route": selected, "routed_by": source})
    return {"route": selected, "routed_by": source}


async def _rag_node(state: AgentState) -> dict[str, Any]:
    principal = state["principal"]
    question = state["question"]
    history = state.get("history") or []
    writer = get_stream_writer()
    shape = await query_shape(principal, question, history)
    writer({"type": "meta", "pii": shape.pii})

    # "" means no marker to release; None means a peer is already answering this
    # exact question, so wait for its answer instead of repeating the rerank and
    # the LLM call.
    marker = ""
    try:
        if shape.coalesce:
            marker = await claim_inflight(shape.scope, shape.text)
            if marker is None:
                # Set before waiting: a disconnect must not release a peer's marker.
                marker = ""
                peer = await await_peer(shape.scope, shape.text)
                if peer:
                    metric(guard="cached", cache="hit", coalesce="hit", pre_llm_ms=elapsed_ms())
                    metric(ttft_ms=elapsed_ms())
                    writer(token_event(peer[0]))
                    writer(done_event(peer[1], True))
                    return {}
        async with aclosing(rag_pipeline(principal, question, history, shape)) as events:
            async for event in events:
                writer(event)
    finally:
        # Also covers a client disconnect: the next identical question must not
        # wait on a request that is no longer running.
        await release_inflight(marker)
    return {}


async def _search_node(state: AgentState) -> dict[str, Any]:
    writer = get_stream_writer()
    async with aclosing(
        provider_search_pipeline(state["principal"], state["question"], state.get("history"))
    ) as events:
        async for event in events:
            writer(event)
    return {}


async def _analytics_node(state: AgentState) -> dict[str, Any]:
    writer = get_stream_writer()
    async with aclosing(provider_sql_pipeline(state["principal"], state["question"], state.get("history"))) as events:
        async for event in events:
            writer(event)
    return {}


def _next_node(state: AgentState) -> str:
    if state.get("route") == "blocked":
        return END
    route = state["route"]
    if route == "faq":
        return "rag"
    if route == "providers":
        return "search"
    if route == "analytics":
        return "analytics"
    return END


def _build() -> Any:
    builder = StateGraph(AgentState)
    builder.add_node("router", _router_node)
    builder.add_node("rag", _rag_node)
    builder.add_node("search", _search_node)
    builder.add_node("analytics", _analytics_node)
    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router", _next_node, {"rag": "rag", "search": "search", "analytics": "analytics", END: END}
    )
    builder.add_edge("rag", END)
    builder.add_edge("search", END)
    builder.add_edge("analytics", END)
    return builder.compile()


_graph = _build()


async def agent_events(
    principal: Principal,
    message: str,
    history: list[Turn] | None = None,
    mode: str = "auto",
) -> AsyncIterator[Event]:
    """Run the agent and stream its events; ``meta`` events describe the routing."""
    state: AgentState = {
        "principal": principal,
        "question": message,
        "history": history or [],
        "mode": mode,
    }
    async with aclosing(_graph.astream(state, stream_mode="custom")) as stream:
        async for chunk in stream:
            if isinstance(chunk, dict) and "type" in chunk:
                yield chunk
