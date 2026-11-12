"""Document RAG pipeline (FAQ / policies / manuals)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from ..auth import Principal, auth_scope
from ..cache import cache_get, publish_answer
from ..guardrails import (
    INJECTION_RE,
    REFUSAL,
    ml_guard,
    normalize,
    pii_guard,
    rule_guard,
    self_service_request,
)
from ..llm import condense_query, stream_llm
from ..messages import FRAGMENT_REPLY, NO_HISTORY_MSG, fallback_reply
from ..observability import elapsed_ms, metric
from ..retrieval import embed_query, expand_context, hybrid_search, rerank
from ..runtime import get_runtime, limit
from ..state import (
    SELF_DISCLOSURE_RE,
    Turn,
    context_scope,
    is_contextual_followup,
    is_conversation_question,
    is_followup,
    is_name_intro,
    is_personal_statement,
    last_question,
    retrieval_text,
    search_only,
)
from ..understanding import expand_query
from .events import Event, done_event, release_task, single_reply, token_event


def _ms(start: float) -> int:
    return int((perf_counter() - start) * 1000)


def _blocked_reply(text: str) -> str:
    """A bare fragment has no request to refuse, so ask for a real one instead."""
    return FRAGMENT_REPLY if len(text.split()) <= 2 else REFUSAL["injection"]


@dataclass(frozen=True)
class QueryShape:
    """What the guards decided about one question, and which answers may be shared."""

    text: str
    scope: str
    pii: bool
    disclosure: bool
    about_conversation: bool
    personal: bool
    followup: bool
    has_history: bool

    @property
    def cacheable(self) -> bool:
        """May be read from and written to the shared answer cache."""
        return not (self.pii or self.disclosure or self.about_conversation)

    @property
    def coalesce(self) -> bool:
        """An identical question may wait for a peer's in-flight answer.

        Only questions whose answer is document-grounded and self-contained
        qualify: a follow-up depends on its session, and a personal statement or
        a conversation summary is never shared.
        """
        return self.cacheable and not self.has_history and not self.personal and not self.followup


async def query_shape(principal: Principal, message: str, history: list[Turn] | None = None) -> QueryShape:
    """Single source of truth for the guard flags and the cache identity."""
    text = normalize(message)
    history = history or []
    context = context_scope(history)
    return QueryShape(
        text=text,
        scope=f"{auth_scope(principal)}-{context}" if context else auth_scope(principal),
        pii=(await pii_guard(text)).found,
        disclosure=bool(SELF_DISCLOSURE_RE.match(text)) or is_name_intro(text),
        about_conversation=is_conversation_question(text),
        personal=is_personal_statement(text),
        followup=is_followup(text, history),
        has_history=bool(history),
    )


async def rag_pipeline(
    principal: Principal,
    message: str,
    history: list[Turn] | None = None,
    shape: QueryShape | None = None,
) -> AsyncIterator[Event]:
    history = history or []
    text = normalize(message)
    # The rules are free, so they run before the PII scan they may make unnecessary.
    category = rule_guard(text)
    if category:
        metric(guard=category, cache="skip")
        async for event in single_reply(REFUSAL[category]):
            yield event
        return

    # The agent computed the shape for coalescing; reuse it instead of scanning
    # the question for PII twice.
    shape = shape or await query_shape(principal, message, history)
    text = shape.text
    # PII, personal statements and conversation summaries are never shared.
    disclosure = shape.disclosure
    about_conversation, personal, followup = shape.about_conversation, shape.personal, shape.followup
    # A follow-up names the topic only through earlier turns, so rewrite it into
    # a standalone query before embedding. The heuristic stays as the fallback
    # when the model is unavailable, and the caller's wording still reaches the
    # LLM unchanged. A summary or a personal statement is about the conversation
    # itself, not the corpus, so it is not rewritten.
    standalone: str | None = None
    if history and not (about_conversation or personal):
        if is_contextual_followup(text, history):
            # The rewrite would only restate the previous question; the heuristic
            # below already joins them, so this saves a model call against a
            # tight token budget and still searches the subject being corrected.
            metric(condense="heuristic")
        else:
            condense_started = perf_counter()
            standalone = await condense_query(text, history)
            metric(condense_ms=_ms(condense_started), condensed=bool(standalone))
    # Canonical terms help the index; the user's wording is kept for the LLM.
    search_text = expand_query(standalone or retrieval_text(text, history))
    scope = shape.scope

    runtime = get_runtime()
    use_cache = runtime.cache_ok and shape.cacheable
    started = perf_counter()
    embed_task = asyncio.create_task(embed_query(search_text))
    search_task: asyncio.Task[list] | None = None
    cache_task: asyncio.Task[tuple[str, list] | None] | None = None
    guard_task: asyncio.Task[bool] | None = None
    rerank_task: asyncio.Task[list[dict[str, Any]]] | None = None
    try:
        dense, sparse_indices, sparse_values = await embed_task
        metric(embed_ms=_ms(started))

        # The guard starts once the query is embedded, alongside the cache
        # lookup and retrieval. Starting it earlier spends CPU on every
        # concurrent cache hit, whose answer was screened when it was stored
        # and never reaches the LLM.
        # The injection model reads "summarize what I said", "show me my
        # appointments" and "my name is …" as instructions to the model and
        # blocks them. None of these names the assistant's own instructions, and
        # rule_guard above has already screened the known injection, privacy and
        # identity phrases, so the model adds nothing for them. A self-service
        # request or a personal statement may still carry documents; the skip
        # only covers the model.
        benign = about_conversation or personal or disclosure or shape.pii or self_service_request(text)
        screen_with_model = runtime.guard and not (benign and not INJECTION_RE.search(text))
        guard_task = asyncio.create_task(ml_guard(text)) if screen_with_model else None
        cache_started = perf_counter()
        cache_task = asyncio.create_task(cache_get(dense, scope)) if use_cache else None
        retrieve_started = perf_counter()
        search_task = asyncio.create_task(hybrid_search(principal, dense, sparse_indices, sparse_values))

        if cache_task is not None:
            hit = await cache_task
            metric(cache_ms=_ms(cache_started))
            if hit:
                release_task(guard_task)
                release_task(search_task)
                metric(guard="cached", cache="hit", pre_llm_ms=_ms(started), ttft_ms=elapsed_ms())
                yield token_event(hit[0])  # one chunk: served from cache, nothing to stream
                yield done_event(hit[1], True)
                return
        metric(cache="miss" if use_cache else "skip")

        # A blocked prompt must not wait for retrieval, so take whichever
        # finishes first. Retrieval is a few milliseconds where the guard takes
        # hundreds: the rerank then starts and runs while the guard finishes.
        done, _ = await asyncio.wait(
            {task for task in (search_task, guard_task) if task is not None},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if guard_task in done and guard_task.result():
            release_task(search_task)
            metric(guard="injection_model", cache="skip", pre_llm_ms=_ms(started))
            async for event in single_reply(_blocked_reply(text)):
                yield event
            return

        points = await search_task
        metric(retrieve_ms=_ms(retrieve_started))

        candidates = [point.payload for point in points if point.payload and point.payload.get("text")]
        # Retrieved text is untrusted: drop chunks that try to issue instructions.
        safe = [chunk for chunk in candidates if not INJECTION_RE.search(normalize(chunk["text"]))]
        metric(retrieved=len(candidates), screened_out=len(candidates) - len(safe))

        rerank_started = perf_counter()
        rerank_task = asyncio.create_task(rerank(search_text, safe))
        # False for an already-safe guard; a blocked prompt cancels the rerank.
        if guard_task is not None and await guard_task:
            release_task(rerank_task)
            metric(guard="injection_model", cache="skip", pre_llm_ms=_ms(started))
            async for event in single_reply(_blocked_reply(text)):
                yield event
            return
        metric(guard="ok")
        chunks = await rerank_task
        metric(kept=len(chunks), rerank_ms=_ms(rerank_started))
        if not chunks and history:
            # The borrowed topic can drown the follow-up's own one ("and can i
            # add insurances?" after a provider search). Retry with the words as
            # typed before answering from the conversation instead.
            retry_started = perf_counter()
            retry_text = expand_query(search_only(text))
            if standalone is None and not followup:
                # "How do I make sure it is for her rather than me?" is a
                # follow-up by meaning but too long to be flagged, and on its own
                # it names nothing to search for. Retry with the last question
                # that did name a topic.
                previous = last_question(history)
                if previous and previous != text:
                    retry_text = expand_query(search_only(f"{previous} {text}"))
            retry_dense, retry_indices, retry_values = await embed_query(retry_text)
            retry_points = await hybrid_search(principal, retry_dense, retry_indices, retry_values)
            retry_chunks = [point.payload for point in retry_points if point.payload and point.payload.get("text")]
            chunks = await rerank(
                retry_text,
                [chunk for chunk in retry_chunks if not INJECTION_RE.search(normalize(chunk["text"]))],
            )
            metric(retry_ms=_ms(retry_started), retry_kept=len(chunks))
    except BaseException:
        # Also covers a client disconnect at any yield above.
        release_task(rerank_task)
        release_task(guard_task)
        release_task(embed_task)
        release_task(cache_task)
        release_task(search_task)
        raise

    context_only = False
    if about_conversation:
        # The conversation is the only source for this, even when retrieval
        # returned something: summarizing the caller's words from a PDF is wrong.
        if not history:
            metric(pre_llm_ms=_ms(started))
            async for event in single_reply(NO_HISTORY_MSG):
                yield event
            return
        documents = []
        context_only = True
        metric(context_only=True, pre_llm_ms=_ms(started))
    elif personal:
        # "i'm from new york" matches a few policy chunks by accident; a weak
        # retrieval hit must not turn a fact about the caller into a FAQ answer.
        documents = []
        context_only = True
        metric(context_only=True, pre_llm_ms=_ms(started))
    elif chunks:
        expand_started = perf_counter()
        documents = await expand_context(principal, chunks)
        metric(context_docs=len(documents), expand_ms=_ms(expand_started), pre_llm_ms=_ms(started))
    elif disclosure or followup:
        # No document answers this, but the conversation does ("I live in
        # New York" -> "Where do I live?"). The answer is not document-grounded,
        # so it is never cached and carries no sources.
        documents = []
        context_only = True
        metric(context_only=True, pre_llm_ms=_ms(started))
    else:
        metric(pre_llm_ms=_ms(started), fallback=True)
        async for event in single_reply(fallback_reply(text)):
            yield event
        return

    parts: list[str] = []
    async with limit(runtime.sem_llm):
        async for piece in stream_llm(text, documents, history):
            if not parts:
                metric(ttft_ms=elapsed_ms())
            parts.append(piece)
            yield token_event(piece)  # forwarded immediately, never buffered
    metric(llm_chunks=len(parts))

    answer = "".join(parts).strip()
    if not answer:
        async for event in single_reply(fallback_reply(text)):
            yield event
        return

    # A context-only answer never cites the chunks it ignored.
    sources = (
        []
        if context_only
        else list(
            {
                (chunk["doc"], chunk["page"]): {"document": chunk["doc"], "page": chunk["page"]} for chunk in chunks
            }.values()
        )
    )
    # Only self-contained, document-grounded answers are shared through the
    # cache; a reply that repeats what the caller told us stays in their session.
    # The entry is written before the answer is released, so a repeat or a
    # coalesced peer cannot miss it.
    if use_cache and not context_only and not history:
        await publish_answer(dense, scope, text, answer, sources)
    yield done_event(sources, False)
