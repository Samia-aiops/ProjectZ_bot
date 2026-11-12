"""Hybrid retrieval over Qdrant (read-only).

The same RBAC filter is applied to the dense leg, the sparse leg, the fusion,
and every follow-up read (whole document / neighbour chunks). Retrieved text is
untrusted and is screened again before it reaches the prompt. Indexing lives in
``chat_search.indexing`` and is never imported here.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from qdrant_client import models

from . import config
from .auth import Principal
from .errors import AppError
from .guardrails import INJECTION_RE, normalize
from .messages import BUSY_MSG
from .observability import metric
from .runtime import get_runtime, limit, run_cpu
from .text import embed_text, est_tokens


def rbac_filter(principal: Principal) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchAny(any=[principal.tenant_id, config.GLOBAL_TENANT]),
            ),
            models.FieldCondition(
                key="audience",
                match=models.MatchAny(any=config.ROLE_AUDIENCES[principal.role]),
            ),
        ]
    )


def eq_condition(key: str, value: str) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _embed_query_sync(text: str) -> tuple[np.ndarray, list[int], list[float]]:
    runtime = get_runtime()
    dense = runtime.embedder.encode(
        config.EMBED_QUERY_PREFIX + text, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)
    sparse = next(iter(runtime.sparse.query_embed(text, show_progress_bar=False)))
    return dense, sparse.indices.tolist(), sparse.values.tolist()


async def embed_query(text: str) -> tuple[np.ndarray, list[int], list[float]]:
    runtime = get_runtime()
    async with limit(runtime.sem_rag):
        # A dedicated small pool keeps the short query embedding ahead of long
        # rerank/guard jobs that share the main executor.
        return await run_cpu(_embed_query_sync, text, executor=runtime.embed_executor)


async def hybrid_search(
    principal: Principal,
    dense: np.ndarray,
    sparse_indices: list[int],
    sparse_values: list[float],
) -> list[models.ScoredPoint]:
    query_filter = rbac_filter(principal)
    prefetch = [models.Prefetch(query=dense.tolist(), using="dense", filter=query_filter, limit=config.PREFETCH_LIMIT)]
    if sparse_indices:  # an empty sparse query (e.g. only stop-words) -> dense leg only
        prefetch.append(
            models.Prefetch(
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse",
                filter=query_filter,
                limit=config.PREFETCH_LIMIT,
            )
        )
    runtime = get_runtime()
    async with limit(runtime.sem_rag):
        try:
            result = await asyncio.wait_for(
                runtime.qdrant.query_points(
                    config.QDRANT_COLLECTION,
                    prefetch=prefetch,
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    query_filter=query_filter,
                    limit=config.HYBRID_LIMIT,
                    with_payload=True,
                ),
                config.QDRANT_TIMEOUT,
            )
        except TimeoutError:
            raise AppError(504, "timeout", BUSY_MSG) from None
    return result.points


def _rerank_sync(query: str, texts: list[str]) -> list[float]:
    return (
        get_runtime()
        .reranker.predict(
            [(query, text) for text in texts],
            batch_size=16,
            show_progress_bar=False,
        )
        .tolist()
    )


async def rerank(query: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runtime = get_runtime()
    if not runtime.reranker or len(chunks) < 2:
        return chunks[: config.TOP_K]
    async with limit(runtime.sem_rerank):
        scores = await run_cpu(_rerank_sync, query, [embed_text(chunk["doc"], chunk) for chunk in chunks])
    ranked = sorted(zip(scores, chunks, strict=True), key=lambda pair: pair[0], reverse=True)
    # Cross-encoder logits are not calibrated across phrasings (the same chunk
    # scores from +9 to -11), so the best chunk sets the scale: keep what is
    # close to it, with the absolute floor only as a safety net.
    cutoff = max(ranked[0][0] - config.RERANK_RELATIVE_MARGIN, config.RERANK_MIN_SCORE)
    return [{**chunk, "_score": score} for score, chunk in ranked if score >= cutoff][: config.TOP_K]


def document_entry(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """One <document> for the LLM: chunks in reading order, with their section headings."""
    parts: list[str] = []
    last_section: str | None = None
    for chunk in chunks:
        section = chunk.get("section") or ""
        if section and section != last_section:
            parts.append(f"[{section}]")
        parts.append(chunk["text"])
        last_section = section
    low, high = min(chunk["page"] for chunk in chunks), max(chunk["page_end"] for chunk in chunks)
    return {
        "doc": chunks[0]["doc"],
        "page": str(low) if low == high else f"{low}-{high}",
        "text": "\n".join(parts),
    }


async def fetch_chunks(principal: Principal, conditions: list[Any], max_points: int) -> list[dict[str, Any]]:
    """Extra Qdrant reads (whole document / neighbours) carry the RBAC filter too."""
    base = rbac_filter(principal)
    query_filter = models.Filter(must=[*(base.must or []), *conditions])
    runtime = get_runtime()
    async with limit(runtime.sem_rag):
        try:
            points, _ = await asyncio.wait_for(
                runtime.qdrant.scroll(
                    config.QDRANT_COLLECTION,
                    scroll_filter=query_filter,
                    limit=max_points,
                    with_payload=True,
                ),
                config.QDRANT_TIMEOUT,
            )
        except TimeoutError:
            raise AppError(504, "timeout", BUSY_MSG) from None
    return [point.payload for point in points if point.payload]


async def expand_context(principal: Principal, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Search small, answer big: the LLM gets the whole short document, or the neighbours of each hit."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:  # reranked order: best document first
        groups.setdefault(chunk["source"], []).append(chunk)

    best = max((chunk.get("_score", 0.0) for chunk in chunks), default=0.0)
    plans: list[tuple[list[dict[str, Any]], list[Any] | None, int]] = []
    for rank, (source, hits) in enumerate(groups.items()):
        picked = sorted(hits, key=lambda chunk: chunk["order"])
        strong = max(chunk.get("_score", 0.0) for chunk in hits) >= best - config.EXPAND_SCORE_MARGIN
        conditions: list[Any] | None = None
        max_points = 0
        if rank < config.EXPAND_DOCS and strong and hits[0].get("kind") != "faq":  # an FAQ chunk already has its answer
            try:
                if hits[0]["doc_tokens"] <= config.EXPAND_DOC_TOKENS:
                    conditions, max_points = [eq_condition("source", source)], 64
                else:
                    orders = sorted(
                        {
                            chunk["order"] + delta
                            for chunk in hits
                            for delta in (-1, 0, 1)
                            if chunk["order"] + delta >= 0
                        }
                    )
                    order_filter = models.FieldCondition(key="order", match=models.MatchAny(any=orders))
                    conditions, max_points = [eq_condition("source", source), order_filter], len(orders)
            except Exception as exc:  # expansion is an optimisation: fall back to the hits
                metric(expand_error=type(exc).__name__)
        plans.append((picked, conditions, max_points))

    # Independent Qdrant reads run together; a failed read falls back to its hits.
    pending = [plan for plan in plans if plan[1] is not None]
    reads: list[Any] = []
    if pending:
        reads = await asyncio.gather(
            *(fetch_chunks(principal, conditions, max_points) for _, conditions, max_points in pending),
            return_exceptions=True,
        )
    fetched = iter(reads)

    documents: list[dict[str, Any]] = []
    budget = config.MAX_CONTEXT_TOKENS
    for picked, conditions, _ in plans:
        if conditions is not None:
            extra = next(fetched)
            if isinstance(extra, Exception):
                metric(expand_error=type(extra).__name__)
            else:
                # Retrieved text is untrusted, including expansion reads.
                extra = [chunk for chunk in extra if not INJECTION_RE.search(normalize(chunk["text"]))]
                merged = {chunk["order"]: chunk for chunk in extra}
                merged.update({chunk["order"]: chunk for chunk in picked})
                candidate = [merged[order] for order in sorted(merged)]
                if sum(est_tokens(chunk["text"]) for chunk in candidate) <= budget:
                    picked = candidate
        budget -= sum(est_tokens(chunk["text"]) for chunk in picked)
        documents.append(document_entry(picked))
    return documents
