"""Semantic answer cache (Redis Stack / RediSearch vectors).

Every failure degrades to "no cache": the cache is an optimisation, never a
correctness dependency.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from typing import Any

import numpy as np
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

from . import config
from .observability import metric
from .runtime import get_runtime


async def init_cache(dim: int) -> None:
    index = f"semcache_idx_{dim}"
    runtime = get_runtime()
    try:
        await runtime.redis.ft(index).info()
    except ResponseError:
        try:
            await runtime.redis.ft(index).create_index(
                [
                    TagField("scope"),
                    TagField("corpus"),
                    VectorField(
                        "embedding",
                        "FLAT",
                        {"TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": "COSINE"},
                    ),
                ],
                definition=IndexDefinition(prefix=[config.CACHE_PREFIX], index_type=IndexType.HASH),
            )
        except ResponseError as exc:
            # Replicas can race on a cold start: the loser sees the index the
            # winner just created and must keep using it, not disable the cache.
            if "already exists" not in str(exc).lower():
                raise
    runtime.cache_index = index


def _as_str(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _tag(value: str) -> str:
    """Escape a TAG value for a query; a scope like ``default-patient`` has a
    hyphen, which the query parser otherwise reads as syntax."""
    return "".join(char if char.isalnum() else f"\\{char}" for char in value)


async def cache_get(dense: np.ndarray, scope: str) -> tuple[str, list[dict[str, Any]]] | None:
    runtime = get_runtime()
    try:
        query = (
            Query(
                f"(@scope:{{{_tag(scope)}}} @corpus:{{{_tag(runtime.corpus)}}})"
                "=>[KNN 1 @embedding $vec AS dist]"
            )
            .sort_by("dist")
            .return_fields("answer", "sources", "dist")
            .dialect(2)
        )
        result = await asyncio.wait_for(
            runtime.redis.ft(runtime.cache_index).search(query, query_params={"vec": dense.tobytes()}),
            config.REDIS_TIMEOUT,
        )
        if not result.docs:
            return None
        document = result.docs[0]
        if float(_as_str(document.dist)) > 1 - config.CACHE_SIMILARITY:  # cosine distance
            return None
        return _as_str(document.answer), json.loads(_as_str(document.sources))
    except Exception as exc:
        metric(cache_error=type(exc).__name__)
        return None


async def cache_put(
    dense: np.ndarray,
    scope: str,
    answer: str,
    sources: list[dict[str, Any]],
    timeout: float | None = None,
) -> None:
    runtime = get_runtime()
    try:
        key = f"{config.CACHE_PREFIX}{scope}:{uuid.uuid4().hex}"  # scope is part of the key identity
        pipeline = runtime.redis.pipeline(transaction=False)
        pipeline.hset(
            key,
            mapping={
                "scope": scope,
                "corpus": runtime.corpus,
                "embedding": dense.tobytes(),
                "answer": answer,
                "sources": json.dumps(sources),
            },
        )
        pipeline.expire(key, config.CACHE_TTL)
        budget = config.CACHE_WRITE_TIMEOUT if timeout is None else timeout
        await asyncio.wait_for(pipeline.execute(), budget)
    except Exception as exc:
        metric(cache_write_error=type(exc).__name__)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def inflight_key(scope: str, text: str) -> str:
    return f"{config.CACHE_PREFIX}inflight:{scope}:{_digest(text)}"


def peer_key(scope: str, text: str) -> str:
    return f"{config.CACHE_PREFIX}peer:{scope}:{_digest(text)}"


async def claim_inflight(scope: str, text: str) -> str | None:
    """Decide who answers an identical question arriving at the same moment.

    Returns the marker key when this request must compute and publish, "" when
    coalescing is off or Redis failed (compute anyway, release nothing) and
    None when a peer is already answering, so the caller should wait for it.
    """
    if config.CACHE_COALESCE_WAIT <= 0:
        return ""
    try:
        runtime = get_runtime()
    except RuntimeError:
        return ""
    if not getattr(runtime, "cache_ok", False):
        return ""
    marker = inflight_key(scope, text)
    try:
        claimed = await asyncio.wait_for(
            runtime.redis.set(marker, "1", nx=True, px=int(config.CACHE_COALESCE_TTL * 1000)),
            config.REDIS_TIMEOUT,
        )
    except Exception:
        metric(coalesce="unavailable")
        return ""
    metric(coalesce="leader" if claimed else "follower")
    return marker if claimed else None


async def release_inflight(marker: str) -> None:
    if not marker:
        return
    try:
        await asyncio.wait_for(get_runtime().redis.delete(marker), config.REDIS_TIMEOUT)
    except Exception:
        metric(coalesce="release_error")  # the marker TTL clears it


async def await_peer(scope: str, text: str) -> tuple[str, list[dict[str, Any]]] | None:
    """Wait for the peer answering this exact question; None means answer it here."""
    runtime = get_runtime()
    marker, peer = inflight_key(scope, text), peer_key(scope, text)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + config.CACHE_COALESCE_WAIT
    while True:
        try:
            payload = await asyncio.wait_for(runtime.redis.get(peer), config.REDIS_TIMEOUT)
        except Exception:
            metric(coalesce="unavailable")
            return None
        if payload:
            try:
                parsed = json.loads(_as_str(payload))
                answer, sources = parsed["answer"], parsed["sources"]
            except (ValueError, KeyError, TypeError):
                metric(coalesce="bad_payload")
                return None
            if isinstance(answer, str) and isinstance(sources, list):
                return answer, sources
            return None
        if loop.time() >= deadline:
            metric(coalesce="timeout")
            return None
        await asyncio.sleep(0.1)
        try:
            if not await asyncio.wait_for(runtime.redis.exists(marker), config.REDIS_TIMEOUT):
                metric(coalesce="miss")  # the leader finished without a shared answer
                return None
        except Exception:
            metric(coalesce="unavailable")
            return None


async def publish_peer(scope: str, text: str, answer: str, sources: list[dict[str, Any]]) -> None:
    """Hand the finished answer to anyone waiting on this exact question."""
    if config.CACHE_COALESCE_WAIT <= 0:
        return
    try:
        await asyncio.wait_for(
            get_runtime().redis.set(
                peer_key(scope, text), json.dumps({"answer": answer, "sources": sources}), ex=config.CACHE_PEER_TTL
            ),
            config.CACHE_WRITE_TIMEOUT,
        )
    except Exception as exc:
        metric(peer_write_error=type(exc).__name__)


async def publish_answer(
    dense: np.ndarray,
    scope: str,
    text: str,
    answer: str,
    sources: list[dict[str, Any]],
) -> None:
    """Store the shared entry before the answer is released.

    Awaiting here (bounded) instead of spawning means a repeat or a waiting peer
    is guaranteed to find the entry this answer filled.
    """
    await asyncio.gather(
        cache_put(dense, scope, answer, sources),
        publish_peer(scope, text, answer, sources),
    )
