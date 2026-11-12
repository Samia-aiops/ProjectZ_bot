"""Corpus identity shared by the index job and the serving process.

The index job writes the fingerprint to Redis after a successful run; serving
reads it for the semantic-cache key. Without the key (Redis unavailable, job
never ran) serving falls back to a settings-only fingerprint.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from . import config

log = logging.getLogger(__name__)

CORPUS_KEY = f"{config.CACHE_PREFIX}corpus"


def settings_salt() -> str:
    return (
        f"{config.EMBEDDING_MODEL}|{config.SPARSE_MODEL}|chunker{config.CHUNKER_VERSION}"
        f"|{config.CHUNK_TOKENS}|{config.CHUNK_MAX_TOKENS}|{config.CHUNK_OVERLAP_TOKENS}"
        f"|{config.MIN_CHUNK_TOKENS}|{config.WHOLE_DOC_TOKENS}"
    )


def fallback_corpus() -> str:
    """Settings-only fingerprint used until the index job publishes one."""
    return hashlib.sha256(settings_salt().encode()).hexdigest()[:16]


async def _published(redis: Any) -> str | None:
    try:
        value = await asyncio.wait_for(redis.get(CORPUS_KEY), config.REDIS_TIMEOUT)
    except Exception:
        return None
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


async def published_corpus(redis: Any) -> str | None:
    """The fingerprint the index job published, or None when it is absent/unreachable."""
    return await _published(redis)


async def read_corpus(redis: Any) -> str:
    return await _published(redis) or fallback_corpus()


async def reload_corpus(runtime: Any, interval: float) -> None:
    """Follow the index job while serving, so a reindex invalidates the cache.

    Only a published fingerprint replaces the one in use: a missing key means
    Redis is unavailable or the job has not run, and the startup value stands.
    """
    while True:
        await asyncio.sleep(interval)
        value = await published_corpus(runtime.redis)
        if value and value != runtime.corpus:
            runtime.corpus = value
            log.info("Corpus fingerprint refreshed corpus=%s", value)


async def write_corpus(redis: Any, corpus: str) -> None:
    await redis.set(CORPUS_KEY, corpus)
