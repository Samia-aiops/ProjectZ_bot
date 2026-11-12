from __future__ import annotations

import asyncio
from contextlib import suppress
from types import SimpleNamespace

from chat_search import config, corpus


class FakeRedis:
    def __init__(self, value: bytes | None = None) -> None:
        self.value = value
        self.keys: list[str] = []

    async def get(self, key: str) -> bytes | None:
        self.keys.append(key)
        return self.value

    async def set(self, key: str, value: str) -> None:
        self.keys.append(key)
        self.value = value.encode()


def test_settings_salt_tracks_model_configuration(monkeypatch):
    before = corpus.settings_salt()
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "other/model")
    assert corpus.settings_salt() != before


def test_fallback_corpus_is_stable():
    assert corpus.fallback_corpus() == corpus.fallback_corpus()
    assert len(corpus.fallback_corpus()) == 16


async def test_read_corpus_prefers_published_fingerprint():
    redis = FakeRedis(b"abc123")
    assert await corpus.read_corpus(redis) == "abc123"
    assert redis.keys == [corpus.CORPUS_KEY]


async def test_read_corpus_falls_back_without_key():
    redis = FakeRedis()
    assert await corpus.read_corpus(redis) == corpus.fallback_corpus()


async def test_read_corpus_survives_redis_errors():
    class Broken:
        async def get(self, key: str) -> bytes | None:
            raise ConnectionError

    assert await corpus.read_corpus(Broken()) == corpus.fallback_corpus()


async def test_write_corpus_sets_the_shared_key():
    redis = FakeRedis()
    await corpus.write_corpus(redis, "deadbeef")
    assert redis.value == b"deadbeef"
    assert redis.keys == [corpus.CORPUS_KEY]


async def test_published_corpus_reports_a_missing_or_unreachable_key():
    class Broken:
        async def get(self, key: str) -> bytes | None:
            raise ConnectionError

    assert await corpus.published_corpus(FakeRedis()) is None
    assert await corpus.published_corpus(Broken()) is None
    assert await corpus.published_corpus(FakeRedis(b"abc123")) == "abc123"


async def test_reload_corpus_follows_the_index_job():
    runtime = SimpleNamespace(redis=FakeRedis(b"new-fingerprint"), corpus="old-fingerprint")
    task = asyncio.create_task(corpus.reload_corpus(runtime, 0.01))
    try:
        for _ in range(100):
            if runtime.corpus == "new-fingerprint":
                break
            await asyncio.sleep(0.005)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    assert runtime.corpus == "new-fingerprint"


async def test_reload_corpus_keeps_the_startup_value_when_redis_is_down():
    class Broken:
        async def get(self, key: str) -> bytes | None:
            raise ConnectionError

    runtime = SimpleNamespace(redis=Broken(), corpus="startup-fingerprint")
    task = asyncio.create_task(corpus.reload_corpus(runtime, 0.01))
    await asyncio.sleep(0.03)
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task

    assert runtime.corpus == "startup-fingerprint"
