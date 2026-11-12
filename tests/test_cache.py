from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from redis.exceptions import ResponseError

from chat_search import cache
from chat_search.runtime import set_runtime


class FakeSearch:
    def __init__(self, *, exists: bool, create_error: str | None = None) -> None:
        self.exists = exists
        self.create_error = create_error
        self.created = False

    async def info(self) -> dict:
        if not self.exists:
            raise ResponseError("Unknown index name")
        return {}

    async def create_index(self, *_args, **_kwargs) -> None:
        if self.create_error:
            raise ResponseError(self.create_error)
        self.created = True
        self.exists = True


def run_init(search: FakeSearch) -> SimpleNamespace:
    runtime = SimpleNamespace(redis=SimpleNamespace(ft=lambda _index: search), cache_index="")
    set_runtime(runtime)
    return runtime


async def test_init_cache_creates_a_missing_index():
    search = FakeSearch(exists=False)
    runtime = run_init(search)

    await cache.init_cache(384)

    assert search.created is True
    assert runtime.cache_index == "semcache_idx_384"


async def test_init_cache_keeps_the_index_a_peer_created_first():
    search = FakeSearch(exists=False, create_error="Index already exists")
    runtime = run_init(search)

    await cache.init_cache(384)

    assert search.created is False
    assert runtime.cache_index == "semcache_idx_384"


async def test_init_cache_still_fails_when_redis_lacks_the_search_module():
    search = FakeSearch(exists=False, create_error="unknown command 'FT.CREATE'")
    run_init(search)

    with pytest.raises(ResponseError):
        await cache.init_cache(384)


class FakeCoalesce:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.calls: list[tuple] = []
        self.failure: Exception | None = None

    async def set(self, key, value, nx=False, px=None, ex=None):
        if self.failure:
            raise self.failure
        self.calls.append(("set", key, nx, px, ex))
        if nx and key in self.values:
            return None
        self.values[key] = value
        return True

    async def get(self, key):
        if self.failure:
            raise self.failure
        return self.values.get(key)

    async def exists(self, key):
        if self.failure:
            raise self.failure
        return 1 if key in self.values else 0

    async def delete(self, key):
        return 1 if self.values.pop(key, None) is not None else 0


def run_coalesce(redis) -> SimpleNamespace:
    runtime = SimpleNamespace(redis=redis, cache_ok=True)
    set_runtime(runtime)
    return runtime


async def test_the_first_caller_claims_and_the_second_follows():
    redis = FakeCoalesce()
    run_coalesce(redis)
    marker = cache.inflight_key("scope", "how do I reset my password")

    first = await cache.claim_inflight("scope", "how do I reset my password")
    second = await cache.claim_inflight("scope", "how do I reset my password")

    assert first == marker
    assert second is None
    assert redis.calls[0] == ("set", marker, True, int(cache.config.CACHE_COALESCE_TTL * 1000), None)


async def test_coalescing_can_be_disabled(monkeypatch):
    redis = FakeCoalesce()
    run_coalesce(redis)
    monkeypatch.setattr(cache.config, "CACHE_COALESCE_WAIT", 0)

    assert await cache.claim_inflight("scope", "question") == ""
    assert redis.calls == []


async def test_a_claim_failure_means_answer_it_here():
    redis = FakeCoalesce()
    redis.failure = ConnectionError()
    run_coalesce(redis)

    assert await cache.claim_inflight("scope", "question") == ""


async def test_a_follower_receives_the_peer_answer():
    redis = FakeCoalesce()
    run_coalesce(redis)

    await cache.publish_peer("scope", "question", "answer", [{"document": "Policy", "page": 1}])

    assert await cache.await_peer("scope", "question") == ("answer", [{"document": "Policy", "page": 1}])
    assert redis.calls[0][4] == cache.config.CACHE_PEER_TTL  # expires on its own


async def test_a_follower_stops_when_the_leader_finishes_without_an_answer(monkeypatch):
    redis = FakeCoalesce()
    run_coalesce(redis)
    monkeypatch.setattr(cache.config, "CACHE_COALESCE_WAIT", 2)
    marker = await cache.claim_inflight("scope", "question")
    assert marker is not None

    async def leader_finishes():
        await asyncio.sleep(0.05)
        await cache.release_inflight(marker)

    leader = asyncio.create_task(leader_finishes())
    assert await cache.await_peer("scope", "question") is None
    await leader


async def test_a_follower_gives_up_after_the_wait_budget(monkeypatch):
    redis = FakeCoalesce()
    run_coalesce(redis)
    monkeypatch.setattr(cache.config, "CACHE_COALESCE_WAIT", 0.15)
    assert await cache.claim_inflight("scope", "question") is not None

    started = asyncio.get_running_loop().time()
    assert await cache.await_peer("scope", "question") is None
    assert asyncio.get_running_loop().time() - started < 1


async def test_releasing_a_marker_deletes_it():
    redis = FakeCoalesce()
    run_coalesce(redis)
    marker = await cache.claim_inflight("scope", "question")

    await cache.release_inflight(marker)

    assert marker not in redis.values


async def test_a_peer_answer_with_a_broken_payload_is_not_served():
    redis = FakeCoalesce()
    run_coalesce(redis)
    redis.values[cache.peer_key("scope", "question")] = "{not json"

    assert await cache.await_peer("scope", "question") is None
