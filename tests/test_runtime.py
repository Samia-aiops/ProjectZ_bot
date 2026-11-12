from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from chat_search import config
from chat_search.errors import AppError
from chat_search.observability import request_ctx
from chat_search.runtime import UserSlots, run_cpu, set_runtime, track


class SharedRedis:
    """Just enough of the two Lua scripts to model the shared counter."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.expiries: dict[str, int] = {}
        self.broken = False

    async def eval(self, script: str, numkeys: int, key: str, *args):
        if self.broken:
            raise ConnectionError("redis is down")
        if "ulock:acquire" in script:
            count = self.counts.get(key, 0) + 1
            self.counts[key] = count
            self.expiries[key] = int(args[0])
            return count
        if "ulock:release" in script:
            count = self.counts.get(key, 0)
            if count <= 0:
                return 0
            count -= 1
            if count <= 0:
                self.counts.pop(key, None)
                return 0
            self.counts[key] = count
            return count
        raise AssertionError(f"unexpected script: {script}")

    async def pexpire(self, key: str, ttl_ms: int) -> int:
        if self.broken:
            raise ConnectionError("redis is down")
        if key in self.counts:
            self.expiries[key] = ttl_ms
            return 1
        return 0


async def test_a_second_question_from_the_same_user_is_rejected():
    slots = UserSlots(1)

    async with slots.hold("acme:user-1", 0):
        with pytest.raises(AppError) as rejected:
            async with slots.hold("acme:user-1", 0):
                pass

    assert rejected.value.status_code == 429
    assert rejected.value.code == "request_in_progress"
    async with slots.hold("acme:user-1", 0):
        pass


async def test_another_user_gets_its_own_slot():
    slots = UserSlots(1)

    async with slots.hold("acme:user-1", 0):
        async with slots.hold("acme:user-2", 0):
            pass


async def test_the_shared_lock_rejects_a_question_from_another_replica():
    redis = SharedRedis()
    replica_a = UserSlots(1, redis=redis, ttl=30)
    replica_b = UserSlots(1, redis=redis, ttl=30)

    async with replica_a.hold("acme:user-1", 0):
        with pytest.raises(AppError) as rejected:
            async with replica_b.hold("acme:user-1", 0):
                pass
        assert rejected.value.code == "request_in_progress"

    # The denial did not leak replica B's local slot, and the release freed Redis.
    assert redis.counts == {}
    async with replica_b.hold("acme:user-1", 0):
        pass


async def test_the_shared_lock_counts_up_to_the_configured_limit():
    redis = SharedRedis()
    slots = UserSlots(2, redis=redis, ttl=30)

    async with slots.hold("acme:user-1", 0):
        async with slots.hold("acme:user-1", 0):
            with pytest.raises(AppError):
                async with slots.hold("acme:user-1", 0):
                    pass

    assert redis.counts == {}


async def test_a_redis_failure_keeps_the_local_slot_working():
    redis = SharedRedis()
    redis.broken = True
    slots = UserSlots(1, redis=redis, ttl=30)

    async with slots.hold("acme:user-1", 0):
        pass
    async with slots.hold("acme:user-1", 0):
        pass


async def test_the_shared_lock_is_off_without_a_ttl():
    redis = SharedRedis()
    slots = UserSlots(1, redis=redis, ttl=0)

    async with slots.hold("acme:user-1", 0):
        pass

    assert redis.counts == {}


class SlowReleaseRedis(SharedRedis):
    """Blocks the first shared release so a disconnect can interrupt it."""

    def __init__(self) -> None:
        super().__init__()
        self.release_started = asyncio.Event()
        self.release_calls = 0

    async def eval(self, script: str, numkeys: int, key: str, *args):
        if "ulock:release" in script:
            self.release_calls += 1
            if self.release_calls == 1:
                self.release_started.set()
                await asyncio.sleep(10)
        return await super().eval(script, numkeys, key, *args)


async def test_a_cancelled_holder_still_returns_the_shared_slot():
    redis = SlowReleaseRedis()
    slots = UserSlots(1, redis=redis, ttl=30)
    entered = asyncio.Event()

    async def holder():
        async with slots.hold("acme:user-1", 0):
            entered.set()
            await asyncio.sleep(10)

    task = asyncio.create_task(holder())
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    await asyncio.wait_for(redis.release_started.wait(), 1)
    task.cancel()  # a disconnect can cancel the cleanup await too
    with pytest.raises(asyncio.CancelledError):
        await task

    for _ in range(100):
        if not redis.counts:
            break
        await asyncio.sleep(0.01)

    assert redis.counts == {}
    async with slots.hold("acme:user-1", 0):  # the caller is not locked out
        pass


async def test_a_waiter_is_admitted_when_the_holder_finishes():
    slots = UserSlots(1)
    order: list[str] = []

    async def first():
        async with slots.hold("acme:user-1", 0):
            order.append("first")
            await asyncio.sleep(0.02)

    task = asyncio.create_task(first())
    await asyncio.sleep(0.005)
    async with slots.hold("acme:user-1", 1):
        order.append("second")
    await task

    assert order == ["first", "second"]


async def test_a_cpu_task_that_waited_in_the_queue_is_reported(monkeypatch):
    pool = ThreadPoolExecutor(max_workers=1)
    set_runtime(SimpleNamespace(executor=pool, tasks=set()))
    monkeypatch.setattr(config, "CPU_QUEUE_WARN_MS", 0)
    ctx: dict = {}
    token = request_ctx.set(ctx)

    def work():
        return "ok"

    try:
        busy = asyncio.get_running_loop().run_in_executor(pool, time.sleep, 0.05)
        await asyncio.sleep(0.005)
        assert await run_cpu(work, executor=pool) == "ok"
        await busy
    finally:
        request_ctx.reset(token)
        pool.shutdown(wait=False)

    assert ctx["cpu_queue_ms"] >= 10  # real time spent waiting for the single worker
    assert ctx["cpu_task"] == "work"


async def test_a_tracked_task_is_dropped_from_the_runtime_when_it_finishes():
    runtime = SimpleNamespace(tasks=set())
    set_runtime(runtime)
    finished = asyncio.Event()

    async def work():
        await asyncio.sleep(0)
        finished.set()

    task = track(asyncio.create_task(work()))
    assert task in runtime.tasks

    await task
    await asyncio.sleep(0)
    assert finished.is_set()
    assert task not in runtime.tasks
