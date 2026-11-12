"""Process-wide runtime object and concurrency helpers.

Heavy resources (models, clients, semaphores) are created once during the
FastAPI lifespan and reached through ``get_runtime()``. Nothing in this module
imports the application layers, which keeps the dependency graph acyclic.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from . import config
from .errors import AppError
from .logging_setup import log_failure
from .messages import BUSY_MSG
from .observability import metric

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    import redis.asyncio as aioredis
    from elasticsearch import AsyncElasticsearch
    from qdrant_client import AsyncQdrantClient

    from .auth import Principal


# One counter per user, shared by every replica. Acquire increments and renews
# the TTL; a count above the limit means the caller must undo its own increment.
# Release decrements only while the key exists, so a holder whose TTL lapsed
# never drives the counter negative.
_ACQUIRE_SLOT = """
-- ulock:acquire
local count = redis.call('incr', KEYS[1])
redis.call('pexpire', KEYS[1], ARGV[1])
return count
"""
_RELEASE_SLOT = """
-- ulock:release
if redis.call('exists', KEYS[1]) == 0 then return 0 end
local count = redis.call('decr', KEYS[1])
if count <= 0 then
  redis.call('del', KEYS[1])
  return 0
end
return count
"""


class UserSlots:
    """One semaphore per user, created on demand and dropped when idle.

    Guarantees a user has at most ``limit`` questions in flight, so a second
    question cannot race the first (interleaved streams, duplicate LLM spend,
    cache writes). With a Redis client the limit is also shared by every
    replica: each holder takes a TTL-bounded counter slot under
    ``ulock:<tenant>:<subject>``, refreshed while the answer streams. A Redis
    failure degrades to the in-process slots (the shared rate limiter still
    bounds total traffic); a crashed holder's slot expires after the TTL.
    """

    def __init__(self, limit: int, redis: Any = None, ttl: float = 0.0) -> None:
        self.limit = max(limit, 1)
        self._redis = redis if ttl > 0 else None
        self._ttl = ttl
        self._slots: dict[str, list[Any]] = {}  # key -> [semaphore, holders + waiters]
        self._releasing: set[asyncio.Task[None]] = set()  # background decrements after a cancel

    @asynccontextmanager
    async def hold(self, key: str, wait: float) -> AsyncIterator[None]:
        entry = self._slots.get(key)
        if entry is None:
            entry = self._slots[key] = [asyncio.Semaphore(self.limit), 0]
        entry[1] += 1
        busy = AppError(
            429,
            "request_in_progress",
            "Your previous question is still being answered. Please wait for it to finish and try again.",
            {"Retry-After": "2"},
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(wait, 0.0)
        try:
            semaphore: asyncio.Semaphore = entry[0]
            if wait <= 0:
                if semaphore.locked():
                    metric(user_slot="rejected")
                    raise busy
                await semaphore.acquire()  # free slot: this never suspends (no race window)
            else:
                try:
                    await asyncio.wait_for(semaphore.acquire(), wait)
                except TimeoutError:
                    metric(user_slot="rejected")
                    raise busy from None
            try:
                remaining = max(deadline - loop.time(), 0.0) if wait > 0 else 0.0
                shared = await self._hold_shared(key, remaining, busy)
                try:
                    yield
                finally:
                    await self._release_or_detach(shared)
            finally:
                semaphore.release()
        finally:
            entry[1] -= 1
            if entry[1] == 0 and self._slots.get(key) is entry:
                del self._slots[key]  # no memory growth for idle users

    async def _hold_shared(self, key: str, wait: float, busy: AppError) -> tuple[str, asyncio.Task[None]] | None:
        """Take the cross-replica slot, or None when the shared lock is off/unavailable."""
        if self._redis is None:
            return None
        slot = f"ulock:{key}"
        ttl_ms = max(int(self._ttl * 1000), 1000)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait
        while True:
            try:
                count = await asyncio.wait_for(self._redis.eval(_ACQUIRE_SLOT, 1, slot, ttl_ms), config.REDIS_TIMEOUT)
            except Exception:
                metric(user_lock="unavailable")
                return None
            if int(count) <= self.limit:
                # Renew the TTL while the answer streams so a slow reply cannot
                # outlive its slot and let a second question in.
                return slot, asyncio.create_task(self._refresh_shared(slot, ttl_ms))
            await self._release_or_detach((slot, None))  # undo our increment
            if wait <= 0 or loop.time() >= deadline:
                metric(user_lock="rejected")
                raise busy
            await asyncio.sleep(min(0.05, max(deadline - loop.time(), 0.0)))

    async def _release_or_detach(self, shared: tuple[str, asyncio.Task[None] | None] | None) -> None:
        """Return the shared slot, finishing the decrement outside a cancelled task.

        A client disconnect can cancel this task at any await, including the
        Redis call. Without the hand-off the counter stays above the limit and
        the caller is locked out until the TTL expires.
        """
        try:
            await self._release_shared(shared)
        except asyncio.CancelledError:
            if shared is not None and self._redis is not None:
                task = asyncio.create_task(self._release_shared(shared))
                self._releasing.add(task)
                task.add_done_callback(self._released)
            raise

    def _released(self, task: asyncio.Task[None]) -> None:
        self._releasing.discard(task)
        if not task.cancelled():
            task.exception()  # retrieve: the release itself never fails loudly

    async def _release_shared(self, shared: tuple[str, asyncio.Task[None] | None] | None) -> None:
        if shared is None or self._redis is None:
            return
        slot, refresh = shared
        if refresh is not None:
            refresh.cancel()
            try:
                await refresh
            except asyncio.CancelledError:
                pass
        try:
            await asyncio.wait_for(self._redis.eval(_RELEASE_SLOT, 1, slot), config.REDIS_TIMEOUT)
        except Exception:
            metric(user_lock="release_error")

    async def _refresh_shared(self, slot: str, ttl_ms: int) -> None:
        while True:
            await asyncio.sleep(max(self._ttl / 3, 0.1))
            try:
                await asyncio.wait_for(self._redis.pexpire(slot, ttl_ms), config.REDIS_TIMEOUT)
            except Exception:
                return


@dataclass
class Runtime:
    embedder: Any
    sparse: Any
    reranker: Any
    guard: Any
    executor: ThreadPoolExecutor
    embed_executor: ThreadPoolExecutor
    qdrant: AsyncQdrantClient
    redis: aioredis.Redis
    http: httpx.AsyncClient
    sem_guard: asyncio.Semaphore
    sem_rag: asyncio.Semaphore
    sem_rerank: asyncio.Semaphore
    sem_llm: asyncio.Semaphore
    user_slots: UserSlots
    corpus: str
    cache_ok: bool = False
    cache_index: str = ""
    tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    pii: Any = None  # Presidio AnalyzerEngine; None falls back to the regex detector
    es: AsyncElasticsearch | None = None  # only when ES_URL is set
    mysql: Any = None  # aiomysql pool for conversation history; None keeps chat stateless
    provider_mysql: Any = None  # dedicated pool for provider operational data; None disables Text-to-SQL


_runtime: Runtime | None = None


def set_runtime(runtime: Runtime) -> None:
    global _runtime
    _runtime = runtime


def get_runtime() -> Runtime:
    if _runtime is None:
        raise RuntimeError("Runtime is not initialized")
    return _runtime


def clear_runtime() -> None:
    global _runtime
    _runtime = None


async def run_cpu(
    fn: Any,
    *args: Any,
    timeout: float | None = None,
    executor: ThreadPoolExecutor | None = None,
) -> Any:
    """Run blocking/CPU work on a shared bounded executor with a timeout."""
    if timeout is None:
        timeout = config.MODEL_TIMEOUT
    loop = asyncio.get_running_loop()
    pool = executor or get_runtime().executor
    name = getattr(fn, "__name__", "cpu")
    enqueued = time.perf_counter()
    started_at: list[float] = []

    def _call() -> Any:
        started_at.append(time.perf_counter())
        return fn(*args)

    try:
        result = await asyncio.wait_for(loop.run_in_executor(pool, _call), timeout)
    except TimeoutError:
        metric(timeout=name)
        raise AppError(503, "busy", BUSY_MSG) from None
    if started_at and (queued := int((started_at[0] - enqueued) * 1000)) >= config.CPU_QUEUE_WARN_MS:
        # Measured in the event loop: the worker thread has no request context.
        metric(cpu_queue_ms=queued, cpu_task=name)
    return result


@asynccontextmanager
async def limit(semaphore: asyncio.Semaphore) -> AsyncIterator[None]:
    """Bounded wait for a slot: overload becomes a fast, safe 503 instead of a pile-up."""
    try:
        await asyncio.wait_for(semaphore.acquire(), config.QUEUE_TIMEOUT)
    except TimeoutError:
        metric(overloaded=True)
        raise AppError(503, "busy", BUSY_MSG) from None
    try:
        yield
    finally:
        semaphore.release()


def user_key(principal: Principal) -> str:
    digest = hashlib.sha256(principal.subject.encode()).hexdigest()[:16]
    return f"{principal.tenant_id}:{digest}"


def track(task: asyncio.Task[Any], name: str = "") -> asyncio.Task[Any]:
    """Keep a background task alive past the request and report its failure."""
    tasks = get_runtime().tasks
    tasks.add(task)
    name = name or getattr(task.get_coro(), "cr_qualname", "background task")

    def _done(finished: asyncio.Task[Any]) -> None:
        tasks.discard(finished)
        if finished.cancelled():
            return
        try:
            exc = finished.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            log_failure(log, "background task failed", exc, task=name)

    task.add_done_callback(_done)
    return task


def spawn(coro: Any) -> None:
    track(asyncio.create_task(coro))
