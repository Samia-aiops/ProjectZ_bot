from __future__ import annotations

from types import SimpleNamespace

import pytest

from chat_search import config
from chat_search.auth import Principal
from chat_search.errors import AppError
from chat_search.observability import client_ip_ctx
from chat_search.rate_limit import _BUCKETS, enforce_rate_limit
from chat_search.runtime import set_runtime


def principal() -> Principal:
    return Principal(
        subject="user-1",
        role="patient",
        tenant_id="acme",
        patient_id="patient-1",
        provider_id=None,
        permissions=frozenset({"docs:read"}),
    )


class FakeRedis:
    def __init__(self, reply=(1, 0), failure: Exception | None = None) -> None:
        self.reply = reply
        self.failure = failure
        self.calls: list[tuple] = []

    async def eval(self, script, numkeys, *keys_and_args):
        if self.failure:
            raise self.failure
        self.calls.append((script, numkeys, keys_and_args))
        return list(self.reply)


def run(redis: FakeRedis) -> None:
    set_runtime(SimpleNamespace(redis=redis))


async def test_a_request_with_a_token_is_allowed():
    redis = FakeRedis()
    run(redis)

    await enforce_rate_limit(principal())

    script, numkeys, args = redis.calls[0]
    assert script == _BUCKETS
    assert numkeys == 2
    assert args[0].startswith("rl:u:acme:") and len(args[0]) == len("rl:u:acme:") + 16
    assert args[1] == "rl:t:acme"
    assert args[2:] == (20, 20 / 60, 300, 300 / 60, 60000)


async def test_a_denied_request_reports_an_exact_retry_after():
    run(FakeRedis(reply=(0, 5200)))

    with pytest.raises(AppError) as caught:
        await enforce_rate_limit(principal())

    assert caught.value.status_code == 429
    assert caught.value.code == "rate_limited"
    assert caught.value.headers == {"Retry-After": "6"}


async def test_the_ip_bucket_is_used_only_when_enabled_and_known(monkeypatch):
    monkeypatch.setattr(config, "RATE_LIMIT_IP", 5)
    redis = FakeRedis()
    run(redis)
    client_ip_ctx.set("203.0.113.5")

    await enforce_rate_limit(principal())

    _, numkeys, args = redis.calls[0]
    assert numkeys == 3
    assert args[2].startswith("rl:i:")
    assert "203.0.113.5" not in args[2]  # hashed, never stored raw
    assert args[3:] == (20, 20 / 60, 300, 300 / 60, 5, 5 / 60, 60000)

    client_ip_ctx.set(None)
    await enforce_rate_limit(principal())
    assert redis.calls[1][1] == 2


async def test_a_broken_redis_fails_open_by_default():
    run(FakeRedis(failure=ConnectionError()))

    await enforce_rate_limit(principal())  # no exception


async def test_a_broken_redis_can_fail_closed(monkeypatch):
    monkeypatch.setattr(config, "RATE_LIMIT_FAIL_OPEN", False)
    run(FakeRedis(failure=ConnectionError()))

    with pytest.raises(AppError) as caught:
        await enforce_rate_limit(principal())

    assert caught.value.status_code == 503


def test_the_clock_comes_from_redis_not_the_app():
    # Replicas with skewed clocks must agree on how full a bucket is.
    assert "redis.call('time')" in _BUCKETS
    assert "'t', 's'" in _BUCKETS
