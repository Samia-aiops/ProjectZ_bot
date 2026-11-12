from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import httpx
import pytest

import chat_search.api.app as app_module
from chat_search import config
from chat_search.messages import GREETING_REPLY
from chat_search.runtime import set_runtime


class FakeQdrant:
    async def collection_exists(self, _: str) -> bool:
        return True


class FakeRedis:
    async def ping(self) -> bool:
        return True


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


def use_dev_bypass(monkeypatch) -> None:
    monkeypatch.setattr(config, "DEV_BYPASS_ROLE", "patient")
    monkeypatch.setattr(config, "APP_ENV", "test")


def use_health_details(monkeypatch) -> None:
    monkeypatch.setattr(config, "HEALTH_DETAILS", True)


async def test_home_page_is_served_with_security_headers(client):
    response = await client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<!doctype html>" in response.text.lower()
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-request-id"]


async def test_chat_requires_authentication_and_returns_the_request_id(client, monkeypatch):
    monkeypatch.setattr(config, "DEV_BYPASS_ROLE", "")

    response = await client.post(
        "/v1/chat/stream",
        json={"message": "hello"},
        headers={"X-Request-ID": "request-12345"},
    )

    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "unauthorized"
    assert body["error"]["request_id"] == "request-12345"
    assert response.headers["x-request-id"] == "request-12345"
    assert response.headers["www-authenticate"] == "Bearer"


async def test_invalid_request_id_is_replaced(client):
    response = await client.get("/", headers={"X-Request-ID": "bad id"})

    assert response.headers["x-request-id"] != "bad id"


async def test_validation_errors_do_not_echo_input(client, monkeypatch):
    use_dev_bypass(monkeypatch)

    response = await client.post(
        "/v1/chat",
        json={"message": "", "unexpected": "do-not-echo-this"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert "do-not-echo-this" not in response.text


async def test_greeting_shortcut_works_without_runtime(client, monkeypatch):
    use_dev_bypass(monkeypatch)

    response = await client.post("/v1/chat", json={"message": "hello"})

    assert response.status_code == 200
    assert response.json()["answer"] == GREETING_REPLY


async def test_greeting_streams_sse_events(client, monkeypatch):
    use_dev_bypass(monkeypatch)

    response = await client.post("/v1/chat/stream", json={"message": "hello"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: token" in response.text
    assert "event: done" in response.text


async def test_health_is_minimal_by_default(client):
    set_runtime(
        SimpleNamespace(
            qdrant=FakeQdrant(),
            redis=FakeRedis(),
            cache_ok=True,
            es=None,
        )
    )

    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_health_reports_dependency_state_when_details_are_enabled(client, monkeypatch):
    use_health_details(monkeypatch)
    set_runtime(
        SimpleNamespace(
            qdrant=FakeQdrant(),
            redis=FakeRedis(),
            cache_ok=True,
            es=None,
        )
    )

    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "qdrant": True, "redis": True, "cache": True}


async def test_health_includes_elasticsearch_when_enabled(client, monkeypatch):
    use_health_details(monkeypatch)
    calls = []

    class FakeIndices:
        async def exists(self, index: str) -> bool:
            calls.append(index)
            return False

    class FakeES:
        indices = FakeIndices()

    set_runtime(
        SimpleNamespace(
            qdrant=FakeQdrant(),
            redis=FakeRedis(),
            cache_ok=False,
            es=FakeES(),
        )
    )

    response = await client.get("/health")
    body = response.json()

    # Elasticsearch is optional: provider search degrades, the service is healthy.
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["elasticsearch"] is False
    assert body["cache"] is False
    assert calls == [config.ES_PROVIDERS_INDEX]


async def test_health_probes_dependencies_in_parallel(client, monkeypatch):
    use_health_details(monkeypatch)

    class SlowQdrant:
        async def collection_exists(self, _: str) -> bool:
            await asyncio.sleep(0.5)
            return True

    class SlowRedis:
        async def ping(self) -> bool:
            await asyncio.sleep(0.5)
            return True

    set_runtime(SimpleNamespace(qdrant=SlowQdrant(), redis=SlowRedis(), cache_ok=True, es=None))

    started = time.perf_counter()
    body = (await client.get("/health")).json()
    elapsed = time.perf_counter() - started

    assert body["qdrant"] is True and body["redis"] is True
    assert elapsed < 0.95  # sequential probes would need 1.0s


async def test_a_dead_critical_dependency_degrades_and_returns_503(client):
    class DeadQdrant:
        async def collection_exists(self, _: str) -> bool:
            raise ConnectionError

    set_runtime(SimpleNamespace(qdrant=DeadQdrant(), redis=FakeRedis(), cache_ok=True, es=None))

    response = await client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "degraded"}


async def test_a_dead_dependency_only_flips_its_own_flag(client, monkeypatch):
    use_health_details(monkeypatch)

    class DeadQdrant:
        async def collection_exists(self, _: str) -> bool:
            raise ConnectionError

    set_runtime(SimpleNamespace(qdrant=DeadQdrant(), redis=FakeRedis(), cache_ok=True, es=None))

    response = await client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "qdrant": False, "redis": True, "cache": True}


async def test_non_streaming_error_without_status_falls_back_to_500(client, monkeypatch):
    use_dev_bypass(monkeypatch)

    async def fake_open_events(principal, message, mode, session_id=None):
        async def events():
            yield {"type": "error", "code": "busy", "message": "busy"}

        return events()

    monkeypatch.setattr(app_module, "open_events", fake_open_events)

    response = await client.post("/v1/chat", json={"message": "tell me about refunds"})

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "busy"


async def test_non_streaming_error_honours_its_status(client, monkeypatch):
    use_dev_bypass(monkeypatch)

    async def fake_open_events(principal, message, mode, session_id=None):
        async def events():
            yield {"type": "error", "code": "rate_limited", "message": "slow down", "status": 429}

        return events()

    monkeypatch.setattr(app_module, "open_events", fake_open_events)

    response = await client.post("/v1/chat", json={"message": "tell me about refunds"})

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "rate_limited"


async def test_the_session_id_reaches_the_pipeline(client, monkeypatch):
    use_dev_bypass(monkeypatch)
    seen = {}

    async def fake_open_events(principal, message, mode, session_id=None):
        seen["session_id"] = session_id

        async def events():
            yield {"type": "done", "sources": [], "cached": False}

        return events()

    monkeypatch.setattr(app_module, "open_events", fake_open_events)

    response = await client.post("/v1/chat", json={"message": "tell me about refunds", "session_id": "thread-2"})

    assert response.status_code == 200
    assert seen["session_id"] == "thread-2"


async def test_session_reset_clears_only_the_callers_conversation(client, monkeypatch):
    use_dev_bypass(monkeypatch)
    cleared = []

    async def fake_clear(principal, session_id=None):
        cleared.append((principal.tenant_id, session_id))
        return True

    monkeypatch.setattr(app_module, "clear_history", fake_clear)

    response = await client.delete("/v1/chat/session", params={"session_id": "thread-2"})

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "cleared": True}
    assert cleared == [("default", "thread-2")]


async def test_session_reset_reports_when_history_is_not_stored(client, monkeypatch):
    use_dev_bypass(monkeypatch)

    async def no_store(principal, session_id=None):
        return False

    monkeypatch.setattr(app_module, "clear_history", no_store)

    response = await client.delete("/v1/chat/session")

    assert response.status_code == 200
    assert response.json() == {"status": "not_stored", "cleared": False}


async def test_session_reset_rejects_an_unsafe_session_id(client, monkeypatch):
    use_dev_bypass(monkeypatch)

    response = await client.delete("/v1/chat/session", params={"session_id": "bad id"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
