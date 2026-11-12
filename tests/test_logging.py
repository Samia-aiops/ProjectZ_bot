from __future__ import annotations

import io
import logging

import httpx
import pytest

import chat_search.api.app as app_module
from chat_search import config, logging_setup
from chat_search.errors import AppError
from chat_search.logging_setup import RedactingFormatter, log_failure, redact, sanitize_url
from chat_search.messages import ERROR_MSG
from chat_search.observability import request_ctx
from chat_search.pipeline.events import safe_events

SECRET = "super-secret-value-123"


def test_redact_removes_registered_secret_values():
    text = f"connect failed with token {SECRET} for tenant acme"
    assert SECRET not in redact(text, secrets=[SECRET])
    assert redact(text, secrets=[SECRET]).startswith("connect failed with token [REDACTED]")


def test_redact_removes_credentials_by_shape():
    text = (
        "Authorization: Bearer abcdefghijklmnop "
        "postgres://admin:hunter2@db:5432/faq "
        "api_key=plainvalue client_secret='quoted' sk-proj-abcdefghijklmnopqrstuvwxyz"
    )
    cleaned = redact(text)
    for leaked in (
        "abcdefghijklmnop",
        "hunter2",
        "plainvalue",
        "quoted",
        "sk-proj-abcdefghijklmnopqrstuvwxyz",
    ):
        assert leaked not in cleaned
    assert "[redacted]" in cleaned


def test_sanitize_url_drops_userinfo_and_query():
    safe = sanitize_url("https://user:pass@es.internal:9200/providers?token=abc#frag")
    assert safe == "https://es.internal:9200/providers"
    assert sanitize_url("not a url") == "not a url"


def test_formatter_redacts_traceback_and_message(monkeypatch):
    monkeypatch.setattr(logging_setup, "_SECRETS", (SECRET,))
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(RedactingFormatter("%(message)s"))
    logger = logging.getLogger("chat_search.tests.redaction")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.ERROR)

    try:
        raise RuntimeError(f"upstream rejected {SECRET}")
    except RuntimeError:
        logger.exception("call to https://user:%s@api.example/v1?key=abc failed", SECRET)

    output = stream.getvalue()
    assert SECRET not in output
    assert "user:" not in output
    assert "RuntimeError" in output  # the type stays useful for debugging


@pytest.mark.parametrize("tracebacks", [False, True])
def test_log_failure_never_emits_exception_text(monkeypatch, caplog, tracebacks):
    monkeypatch.setattr(config, "LOG_TRACEBACKS", tracebacks)
    exc = ValueError(f"bad payload for patient 42: {SECRET}")

    with caplog.at_level(logging.ERROR, logger="chat_search.tests.failure"):
        log_failure(logging.getLogger("chat_search.tests.failure"), "pipeline failed", exc, request_id="req-1234")

    record = caplog.records[-1]
    message = record.getMessage()
    assert "error_type=ValueError" in message
    assert "request_id=req-1234" in message
    assert SECRET not in message
    assert "patient 42" not in message
    assert ("exc_info" in record.__dict__ and record.exc_info) if tracebacks else record.exc_info is None


def test_log_failure_redacts_before_truncating(monkeypatch, caplog):
    monkeypatch.setattr(config, "LOG_TRACEBACKS", False)
    # The userinfo runs past the truncation limit, so redacting after trimming
    # would leave the password fragment in the log line.
    password = "Q" * 300
    long_url = "x" * 100 + f"https://user:{password}@db.internal/x"

    with caplog.at_level(logging.ERROR, logger="chat_search.tests.fields"):
        log_failure(logging.getLogger("chat_search.tests.fields"), "sync failed", url=long_url)

    message = caplog.records[-1].getMessage()
    assert "user:" not in message
    assert "QQQQQQQQQQ" not in message


def test_secret_values_are_collected_longest_first(monkeypatch):
    monkeypatch.setenv("SYNC_TOKEN", SECRET)
    monkeypatch.setattr(config, "LLM_API_KEY", "short-key")
    values = logging_setup.secret_values()
    assert SECRET in values
    assert values == tuple(sorted(values, key=len, reverse=True))


async def _failing_events():
    raise ValueError(f"provider returned PHI for patient 42: {SECRET}")
    yield {}  # pragma: no cover


async def _app_error_events():
    raise AppError(409, "conflict", "That request conflicts with an earlier one.")
    yield {}  # pragma: no cover


async def test_safe_events_hides_exception_text(caplog):
    token = request_ctx.set({"request_id": "req-abcdef12"})
    try:
        with caplog.at_level(logging.ERROR, logger="chat_search.pipeline.events"):
            events = [event async for event in safe_events(_failing_events())]
    finally:
        request_ctx.reset(token)

    assert events == [
        {
            "type": "error",
            "code": "internal_error",
            "message": ERROR_MSG,
            "status": 500,
            "request_id": "req-abcdef12",
        }
    ]
    messages = [record.getMessage() for record in caplog.records]
    assert any("error_type=ValueError" in message for message in messages)
    assert all(SECRET not in message for message in messages)
    assert all("patient 42" not in message for message in messages)


async def test_safe_events_passes_client_errors_without_logging(caplog):
    token = request_ctx.set({"request_id": "req-abcdef12"})
    try:
        with caplog.at_level(logging.WARNING, logger="chat_search.pipeline.events"):
            events = [event async for event in safe_events(_app_error_events())]
    finally:
        request_ctx.reset(token)

    assert events == [
        {
            "type": "error",
            "code": "conflict",
            "message": "That request conflicts with an earlier one.",
            "status": 409,
            "request_id": "req-abcdef12",
        }
    ]
    logged = [record for record in caplog.records if record.name == "chat_search.pipeline.events"]
    assert not logged


async def test_unhandled_route_error_returns_a_generic_500(monkeypatch, caplog):
    monkeypatch.setattr(config, "DEV_BYPASS_ROLE", "patient")
    monkeypatch.setattr(config, "APP_ENV", "test")

    async def boom(*_):
        raise RuntimeError(f"backend exploded with {SECRET}")

    monkeypatch.setattr(app_module, "open_events", boom)
    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        with caplog.at_level(logging.ERROR):
            response = await http.post("/v1/chat", json={"message": "tell me about refunds"})

    assert response.status_code == 500
    assert response.json()["error"] == {
        "code": "internal_error",
        "message": ERROR_MSG,
        "request_id": response.headers["x-request-id"],
    }
    assert SECRET not in response.text
    assert "exploded" not in response.text
    assert any("error_type=RuntimeError" in record.getMessage() for record in caplog.records)
