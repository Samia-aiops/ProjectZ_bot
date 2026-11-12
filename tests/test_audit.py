from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from chat_search import audit, config, logging_setup
from chat_search.auth import Principal
from chat_search.errors import AppError
from chat_search.messages import BUSY_MSG
from chat_search.observability import request_ctx
from chat_search.pipeline import agent, chat
from chat_search.runtime import UserSlots

SECRET = "super-secret-value-123"


def patient() -> Principal:
    return Principal(
        subject="user-1",
        role="patient",
        tenant_id="acme",
        patient_id="patient-1",
        provider_id=None,
        permissions=frozenset({"docs:read"}),
    )


def provider() -> Principal:
    return Principal(
        subject="user-9",
        role="provider",
        tenant_id="acme",
        patient_id=None,
        provider_id="provider-9",
        permissions=frozenset({"docs:read"}),
    )


def records(caplog) -> list[dict]:
    return [json.loads(record.getMessage()) for record in caplog.records if record.name == "chat_search.audit"]


def emit(**overrides) -> None:
    fields = {
        "question": "show me my appointments",
        "answer": "Open Appointments in the dashboard.",
        "node": "faq",
        "session_id": "session-1",
        "mode": "auto",
    }
    fields.update(overrides)
    audit.audit_chat(patient(), **fields)


def test_a_chat_is_audited_with_identity_text_sources_and_timestamp(caplog):
    token = request_ctx.set({"request_id": "req-audit-1"})
    try:
        with caplog.at_level(logging.INFO, logger="chat_search.audit"):
            emit(
                sources=[{"document": "Patient Onboarding Guide", "page": 21}],
                latency_ms=123,
                history_turns=0,
            )
    finally:
        request_ctx.reset(token)

    record = records(caplog)[0]

    assert record["event"] == "chat"
    assert record["ts"].endswith("Z") and "T" in record["ts"]
    assert record["request_id"] == "req-audit-1"
    assert record["tenant_id"] == "acme"
    assert record["role"] == "patient"
    assert record["subject"] == "user-1"
    assert record["patient_id"] == "patient-1"
    assert record["provider_id"] is None
    assert record["session_id"] == "session-1"
    assert record["input"] == "show me my appointments"
    assert record["output"] == "Open Appointments in the dashboard."
    assert record["sources"] == [{"document": "Patient Onboarding Guide", "page": 21}]
    assert record["cached"] is False
    assert record["outcome"] == "ok"
    assert record["latency_ms"] == 123
    assert record["history_turns"] == 0
    assert "error" not in record


def test_a_provider_account_is_identified_by_provider_id(caplog):
    with caplog.at_level(logging.INFO, logger="chat_search.audit"):
        audit.audit_chat(
            provider(),
            question="what is my next appointment?",
            answer="Open the schedule.",
            node="faq",
            session_id=None,
            mode="faq",
        )

    record = records(caplog)[0]

    assert record["provider_id"] == "provider-9"
    assert record["patient_id"] is None


def test_the_audit_text_is_redacted_and_bounded(monkeypatch, caplog):
    monkeypatch.setattr(logging_setup, "_SECRETS", (SECRET,))
    monkeypatch.setattr(config, "AUDIT_MAX_CHARS", 40)

    with caplog.at_level(logging.INFO, logger="chat_search.audit"):
        emit(
            question=f"my token is {SECRET} please list appointments",
            answer="sk-proj-abcdefghijklmnopqrstuvwxyz " + "y" * 200,
        )

    record = records(caplog)[0]

    assert SECRET not in record["input"]
    assert "[REDACTED]" in record["input"]
    assert len(record["input"]) <= 40
    assert "sk-proj-abcdefghijklmnopqrstuvwxyz" not in record["output"]
    assert len(record["output"]) <= 40


def test_ids_and_sources_survive_when_content_is_disabled(monkeypatch, caplog):
    monkeypatch.setattr(config, "AUDIT_LOG_CONTENT", False)

    with caplog.at_level(logging.INFO, logger="chat_search.audit"):
        emit(sources=[{"document": "Patient Guide", "page": 45}])

    record = records(caplog)[0]

    assert record["input"] is None
    assert record["output"] is None
    assert record["patient_id"] == "patient-1"
    assert record["sources"] == [{"document": "Patient Guide", "page": 45}]


def test_no_record_is_written_when_auditing_is_disabled(monkeypatch, caplog):
    monkeypatch.setattr(config, "AUDIT_LOG_ENABLED", False)

    with caplog.at_level(logging.INFO, logger="chat_search.audit"):
        emit()

    assert records(caplog) == []


def test_directory_answers_keep_public_provider_labels(caplog):
    providers = [
        {"provider_id": "p-1", "name": "Dr A", "specialty": "Cardiology", "city": "Boston", "bio": "private"}
    ] + [{"name": f"Dr {index}", "specialty": "Cardiology"} for index in range(20)]

    with caplog.at_level(logging.INFO, logger="chat_search.audit"):
        emit(node="providers", providers=providers)

    record = records(caplog)[0]

    assert record["providers"][0] == {"name": "Dr A", "specialty": "Cardiology", "city": "Boston"}
    assert len(record["providers"]) == 10
    assert "bio" not in record["providers"][0]


async def test_a_chat_request_emits_one_audit_record(monkeypatch, caplog):
    async def history(*_):
        return []

    async def append(*_):
        return None

    async def pipeline(_principal, _message, _history, _shape=None):
        yield {"type": "token", "text": "Open Appointments."}
        yield {"type": "done", "sources": [{"document": "Guide", "page": 21}], "cached": False}

    monkeypatch.setattr(chat, "load_history", history)
    monkeypatch.setattr(chat, "append_turns", append)
    monkeypatch.setattr(agent, "rag_pipeline", pipeline)
    monkeypatch.setattr(chat, "get_runtime", lambda: SimpleNamespace(user_slots=UserSlots(1)))

    token = request_ctx.set({"request_id": "req-audit-2"})
    try:
        with caplog.at_level(logging.INFO, logger="chat_search.audit"):
            events = [event async for event in chat.rag_events(patient(), "show me my appointments", "faq", "s-2")]
    finally:
        request_ctx.reset(token)

    assert events[-1]["type"] == "done"
    record = records(caplog)[0]
    assert record["input"] == "show me my appointments"
    assert record["output"] == "Open Appointments."
    assert record["sources"] == [{"document": "Guide", "page": 21}]
    assert record["session_id"] == "s-2"
    assert record["outcome"] == "ok"
    assert record["history_turns"] == 0
    assert record["latency_ms"] >= 0


async def test_a_failed_request_is_audited_with_the_error_code(monkeypatch, caplog):
    async def history(*_):
        return []

    async def pipeline(_principal, _message, _history, _shape=None):
        raise AppError(503, "busy", BUSY_MSG)
        yield {}  # pragma: no cover

    monkeypatch.setattr(chat, "load_history", history)
    monkeypatch.setattr(agent, "rag_pipeline", pipeline)
    monkeypatch.setattr(chat, "get_runtime", lambda: SimpleNamespace(user_slots=UserSlots(1)))

    with caplog.at_level(logging.INFO, logger="chat_search.audit"):
        with pytest.raises(AppError):
            async for _ in chat.rag_events(patient(), "show me my appointments", "faq"):
                pass

    record = records(caplog)[0]
    assert record["outcome"] == "error"
    assert record["error"] == "busy"


async def test_a_greeting_is_audited_before_the_pipeline(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="chat_search.audit"):
        events = [event async for event in chat.rag_events(patient(), "hello", "auto")]

    assert events[-1]["type"] == "done"
    record = records(caplog)[0]
    assert record["node"] == "greeting"
    assert record["input"] == "hello"
    assert record["patient_id"] == "patient-1"


async def test_a_misspelled_greeting_still_bypasses_the_rate_limiter(monkeypatch, caplog):
    async def deny(_principal):
        raise AssertionError("a greeting must not reach the rate limiter")

    monkeypatch.setattr(chat, "enforce_rate_limit", deny)
    monkeypatch.setattr(chat, "spell_correct", lambda text: "hello")

    with caplog.at_level(logging.INFO, logger="chat_search.audit"):
        events = [event async for event in await chat.open_events(patient(), "helllo")]

    assert events[-1]["type"] == "done"
    record = records(caplog)[0]
    assert record["node"] == "greeting"
    assert record["input"] == "helllo"


async def test_a_rate_limited_request_is_audited(monkeypatch, caplog):
    async def deny(_principal):
        raise AppError(429, "rate_limited", BUSY_MSG, {"Retry-After": "3"})

    monkeypatch.setattr(chat, "enforce_rate_limit", deny)

    with caplog.at_level(logging.INFO, logger="chat_search.audit"):
        with pytest.raises(AppError):
            await chat.open_events(patient(), "show me my appointments")

    record = records(caplog)[0]
    assert record["node"] == "rate_limit"
    assert record["outcome"] == "rate_limited"
    assert record["error"] == "rate_limited"
    assert record["input"] == "show me my appointments"
