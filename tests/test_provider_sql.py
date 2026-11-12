"""Unit tests for Provider Text-to-SQL MCP service and data isolation."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from chat_search import config, mcp_server
from chat_search.auth import dev_principal
from chat_search.errors import AppError
from chat_search.pipeline.provider_sql import provider_sql_pipeline
from chat_search.providers import sql_service
from chat_search.providers.sql_service import execute_provider_sql, validate_and_sanitize_sql


def test_sql_validation_scopes_and_binds():
    safe = validate_and_sanitize_sql("SELECT * FROM appointments", "prov-123", "acme")
    assert "provider_id = %(provider_id)s" in safe
    assert "tenant_id = %(tenant_id)s" in safe
    assert "LIMIT 100" in safe
    assert ":provider_id" not in safe  # never the broken named-param style


def test_sql_validation_provider_table_uses_id():
    safe = validate_and_sanitize_sql("SELECT * FROM provider", "prov-123", "acme")
    assert "id = %(provider_id)s" in safe
    assert "provider_id = %(provider_id)s" not in safe


def test_sql_validation_caps_limit():
    safe = validate_and_sanitize_sql("SELECT * FROM appointments LIMIT 1000000", "p", "t")
    assert "LIMIT 100" in safe
    assert "1000000" not in safe


def test_sql_validation_order_by_does_not_crash():
    safe = validate_and_sanitize_sql("SELECT * FROM appointments ORDER BY appointment_date DESC", "p", "t")
    assert "ORDER BY appointment_date DESC" in safe
    assert "tenant_id = %(tenant_id)s" in safe


def test_sql_validation_parenthesises_or_tautology():
    safe = validate_and_sanitize_sql("SELECT * FROM appointments WHERE status = 'x' OR 1=1", "p", "t")
    # The tautology stays inside the caller's scope; it cannot widen isolation.
    assert "(status = 'x' OR 1 = 1) AND (provider_id = %(provider_id)s" in safe


def test_sql_validation_escapes_literal_percent_for_pyformat():
    # pymysql binds with pyformat: a literal % (LIKE) must become %% so it is not
    # mistaken for a format specifier, while the injected placeholders stay single.
    safe = validate_and_sanitize_sql("SELECT * FROM insurances WHERE insurance_name LIKE '%blue%'", "p", "t")
    assert "%%blue%%" in safe
    assert "%(provider_id)s" in safe


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE appointments",
        "SELECT * FROM appointments; DROP TABLE appointments",
        "SELECT * FROM patients",
        "SELECT * FROM specialties",
        "SELECT * FROM users",
        "SELECT * FROM otherdb.appointments",
        "SELECT * FROM appointments WHERE 1=1 -- :provider_id :tenant_id",
        "SELECT * FROM appointments /* comment */",
        "SELECT * FROM provider WHERE id=:provider_id",
        "SELECT ':provider_id' AS p, ':tenant_id' AS t, email FROM patients",
        "SELECT * FROM appointments UNION SELECT * FROM patients",
        "SELECT * FROM appointments WHERE provider_id IN (SELECT provider_id FROM appointments)",
        "WITH x AS (SELECT * FROM appointments) SELECT * FROM x",
        "SELECT load_file('/etc/passwd')",
        "SELECT load_file('/etc/passwd') FROM appointments",
        "SELECT sleep(10)",
        "SELECT * FROM appointments FOR UPDATE",
        "SELECT '__cs_provider_id__' AS p FROM appointments",
        "SELECT * FROM appointments INTO OUTFILE '/tmp/x'",
        "SELECT * FROM appointments a JOIN patients b ON a.patient_id = b.id",
        "",
    ],
)
def test_sql_validation_blocks_unsafe(sql):
    with pytest.raises(AppError) as exc:
        validate_and_sanitize_sql(sql, "p", "t")
    assert exc.value.status_code == 400


async def test_provider_pool_never_falls_back_to_the_conversation_store(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(config, "PROVIDER_DB_URL", "")
    monkeypatch.setattr(config, "MYSQL_URL", "mysql://chat@db/conversations")
    conversation_store = object()
    monkeypatch.setattr(
        sql_service, "get_runtime", lambda: SimpleNamespace(provider_mysql=None, mysql=conversation_store)
    )

    assert sql_service._provider_pool() is None


async def test_execute_rejects_patients():
    with pytest.raises(PermissionError):
        await execute_provider_sql(dev_principal("patient", "acme"), "show my appointments today")


async def test_execute_rejects_provider_without_permission(monkeypatch):
    monkeypatch.setattr(config, "PROVIDER_SQL_PERMISSION", "providers:analytics")
    principal = dataclasses.replace(
        dev_principal("provider", "acme"), permissions=frozenset({config.REQUIRED_PERMISSION})
    )
    with pytest.raises(PermissionError):
        await execute_provider_sql(principal, "how many appointments today?")


async def test_execute_mock_is_marked(monkeypatch):
    monkeypatch.setattr(config, "ANALYTICS_API_URL", "")
    monkeypatch.setattr(config, "PROVIDER_DB_URL", "")
    monkeypatch.setattr(config, "MYSQL_URL", "")
    result = await execute_provider_sql(dev_principal("provider", "acme"), "how many appointments do I have today?")
    assert result["source"] == "mock"
    assert result["sql"] is None
    assert result["rows"]
    assert result["question"] == "how many appointments do I have today?"


async def test_chat_pipeline_denies_patients():
    events = [
        event
        async for event in provider_sql_pipeline(
            dev_principal("patient", "acme"), "how many appointments do I have today?"
        )
    ]
    text = "".join(event.get("text", "") for event in events if event.get("type") == "token")
    assert "provider accounts only" in text


async def test_chat_pipeline_denies_provider_without_permission(monkeypatch):
    monkeypatch.setattr(config, "PROVIDER_SQL_PERMISSION", "providers:analytics")
    principal = dataclasses.replace(
        dev_principal("provider", "acme"), permissions=frozenset({config.REQUIRED_PERMISSION})
    )
    events = [event async for event in provider_sql_pipeline(principal, "how many appointments?")]
    text = "".join(event.get("text", "") for event in events if event.get("type") == "token")
    assert "analytics permission" in text


class _FakeCursor:
    def __init__(self):
        self.description = [("count",)]
        self.executed = None
        self.args = None

    async def execute(self, sql, args):
        self.executed = sql
        self.args = args

    async def fetchmany(self, n):
        return [(7,)]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, cursor):
        self._cursor = cursor

    def acquire(self):
        return _FakeConn(self._cursor)


@pytest.mark.parametrize("tenant", ["tenant-a", "tenant-b"])
async def test_execute_binds_caller_identity(monkeypatch, tenant):
    principal = dev_principal("provider", tenant)
    cursor = _FakeCursor()
    monkeypatch.setattr(config, "ANALYTICS_API_URL", "")
    monkeypatch.setattr(sql_service, "_provider_pool", lambda: _FakePool(cursor))

    async def fake_translate(question, history):
        return "SELECT COUNT(*) FROM appointments"

    monkeypatch.setattr(sql_service, "translate_question_to_sql", fake_translate)
    result = await execute_provider_sql(principal, "how many appointments?")
    assert result["source"] == "sql"
    # Isolation values come from the verified principal, bound as parameters.
    assert cursor.args == {"provider_id": principal.provider_id, "tenant_id": tenant}
    assert "provider_id = %(provider_id)s" in cursor.executed
    assert "tenant_id = %(tenant_id)s" in cursor.executed
    assert result["rows"] == [{"count": 7}]


async def test_execute_uses_the_analytics_api_first(monkeypatch):
    principal = dev_principal("provider", "acme")
    calls = []

    class _Response:
        status_code = 200

        def json(self):
            return {
                "status": 200,
                "message": "ok",
                "data": {
                    "appointment_analytics": {"appointment_counts": {"today": 4}},
                    "availability": {"appointment_availability": 1},
                },
            }

    class _Http:
        async def get(self, url, params=None, headers=None):
            calls.append((url, params, headers))
            return _Response()

    monkeypatch.setattr(sql_service, "get_runtime", lambda: SimpleNamespace(http=_Http()))
    monkeypatch.setattr(config, "ANALYTICS_API_URL", "http://analytics.test/api")
    monkeypatch.setattr(config, "PROVIDER_DB_TOKEN", "")
    monkeypatch.setattr(config, "PROVIDERS_API_TOKEN", "tok")

    def no_sql():
        raise AssertionError("the API answer must not fall through to SQL")

    monkeypatch.setattr(sql_service, "_provider_pool", no_sql)

    result = await execute_provider_sql(principal, "how many appointments today?")

    assert result["source"] == "api"
    # The platform's {"status","message","data"} envelope is unwrapped for callers.
    assert result["data"]["appointment_analytics"]["appointment_counts"]["today"] == 4
    url, params, headers = calls[0]
    assert params == {
        "provider_id": principal.provider_id,
        "tenant_id": "acme",
        "question": "how many appointments today?",
    }
    assert headers == {"Authorization": "Bearer tok"}


async def test_mcp_query_provider_data_mock(monkeypatch):
    token = mcp_server._principal.set(dev_principal("provider", "acme"))
    try:
        monkeypatch.setattr(config, "ANALYTICS_API_URL", "")
        monkeypatch.setattr(config, "PROVIDER_DB_URL", "")
        monkeypatch.setattr(config, "MYSQL_URL", "")
        monkeypatch.setattr(config, "MCP_SQL_ENABLED", True)
        result = await mcp_server.query_provider_data("show my schedule")
        assert result["source"] == "mock"
        assert "rows" in result
    finally:
        mcp_server._principal.reset(token)


async def test_mcp_query_provider_data_rejects_patients():
    token = mcp_server._principal.set(dev_principal("patient", "acme"))
    try:
        with pytest.raises(PermissionError):
            await mcp_server.query_provider_data("show my schedule")
    finally:
        mcp_server._principal.reset(token)


async def test_mcp_query_provider_data_requires_permission(monkeypatch):
    monkeypatch.setattr(config, "PROVIDER_SQL_PERMISSION", "providers:analytics")
    principal = dataclasses.replace(
        dev_principal("provider", "acme"), permissions=frozenset({config.REQUIRED_PERMISSION})
    )
    token = mcp_server._principal.set(principal)
    try:
        monkeypatch.setattr(config, "MCP_SQL_ENABLED", True)
        with pytest.raises(PermissionError):
            await mcp_server.query_provider_data("show my schedule")
    finally:
        mcp_server._principal.reset(token)


async def test_mcp_query_provider_data_guardrail(monkeypatch):
    token = mcp_server._principal.set(dev_principal("provider", "acme"))
    try:
        monkeypatch.setattr(config, "MCP_SQL_ENABLED", True)
        result = await mcp_server.query_provider_data("I want to kill myself")
        assert result["refused"] is True
        assert result["category"] == "emergency"
    finally:
        mcp_server._principal.reset(token)
