"""Text-to-SQL service and MCP tool backend for providers.

Enforces strict multi-tenant and multi-provider data isolation. The generated
SQL is parsed into an AST and rewritten so that every query is a single SELECT
against one whitelisted table, with the verified caller's ``provider_id`` and
``tenant_id`` bound as parameters at execution time. The model never supplies
the isolation values and cannot smuggle joins, subqueries, comments or extra
statements past the validator. Patients and unauthorized roles are denied.
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import time
from typing import Any

import sqlglot
from sqlglot import exp

from .. import config
from ..audit import audit_provider_sql
from ..auth import Principal
from ..errors import AppError
from ..observability import metric
from ..runtime import get_runtime, limit
from ..state import Turn

log = logging.getLogger(__name__)

# Each queryable table maps to the column that identifies the calling provider.
# Every table also carries tenant_id. Tables without a provider key (patients,
# specialties) are intentionally absent: they cannot be scoped to one provider
# without a join, so they are not queryable here. Patient-level data is served
# by the analytics API, which enforces provider scope server-side.
_TABLE_SCOPE: dict[str, str] = {
    "provider": "id",
    "providers": "id",
    "appointment_analytics": "provider_id",
    "appointments": "provider_id",
    "availability": "provider_id",
    "insurances": "provider_id",
    "working_schedule": "provider_id",
}

# Functions that can read files, stall the connection, or exfiltrate data.
_FORBIDDEN_FUNCS = frozenset(
    {
        "load_file",
        "sleep",
        "benchmark",
        "extractvalue",
        "updatexml",
        "sys_eval",
        "sys_exec",
        "sys_get",
        "get_lock",
        "release_lock",
        "release_all_locks",
        "load_extension",
        "master_pos_wait",
        "source_pos_wait",
        "pg_sleep",
        "xp_cmdshell",
    }
)

# Constructs that could escape the single-table, single-statement scope.
_REJECT_NODES = (
    exp.Subquery,
    exp.CTE,
    exp.Union,
    exp.Intersect,
    exp.Except,
    exp.With,
    exp.Join,
    exp.Into,
    exp.Lock,
    exp.Placeholder,
    exp.Parameter,
    exp.Command,
)

_COMMENT_RE = re.compile(r"--|/\*|\*/|#")
# Identifiers carrying the bound values through the AST use a per-call random
# suffix, so the model can neither emit nor predict them. Any model SQL that
# contains the prefix at all is rejected.
_RESERVED_PREFIX = "__cs_"

PROVIDER_SQL_SCHEMA_PROMPT = """
You are a secure, read-only Text-to-SQL assistant for a healthcare platform.
Translate the provider's natural language question into a safe, valid MySQL SELECT query.

Queryable schema (the system automatically restricts every table to the calling provider and tenant):
1. provider (id, user_id, firstname, lastname, status, verification_status, degree, npi, practice_role, practice_name, practice_size, booking_type, primary_specialty, rating, reviews_count, tenant_id)
2. appointment_analytics (id, provider_id, tenant_id, period, today_count, this_week_count, last_week_count, this_month_count, last_month_count, this_year_count, last_year_count, start_date, end_date, selected_count)
3. availability (id, provider_id, tenant_id, working_days, appointment_availability, weekly_working_schedule_json)
4. appointments (id, provider_id, patient_id, tenant_id, appointment_type, status, appointment_date, created_at)
5. insurances (id, provider_id, tenant_id, insurance_name)

Rules for SQL generation:
1. Return ONLY the SQL query: no markdown fences, no explanation, no trailing semicolon.
2. The query must be a single SELECT against exactly ONE table above. No joins, subqueries, unions, CTEs or comments.
3. Do NOT write provider_id/tenant_id isolation filters and do NOT use bind placeholders (:name or %(name)s). The system injects and binds the verified caller's identity automatically.
4. Use only the columns listed for the chosen table. Add business filters (status, dates, period) as needed.
5. Use plain columns, comparisons and the aggregates COUNT, SUM, AVG, MIN, MAX only; call no other functions.
6. Keep results small; the system caps the number of rows automatically.
""".strip()


def _single_select(sql: str) -> exp.Select:
    try:
        statements = [stmt for stmt in sqlglot.parse(sql, read="mysql") if stmt is not None]
    except Exception as exc:
        raise AppError(400, "invalid_sql", "Could not parse the SQL query.") from exc
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise AppError(400, "invalid_sql", "Only a single SELECT statement is allowed.")
    return statements[0]


def _reject_unsafe(tree: exp.Expression) -> None:
    if next(tree.find_all(*_REJECT_NODES), None) is not None:
        raise AppError(
            400,
            "invalid_sql",
            "Joins, subqueries, unions, locking clauses and parameters are not allowed.",
        )
    for func in tree.find_all(exp.Func):
        # Unknown functions parse as Anonymous and report "ANONYMOUS" as their
        # SQL name, so the denylist must read the name they were called with.
        name = func.name if isinstance(func, exp.Anonymous) else func.sql_name()
        if name.lower() in _FORBIDDEN_FUNCS:
            raise AppError(400, "invalid_sql", "Forbidden SQL function detected.")


def _cap_limit(tree: exp.Select, max_rows: int) -> None:
    limit_node = tree.args.get("limit")
    if limit_node is None:
        tree.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
        return
    current = limit_node.expression
    try:
        count = int(current.this) if isinstance(current, exp.Literal) else max_rows
    except (TypeError, ValueError):
        count = max_rows
    limit_node.set("expression", exp.Literal.number(min(max(count, 1), max_rows)))


def _verify_rewritten(
    sql: str, table: str, provider_column: str, provider_nonce: str, tenant_nonce: str, max_rows: int
) -> None:
    """Re-parse the rewritten statement and demand the scope this service promises.

    The rewrite is an assumption about the AST; this check is the evidence. If
    rendering ever loses the predicates, adds a table or unbounds the LIMIT, the
    query is refused instead of reaching the driver.
    """
    tree = _single_select(sql)
    _reject_unsafe(tree)
    tables = list(tree.find_all(exp.Table))
    where = tree.args.get("where")
    limit = tree.args.get("limit")
    scoped = {
        (left.name.lower(), right.name.lower())
        for left, right in ((eq.this, eq.expression) for eq in tree.find_all(exp.EQ))
        if isinstance(left, exp.Column) and isinstance(right, exp.Column)
    }
    bounded = (
        limit is not None
        and isinstance(limit.expression, exp.Literal)
        and limit.expression.is_int
        and 0 < int(limit.expression.this) <= max_rows
    )
    if (
        len(tables) != 1
        or tables[0].name.lower() != table
        or sql.count(provider_nonce) != 1
        or sql.count(tenant_nonce) != 1
        or where is None
        or (provider_column, provider_nonce) not in scoped
        or ("tenant_id", tenant_nonce) not in scoped
        or not bounded
    ):
        raise AppError(400, "invalid_sql", "Could not scope the query safely.")


def validate_and_sanitize_sql(sql: str, provider_id: str, tenant_id: str) -> str:
    """Parse, validate and rewrite the generated SQL into a scoped, bound SELECT.

    ``provider_id`` and ``tenant_id`` are not interpolated here: the rewritten
    query carries pyformat placeholders that pymysql binds at execution time.
    The arguments are accepted for signature compatibility and future use.
    """
    cleaned = sql.strip()
    if not cleaned:
        raise AppError(400, "invalid_sql", "Empty SQL query.")
    if _COMMENT_RE.search(cleaned) or ";" in cleaned.rstrip(";"):
        raise AppError(400, "invalid_sql", "Comments and multiple statements are not allowed.")
    if _RESERVED_PREFIX in cleaned:
        raise AppError(400, "invalid_sql", "Reserved identifiers are not allowed.")

    tree = _single_select(cleaned)
    _reject_unsafe(tree)

    tables = list(tree.find_all(exp.Table))
    if len(tables) != 1:
        raise AppError(400, "invalid_sql", "Exactly one table is allowed per query.")
    table_node = tables[0]
    if table_node.args.get("db") or table_node.args.get("catalog"):
        raise AppError(400, "invalid_sql", "Cross-database queries are not allowed.")
    table = table_node.name.lower()
    if table not in _TABLE_SCOPE:
        raise AppError(400, "invalid_sql", f"Table '{table}' is not queryable.")
    provider_column = _TABLE_SCOPE[table]

    # Random per call: the model cannot emit or guess these identifiers.
    provider_nonce = f"{_RESERVED_PREFIX}{secrets.token_hex(12)}_provider"
    tenant_nonce = f"{_RESERVED_PREFIX}{secrets.token_hex(12)}_tenant"
    # AND the caller's filters (parenthesised so an OR cannot widen the scope)
    # with the mandatory provider + tenant predicate.
    scope = sqlglot.parse_one(
        f"{provider_column} = {provider_nonce} AND tenant_id = {tenant_nonce}",
        read="mysql",
    )
    where = tree.args.get("where")
    if where is not None:
        combined = exp.And(this=exp.Paren(this=where.this), expression=exp.Paren(this=scope))
    else:
        combined = scope
    tree.set("where", exp.Where(this=combined))

    # Force a bounded LIMIT regardless of what the model emitted.
    max_rows = max(config.PROVIDER_SQL_MAX_ROWS, 1)
    _cap_limit(tree, max_rows)

    rendered = tree.sql(dialect="mysql")
    _verify_rewritten(rendered, table, provider_column, provider_nonce, tenant_nonce, max_rows)

    # pymysql binds with pyformat, so a literal % from the model (e.g. LIKE '%x%')
    # must be escaped to %% or the query is treated as a broken format string.
    # The nonces carry no %, so escaping first leaves our placeholders intact.
    rendered = rendered.replace("%", "%%")
    return rendered.replace(provider_nonce, "%(provider_id)s").replace(tenant_nonce, "%(tenant_id)s")


async def translate_question_to_sql(question: str, history: list[Turn]) -> str:
    """Use LLM to translate natural language into SQL."""
    history_transcript = "\n".join(f"{t.role}: {t.text}" for t in history[-4:])
    prompt = (
        f"Conversation History:\n{history_transcript}\n\nQuestion: {question}"
        if history_transcript
        else f"Question: {question}"
    )

    messages = [
        {"role": "system", "content": PROVIDER_SQL_SCHEMA_PROMPT},
        {"role": "user", "content": prompt},
    ]

    runtime = get_runtime()
    payload = {
        "model": config.LLM_MODEL,
        "temperature": 0.0,
        "max_tokens": 400,
        "messages": messages,
    }
    if config.LLM_REASONING_EFFORT:
        payload["reasoning_effort"] = config.LLM_REASONING_EFFORT

    try:
        async with limit(runtime.sem_llm):
            response = await runtime.http.post(
                f"{config.LLM_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
                json=payload,
                timeout=config.LLM_READ_TIMEOUT,
            )
    except Exception as exc:
        metric(provider_sql_trans_error=type(exc).__name__)
        raise AppError(502, "sql_translation_unavailable", "Failed to translate question to SQL query.") from exc

    if response.status_code >= 400:
        metric(provider_sql_trans_error=str(response.status_code))
        raise AppError(502, "sql_translation_failed", "SQL translation service returned an error.")

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        # Clean up possible markdown code blocks
        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(line for line in lines if not line.startswith("```")).strip()
        return content
    except Exception as exc:
        metric(provider_sql_parse_error=type(exc).__name__)
        raise AppError(502, "sql_parse_failed", "Failed to parse SQL translation output.") from exc


def _provider_pool() -> Any | None:
    """The dedicated provider DB pool only; None when it is not configured.

    Never the shared conversation pool: the two databases have different
    owners and schemas, so falling back would widen the isolation boundary
    beyond the provider operational data this service is meant to read.
    """
    if not config.PROVIDER_DB_URL:
        return None
    try:
        runtime = get_runtime()
    except RuntimeError:
        return None
    return runtime.provider_mysql


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


async def execute_provider_sql(
    principal: Principal, question: str, history: list[Turn] | None = None
) -> dict[str, Any]:
    """Execute Text-to-SQL query for an authenticated provider with strict isolation."""
    if principal.role != "provider" or not principal.provider_id:
        raise PermissionError("Provider Text-to-SQL analytics are available to provider accounts only.")
    if config.PROVIDER_SQL_PERMISSION and config.PROVIDER_SQL_PERMISSION not in principal.permissions:
        raise PermissionError("Provider data query requires analytics permission.")

    started = time.perf_counter()

    # Production path: the analytics API enforces provider scope server-side.
    if config.ANALYTICS_API_URL:
        try:
            runtime = get_runtime()
            async with asyncio.timeout(config.PROVIDER_DB_TIMEOUT):
                params = {
                    "provider_id": principal.provider_id,
                    "tenant_id": principal.tenant_id,
                    "question": question,
                }
                headers = {}
                if config.PROVIDER_DB_TOKEN:
                    headers["Authorization"] = f"Bearer {config.PROVIDER_DB_TOKEN}"
                elif config.PROVIDERS_API_TOKEN:
                    headers["Authorization"] = f"Bearer {config.PROVIDERS_API_TOKEN}"

                response = await runtime.http.get(config.ANALYTICS_API_URL, params=params, headers=headers)
                if response.status_code == 200:
                    payload = response.json()
                    # The platform wraps its responses as {"status","message","data"}.
                    data = payload.get("data", payload) if isinstance(payload, dict) else payload
                    metric(provider_analytics_success=True)
                    result = {"question": question, "data": data, "source": "api"}
                    audit_provider_sql(principal, question=question, result=result, latency_ms=_ms(started))
                    return result
        except Exception as exc:
            log.warning("Analytics API request failed (%s); falling back to SQL if available", type(exc).__name__)

    pool = _provider_pool()
    if pool is None:
        # No database configured: structured dev placeholder, explicitly marked
        # as mock so callers never mistake fabricated counts for real data.
        result = {
            "question": question,
            "source": "mock",
            "sql": None,
            "rows": [
                {
                    "provider_id": principal.provider_id,
                    "tenant_id": principal.tenant_id,
                    "today_count": 0,
                    "this_week_count": 0,
                    "this_month_count": 0,
                    "appointment_availability": True,
                }
            ],
            "row_count": 1,
        }
        audit_provider_sql(principal, question=question, result=result, latency_ms=_ms(started))
        return result

    history_turns = history or []
    sql_query = await translate_question_to_sql(question, history_turns)
    safe_sql = validate_and_sanitize_sql(sql_query, str(principal.provider_id), principal.tenant_id)

    rows: list[dict[str, Any]] = []
    try:
        async with asyncio.timeout(config.PROVIDER_DB_TIMEOUT):
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    # pymysql/aiomysql use pyformat: %(name)s bound from this dict.
                    await cursor.execute(
                        safe_sql, {"provider_id": principal.provider_id, "tenant_id": principal.tenant_id}
                    )
                    columns = [col[0] for col in cursor.description] if cursor.description else []
                    # fetchmany caps memory even if the rewritten LIMIT is bypassed.
                    raw_rows = await cursor.fetchmany(max(config.PROVIDER_SQL_MAX_ROWS, 1))
                    for row in raw_rows:
                        rows.append(dict(zip(columns, row, strict=False)))
    except Exception as exc:
        metric(provider_sql_exec_error=type(exc).__name__)
        log.error("Provider SQL execution failed: %s", type(exc).__name__)
        audit_provider_sql(
            principal,
            question=question,
            sql=safe_sql,
            outcome="error",
            error=type(exc).__name__,
            latency_ms=_ms(started),
        )
        raise AppError(502, "sql_execution_failed", "Failed to execute provider query safely.") from exc

    metric(provider_sql_success=True, row_count=len(rows))
    result = {"question": question, "sql": safe_sql, "rows": rows, "row_count": len(rows), "source": "sql"}
    audit_provider_sql(principal, question=question, sql=safe_sql, result=result, latency_ms=_ms(started))
    return result
