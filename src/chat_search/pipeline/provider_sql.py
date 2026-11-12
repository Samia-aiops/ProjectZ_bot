"""Provider Text-to-SQL pipeline for analytics, appointments, and availability."""

from __future__ import annotations

from collections.abc import AsyncIterator

from .. import config
from ..auth import Principal
from ..guardrails import normalize, rule_guard
from ..messages import REFUSAL
from ..observability import metric
from ..providers.sql_service import execute_provider_sql
from ..state import Turn
from .events import Event, done_event, single_reply, token_event


async def provider_sql_pipeline(
    principal: Principal,
    message: str,
    history: list[Turn] | None = None,
) -> AsyncIterator[Event]:
    """Execute text-to-SQL for providers and stream formatted results."""
    metric(cache="skip")
    if principal.role != "provider" or not principal.provider_id:
        async for event in single_reply("Provider data queries are available to provider accounts only."):
            yield event
        return
    if config.PROVIDER_SQL_PERMISSION and config.PROVIDER_SQL_PERMISSION not in principal.permissions:
        async for event in single_reply("Provider data queries require the analytics permission."):
            yield event
        return

    question = normalize(message)
    category = rule_guard(question)
    if category:
        metric(guard=category)
        async for event in single_reply(REFUSAL[category]):
            yield event
        return

    try:
        result = await execute_provider_sql(principal, question, history)
    except Exception as exc:
        msg = getattr(exc, "message", "Database query failed.")
        async for event in single_reply(f"I couldn't retrieve that data: {msg}"):
            yield event
        return

    if result.get("source") == "api":
        # Format the analytics data from the API result
        data = result.get("data") or {}
        analytics = data.get("appointment_analytics", {})
        counts = analytics.get("appointment_counts", {})

        answer = "Here is your appointment summary:\n\n"
        answer += f"- Today: {counts.get('today', 0)}\n"
        answer += f"- This Week: {counts.get('this_week', 0)}\n"
        answer += f"- This Month: {counts.get('this_month', 0)}\n"
        answer += f"- This Year: {counts.get('this_year', 0)}\n"

        availability = data.get("availability", {})
        if availability.get("appointment_availability"):
            answer += "\nYou are currently accepting new appointments."
        else:
            answer += "\nYou are not currently accepting new appointments."
    else:
        rows = result.get("rows", [])
        prefix = ""
        if result.get("source") == "mock":
            prefix = (
                "Note: no analytics database is configured, so this is placeholder data, not your real records.\n\n"
            )
        if not rows:
            answer = prefix + "I didn't find any records matching your request."
        else:
            answer = prefix + f"Here are the results for your query ({len(rows)} record(s)):\n\n"
            for i, row in enumerate(rows[:10], start=1):
                details = ", ".join(f"{k}: {v}" for k, v in row.items() if v is not None)
                answer += f"{i}. {details}\n"
            if len(rows) > 10:
                answer += f"\n...and {len(rows) - 10} more rows."

    for chunk in answer.splitlines(keepends=True):
        yield token_event(chunk)

    yield done_event([], False, {"provider_sql": result})
