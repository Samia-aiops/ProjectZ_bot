"""Event stream primitives shared by both pipelines."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

from ..errors import AppError
from ..logging_setup import log_failure
from ..messages import ERROR_MSG
from ..observability import metric, request_ctx

log = logging.getLogger(__name__)

Event = dict[str, Any]


def token_event(text: str) -> Event:
    return {"type": "token", "text": text}


def done_event(sources: list[dict[str, Any]], cached: bool, data: dict[str, Any] | None = None) -> Event:
    event: Event = {"type": "done", "sources": sources, "cached": cached}
    if data:
        event["data"] = data  # structured payload for the UI (e.g. {"providers": [...]})
    return event


def error_event(code: str, message: str, status: int, request_id: str | None = None) -> Event:
    event: Event = {"type": "error", "code": code, "message": message, "status": status}
    if request_id:
        event["request_id"] = request_id
    return event


async def single_reply(text: str) -> AsyncIterator[Event]:
    yield token_event(text)
    yield done_event([], False)


def release_task(task: asyncio.Task[Any] | None) -> None:
    if task is None:
        return
    if not task.done():
        task.cancel()
    elif not task.cancelled():
        task.exception()  # mark as retrieved


async def safe_events(events: AsyncIterator[Event]) -> AsyncIterator[Event]:
    """Turn failures into a final error event with a safe message."""
    try:
        async with aclosing(events) as iterator:
            async for event in iterator:
                yield event
    except AppError as exc:
        request_id = (request_ctx.get() or {}).get("request_id")
        metric(error=exc.code)
        if exc.status_code >= 500:
            log_failure(log, "pipeline failed", code=exc.code, status=exc.status_code, request_id=request_id)
        yield error_event(exc.code, exc.message, exc.status_code, request_id)
    except asyncio.CancelledError:
        metric(outcome="cancelled")
        raise
    except Exception as exc:
        request_id = (request_ctx.get() or {}).get("request_id")
        metric(error=type(exc).__name__)
        log_failure(log, "pipeline failed", exc, request_id=request_id)
        yield error_event("internal_error", ERROR_MSG, 500, request_id)


async def sse(events: AsyncIterator[Event]) -> AsyncIterator[str]:
    async with aclosing(events) as iterator:
        async for event in iterator:
            yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
