"""Request-scoped context, metrics and the outermost ASGI middleware.

The structured request line contains only metadata (method, path, status,
timing, error code type) and never the query string, user input or PHI.
Conversation text is written by the separate audit log in ``audit.py``.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders

from . import config
from .logging_setup import log_failure
from .messages import ERROR_MSG

log = logging.getLogger(__name__)

request_ctx: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("request_ctx", default=None)
# Never logged: the rate limiter hashes it into a key and nothing else reads it.
client_ip_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("client_ip", default=None)

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _client_address(scope: dict[str, Any]) -> str | None:
    """The caller's address, or the edge proxy's view of it when that is trusted."""
    if config.TRUST_FORWARDED_FOR:
        forwarded = dict(scope["headers"]).get(b"x-forwarded-for", b"").decode("latin-1")
        # The last entry is the one our own proxy appended; earlier ones are client claims.
        if last := forwarded.split(",")[-1].strip():
            return last
    client = scope.get("client")
    return client[0] if client else None


def _loggable(value: str, limit: int) -> str:
    return _CONTROL_RE.sub("?", value)[:limit]


def metric(**fields: Any) -> None:
    ctx = request_ctx.get()
    if ctx is not None:
        ctx.update(fields)


def elapsed_ms() -> int:
    ctx = request_ctx.get()
    started = ctx.get("t0") if ctx else None
    return int((time.perf_counter() - started) * 1000) if started else 0


@contextmanager
def request_context(request_id: str | None = None) -> Iterator[dict[str, Any]]:
    """Give a unit of work its own metrics context (MCP tool calls have no HTTP request).

    The task that runs an MCP tool was started by an earlier request (the session
    initialize), so it must not reuse that request's dict: its timing and fields
    belong to the tool call, not to the handshake.
    """
    ctx: dict[str, Any] = {
        "request_id": request_id or uuid.uuid4().hex,
        "t0": time.perf_counter(),
        "status": 0,
    }
    token = request_ctx.set(ctx)
    try:
        yield ctx
    finally:
        request_ctx.reset(token)


def error_response(
    status_code: int,
    code: str,
    message: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    ctx = request_ctx.get() or {}
    metric(error=code)
    return JSONResponse(
        {"error": {"code": code, "message": message, "request_id": ctx.get("request_id")}},
        status_code=status_code,
        headers=headers,
    )


class RequestContextMiddleware:
    """Pure ASGI middleware (safe for streaming): request id, headers, structured log line."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if re.fullmatch(r"[A-Za-z0-9-]{8,64}", incoming) else uuid.uuid4().hex
        started_at = time.perf_counter()
        ctx: dict[str, Any] = {"request_id": request_id, "t0": started_at, "status": 0}
        token = request_ctx.set(ctx)
        ip_token = client_ip_ctx.set(_client_address(scope))
        started = False

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                ctx["status"] = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                headers.setdefault("X-Content-Type-Options", "nosniff")
                headers.setdefault("Referrer-Policy", "no-referrer")
                headers.setdefault("Cache-Control", "no-store")
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except asyncio.CancelledError:
            ctx["outcome"] = "cancelled"  # client disconnected
            raise
        except Exception as exc:
            ctx["outcome"] = "error"
            ctx["error"] = type(exc).__name__
            ctx["status"] = 500
            # Type + request id only; the exception text may contain user data.
            log_failure(log, "unhandled request error", exc, request_id=request_id)
            if not started:
                await error_response(500, "internal_error", ERROR_MSG)(scope, receive, send_wrapper)
        finally:
            ctx.setdefault("outcome", "ok" if ctx["status"] < 400 else "error")
            ctx["latency_ms"] = int((time.perf_counter() - started_at) * 1000)
            ctx["method"] = _loggable(str(scope.get("method", "")), 16)
            ctx["path"] = _loggable(str(scope.get("path", "")), 256)  # never the query string
            # Copy without t0: the live dict is shared with tasks spawned during the
            # request, so it is never mutated here.
            record = {key: value for key, value in ctx.items() if key != "t0"}
            try:
                log.info("request %s", json.dumps(record, default=str, separators=(",", ":")))
            except Exception:
                pass  # logging must never break a response
            client_ip_ctx.reset(ip_token)
            request_ctx.reset(token)
