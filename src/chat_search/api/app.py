"""FastAPI application factory and routes."""

from __future__ import annotations

import asyncio
import logging
from contextlib import aclosing, asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Query
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__, config
from ..auth import Principal, current_principal
from ..errors import AppError
from ..logging_setup import log_failure
from ..messages import HTTP_ERROR_MESSAGES
from ..observability import RequestContextMiddleware, error_response, request_ctx
from ..pipeline.chat import open_events
from ..pipeline.events import sse
from ..runtime import get_runtime
from ..state import SESSION_RE, clear_history
from .schemas import ChatRequest, ChatResponse
from .ui import CHAT_PAGE_HTML

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    from ..bootstrap import shutdown, startup

    await startup()
    try:
        if config.MCP_ENABLED:
            # Mounted sub-app lifespans do not run, so the MCP session manager
            # is entered here, alongside the shared runtime.
            from ..mcp_server import session

            async with session():
                yield
        else:
            yield
    finally:
        await shutdown()


async def _probe(coro: Any) -> bool:
    try:
        return bool(await asyncio.wait_for(coro, 1.0))
    except Exception:
        return False


def create_app() -> FastAPI:
    config.validate_cors()
    application = FastAPI(
        title=config.APP_NAME,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if config.ENABLE_DOCS else None,
        redoc_url=None,
        openapi_url="/openapi.json" if config.ENABLE_DOCS else None,
    )

    if config.CORS_ORIGINS:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=config.CORS_ORIGINS,
            allow_credentials=config.CORS_ALLOW_CREDENTIALS,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID", "Retry-After"],
            max_age=600,
        )
    application.add_middleware(RequestContextMiddleware)  # added last = outermost

    @application.exception_handler(AppError)
    async def _app_error(_, exc: AppError):
        return error_response(exc.status_code, exc.code, exc.message, exc.headers)

    @application.exception_handler(StarletteHTTPException)
    async def _http_error(_, exc: StarletteHTTPException):
        if exc.status_code >= 500:
            log_failure(
                log,
                "http error",
                status=exc.status_code,
                request_id=(request_ctx.get() or {}).get("request_id"),
                level=logging.WARNING,
            )
        message = HTTP_ERROR_MESSAGES.get(exc.status_code, "Request failed")
        return error_response(exc.status_code, "http_error", message)

    @application.exception_handler(RequestValidationError)
    async def _validation_error(_, __):
        return error_response(422, "invalid_request", "Invalid request")  # never echo the input back

    @application.post("/v1/chat/stream")
    async def chat_stream(req: ChatRequest, principal: Principal = Depends(current_principal)):
        events = await open_events(principal, req.message, req.mode, req.session_id)
        return StreamingResponse(
            sse(events),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    @application.post("/v1/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, principal: Principal = Depends(current_principal)):
        """Non-streaming variant; consumes the same pipeline."""
        events = await open_events(principal, req.message, req.mode, req.session_id)
        parts: list[str] = []
        sources: list[dict[str, Any]] = []
        cached = False
        data: dict[str, Any] | None = None
        async with aclosing(events) as iterator:
            async for event in iterator:
                if event["type"] == "token":
                    parts.append(event["text"])
                elif event["type"] == "done":
                    sources = event["sources"]
                    cached = event["cached"]
                    data = event.get("data")
                elif event["type"] == "error":
                    raise AppError(event.get("status", 500), event["code"], event["message"])
        return ChatResponse(answer="".join(parts), sources=sources, cached=cached, data=data)

    @application.delete("/v1/chat/session")
    async def reset_session(
        session_id: str | None = Query(default=None, max_length=64, pattern=SESSION_RE.pattern),
        principal: Principal = Depends(current_principal),
    ):
        """Forget the caller's conversation so the next question starts fresh."""
        cleared = await clear_history(principal, session_id)
        # No history store means there was nothing to forget; say so instead of
        # claiming a reset that never ran.
        return {"status": "ok" if cleared else "not_stored", "cleared": cleared}

    @application.get("/health")
    async def health():
        runtime = get_runtime()
        probes: dict[str, Any] = {
            "qdrant": runtime.qdrant.collection_exists(config.QDRANT_COLLECTION),
            "redis": runtime.redis.ping(),
        }
        if runtime.es is not None:
            # An index-scoped read key is denied on HEAD / (ping), so probe the index itself.
            probes["elasticsearch"] = runtime.es.indices.exists(index=config.ES_PROVIDERS_INDEX)
        # Probed together: one slow dependency must not delay the verdict on the others.
        results = await asyncio.gather(*(_probe(coro) for coro in probes.values()))
        states = dict(zip(probes, results, strict=True))
        # RAG needs both stores; Elasticsearch only degrades provider search.
        healthy = all(states.get(name, False) for name in ("qdrant", "redis"))
        # Public callers get a verdict, not an inventory of our infrastructure.
        body: dict[str, Any] = {"status": "ok" if healthy else "degraded"}
        if config.HEALTH_DETAILS:
            body.update(states)
            body["cache"] = runtime.cache_ok
        return JSONResponse(body, status_code=200 if healthy else 503)

    @application.get("/", response_class=HTMLResponse)
    async def home():
        return HTMLResponse(CHAT_PAGE_HTML)

    if config.MCP_ENABLED:
        from ..mcp_server import http_app

        application.mount(config.MCP_PATH, http_app())

    return application


app = create_app()
