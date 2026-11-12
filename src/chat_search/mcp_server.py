"""MCP server: the RAG corpus and the provider directory as tools.

The streamable HTTP transport is mounted on the API port and reuses the same
warm runtime as the chat service; the caller's identity comes from the app's
JWT. ``chat-search-mcp`` serves the same tools over stdio for a local client and
identifies the caller with ``MCP_ROLE``/``MCP_TENANT_ID`` instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import aclosing
from contextvars import ContextVar
from dataclasses import replace
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.datastructures import Headers

from . import config
from .auth import Principal, dev_bypass_enabled, dev_principal, principal_from_token
from .errors import AppError
from .guardrails import INJECTION_RE, normalize, rule_guard
from .messages import REFUSAL, SEARCH_UNAVAILABLE_MSG
from .observability import request_context
from .pipeline.chat import open_events
from .providers.sql_service import execute_provider_sql
from .que import search_providers as provider_search
from .que import understand_provider_query
from .retrieval import embed_query, hybrid_search, rerank
from .runtime import get_runtime
from .state import load_history
from .understanding import expand_query

log = logging.getLogger(__name__)

_ALLOWED_HOSTS = [host.strip() for host in config.MCP_ALLOWED_HOSTS.split(",") if host.strip()]

mcp = FastMCP(
    "chat-search",
    instructions=(
        "Search the clinic's FAQ, policy and manual corpus and the public provider directory. "
        "Use search_knowledge for documents, search_providers for doctors and ask for a written "
        "answer with citations."
    ),
    streamable_http_path="/",
    # Each POST is served in its own request task. In stateful mode every tool
    # call runs in the task created by the session's `initialize`, so it would
    # execute under the identity and metrics context of whichever caller opened
    # the session instead of the caller that sent the call.
    stateless_http=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=bool(_ALLOWED_HOSTS),
        allowed_hosts=_ALLOWED_HOSTS,
    ),
)

_principal: ContextVar[Principal | None] = ContextVar("mcp_principal", default=None)
_HTTP_APP: Any = None


def _require_principal() -> Principal:
    principal = _principal.get()
    if principal is None:
        raise RuntimeError("MCP request is not authenticated")
    return principal


@mcp.tool()
async def search_knowledge(query: str, top_k: int = 5) -> dict[str, Any]:
    """Search FAQs, policies and manuals. Returns the best-matching passages with document and page."""
    principal = _require_principal()
    with request_context():
        question = query.strip()
        if not question:
            raise ValueError("query must not be empty")
        limit = max(1, min(top_k, 20))
        # Canonical terms help the index, exactly as on the chat path.
        text = expand_query(normalize(question))
        dense, indices, values = await embed_query(text)
        points = await hybrid_search(principal, dense, indices, values)
        candidates = [point.payload for point in points if point.payload and point.payload.get("text")]
        # Retrieved text is untrusted: it is screened before it leaves the tool.
        safe = [chunk for chunk in candidates if not INJECTION_RE.search(normalize(chunk["text"]))]
        chunks = await rerank(text, safe)
        return {
            "query": question,
            "canonical_query": text,
            "results": [
                {
                    "document": chunk["doc"],
                    "page": chunk["page"],
                    "section": chunk.get("section") or None,
                    "text": chunk["text"],
                    "score": round(float(chunk.get("_score", 0.0)), 3),
                }
                for chunk in chunks[:limit]
            ],
        }


@mcp.tool()
async def search_providers(query: str, languages: list[str] | None = None, limit: int = 5) -> dict[str, Any]:
    """Search the public provider directory by specialty, name, city or language."""
    principal = _require_principal()
    with request_context():
        if principal.role != "patient" or (
            config.PROVIDER_SEARCH_PERMISSION and config.PROVIDER_SEARCH_PERMISSION not in principal.permissions
        ):
            raise PermissionError("Provider search is available to patients with the directory permission")
        if get_runtime().es is None:
            raise RuntimeError(SEARCH_UNAVAILABLE_MSG)
        text = normalize(query)
        structured = understand_provider_query(text)
        if languages:
            structured = replace(structured, languages=tuple(languages))
        hits, relaxed, fallback = await provider_search(principal, structured)
        return {
            "query": query,
            "structured": structured.as_dict(),
            "terms": structured.terms,
            "languages": list(structured.languages),
            "relaxed": relaxed,
            "language_fallback": fallback,
            "providers": hits[: max(1, min(limit, config.PROVIDER_RESULTS))],
        }


@mcp.tool()
async def query_provider_data(question: str, session_id: str | None = None) -> dict[str, Any]:
    """Query provider-specific operational analytics, appointments, and schedule data via natural language (Text-to-SQL)."""
    principal = _require_principal()
    with request_context():
        if not config.MCP_SQL_ENABLED:
            raise RuntimeError("Text-to-SQL MCP provider tool is disabled in configuration.")
        if principal.role != "provider" or not principal.provider_id:
            raise PermissionError("Provider data query is available to provider accounts only.")
        if config.PROVIDER_SQL_PERMISSION and config.PROVIDER_SQL_PERMISSION not in principal.permissions:
            raise PermissionError("Provider data query requires analytics permission.")

        text = normalize(question)
        if not text:
            raise ValueError("question must not be empty")
        # Screen the question exactly as the chat pipeline does before it reaches the model.
        category = rule_guard(text)
        if category:
            return {"question": text, "refused": True, "category": category, "answer": REFUSAL[category]}
        history = await load_history(principal, session_id)
        return await execute_provider_sql(principal, text, history)


@mcp.tool()
async def ask(question: str) -> dict[str, Any]:
    """Answer a question from the corpus (RAG). Returns the answer, its sources and the parsed intent."""
    principal = _require_principal()
    with request_context():
        events = await open_events(principal, question.strip(), "auto", None)
        parts: list[str] = []
        async with aclosing(events) as iterator:
            async for event in iterator:
                kind = event.get("type")
                if kind == "token":
                    parts.append(event["text"])
                elif kind == "done":
                    data = event.get("data") or {}
                    return {
                        "answer": "".join(parts).strip(),
                        "sources": event.get("sources", []),
                        "cached": bool(event.get("cached")),
                        "understanding": data.get("understanding"),
                    }
                elif kind == "error":
                    raise RuntimeError(event.get("message") or "chat pipeline failed")
        return {"answer": "".join(parts).strip(), "sources": [], "cached": False, "understanding": None}


class _Authenticated:
    """ASGI wrapper: verify the app's JWT before the MCP transport sees a request."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        try:
            principal = _principal_for(Headers(scope=scope).get("authorization"))
        except AppError as exc:
            await _deny(send, exc.status_code, exc.message, exc.headers)
            return
        token = _principal.set(principal)
        try:
            await self.app(scope, receive, send)
        finally:
            _principal.reset(token)


def _principal_for(authorization: str | None) -> Principal:
    # A presented token is always verified; the dev bypass only applies when the
    # client sent none *and* the process declares a development environment.
    if authorization and authorization.lower().startswith("bearer "):
        return principal_from_token(authorization.split(" ", 1)[1].strip())
    if dev_bypass_enabled():
        return dev_principal(config.DEV_BYPASS_ROLE, config.MCP_TENANT_ID)
    if config.DEV_BYPASS_ROLE:
        raise AppError(500, "internal_error", "Authentication is misconfigured")
    raise AppError(401, "unauthorized", "Authentication required", {"WWW-Authenticate": "Bearer"})


async def _deny(send: Any, status: int, message: str, headers: dict[str, str] | None = None) -> None:
    body = json.dumps({"error": message}).encode()
    raw = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    raw.extend((key.lower().encode(), value.encode()) for key, value in (headers or {}).items())
    await send({"type": "http.response.start", "status": status, "headers": raw})
    await send({"type": "http.response.body", "body": body})


def http_app() -> Any:
    """Streamable HTTP transport; mount it in the FastAPI application."""
    global _HTTP_APP
    if _HTTP_APP is None:
        _HTTP_APP = _Authenticated(mcp.streamable_http_app())
    return _HTTP_APP


def session() -> Any:
    """Session-manager context; the parent app's lifespan must run it when mounted."""
    return mcp.session_manager.run()


def _stdio_principal() -> Principal:
    if config.MCP_ROLE not in config.ROLE_AUDIENCES:
        raise SystemExit(f"MCP_ROLE must be one of {sorted(config.ROLE_AUDIENCES)}")
    return dev_principal(config.MCP_ROLE, config.MCP_TENANT_ID)


async def _serve_stdio() -> None:
    from .bootstrap import shutdown, startup

    await startup()
    _principal.set(_stdio_principal())
    try:
        await mcp.run_stdio_async()
    finally:
        await shutdown()


def stdio_main() -> None:
    logging.basicConfig(level=config.LOG_LEVEL)
    asyncio.run(_serve_stdio())
