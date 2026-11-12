"""Command-line entry points.

python -m chat_search                              # run the API (read-only serving)
python -m chat_search token patient [tenant]       # mint a dev JWT (HS* only)
python -m chat_search index-documents              # PDFs -> Qdrant (batch job)
python -m chat_search seed-providers [tenant]      # dev: sample providers into Elasticsearch
python -m chat_search sync-providers               # one provider sync run (Airflow)
python -m chat_search sync-provider-synonyms       # update the ES synonym_graph rules
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from . import config
from .logging_setup import configure_logging, log_failure

log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chat-search", description="Platform FAQ assistant")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="run the HTTP API (default)")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=None)

    token = subparsers.add_parser("token", help="mint a development JWT")
    token.add_argument("role", choices=sorted(config.ROLE_AUDIENCES))
    token.add_argument("tenant", nargs="?", default="default")

    subparsers.add_parser("index-documents", help="index DATA_DIR PDFs into Qdrant")

    seed = subparsers.add_parser("seed-providers", help="load sample providers (development)")
    seed.add_argument("tenant", nargs="?", default="default")

    subparsers.add_parser("sync-providers", help="run one provider sync into Elasticsearch")
    subparsers.add_parser(
        "sync-provider-synonyms",
        help="apply the synonym catalogue to the provider index's search analyzer",
    )
    return parser


def _serve(host: str, port: int) -> int:
    import uvicorn

    from .api.app import app

    uvicorn.run(
        app,
        host=host,
        port=port,
        access_log=False,  # RequestContextMiddleware writes the structured log line
        log_config=None,  # keep our redacting handlers instead of uvicorn's defaults
        limit_concurrency=config.UVICORN_LIMIT_CONCURRENCY,
        timeout_keep_alive=15,
    )
    return 0


async def _remember_corpus(corpus: str) -> None:
    import redis.asyncio as aioredis

    from .corpus import write_corpus

    redis = aioredis.Redis.from_url(
        config.REDIS_URL,
        socket_timeout=config.REDIS_TIMEOUT,
        socket_connect_timeout=config.REDIS_TIMEOUT,
    )
    try:
        await write_corpus(redis, corpus)
    finally:
        await redis.aclose()


def _index_documents() -> int:
    from .indexing.qdrant import index_documents
    from .models import load_index_models

    if not config.DATA_DIR.is_dir():
        log.error("DATA_DIR not found: %s", config.DATA_DIR.resolve())
        return 1

    try:
        models = load_index_models()
        corpus = index_documents(models.embedder, models.sparse)
    except Exception as exc:
        log_failure(log, "document indexing failed", exc)
        return 1

    try:
        asyncio.run(_remember_corpus(corpus))
    except Exception as exc:  # the index is ready; serving falls back to settings identity
        log.warning("Indexed corpus %s but could not publish it to Redis (%s)", corpus, type(exc).__name__)
    else:
        log.info("Document index ready corpus=%s", corpus)
    return 0


def _sync_providers() -> int:
    from .providers.indexing.sync import sync_providers

    try:
        return 0 if asyncio.run(sync_providers()) else 1
    except Exception as exc:
        log_failure(log, "provider sync failed", exc)
        return 1


async def _apply_provider_synonyms() -> None:
    from .providers.client import make_es_client
    from .providers.indexing.schema import ensure_providers_index

    client = make_es_client(write=True)
    try:
        await ensure_providers_index(client)
    finally:
        await client.close()


def _sync_provider_synonyms() -> int:
    try:
        asyncio.run(_apply_provider_synonyms())
    except Exception as exc:
        log_failure(log, "provider synonym sync failed", exc)
        return 1
    return 0


def _seed_providers(tenant: str) -> int:
    from .providers.indexing.seed import seed_providers

    try:
        asyncio.run(seed_providers(tenant))
    except RuntimeError as exc:
        log_failure(log, "provider seed failed", exc)
        return 1
    return 0


def _mint_token(role: str, tenant: str) -> int:
    from .auth import mint_dev_token

    if not config.JWT_ALGORITHM.startswith("HS"):
        print("Dev tokens only work with HS* (shared secret)", file=sys.stderr)
        return 2
    if not config.JWT_KEY:
        print("Set JWT_SECRET first", file=sys.stderr)
        return 2
    print(mint_dev_token(role, tenant))
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = _build_parser().parse_args(argv)

    if args.command in (None, "serve"):
        # Bare `python -m chat_search` has no subparser, so host/port are unset.
        host = getattr(args, "host", "0.0.0.0")
        port = getattr(args, "port", None) or config.PORT
        return _serve(host, port)
    if args.command == "token":
        return _mint_token(args.role, args.tenant)
    if args.command == "index-documents":
        return _index_documents()
    if args.command == "seed-providers":
        return _seed_providers(args.tenant)
    if args.command == "sync-providers":
        return _sync_providers()
    if args.command == "sync-provider-synonyms":
        return _sync_provider_synonyms()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
