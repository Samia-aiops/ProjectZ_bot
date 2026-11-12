"""Application startup and shutdown.

Only read paths are opened here; the serving process never indexes documents or
creates search indexes. Run ``index-documents`` and ``sync-providers`` as
separate jobs. Startup fails fast on missing security-critical configuration.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import httpx
import redis.asyncio as aioredis
from qdrant_client import AsyncQdrantClient

from . import config
from .cache import init_cache
from .corpus import read_corpus, reload_corpus
from .guardrails import load_pii_engine
from .logging_setup import configure_logging, log_failure
from .models import load_models
from .pipeline.agent import warm_router
from .providers.client import make_es_client
from .que import build_query, understand_provider_query
from .runtime import Runtime, UserSlots, clear_runtime, get_runtime, set_runtime, spawn
from .spelling import warm_spelling
from .state import close_store, open_provider_store, open_store
from .synonyms import warm_synonyms
from .understanding import warm_understanding

log = logging.getLogger(__name__)


async def _warn_missing_indexes(runtime: Runtime) -> None:
    try:
        exists = await asyncio.wait_for(runtime.qdrant.collection_exists(config.QDRANT_COLLECTION), 3)
    except Exception as exc:
        log.warning("Qdrant not reachable at startup (%s)", type(exc).__name__)
    else:
        if not exists:
            log.warning(
                "Qdrant collection '%s' is missing; run `python -m chat_search index-documents`",
                config.QDRANT_COLLECTION,
            )

    if runtime.es is None:
        return
    try:
        exists = await asyncio.wait_for(runtime.es.indices.exists(index=config.ES_PROVIDERS_INDEX), 3)
    except Exception as exc:
        log.warning("Elasticsearch not reachable at startup (%s)", type(exc).__name__)
    else:
        if not exists:
            log.warning(
                "Elasticsearch index '%s' is missing; run `python -m chat_search sync-providers`",
                config.ES_PROVIDERS_INDEX,
            )
            return
    # The first directory query of a fresh process pays ~10x (cold ES query
    # structures); warm the real query shape so the first patient does not.
    try:
        await asyncio.wait_for(
            runtime.es.search(
                index=config.ES_PROVIDERS_INDEX,
                query=build_query(understand_provider_query("cardiology"), strict=True, tenant_id=""),
                size=0,
                track_total_hits=False,
            ),
            3,
        )
    except Exception as exc:
        log.warning("Elasticsearch warm-up query failed (%s)", type(exc).__name__)


async def _load_pii() -> object | None:
    """Presidio is optional at runtime: without it PII falls back to the regex."""
    if not config.PII_ENABLED:
        return None
    try:
        return await asyncio.to_thread(load_pii_engine)
    except Exception as exc:
        log.warning(
            "Presidio PII detection unavailable (%s); using the regex fallback. "
            "Install presidio-analyzer and `python -m spacy download %s`.",
            type(exc).__name__,
            config.PII_SPACY_MODEL,
        )
        return None


def _warm_spacy() -> None:
    """Initialize spaCy once, before the startup threads run.

    spaCy's Cython modules are not safe to initialize from several threads at
    once: the parallel warmups below then fail with ``KeyError: 'spacy'`` (or a
    partial-module ImportError) and quietly fall back to regex, so Presidio,
    the phrase matcher and clinical NER all appear "unavailable".
    """
    try:
        import spacy

        spacy.blank("en")
    except Exception as exc:
        log.warning("spaCy unavailable (%s); understanding stays on the catalogue regex", type(exc).__name__)


async def startup() -> None:
    configure_logging()
    config.validate_cors()
    config.validate_runtime()

    if config.DEV_BYPASS_ROLE:
        log.warning(
            "AUTH BYPASS ACTIVE as role=%s in APP_ENV=%s - never use outside local dev",
            config.DEV_BYPASS_ROLE,
            config.APP_ENV,
        )

    _warm_spacy()
    models, catalog, pii, understanding = await asyncio.gather(
        asyncio.to_thread(load_models),
        asyncio.to_thread(warm_synonyms),
        _load_pii(),
        asyncio.to_thread(warm_understanding),
    )
    spelling_ready = await asyncio.to_thread(warm_spelling)
    dimension = models.embedder.get_sentence_embedding_dimension()

    redis = aioredis.Redis.from_url(
        config.REDIS_URL,
        socket_timeout=config.REDIS_TIMEOUT,
        socket_connect_timeout=config.REDIS_TIMEOUT,
        max_connections=config.REDIS_MAX_CONNECTIONS,
    )
    corpus = await read_corpus(redis)  # published by the index job; settings fallback

    runtime = Runtime(
        embedder=models.embedder,
        sparse=models.sparse,
        reranker=models.reranker,
        guard=models.guard,
        pii=pii,
        executor=ThreadPoolExecutor(max_workers=config.EXECUTOR_WORKERS, thread_name_prefix="cpu"),
        embed_executor=ThreadPoolExecutor(max_workers=config.EMBED_WORKERS, thread_name_prefix="embed"),
        qdrant=AsyncQdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            timeout=max(int(config.QDRANT_TIMEOUT), 1),
        ),
        redis=redis,
        http=httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=config.LLM_CONNECT_TIMEOUT,
                read=config.LLM_READ_TIMEOUT,
                write=10,
                pool=5,
            ),
            limits=httpx.Limits(max_connections=config.LLM_MAX_CONNECTIONS, max_keepalive_connections=20),
        ),
        sem_guard=asyncio.Semaphore(config.SEM_GUARD),
        sem_rag=asyncio.Semaphore(config.SEM_RAG),
        sem_rerank=asyncio.Semaphore(config.SEM_RERANK),
        sem_llm=asyncio.Semaphore(config.SEM_LLM),
        user_slots=UserSlots(config.USER_CONCURRENCY, redis=redis, ttl=config.USER_LOCK_TTL),
        corpus=corpus,
    )
    if config.ES_URL:
        runtime.es = make_es_client()
    if config.MYSQL_URL:
        try:
            runtime.mysql = await asyncio.wait_for(open_store(), 5)
        except Exception as exc:
            log.warning("Conversation history disabled (%s). Multi-turn context needs MySQL.", type(exc).__name__)
    if config.PROVIDER_DB_URL:
        try:
            runtime.provider_mysql = await asyncio.wait_for(open_provider_store(), 5)
        except Exception as exc:
            log.warning(
                "Provider SQL database unavailable (%s); analytics falls back to the API or mock.", type(exc).__name__
            )
    set_runtime(runtime)

    if config.CORPUS_RELOAD_SECONDS > 0:
        spawn(reload_corpus(runtime, config.CORPUS_RELOAD_SECONDS))

    # Loading the router checkpoint takes seconds; until it finishes the agent
    # asks the LLM, so this must not hold up startup.
    spawn(warm_router())

    await _warn_missing_indexes(runtime)

    if config.CACHE_ENABLED:
        try:
            await asyncio.wait_for(init_cache(dimension), 5)
            runtime.cache_ok = True
        except Exception as exc:
            log.warning("Semantic cache disabled (%s). Needs Redis Stack with RediSearch.", type(exc).__name__)

    log.info(
        "Ready corpus=%s cache=%s history=%s guard=%s pii=%s rerank=%s providers=%s synonyms=%d understanding=%s spelling=%s router=%s audit=%s cpu_budget=%d",
        corpus,
        runtime.cache_ok,
        bool(runtime.mysql),
        bool(models.guard),
        "presidio" if pii else ("regex" if config.PII_ENABLED else "off"),
        bool(models.reranker),
        bool(runtime.es),
        len(catalog.groups),
        understanding,
        spelling_ready,
        config.AGENT_ROUTER_BACKEND,
        config.AUDIT_LOG_ENABLED,
        config.CPU_BUDGET,
    )


async def shutdown() -> None:
    try:
        runtime = get_runtime()
    except RuntimeError:
        return

    for task in list(runtime.tasks):
        task.cancel()
    closers = [("http", runtime.http.aclose), ("qdrant", runtime.qdrant.close), ("redis", runtime.redis.aclose)]
    if runtime.es:
        closers.append(("elasticsearch", runtime.es.close))
    if runtime.mysql is not None:
        closers.append(("mysql", lambda: close_store(runtime.mysql)))
    if runtime.provider_mysql is not None:
        closers.append(("provider_mysql", lambda: close_store(runtime.provider_mysql)))
    for name, closer in closers:
        try:
            await closer()
        except Exception as exc:
            log_failure(log, "shutdown cleanup failed", exc, resource=name, level=logging.WARNING)
    runtime.executor.shutdown(wait=False, cancel_futures=True)
    runtime.embed_executor.shutdown(wait=False, cancel_futures=True)
    clear_runtime()
