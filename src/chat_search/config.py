"""Environment-driven configuration.

Every environment variable is read here, once, at import time. Application
modules import this module and read attributes dynamically (``config.TOP_K``)
so tests can override values without reloading the process.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

# =============================================================================
# Environment helpers
# =============================================================================


def _env(name: str, default: str = "") -> str:
    """Return an environment value; a variable that is set but blank counts as unset."""
    value = os.getenv(name)
    return default if value is None or not value.strip() else value.strip()


def _int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in ("1", "true", "yes")


def _list(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in _env(name, default).split(",") if item.strip()]


def _cpu_budget() -> int:
    """CPUs this process may actually use.

    A container's cgroup quota is often far below the host's core count, and
    ``os.cpu_count()`` reports the host. Sizing thread pools from the host
    oversubscribes a small task on a big node, which shows up as p95 latency
    with no error to explain it.
    """
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="ascii").split()[:2]
        if quota != "max":
            return max(round(int(quota) / int(period)), 1)
    except (OSError, ValueError):
        pass
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text(encoding="ascii"))
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text(encoding="ascii"))
        if quota > 0 and period > 0:
            return max(round(quota / period), 1)
    except (OSError, ValueError):
        pass
    try:
        return max(len(os.sched_getaffinity(0)), 1)  # Windows has no affinity mask API
    except (AttributeError, OSError):
        return max(os.cpu_count() or 2, 1)


def _json(name: str) -> dict[str, Any]:
    try:
        value = json.loads(_env(name, "{}"))
    except ValueError:
        raise RuntimeError(f'{name} must be valid JSON, e.g. {{"key": "value"}}') from None
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a JSON object")
    return value


# =============================================================================
# Application
# =============================================================================

APP_NAME = _env("APP_NAME", "Platform FAQ Assistant")
PORT = _int("PORT", 8000)
DATA_DIR = Path(_env("DATA_DIR", "data"))  # read only by the document index job
ENABLE_DOCS = _bool("ENABLE_DOCS", False)
LOG_TRACEBACKS = _bool("LOG_TRACEBACKS", False)
LOG_LEVEL = _env("LOG_LEVEL", "INFO")
MAX_MESSAGE_CHARS = _int("MAX_MESSAGE_CHARS", 1000)
# MCP (Model Context Protocol) tools for the RAG corpus and provider directory.
# The streamable HTTP transport is mounted on the API port and verifies the same
# JWT; the stdio transport (`chat-search-mcp`) identifies the caller with
# MCP_ROLE/MCP_TENANT_ID instead. Host validation stays off unless hosts are
# listed, because the directory is reached by IP (the JWT is the real gate).
MCP_ENABLED = _bool("MCP_ENABLED", True)
MCP_PATH = _env("MCP_PATH", "/mcp")
MCP_ALLOWED_HOSTS = _env("MCP_ALLOWED_HOSTS", "")  # csv; enables DNS-rebinding host checks
MCP_ROLE = _env("MCP_ROLE", "patient")
MCP_TENANT_ID = _env("MCP_TENANT_ID", "default")
# One audit record per chat request: caller ids, question, answer and sources.
# It is the only log line that carries conversation text (PHI), so route the
# "chat_search.audit" logger to a sink with its own access control, encryption
# and retention. AUDIT_LOG_CONTENT=false keeps ids, sources and timing only.
AUDIT_LOG_ENABLED = _bool("AUDIT_LOG_ENABLED", True)
AUDIT_LOG_CONTENT = _bool("AUDIT_LOG_CONTENT", True)
AUDIT_MAX_CHARS = _int("AUDIT_MAX_CHARS", 4000)  # per input/output field, after redaction

# =============================================================================
# Stores
# =============================================================================

QDRANT_URL = _env("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
# v2 uses named dense+sparse vectors; the old unnamed-vector collection is incompatible.
QDRANT_COLLECTION = _env("QDRANT_COLLECTION", "faq_documents_v2")
REDIS_URL = _env("REDIS_URL", "redis://localhost:6379/0")
# Shared by rate limits, the semantic cache and the per-user lock. Each task
# opens its own pool, so the account-wide total is this value times task count.
REDIS_MAX_CONNECTIONS = _int("REDIS_MAX_CONNECTIONS", 100)
# Durable conversation history; empty MYSQL_URL keeps chat stateless (single turn).
MYSQL_URL = os.getenv("MYSQL_URL", "")
# One pooled connection per in-flight history read; each task opens its own pool.
MYSQL_POOL_SIZE = _int("MYSQL_POOL_SIZE", 20)
MYSQL_TIMEOUT = _float("MYSQL_TIMEOUT", 2)  # conversation read/write; a slow store degrades, never blocks

# =============================================================================
# Models
# =============================================================================

EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")  # 384-dim, 512-token limit
# bge models are trained with this prefix on short search queries (not on documents).
EMBED_QUERY_PREFIX = os.getenv(
    "EMBED_QUERY_PREFIX",
    "Represent this sentence for searching relevant passages: " if "bge" in EMBEDDING_MODEL.lower() else "",
)
SPARSE_MODEL = _env("SPARSE_MODEL", "Qdrant/bm25")
RERANK_ENABLED = _bool("RERANK_ENABLED", True)
RERANK_MODEL = _env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# Ranking cost grows superlinearly with the token count (12 chunks at 512 tokens
# take ~2s on CPU, at 256 ~0.8s). Ranking only needs the head of a chunk; the
# LLM still receives the full text after expansion.
RERANK_MAX_LEN = _int("RERANK_MAX_LEN", 256)
# Raw logit scale for the default model, and a safety floor rather than a
# relevance cut: the same chunk scores from +9 to -11 depending on phrasing, so
# the source of truth is the rank-relative margin below.
RERANK_MIN_SCORE = _float("RERANK_MIN_SCORE", -10.0)
# Keep chunks within this distance of the best score. An absolute threshold
# cannot separate "relevant but weakly worded" from "unrelated" across phrasings.
RERANK_RELATIVE_MARGIN = _float("RERANK_RELATIVE_MARGIN", 6.0)
# ProtectAI's LLM Guard prompt-injection classifier (INJECTION / SAFE), loaded
# through transformers rather than the llm-guard package, which pins an older
# transformers than sentence-transformers 6 requires.
GUARDRAIL_MODEL = os.getenv("GUARDRAIL_MODEL", "protectai/deberta-v3-base-prompt-injection-v2")
GUARD_BLOCK_LABELS = {label.upper() for label in _list("GUARD_BLOCK_LABELS", "INJECTION,JAILBREAK,UNSAFE")}
GUARD_THRESHOLD = _float("GUARD_THRESHOLD", 0.92)
# Presidio PII detection. It feeds cache/history decisions, not refusals: text
# with personal data is answered but never shared or persisted.
PII_ENABLED = _bool("PII_ENABLED", True)
PII_SPACY_MODEL = _env("PII_SPACY_MODEL", "en_core_web_sm")
PII_SCORE_THRESHOLD = _float("PII_SCORE_THRESHOLD", 0.5)
# PERSON, LOCATION and DATE_TIME are deliberately absent: they match doctors,
# cities and ordinary dates in product questions and would block the cache for
# harmless text. An empty list means every entity Presidio knows.
PII_ENTITIES = tuple(
    _list(
        "PII_ENTITIES",
        "EMAIL_ADDRESS,PHONE_NUMBER,CREDIT_CARD,US_SSN,US_DRIVER_LICENSE,US_PASSPORT,"
        "US_ITIN,US_BANK_NUMBER,MEDICAL_LICENSE,IBAN_CODE,CRYPTO,UK_NHS,"
        "SELF_IDENTIFIER,LONG_NUMBER",
    )
)

# =============================================================================
# JWT (issued by Laravel)
# =============================================================================

JWT_ALGORITHM = _env("JWT_ALGORITHM", "HS256")
JWT_KEY = (
    os.getenv("JWT_PUBLIC_KEY", "").replace("\\n", "\n")
    if JWT_ALGORITHM.startswith(("RS", "ES"))
    else os.getenv("JWT_SECRET", "")
)
JWT_ISSUER = os.getenv("JWT_ISSUER") or None  # verified only when set
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE") or None  # verified only when set
JWT_LEEWAY = _int("JWT_LEEWAY_SECONDS", 10)
REQUIRED_PERMISSION = _env("REQUIRED_PERMISSION", "docs:read")
# The auth bypass is refused unless the process explicitly declares one of these
# environments, so a stray DEV_BYPASS_ROLE fails startup instead of turning the
# flag alone into a switch that authenticates every caller.
APP_ENV = os.getenv("APP_ENV", "production").strip().lower()
DEV_ENVIRONMENTS = frozenset({"dev", "development", "local", "test"})
DEV_BYPASS_ROLE = os.getenv("DEV_BYPASS_ROLE", "").strip().lower()  # "", "patient" or "provider"
# Optional provider id for the bypass identity. The analytics API requires the
# platform's numeric provider id, which no real JWT can be minted for in dev.
DEV_PROVIDER_ID = os.getenv("DEV_PROVIDER_ID", "").strip()

# =============================================================================
# LLM
# =============================================================================

LLM_BASE_URL = _env("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = _env("LLM_MODEL", "openai/gpt-oss-120b")
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 700)
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low")  # "" to omit (non gpt-oss)
LLM_CONNECT_TIMEOUT = _float("LLM_CONNECT_TIMEOUT", 5)
LLM_READ_TIMEOUT = _float("LLM_READ_TIMEOUT", 30)  # max silence, also bounds time-to-first-token
LLM_TOTAL_TIMEOUT = _float("LLM_TOTAL_TIMEOUT", 90)
LLM_MAX_CONNECTIONS = _int("LLM_MAX_CONNECTIONS", 50)

# =============================================================================
# Retrieval
# =============================================================================

TOP_K = _int("TOP_K", 5)
PREFETCH_LIMIT = _int("PREFETCH_LIMIT", 20)  # per leg (dense / sparse)
HYBRID_LIMIT = _int("HYBRID_LIMIT", 12)  # after RRF, before rerank

# =============================================================================
# Query understanding: spelling and dynamic synonyms
# =============================================================================

SPELLCHECK_ENABLED = _bool("SPELLCHECK_ENABLED", True)
# Maximum edit distance for a correction (longer words reach 2; short words stay at 1).
SPELL_MAX_EDIT_DISTANCE = _int("SPELL_MAX_EDIT_DISTANCE", 2)
SPELL_MIN_FREQUENCY = _int("SPELL_MIN_FREQUENCY", 20)  # reject rare candidates
SPELL_MAX_CORRECTIONS = _int("SPELL_MAX_CORRECTIONS", 4)  # cap changes per query
SPELL_DOMAIN_FREQUENCY = _int("SPELL_DOMAIN_FREQUENCY", 1000)  # frequency for catalogue terms

# Master switch for the concept catalogue; SYNONYMS_ENABLED remains honoured so
# an existing deployment can disable it without changing its environment.
CONCEPTS_ENABLED = _bool("CONCEPTS_ENABLED", True)
SYNONYMS_ENABLED = _bool("SYNONYMS_ENABLED", True)
# Structured catalogue (concepts.json). The loader also accepts the legacy
# synonyms.json shape, and SYNONYMS_FILE still overrides the source when set to
# anything other than the default name.
CONCEPTS_FILE = _env("CONCEPTS_FILE", "concepts.json")
CONCEPTS_JSON = os.getenv("CONCEPTS_JSON", "")
CONCEPTS_RELOAD_SECONDS = _float("CONCEPTS_RELOAD_SECONDS", 60)  # 0 = check on every query
SYNONYMS_FILE = _env("SYNONYMS_FILE", "synonyms.json")
SYNONYMS_JSON = os.getenv("SYNONYMS_JSON", "")
SYNONYMS_RELOAD_SECONDS = _float("SYNONYMS_RELOAD_SECONDS", 60)
SYNONYM_MAX_EXPANSIONS = _int("SYNONYM_MAX_EXPANSIONS", 4)  # canonical terms added per query

# Intent detection and medical concept recognition.
UNDERSTANDING_ENABLED = _bool("UNDERSTANDING_ENABLED", True)
# Standalone calls to understanding.understand() correct spelling; the chat
# pipeline passes the already-corrected text and leaves this off.
UNDERSTANDING_SPELLING = _bool("UNDERSTANDING_SPELLING", True)
# Clinical NER (scispaCy en_core_sci_sm). Missing package/model degrades to the
# catalogue concepts with a single warning; routing never depends on it.
SCISPACY_ENABLED = _bool("SCISPACY_ENABLED", True)
SCISPACY_MODEL = _env("SCISPACY_MODEL", "en_core_sci_sm")
# UMLS linking needs a local licensed dataset (or a downloaded linker bundle).
# Off by default so startup never reaches the network.
UMLS_ENABLED = _bool("UMLS_ENABLED", False)
UMLS_DATA_PATH = _env("UMLS_DATA_PATH", "")

# Rewrite a follow-up ("make it for next friday") into a standalone search query
# from the recent turns. Fail-open: a slow or failed call falls back to the
# heuristic retrieval text.
CONDENSE_ENABLED = _bool("CONDENSE_ENABLED", True)
CONDENSE_TIMEOUT = _float("CONDENSE_TIMEOUT", 4)

# The LangGraph agent's router node picks the pipeline. Routing is a System 1
# decision, so the default backend is Laya: a small non-autoregressive model that
# reads the message and recent turns and picks faq or providers in one forward
# pass, leaving the token budget to the answer. "llm" restores the label-only
# chat completion; "rules" keeps every decision deterministic. A slow, unloaded
# or unsure model falls back to the LLM, then to the rules.
AGENT_ROUTER_ENABLED = _bool("AGENT_ROUTER_ENABLED", True)
AGENT_ROUTER_BACKEND = _env("AGENT_ROUTER_BACKEND", "laya").lower()
AGENT_ROUTER_MODEL = _env("AGENT_ROUTER_MODEL", "convaiinnovations/laya")
# "" is the English checkpoint; "multilingual" is smaller and faster but weaker
# on platform questions. Keep the Dockerfile build args in step with these two.
AGENT_ROUTER_SUBFOLDER = os.getenv("AGENT_ROUTER_SUBFOLDER", "").strip()
AGENT_ROUTER_DEVICE = _env("AGENT_ROUTER_DEVICE", "cpu")
AGENT_ROUTER_TIMEOUT = _float("AGENT_ROUTER_TIMEOUT", 4)  # LLM label call
# Laya's forward pass is slower than a hosted completion on CPU, so it gets its
# own bound. A timeout is a rules fallback, not a retry.
AGENT_ROUTER_LAYA_TIMEOUT = _float("AGENT_ROUTER_LAYA_TIMEOUT", 8)
# The checkpoint ships uncalibrated temperatures (laya warns on load), so the
# floor is off by default. When set, a weaker answer falls back to the rules.
AGENT_ROUTER_MIN_CONFIDENCE = _float("AGENT_ROUTER_MIN_CONFIDENCE", 0.0)

# =============================================================================
# Semantic cache
# =============================================================================

CACHE_ENABLED = _bool("CACHE_ENABLED", True)
CACHE_SIMILARITY = _float("CACHE_SIMILARITY", 0.92)
CACHE_TTL = _int("CACHE_TTL_SECONDS", 86400)
CACHE_PER_USER = _bool("CACHE_PER_USER", False)  # True: never share answers across users
CACHE_PREFIX = "semcache:"
# The answer is stored before it is released, so a repeat (same user or a peer
# waiting on this question) cannot miss the entry the previous answer filled.
CACHE_WRITE_TIMEOUT = _float("CACHE_WRITE_TIMEOUT", 0.25)
# Identical questions arriving together are answered once: the leader computes,
# followers wait up to this long for its answer instead of repeating rerank and
# the LLM call. 0 disables coalescing and every caller answers independently.
CACHE_COALESCE_WAIT = _float("CACHE_COALESCE_WAIT_SECONDS", 5)
# Safety net for a leader that dies mid-answer (the marker is deleted normally).
CACHE_COALESCE_TTL = _float("CACHE_COALESCE_TTL_SECONDS", 120)
CACHE_PEER_TTL = _int("CACHE_PEER_TTL_SECONDS", 60)  # the in-flight answer channel, not a cache
# The corpus fingerprint scopes cache entries; re-read it after a reindex so a
# running replica stops serving answers built from the previous corpus.
# 0 disables the refresh (the fingerprint is then read once at startup).
CORPUS_RELOAD_SECONDS = _float("CORPUS_RELOAD_SECONDS", 60)

# =============================================================================
# Conversation state (multi-turn context, MySQL; see state.py)
# =============================================================================

HISTORY_ENABLED = _bool("HISTORY_ENABLED", True)
HISTORY_TURNS = _int("HISTORY_TURNS", 4)  # messages kept per conversation (user + assistant)
HISTORY_TTL = _int("HISTORY_TTL_SECONDS", 1800)  # an idle conversation is forgotten
HISTORY_MAX_CHARS = _int("HISTORY_MAX_CHARS", 400)  # per stored turn, bounds the prompt

# =============================================================================
# Rate limiting (token bucket, Redis; atomic per request)
# =============================================================================

# Capacity is a burst of this many requests, refilled evenly over the window.
RATE_LIMIT_WINDOW = _int("RATE_LIMIT_WINDOW_SECONDS", 60)
RATE_LIMIT_USER = _int("RATE_LIMIT_USER_REQUESTS", 20)
RATE_LIMIT_TENANT = _int("RATE_LIMIT_TENANT_REQUESTS", 300)
# Per source address; 0 disables. Behind a load balancer set TRUST_FORWARDED_FOR
# so the bucket keys on the client, and leave it off when many patients share
# one NAT address.
RATE_LIMIT_IP = _int("RATE_LIMIT_IP_REQUESTS", 0)
TRUST_FORWARDED_FOR = _bool("TRUST_FORWARDED_FOR", False)
RATE_LIMIT_FAIL_OPEN = _bool("RATE_LIMIT_FAIL_OPEN", True)

# =============================================================================
# Concurrency and timeouts
# =============================================================================

# What this process may use, not the host's core count; every pool below is
# sized from it so a 2-vCPU task does not run a 16-core default.
CPU_BUDGET = _int("CPU_BUDGET", _cpu_budget())
# One core is left to the event loop; the rest run guard, rerank and expansion.
EXECUTOR_WORKERS = _int("EXECUTOR_WORKERS", max(CPU_BUDGET - 1, 2))
EMBED_WORKERS = _int("EMBED_WORKERS", max(CPU_BUDGET // 4, 1))  # must not queue behind rerank/guard
# 0 leaves torch's default (all cores per inference), which gives the best
# single-request latency. Set it lower (e.g. 2 on a 4-core host) when many
# callers run guard/rerank inference at once, to reduce oversubscription.
TORCH_NUM_THREADS = _int("TORCH_NUM_THREADS", 0)
SEM_GUARD = _int("SEM_GUARD", max(CPU_BUDGET, 2))
SEM_RAG = _int("SEM_RAG", max(CPU_BUDGET * 2, 4))  # mostly waiting on Qdrant/Elasticsearch
SEM_RERANK = _int("SEM_RERANK", max(CPU_BUDGET // 2, 1))
SEM_LLM = _int("SEM_LLM", 16)
# A CPU task that waited this long in the executor queue is logged with the
# request line, which is what turns oversubscription into a visible signal.
CPU_QUEUE_WARN_MS = _int("CPU_QUEUE_WARN_MS", 50)
USER_CONCURRENCY = _int("USER_CONCURRENCY", 1)  # in-flight questions allowed per user
USER_WAIT_TIMEOUT = _float("USER_WAIT_TIMEOUT", 0)  # 0 = reject at once; >0 = wait this long
# The per-user limit is enforced across replicas through Redis; 0 disables the
# shared lock and leaves only in-process slots. Must exceed the longest answer
# (pre-LLM work plus LLM_TOTAL_TIMEOUT); it is renewed while the answer streams.
USER_LOCK_TTL = _float("USER_LOCK_TTL_SECONDS", 180)
QUEUE_TIMEOUT = _float("QUEUE_TIMEOUT", 5)  # max wait for a semaphore slot
REDIS_TIMEOUT = _float("REDIS_TIMEOUT", 0.5)
QDRANT_TIMEOUT = _float("QDRANT_TIMEOUT", 5)
MODEL_TIMEOUT = _float("MODEL_TIMEOUT", 10)
UVICORN_LIMIT_CONCURRENCY = _int("UVICORN_LIMIT_CONCURRENCY", 200)

# =============================================================================
# Provider Operational DB: analytics, appointments and schedule (Text-to-SQL)
# =============================================================================

# MySQL DSN for the provider operational database (Text-to-SQL). Opens a
# dedicated read-only pool. If empty, SQL falls back to the shared MYSQL_URL
# pool, or to a marked mock when no database is configured. The analytics API
# below is independent of this setting.
PROVIDER_DB_URL = os.getenv("PROVIDER_DB_URL", "")
# Dedicated analytics endpoint for provider metrics.
ANALYTICS_API_URL = os.getenv("ANALYTICS_API_URL", "")
# Auth token for the provider database API (if separate from JWT).
PROVIDER_DB_TOKEN = os.getenv("PROVIDER_DB_TOKEN", "")
PROVIDER_DB_TIMEOUT = _float("PROVIDER_DB_TIMEOUT", 10)
# Required JWT permission for a provider to query their analytics and schedule.
PROVIDER_SQL_PERMISSION = os.getenv("PROVIDER_SQL_PERMISSION", "providers:analytics")
# Enable the text-to-SQL Model Context Protocol (MCP) tool.
MCP_SQL_ENABLED = _bool("MCP_SQL_ENABLED", True)
# Maximum rows to return from a single natural language SQL query.
PROVIDER_SQL_MAX_ROWS = _int("PROVIDER_SQL_MAX_ROWS", 100)
# If True, audit SQL queries and results (PHI). AUDIT_LOG_CONTENT=false overrides this.
PROVIDER_SQL_AUDIT = _bool("PROVIDER_SQL_AUDIT", True)

# =============================================================================
# Elasticsearch: provider search (empty ES_URL disables the feature)
# =============================================================================

ES_URL = os.getenv("ES_URL", "")
ES_API_KEY = os.getenv("ES_API_KEY") or None
ES_USER = os.getenv("ES_USER", "")
ES_PASSWORD = os.getenv("ES_PASSWORD", "")
ES_CA_CERTS = os.getenv("ES_CA_CERTS") or None
ES_PROVIDERS_INDEX = _env("ES_PROVIDERS_INDEX", "providers")
ES_TIMEOUT = _float("ES_TIMEOUT", 3)
PROVIDER_RESULTS = _int("PROVIDER_RESULTS", 5)
PROVIDER_SEARCH_PERMISSION = os.getenv("PROVIDER_SEARCH_PERMISSION", "providers:search")  # "" = no check
# Write access is only used by the sync job; the read-only key stays in ES_API_KEY.
ES_SYNC_API_KEY = os.getenv("ES_SYNC_API_KEY") or None

# =============================================================================
# Query Understanding Engine (provider directory)
# =============================================================================

# spaCy model used for NER and phrase matching. Missing model/package falls
# back to the concept catalogue's matcher; access decisions never depend on it.
QUE_SPACY_MODEL = _env("QUE_SPACY_MODEL", "en_core_web_sm")
# Replaces the default request-word list when set (space separated).
PROVIDER_STOPWORDS = os.getenv("PROVIDER_STOPWORDS", "").strip()
# Strict pass requires every concept; the relaxed pass tolerates a stray word.
PROVIDER_STRICT_MSM = _env("PROVIDER_STRICT_MSM", "2<75%")
PROVIDER_RELAXED_MSM = _env("PROVIDER_RELAXED_MSM", "1")
PROVIDER_FUZZINESS = _env("PROVIDER_FUZZINESS", "AUTO")  # "" disables fuzzy matching
PROVIDER_FUZZY_PREFIX_LENGTH = _int("PROVIDER_FUZZY_PREFIX_LENGTH", 1)
PROVIDER_FUZZY_MAX_EXPANSIONS = _int("PROVIDER_FUZZY_MAX_EXPANSIONS", 50)
PROVIDER_BOOST_SPECIALTY = _float("PROVIDER_BOOST_SPECIALTY", 3.0)
PROVIDER_BOOST_NAME = _float("PROVIDER_BOOST_NAME", 2.0)
PROVIDER_BOOST_CONDITIONS = _float("PROVIDER_BOOST_CONDITIONS", 1.5)
PROVIDER_BOOST_ACCEPTING = _float("PROVIDER_BOOST_ACCEPTING", 1.5)
PROVIDER_RATING_MODIFIER = _env("PROVIDER_RATING_MODIFIER", "sqrt")
PROVIDER_RATING_MISSING = _float("PROVIDER_RATING_MISSING", 3.0)

# =============================================================================
# Provider sync: full pull from the backend API into Elasticsearch
# =============================================================================

PROVIDERS_API_URL = os.getenv("PROVIDERS_API_URL", "")
PROVIDERS_API_TOKEN = os.getenv("PROVIDERS_API_TOKEN", "")  # a service token, not a user's JWT
PROVIDERS_API_TENANT = os.getenv("PROVIDERS_API_TENANT", "")  # for single-tenant APIs without a tenant field
PROVIDERS_API_PAGE_SIZE = _int("PROVIDERS_API_PAGE_SIZE", 100)
PROVIDERS_API_ITEMS_KEY = _env("PROVIDERS_API_ITEMS_KEY", "data")  # key holding the list in the response
PROVIDERS_API_PARAMS = _json("PROVIDERS_API_PARAMS")  # extra query params, JSON
PROVIDERS_API_TIMEOUT = _float("PROVIDERS_API_TIMEOUT", 30)
PROVIDERS_FIELD_MAP = _json("PROVIDERS_FIELD_MAP")  # {"name": "full_name", "city": "location.city"}
PROVIDER_SYNC_MIN_RATIO = _float("PROVIDER_SYNC_MIN_RATIO", 0.5)  # refuse to remove more than this share at once
PROVIDERS_API_MAX_PAGES = 5000

# =============================================================================
# CORS (explicit origins only, never "*")
# =============================================================================

CORS_ORIGINS = _list("CORS_ORIGINS")
CORS_ALLOW_CREDENTIALS = _bool("CORS_ALLOW_CREDENTIALS", False)

# =============================================================================
# Health endpoint
# =============================================================================

# The public response carries only the verdict. Enable details on a private
# network or when the load balancer itself is the only caller.
HEALTH_DETAILS = _bool("HEALTH_DETAILS", False)

# =============================================================================
# Document chunking and context expansion
# =============================================================================

CHUNKER_VERSION = "2"  # bump to force a re-index after changing the chunker
CHUNK_TOKENS = _int("CHUNK_TOKENS", 300)  # target size: a chunk closes once it reaches this
CHUNK_MAX_TOKENS = _int("CHUNK_MAX_TOKENS", 380)  # never grow past this (reranker/embedder limit)
CHUNK_OVERLAP_TOKENS = _int("CHUNK_OVERLAP_TOKENS", 40)  # only when one block is too big
MIN_CHUNK_TOKENS = _int("MIN_CHUNK_TOKENS", 60)  # smaller trailing chunks merge into the previous one
WHOLE_DOC_TOKENS = _int("WHOLE_DOC_TOKENS", 350)  # a policy this short stays a single chunk
EXPAND_DOC_TOKENS = _int("EXPAND_DOC_TOKENS", 1000)  # up to this size the whole document goes to the LLM
EXPAND_DOCS = _int("EXPAND_DOCS", 2)  # how many top documents may be expanded
EXPAND_SCORE_MARGIN = _float("EXPAND_SCORE_MARGIN", 4.0)  # 2nd+ document must score within this of the best
MAX_CONTEXT_TOKENS = _int("MAX_CONTEXT_TOKENS", 1600)  # budget for expansion

# =============================================================================
# RBAC mapping (folder layout decides who can read a document)
# =============================================================================

GLOBAL_TENANT = "global"
ROLE_AUDIENCES = {"patient": ["patient", "shared"], "provider": ["provider", "shared"]}
FOLDER_AUDIENCE = {"patient": "patient", "provider": "provider", "shared": "shared"}
TENANT_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# =============================================================================
# Validation
# =============================================================================


def validate_cors() -> None:
    for origin in CORS_ORIGINS:
        if origin == "*" or not re.match(r"^https?://[^/\s*]+$", origin):
            raise RuntimeError(f"Invalid CORS origin {origin!r}: use explicit http(s)://host[:port] values")


def validate_runtime() -> None:
    """Requirements for the chat service (the sync job deliberately needs neither)."""
    if not JWT_KEY:
        raise RuntimeError("Set JWT_SECRET (HS*) or JWT_PUBLIC_KEY (RS*/ES*)")
    if not LLM_API_KEY:
        raise RuntimeError("Set LLM_API_KEY")
    if DEV_BYPASS_ROLE:
        if APP_ENV not in DEV_ENVIRONMENTS:
            raise RuntimeError(
                f"DEV_BYPASS_ROLE is set but APP_ENV={APP_ENV!r}: the auth bypass is allowed only in "
                f"{sorted(DEV_ENVIRONMENTS)}"
            )
        if DEV_BYPASS_ROLE not in ROLE_AUDIENCES:
            raise RuntimeError(f"DEV_BYPASS_ROLE must be one of {sorted(ROLE_AUDIENCES)}")
