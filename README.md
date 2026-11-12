# Platform FAQ Assistant

Multi-tenant FAQ and provider-directory service. It answers policy, onboarding
and navigation questions from ingested PDFs and lets patients search a provider
directory. Answers stream over SSE; nothing about the caller is taken from the
request body — role, tenant and permissions come only from a verified JWT.

```
POST /v1/chat[/stream]
        |
        v
  JWT -> Principal (role, tenant, permissions)
        |
        v
  greeting shortcut -> Redis rate limit -> MySQL conversation history
        -> spell correction -> LangGraph agent: router node (Laya System 1, LLM/rules fallback)
        |
        +--> providers: emergency/business rules
                                -> Elasticsearch query (synonym_graph analyzer)
        |
        +--> faq: emergency/business rules -> PII scan (sharing)
                                -> synonym expansion -> embed -> semantic cache hit? -> answer
                                -> Qdrant hybrid (dense + BM25 -> RRF)
                                -> rerank || injection guard -> expand -> LLM stream
                                (nothing retrieved: answer from history)
```

## What is in the box

- **Hybrid retrieval**: `BAAI/bge-small-en-v1.5` dense vectors plus `Qdrant/bm25`
  sparse vectors, fused with RRF, then reranked with a cross-encoder.
- **RBAC in the database query**: tenant and audience filters are applied to
  the dense leg, sparse leg, fusion and follow-up reads, not after retrieval.
- **Semantic cache** on Redis Stack (RediSearch vectors), keyed by what the
  caller is allowed to read. Cached answers are never shared across roles,
  tenants or (optionally) users.
- **Streaming**: token/done/error SSE events; the non-streaming endpoint
  consumes the same pipeline. The bundled page shows a search placeholder until
  the first token arrives, so the retrieval and guard wait is visible.
- **Guardrails** run in four stages on every request:
  1. **Normalize input** — NFKC, zero-width characters removed, whitespace
     collapsed.
  2. **Emergency and business rules** — deterministic patterns for emergency,
     injection, identity, privacy, restricted and medical requests. Emergency
     symptoms (chest pain, can't breathe, a racing or "going down" heart,
     droopy face or slurred speech, throat or tongue swelling after a
     medication, self-harm, …) get immediate 911/988 guidance as the highest
     priority; injection and security violations get the fixed
     `Blocked: security violation` refusal. Asking the assistant to read a
     person's details out of the database is a privacy request, as are a named
     doctor's NPI, contact details or appointments, moving every appointment
     onto the caller's profile, a family member's or dependent's records ("my
     adult daughter… show me her lab reports") and a named patient's records.
     Asking how to find a doctor, view a doctor's profile or book with a named
     doctor stays a normal question. Ending the session or "start as Patient B
     without login" is refused with the sign-in route. A stated diagnosis or
     organ damage ("have liver damage") gets the medical-guidance reply.
  3. **LLM Guard injection detection** — ProtectAI's
     `deberta-v3-base-prompt-injection-v2` runs through transformers and
     screens text that would reach the LLM; retrieved chunks are screened
     again. The classifier is skipped for questions about the conversation
     itself, first-person requests about the caller's own account ("show me my
     appointments", "how do I delete a report") and messages that state
     personal details ("my name is…", "my email is…"), which it reads as
     instructions; the rules above have already screened prompts, secrets,
     third-party records and account switches, and a message carrying a known
     injection phrase still reaches the classifier. A request that names the
     assistant's own rules or prompt also reaches it. The provider directory
     has no LLM prompt, so it skips this stage.
  4. **Presidio PII detection** — emails, phone numbers, government and
     medical identifiers, bank/card numbers and self-introductions are
     detected after the free rules clear the prompt. PII never causes a
     refusal: it keeps the turn out of the shared semantic cache and out of
     MySQL conversation history. If Presidio is unavailable the previous
     regex detector is used; if analysis fails, the text is treated as
     personal (fail closed for sharing and storage).
  Privacy and RBAC filters are unchanged by any of these paths.
- **Multi-turn context**: the last turns of a conversation are kept in MySQL,
  keyed by tenant, user and session, and replayed to the LLM so follow-ups
  ("and for the app?") work and stated facts are remembered ("I live in New
  York" → "Where do I live?"). A correction ("actually, I live in Boston",
  "I've moved to…") is a statement in its own right, and the newest one wins.
  Retrieval rewrites a follow-up into a standalone query with the LLM
  (`CONDENSE_ENABLED`), resolving "it", "that" or "the video one" from the
  recent turns; a reply directive ("and just say done") is left out of the
  search text. The heuristic below is the fallback when the rewrite is disabled
  or unavailable: the previous question is borrowed when the new one cannot
  carry its own topic, and a long follow-up that names nothing ("How do I make
  sure it is for her rather than me?") retries with the last question that did.
- **Provider search**: Elasticsearch with a search-time `synonym_graph`
  analyzer, strict-then-relaxed matching and tenant isolation. The app sends
  the request words; the index rewrites "heart doctor" to cardiology before
  matching, so a multi-word phrase stays one concept. No LLM is involved, so
  answers stream line by line and skip the classifier. It is a patient
  feature: directory intent is routed the same way for every role, but the
  pipeline refuses any non-patient with "Provider search isn't available for
  provider accounts." (a patient missing the permission gets "… isn't
  available for your account."), and that role gate runs before the
  Elasticsearch check so a refusal never looks like an outage.
- **Query understanding**: conservative SymSpell correction (domain-aware,
  names and medical terms left alone) and a hot-reloaded synonym catalogue for
  routing and Qdrant retrieval. Provider search receives the same specialty
  groups as an Elasticsearch `synonym_graph`, installed by the sync job, so
  `pediatrician`, `pediatrics doctor` and `child specialist` search the same
  specialty in both corpora.
- **Read/write separation**: the web process only reads Qdrant (RAG) and
  Elasticsearch (providers). PDF ingestion and provider sync are separate
  one-shot jobs; startup never builds an index.
- **Provider sync**: idempotent API-to-Elasticsearch sync with stale removal
  guarded by a safety ratio, run weekly by an Airflow `DockerOperator`.
- **Document indexing**: the Qdrant index job also runs weekly (and on demand)
  through Airflow, and publishes a corpus fingerprint so the cache invalidates.
- **Operations**: request IDs, structured request logs without input/PHI,
  redacted logs and fixed user-facing error envelopes, health checks,
  fixed-window rate limits, semaphores for every heavy stage.

## Repository layout

```
.
├── src/chat_search/
│   ├── api/            FastAPI app factory, schemas, HTML chat page
│   ├── indexing/       PDF extraction, chunking, Qdrant indexing (batch job)
│   ├── pipeline/       LangGraph agent, chat orchestration, events, FAQ and provider pipelines
│   ├── providers/      provider search read path: search, fields, ES client
│   │   └── indexing/   ES schema, normalization, sync, dev seed (batch jobs)
│   ├── auth.py         JWT verification and Principal
│   ├── bootstrap.py    startup/shutdown of models, stores and runtime (reads only)
│   ├── cache.py        semantic cache
│   ├── cli.py          serve / token / index-documents / seed-providers / sync-providers / sync-provider-synonyms
│   ├── concepts.py     structured concept catalogue (accepts the legacy shape)
│   ├── config.py       every environment variable, read once
│   ├── corpus.py       corpus fingerprint shared with the index job
│   ├── guardrails.py   normalize, rules, injection model, PII detection
│   ├── llm.py          OpenAI-compatible streaming client
│   ├── mcp_server.py   MCP tools: knowledge search, provider search, RAG answer
│   ├── rate_limit.py   Redis fixed-window limits
│   ├── retrieval.py    hybrid search, rerank, context expansion (Qdrant reads)
│   ├── routing.py      FAQ vs provider intent (shares understanding.py rules)
│   ├── runtime.py      process-wide resources, semaphores, user slots
│   ├── spelling.py     conservative, domain-aware query spelling correction
│   ├── state.py        per-tenant conversation state (multi-turn context, MySQL)
│   ├── synonyms.py     backward-compatible facade over concepts.py
│   ├── text.py         token estimate and embedding text shared by both sides
│   └── understanding.py normalize, intent, appointment rules, canonical mapping
├── dags/               Airflow DAGs: provider sync + document index
├── concepts.json       structured catalogue: specialties, topics, conditions (editable)
├── data/               PDFs, read only by the index job
├── tests/              unit tests (no external services required)
├── main.py             compatibility launcher: python main.py ... still works
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

Dependency direction is one-way: `config` → `runtime` → domain modules →
pipelines → `api`/`cli`. The serving path never imports `indexing` or
`providers.indexing`; `tests/test_separation.py` enforces that in a subprocess.
Heavy ML imports are lazy, so linting, tests and the provider sync do not need
torch.

## Quick start

```bash
cp .env.example .env
# edit .env: JWT_SECRET, LLM_API_KEY, MYSQL_PASSWORD, ELASTIC_PASSWORD, ES_* credentials
docker compose up --build
```

If the build dies at `pip install ".[all]"` with `Expected sha256 ... Got
e3b0c442...` (the digest of an empty file), the PyPI CDN is throttling the
download and pip cannot resume it. Point the build at a mirror; nothing else
changes:

```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple docker compose build
```

Dependencies are installed in their own layer keyed on `pyproject.toml`, so
later `src/` changes rebuild in seconds instead of re-downloading the ML stack.

Open <http://localhost:8000>. The first start downloads (or loads baked) models;
the healthcheck has a 180 s start period for that reason. The API itself never
indexes documents, so build the Qdrant collection once and refresh it whenever
`data/` changes:

```bash
docker compose --profile indexing run --rm indexer
```

Provider search needs a populated Elasticsearch index; run `sync-providers`
(see below) before serving patients. The same job installs the provider synonym
rules; `sync-provider-synonyms` applies a catalogue change without a full API
pull.

The local catalogue (`concepts.json`, mounted into the app container; the
legacy `synonyms.json` shape is still accepted for other deployments) drives
routing and Qdrant retrieval and is picked up within `CONCEPTS_RELOAD_SECONDS`. Provider rules are
compiled into the index's search analyzer, so after editing specialty wording
run `sync-provider-synonyms` (or wait for the weekly sync). Qdrant documents do
not need to be re-indexed for either change.

For a local chat without a Laravel token, set `DEV_BYPASS_ROLE=patient` in
`.env`. **Never set it in production** — startup logs a warning, but the bypass
is total.

Create an Elasticsearch API key for the app and a write key for the sync job,
then set `ES_API_KEY` and `ES_SYNC_API_KEY`. The sync job creates the
`providers` index when it is missing.

Mint a development JWT (HS* secrets only):

```bash
docker compose exec app python -m chat_search token patient default
# legacy form still works: python main.py token patient default
```

Seed sample providers (development only):

```bash
docker compose exec app python -m chat_search seed-providers default
```

## Local development

Python 3.11+ is required.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,stores,ml,pdf]"
chat-search serve --port 8000
```

Extras are layered so CI and the sync job stay small:

| Extra | Contents |
| --- | --- |
| *(base)* | FastAPI, uvicorn, httpx, PyJWT, pydantic, numpy, rapidfuzz, symspellpy |
| `ml` | sentence-transformers, fastembed, transformers, sentencepiece, protobuf |
| `pdf` | pymupdf4llm |
| `stores` | qdrant-client, redis, elasticsearch, aiomysql |
| `airflow` | Airflow and the Docker provider (install only in the Airflow image) |
| `dev` | pytest, pytest-asyncio, pytest-cov, ruff, mypy |
| `all` | `ml` + `pdf` + `stores` — what the service image installs |

The root `main.py` inserts `src/` on `sys.path`, so existing commands such as
`python main.py token patient` keep working from a checkout. New code should
use `python -m chat_search ...` or the `chat-search` script.

## HTTP API

All chat endpoints require `Authorization: Bearer <JWT>`. The JWT must carry
`exp`, `sub`, `role` (`patient` or `provider`), `tenant_id` (or `org_id`),
`permissions` containing `docs:read` (configurable), and `patient_id` /
`provider_id` matching the role. Laravel puts the identifier into the token at
sign-in; the service reads it only from those verified claims, never from the
request body or query, and rejects a token whose `patient_id` or `provider_id`
is missing with 403 rather than logging an unverified value.

### `POST /v1/chat/stream`

Streams `text/event-stream`:

```
event: token
data: {"type": "token", "text": "You can reset your password "}

event: done
data: {"type": "done", "sources": [{"document": "account-policy", "page": 3}], "cached": false}

event: error
data: {"type": "error", "code": "busy", "message": "...", "status": 503}
```

`mode` is `auto` (default), `faq` or `providers`. Explicit `faq` never falls
through to provider search; only patients are auto-routed to providers.

`session_id` (optional, `[A-Za-z0-9._-]{1,64}`) selects a conversation; omit it
for the caller's rolling one. See [Conversation state](#conversation-state).

### `POST /v1/chat`

Same pipeline without streaming:

```json
{
  "answer": "You can reset your password from Account > Security.",
  "sources": [{"document": "account-policy", "page": 3}],
  "cached": false,
  "data": null
}
```

Provider searches also return `data.providers` with the structured hits.

### `DELETE /v1/chat/session`

Forgets the caller's conversation, so the next question starts from scratch.
Takes an optional `?session_id=…`; without it the rolling conversation is
cleared. Returns `{"status": "ok"}`.

### `GET /health`

Returns `{"status": "ok"}` with HTTP 200 when Qdrant and Redis respond, and
`{"status": "degraded"}` with HTTP 503 when either is down. Elasticsearch is
optional: a failed probe degrades provider search but not the service verdict.
Each dependency check is capped at one second, and the probes run concurrently,
so the endpoint takes about one slow probe rather than their sum.

The public response carries no dependency inventory. Set `HEALTH_DETAILS=true`
only when the caller is the load balancer or another host on a private network;
the response then adds `qdrant`, `redis`, `cache` and `elasticsearch` booleans
(one down dependency only flips its own flag).

### `GET /`

A self-contained chat page. It keeps the token in `sessionStorage`/the URL
fragment and renders SSE tokens as they arrive. Its "New chat" button calls
`DELETE /v1/chat/session`.

## LangGraph agent

Every message runs through a small LangGraph state graph
([pipeline/agent.py](file:///C:/Users/user/Desktop/Chat_Search/src/chat_search/pipeline/agent.py)):

```
START -> router (Laya) --faq-------> rag    (document RAG)
                       --providers-> search (Elasticsearch directory)
rag / search -> END
```

- **`router`** is the System 1 decision node. By default it loads
  [Laya](https://huggingface.co/convaiinnovations/laya), a small
  non-autoregressive checkpoint that reads the message plus the recent turns and
  picks one label, `faq` or `providers`, in a single forward pass. Routing then
  costs no completion tokens, which matters on a rate-limited LLM. The decision
  is written as a `meta` event (`routed_by`: `laya`, `llm`, `mode`, `history` or
  `rules`).
- The checkpoint is loaded in the background at startup; until it is ready, and
  if it fails to load, the LLM router answers. Once it is loaded, a slow or
  unexpected Laya result falls back to the deterministic rules rather than
  spending a second model call.
- The rules are also a precision guard on both sides of the model: a directory
  phrase it misses ("who speaks Urdu?") stays providers, and an explicit
  how-to question that merely names a doctor ("how do I change my primary care
  doctor?") stays documents.
- The **conditional edge** dispatches to the matching node. Explicit
  `"mode": "faq" | "providers"` skips the model, an appointment/booking
  question always goes to `faq` because its steps live in the documents, and a
  correction ("actually it is for my son", "finish the search") follows the
  route of the turn it refers to.
- **`rag`** and **`search`** wrap the existing pipelines; their events are
  forwarded through LangGraph's custom stream, so tokens still reach the client
  as they are produced. The RAG node also owns the in-flight answer coalescing.

`AGENT_ROUTER_ENABLED=false` removes the model round trip and keeps rule-based
routing only; `AGENT_ROUTER_BACKEND=llm` restores the label-only prompt and
`rules` keeps every decision deterministic.

## MCP server

The corpus and the directory are also exposed as Model Context Protocol tools,
so an agent can retrieve without a browser session:

| Tool | Returns |
| --- | --- |
| `search_knowledge(query, top_k=5)` | Ranked corpus passages with document, page, section and score |
| `search_providers(query, languages=None, limit=5)` | Public directory hits, plus `relaxed` and `language_fallback` |
| `ask(question)` | The RAG answer with its sources and the structured understanding |

The streamable HTTP transport is mounted on the API port at `/mcp/` and reuses
the warm runtime, so retrieval, rerank and models are shared with the chat API.
It requires the same bearer token as `/v1/chat` unless `DEV_BYPASS_ROLE` is set.
Each request is served statelessly, so the token is verified per call and the
caller's identity never carries over to another client's call:

```json
{
  "mcpServers": {
    "chat-search": {
      "url": "http://10.50.50.237:8000/mcp/",
      "headers": {"Authorization": "Bearer <laravel-token>"}
    }
  }
}
```

For a local client, `chat-search-mcp` serves the same tools over stdio inside the
container (`docker exec -i chat_search-app-1 chat-search-mcp`); it authenticates
with `MCP_ROLE`/`MCP_TENANT_ID` instead of a token. Set `MCP_ENABLED=false` to
drop the mount, and list `MCP_ALLOWED_HOSTS=host:port,...` to turn on the
transport's DNS-rebinding host check (off by default because the directory is
reached by IP and every request carries a JWT).

## Conversation state

Follow-ups need the previous turns, so the last `HISTORY_TURNS` messages of a
conversation are kept in MySQL and replayed to the LLM as chat messages ahead
of the current question; retrieved documents stay in the final user message.
MySQL is the durable system of record for conversations; Redis keeps only the
cache and rate-limit counters. Without `MYSQL_URL` (or when MySQL is
unreachable) the service answers one turn at a time and never fails.

- **Key**: the tuple `(tenant_id, sha256(subject)[:16], session_id)` in the
  `conversation_turn` table. Tenant and user are both part of it, so no request
  can reach another tenant's or another user's conversation. `session_id` only
  splits one user's parallel threads and defaults to `default`.
- **Context answers**: when retrieval finds nothing and the question is a
  follow-up ("where do i live") or a statement about the caller ("i'm lucas",
  "lucas here"), the LLM answers from the earlier turns with an empty
  `<documents>` block. Such replies carry no sources and are never cached; if no
  earlier turn covers the question the normal "couldn't find that" reply is used.
  An origin or identity statement ("i'm from new york", "my name is …",
  "lucas here") stays context-only even when retrieval returns a weak match,
  because a policy page cannot answer it.
- **Corrections**: an inline correction ("actually, i live in boston",
  "sorry, i mean…", "i've moved to…", "i now live in…") is detected as a
  personal statement, so it overrides a weak document match. The prompt tells
  the model that the newest statement replaces the earlier one, so the
  follow-up "where do i live" answers Boston, not New York.
- **Conversation questions**: "summarize what I said", "what did I ask",
  "do you remember…" are answered from the turns even when retrieval returns
  documents, because a PDF cannot summarize the caller. They also skip the
  prompt-injection model, which reads a summarization request as an instruction;
  the deterministic injection patterns still block first. With no history the
  answer says there is nothing earlier to summarize.
- **Fallbacks**: a reply with nothing found names the topic that was asked about
  — billing/refunds, booking/changing/cancelling (including "move my visit"), or
  uploads/file size — so the next step (usually platform support) is obvious; an
  unknown topic keeps the generic reply. Fallbacks never become conversation
  context.
- **Answer grounding**: the prompt keeps per-flow limits apart (booking
  attachments up to 10 MB each; a lab report uploaded from Settings up to 5 MB),
  forbids promising a refund, reversal or waiver after a cancellation or
  rescheduling, and forbids facts the documents do not support. Billing and
  refund questions point to platform support (providers: Support → Billing). It
  also cannot act on an account: asked to move a visit, it explains the steps
  and never claims the change was made or answers "done".
- **Tool failures**: if Elasticsearch raises during provider search the caller
  sees the normal "temporarily unavailable" reply, not a generic error.
- **Retrieval**: a short question, or one starting with a continuation ("and",
  "what about", "how about", …), is searched together with the previous
  question, so "and for the app?" still finds the password-reset policy. A
  short question that introduces its own topic ("how do I reset my password?"
  after a booking question, "what is the cancellation policy?") is searched
  exactly as asked, so a new subject never inherits the previous one. A
  question that names no topic and only points back ("is a 7MB PDF within the
  documented limit here?") borrows the nearest earlier turn about a product
  flow — booking, attaching, uploads — rather than an older, unrelated topic.
  A longer, self-contained question is searched exactly as asked.
- **Cache**: a multi-turn answer is only looked up when the whole history
  matches, because the same words mean different things after different turns,
  and it is never written back: a reply that repeats what the caller said stays
  in their session. The first question of a conversation keeps the plain scope,
  so the common FAQ path retains its hit rate.
- **Bounds**: a turn is clipped to `HISTORY_MAX_CHARS` (and 2000 characters in
  the column), only the newest `HISTORY_TURNS` rows of a conversation survive
  (older rows are deleted on write), and a once-a-minute sweep deletes rows
  older than `HISTORY_TTL_SECONDS`, so abandoned conversations disappear.
- **Cost**: one indexed MySQL round trip per question, plus at most
  `HISTORY_TURNS × HISTORY_MAX_CHARS` characters of extra prompt.
- **Failure**: history is an optimisation. When MySQL is unavailable the
  question is answered as a single turn and the request log carries
  `history_error` / `history_write_error`; only the explicit reset fails loudly
  (503), because a reset the caller believes happened must have happened.
- **Ordering**: the history read happens after the per-user slot is claimed, not
  before. Reading first would let a second question on another replica read the
  old turns while the first answer was still being written, and then append
  after it, duplicating or reordering turns. The slot serializes the whole
  read → answer → append sequence across replicas.
- Greetings, pre-retrieval refusals and empty answers are not stored, and a
  turn is answered but never persisted when either the question or the answer
  contains personal data (email, phone, date of birth — the pattern that also
  keeps it out of the cache). Checking both sides matters because the model can
  echo identifying data the question did not contain.
  Provider directory answers are stored (so a later FAQ follow-up sees the
  conversation) but the provider path itself never reads history: it has no
  prompt to contextualise.
- The table is created at startup (`CREATE TABLE IF NOT EXISTS`); no migration
  step is needed. The schema lives in `state.py`.

## Serving and indexing

The web process is read-only:

- **RAG** reads Qdrant through `retrieval.py`.
- **Provider search** reads Elasticsearch through `providers/search.py` with the
  read-only `ES_API_KEY`.

Everything that writes is a separate process:

| Job | Command | Writes |
| --- | --- | --- |
| Document index | `python -m chat_search index-documents` | Qdrant collection + corpus fingerprint in Redis |
| Provider sync | `python -m chat_search sync-providers` | Elasticsearch index + provider documents + synonym analysis |
| Provider synonyms | `python -m chat_search sync-provider-synonyms` | Elasticsearch index analysis settings |
| Dev seed | `python -m chat_search seed-providers [tenant]` | Sample provider documents |

Startup opens read clients only. If a collection or index is missing, it logs a
warning and serves; RAG returns no context and provider search reports its
"unavailable" message until the corresponding job runs. Index jobs use a
dedicated model loader (`load_index_models`) that skips the reranker and
guardrail models, and a separate Elasticsearch write client
(`make_es_client(write=True)` with `ES_SYNC_API_KEY`).

The index job publishes a corpus fingerprint to Redis (`semcache:corpus`) after
a successful run. Serving uses it in the semantic-cache key, so cached answers
are not reused after the corpus changes. When the key is missing (Redis down or
the job never ran), serving falls back to a settings-only fingerprint.

### Airflow schedules

Airflow runs from `Dockerfile.airflow` (`faq-airflow:2.7.2`), which preinstalls
`apache-airflow-providers-docker` against the constraints file for this Airflow
version. Nothing is installed at container start, and the service runs as the
image's `airflow` user (uid 50000, gid 0), which can open `docker.sock` without
root. If the `airflow_data` volume was created by an older root-run container,
fix its ownership once:

```bash
docker run --rm -v chat_search_airflow_data:/opt/airflow \
  --entrypoint chown alpine:3 -R 50000:0 /opt/airflow
```

Both write jobs are scheduled weekly by Airflow and can be triggered manually:

| DAG | Job | Schedule | Manual trigger |
| --- | --- | --- | --- |
| `document_index_weekly` | `index-documents` (Qdrant) | Sundays 03:00 | `airflow dags trigger document_index_weekly` |
| `provider_sync_weekly` | `sync-providers` (Elasticsearch) | Sundays 02:00 | `airflow dags trigger provider_sync_weekly` |

Scheduler, webserver and triggerer output streams from the `airflow` service.
Task output is captured by Airflow into files under `/opt/airflow/logs`, so the
`airflow-logs` service follows new task logs and prints them; run
`docker compose logs -f` to watch DAG runs (including `DockerOperator`
container output) as they happen.

Each DAG runs the service image one-shot through `DockerOperator` with
`max_active_runs=1`, so two runs never overlap. Both jobs are idempotent:
document chunks are upserted by deterministic point id, provider documents by
`{tenant}-{provider_id}`.

The document DAG needs `qdrant_url`, `qdrant_api_key` (optional), `redis_url`,
`qdrant_collection`, the model names and the chunk sizes as Airflow Variables;
those values must match the serving process, or the bot queries vectors no run
produced. `DOCUMENTS_PATH` is an environment variable on the Airflow
scheduler/worker (compose fills it from `.env`) and must be an **absolute host
path**, because DockerOperator asks the host Docker daemon to mount it at
`/data` before task templating. The provider DAG needs `es_url`,
`es_sync_api_key`, `providers_api_url` and `providers_api_token`, plus
`es_providers_index` and the `providers_*`/`provider_sync_min_ratio` Variables
for the source API; the index name and field map must match serving. Variable
names containing `token` or `api_key` are masked in the UI and logs. The local
compose stack maps `DOCUMENTS_PATH`, `INDEX_IMAGE` and everything above into
Airflow Variables from `.env`.

## Query understanding

Every message goes through two cheap, local steps before routing and retrieval:

1. **Spelling correction** (`spelling.py`). SymSpell with the packaged English
   frequency dictionary plus domain terms from the synonym catalogue. It is
   deliberately conservative:
   - only unknown words are changed (a word with distance 0 is untouched);
   - one edit for words under eight characters, at most two for longer words;
   - a long word that is missing from the English dictionary may take a third
     edit when it matches a clinical term (`best paditricin` ->
     `best pediatrician`); this fallback never rewrites dictionary words;
   - rare candidates are rejected by a frequency floor;
   - ALL-CAPS tokens, mixed-case tokens, possessives and numbers are skipped;
   - capitalised words are only corrected at the start of the query, so names
     such as `Aetna` in the middle are left alone;
   - words in the medical/protected vocabulary are never changed;
   - at most `SPELL_MAX_CORRECTIONS` words change per query.

   If `symspellpy` or its dictionary is unavailable, correction becomes a no-op
   after one warning; the query is never blocked.

2. **Query understanding** (`understanding.py`) over a **structured concept
   catalogue** (`concepts.json`, loaded by `concepts.py`). Instead of a flat
   synonym map, every entry says what kind of thing it is:

   ```json
   {
     "concepts": [
       {"id": "cardiology", "canonical": "cardiology", "category": "specialty",
        "terms": ["cardiologist", "heart doctor"], "cui": "C0007222"},
       {"id": "hypertension", "canonical": "hypertension", "category": "condition",
        "terms": ["high blood pressure"], "cui": "C0020538"}
     ],
     "provider_nouns": ["doctor", "physician"],
     "appointment_actions": ["book", "cancel", "reschedule"],
     "appointment_objects": ["appointment", "visit", "consultation"],
     "rewrites": {"face to face": "in person"}
   }
   ```

   The pipeline is layered, and a failure in a heavier stage never blocks an
   answer:

   1. **Normalization** — `normalize_query()` lowercases, removes punctuation
      and collapses whitespace, then applies the conservative spell corrector.
   2. **Intent detection** — a spaCy `PhraseMatcher` over the catalogue finds
      specialties/FAQ topics/clinical vocabulary; explicit rules combine an
      appointment action with an appointment object (`book a video
      consultation`, `move my visit`). A named specialty alone is a provider
      search, never an appointment.
   3. **Medical terminology** — scispaCy (`en_core_sci_sm`) adds clinical
      entities that are not in the catalogue. When a local UMLS database is
      configured (`UMLS_ENABLED`, `UMLS_DATA_PATH`), entities also carry a CUI.
      Without scispaCy/UMLS the catalogue concepts are still returned.
   4. **Canonical mapping** — `heart doctor` → `cardiology`, `high blood
      pressure` → `hypertension`. The canonical term is prepended only to the
      text used for local retrieval; the caller's wording is what the LLM sees.

   The result is a structured object attached to the final SSE/JSON `done`
   event, so a caller never has to parse the answer:

   ```json
   {
     "understanding": {
       "original": "I need a heart doctor",
       "normalized": "i need a heart doctor",
       "intent": "provider_search",
       "specialties": ["cardiology"],
       "faq_topics": [],
       "medical_concepts": [
         {"text": "heart doctor", "canonical": "cardiology",
          "category": "specialty", "cui": "C0007222", "source": "catalogue"}
       ],
       "provider_nouns": ["doctor"],
       "appointment": null,
       "languages": [],
       "canonical_terms": ["cardiology"]
     }
   }
   ```

   For routing and Qdrant retrieval, a matching variant has its canonical term
   **prepended** to the search text. Plurals are matched automatically
   (`child specialists`, `heart doctors`, `cardiologists`). For example, `I need
   a child specialist in Boston` becomes `pediatrics I need a child specialist
   in Boston` for retrieval. This is why `pediatrician`, `pediatrics doctor`,
   `child specialist` and `kids doctor` search the same FAQ corpus without
   changing the index.

   Provider search does **not** use that local expansion. `sync-providers`
   compiles the `specialty` concepts into the index's search-time
   `synonym_graph` (`heart doctor => cardiology`, `gynecologist => gynecology,
   ob gyn`, …). Elasticsearch collapses the phrase before matching, so the
   strict pass's `2<75%` rule still means "both concepts"; the app only strips
   request words and language cues. The analyzer removes apostrophes, so
   `children's doctor` and `childrens doctor` hit the same rule.
   `SPECIALTY_LABELS` in `providers/fields.py` supplies the directory wording
   (`OB-GYN`, `eye doctor`) used in those rules.

   The same catalogue carries FAQ vocabulary where callers and documents use
   different words: `file attachment` (`attachment`, `attach a file`, …) adds
   the documents' word `file`, and `appointment booking` (`appointment`,
   `reschedule`, …) adds `booking`. That is what lets "maximum attachment size
   for appointments" reach the booking upload limit instead of falling back.
   `appointment booking reschedule` (`move my visit`, `change my appointment`, …)
   puts both words the FAQ uses into a reschedule question, and
   `book for someone else` (`family member`, `rather than me`, …) reaches the
   proxy-booking FAQ. These are `faq_topic` concepts: they participate in
   retrieval but never become provider synonyms. A canonical name is only
   retrieval wording; it is never shown to the caller.

   The catalogue is re-read at most once every `CONCEPTS_RELOAD_SECONDS`
   (default 60 s), keyed by the file's modification time. Invalid JSON keeps
   the last good catalogue and logs a warning. `CONCEPTS_JSON` can supply
   inline JSON instead of a file, and `CONCEPTS_ENABLED=false` /
   `UNDERSTANDING_ENABLED=false` / `SCISPACY_ENABLED=false` /
   `SPELLCHECK_ENABLED=false` disable individual stages. The loader also
   accepts the legacy catalogue shape (`specialties`, `protected_terms`,
   `rewrites`), so a `SYNONYMS_FILE` or `SYNONYMS_JSON` from an older
   deployment still works and overrides the source. Provider analysis settings follow the same flags: the next sync
   installs an analyzer without the synonym graph when the catalogue is
   disabled.

## Document ingestion

Place PDFs under `data/` and run the index job. It indexes new or changed files
and removes documents whose source file disappeared.

| Location | Tenant | Audience |
| --- | --- | --- |
| `data/shared/**` | `global` | patient + provider |
| `data/patient/**` | `global` | patient |
| `data/provider/**` | `global` | provider |
| `data/<tenant>/<audience>/**` | `<tenant>` | audience folder |

A caller with tenant `acme` can read `acme` documents plus `global` documents.
Anything outside these folders is skipped with a warning.

The file name chooses the chunker:

| Suffix | Strategy |
| --- | --- |
| `*-faqs.pdf`, `*-faq.pdf` | one chunk per question + answer (`Q:`/`A:`, numbered, bold or standalone `?`) |
| `*-manual.pdf` | sections by heading path; each table is its own chunk, header repeated if split |
| anything else (`*-policy.pdf`, ...) | sections; a document under `WHOLE_DOC_TOKENS` becomes one chunk |

Pages are joined before chunking so sections crossing page breaks stay intact.
Page numbers, repeated headers/footers, figure captions and image placeholders
are removed. Chunks store a page range, heading path, kind and reading order.

Each chunk is keyed by a hash of the file **and** the chunker/model settings.
Changing `CHUNKER_VERSION` (or the chunk sizes, embedding model or sparse model)
forces a re-index on the next `index-documents` run. The Qdrant collection uses
named dense and sparse vectors (`faq_documents_v2`); the old single-vector
collection is not compatible.

## Provider search and sync

`ES_URL` empty disables the whole provider feature (route falls back to FAQ for
other questions and returns a 502-style event for explicit provider searches).

**Tenant isolation.** Search filters on `active=true` and
`tenant_id=<JWT tenant>`. The sync writes `tenant_id` from each source record,
or from `PROVIDERS_API_TENANT` when the API is single-tenant. Set
`PROVIDERS_API_TENANT` to the same tenant id your JWTs carry (for example
`default` with `DEV_BYPASS_ROLE`); documents with neither are never returned.
Existing documents created before this rule have no `tenant_id`; run the sync
once after deploying so they become visible.

**Sync safety.** The job pages through `PROVIDERS_API_URL` (Laravel paginator,
nested lists and flat lists are supported; cross-host next links are refused),
normalizes only public directory fields, and bulk-upserts by provider
id with a `sync_run` marker. Stale documents are removed only when the pull
completed, indexing produced no bulk errors, at least one provider was indexed,
and the run kept `PROVIDER_SYNC_MIN_RATIO` of the existing corpus. Otherwise it
logs the reason and exits non-zero; Airflow retries and alerts. The job needs
`ES_URL`, `ES_SYNC_API_KEY`, `PROVIDERS_API_URL` and `PROVIDERS_API_TOKEN` — no
JWT secret, LLM key, Qdrant or Redis. The job also installs the search-time
analyzer: an existing index is closed briefly while its analysis settings and
mapping are upgraded, so the synonym rules are current after every run.

**Synonyms.** `provider_text` (index-time) only normalizes and stems;
`provider_search` (search-time) adds a `synonym_graph` filter built from the
catalogue's `specialty` concepts. The rewrite therefore never inflates the index, and a
multi-word phrase collapses to one concept: `heart doctor boston` is searched
as `cardiology boston`, not as four independent words. The strict pass uses
`minimum_should_match: 2<75%` — two concepts must both match, while longer
queries keep the original 75% tolerance for a stray word. The analyzer strips
apostrophes, so `children's doctor` matches the catalogue's `childrens doctor`
rule. Rules are one-way (`variant => canonical, directory label`) so the
canonical and the stored label both remain reachable.

**Directory labels.** Some specialties are stored under an abbreviation or a
colloquial name (`OB-GYN`, `Eye Doctor`). `SPECIALTY_LABELS` in
`providers/fields.py` adds the stored wording to the analyzer rule, so
`best gynecologist`, `best ob-gyn`, `best eye specialist` and `best eye doctor`
reach the same rows. Add an entry there when a new label diverges from its
catalogue canonical, then run `sync-provider-synonyms` (or the weekly sync).

**Ranking requests.** Rating words ("best ratings", "highest rated") select the
directory and order results by the rating field; they are never search terms,
so `doctors with best ratings` lists the top-rated providers instead of
searching for a word the index does not hold.

**Language requests.** A language in the question is a requirement, not one
more search word. `provider_languages` reads mapped labels ("Urdu", "Farsi" →
Persian, "Haitian Creole"), any language named after "speaks"/"fluent in", and
"whose primary language is X" (the cue words are dropped from the terms).
`provider_terms` drops the language from the text terms, and `provider_query`
adds it to the bool `filter`, so both the exact and the relaxed pass return only
providers who speak it (none rather than a non-speaker). A language-only
question ("who speaks Urdu") or a language attached to a provider noun
("spanish doctors", "English-speaking doctors") routes to the directory. When a
language query also names a specialty or city that nobody matches, the search is
retried with the language alone and the answer says so: `spanish doctor in
newyork` lists the directory's Spanish speakers (there are none in New York)
instead of a bare "no match", and `es_language_fallback=true` is logged. In the
gerund form the language comes before the cue ("urdu speaking dermatologist"),
so the word after "speaking" is the provider; a specialty from the synonym
catalogue is never read as a language. Add a
label to `LANGUAGE_LABELS` when the directory spells a language differently from
its common name.

**Airflow.** `dags/provider_sync_dag.py` runs the same image one-shot:
`python -m chat_search sync-providers`, weekly, with `max_active_runs=1` and
retries. Trigger it manually with
`airflow dags trigger provider_sync_weekly`. Required Airflow Variables are
`es_url`, `es_sync_api_key`, `providers_api_url` and `providers_api_token`;
`es_providers_index`, `providers_api_tenant`, `providers_api_page_size`,
`providers_api_items_key`, `providers_api_params`, `providers_field_map`,
`providers_api_timeout` and `provider_sync_min_ratio` are optional. Names
containing `token` or `api_key` are masked, and `index_image` is optional. Keep
the index name and field map in Variables, not in code, so Airflow writes the
same index FastAPI reads. See [Airflow schedules](#airflow-schedules) for the
document index DAG.

## Performance and concurrency

The request path is arranged so independent work overlaps, and no user waits on
work whose result is already known.

**FAQ path.** The query is embedded first. The ML guard then starts together
with the semantic-cache lookup and the Qdrant hybrid search. A cache hit
returns immediately: the cached answer was screened before it was stored, no
LLM runs, and the in-flight guard and search are cancelled before they consume
more CPU. On a miss the pipeline takes whichever of the guard and the search
finishes first, so a refusal never waits for a slow store, and the rerank
starts as soon as the points arrive: it runs while the guard is still working,
and a late refusal cancels it. Expansion and the LLM stream follow.
`expand_context` fetches each document's chunks concurrently and still merges
them in reading order, so a document that fails to load degrades that one
document instead of failing the request.

The reranker sees at most `RERANK_MAX_LEN` tokens per chunk. Ranking only needs
the head of a chunk, while the LLM receives the full text after expansion;
capping the input is the single largest CPU saving on the FAQ path on a
CPU-only host (12 chunks at 512 tokens take about 2 s, at 256 about 0.8 s).
Every RRF hit is ranked: trimming the list was measured to drop high-scoring
context (for example the best chunk of "How do I add insurance?" ranks 12th),
so the cap is applied to tokens, not to candidates. The cut is relative to the
best chunk (`RERANK_RELATIVE_MARGIN`): cross-encoder logits are not calibrated
across phrasings (the same policy chunk scores from +9 to −11), so an absolute
threshold dropped relevant context for some questions and kept noise for
others. `RERANK_MIN_SCORE` is only a safety floor; the top chunk always
survives unless it falls below it.

**Identical questions.** When two callers ask the same question at the same
time (a double-click, a retry, two patients on one ward terminal), the first
one to claim `semcache:inflight:<scope>:<hash>` becomes the leader and runs the
pipeline; the others wait on `semcache:peer:<scope>:<hash>` for up to
`CACHE_COALESCE_WAIT_SECONDS` and replay the leader's answer as their own SSE
stream. The scope includes tenant, audience and conversation digest, so an
answer is never shared across tenants, roles or a different history. Only
document-grounded FAQ questions without history, personal or follow-up framing
are coalesced; provider searches, refusals and "what did I just ask" answers
always run independently. The leader publishes the answer to the semantic cache
before releasing the marker, so a follower that arrives late still finds it
there. The trade-off is a bounded wait: if the leader fails, followers fall
back to running the pipeline themselves, and a leader that dies without
releasing its marker expires after `CACHE_COALESCE_TTL_SECONDS`. Set
`CACHE_COALESCE_WAIT_SECONDS=0` to turn this off and answer every caller
independently.

**Write-through.** The semantic-cache entry is written before `done` is sent,
bounded by `CACHE_WRITE_TIMEOUT` (0.25 s by default). Previously the write was
fire-and-forget, so a caller who heard the answer and asked the same question
again on another replica could miss the entry it had just filled; now a repeat
after the stream ends is a guaranteed hit. The write never delays the stream by
more than the timeout, and a Redis failure still degrades to a miss.

**Provider path.** Deterministic rules run first, because they are free and can
refuse without any I/O. The prompt-injection classifier and PII scan are
skipped inside this pipeline: the path has no prompt, returns only public
directory fields, and the model costs hundreds of milliseconds per query. The
chat layer still scans the question and answer before it writes a turn to
history. The exact Elasticsearch pass runs alone
(the text was already spell-corrected), and the fuzzy relaxed pass runs only
when it finds nothing. Startup issues one throwaway query so the first patient
does not pay the cold JVM/query cost, and the answer is streamed line by line.
A warm exact search takes tens of milliseconds on a small directory.

**CPU work.** Embedding, reranking, BM25 and the guard model run on thread
pools. Query embedding has its own small pool (`EMBED_WORKERS`) so short
embeddings cannot queue behind long rerank or guard jobs on the main executor
(`EXECUTOR_WORKERS`). Pool and semaphore defaults are derived from
`CPU_BUDGET`, which reads the container's cgroup quota first and the host's
core count only as a fallback: a 2-vCPU task on a 16-core node gets a 2-vCPU
sizing (`EXECUTOR_WORKERS=2`, `EMBED_WORKERS=1`, `SEM_GUARD=2`, `SEM_RAG=4`,
`SEM_RERANK=1`) instead of the host-sized defaults, which is what keeps a
small task from oversubscribing itself. Override `CPU_BUDGET` only when the
cgroup quota is not present (for example a bare-metal host). A CPU task that
sat in the executor queue longer than `CPU_QUEUE_WARN_MS` is logged as
`cpu_queue_ms=<n> cpu_task=<name>` with the request line, so oversubscription
shows up in the log instead of only as slow answers; raise the warn threshold
if you deliberately queue. Torch defaults to using every core per inference;
on a small host with many concurrent users that oversubscribes the CPU. Set
`TORCH_NUM_THREADS` (for example `2` on four cores) to trade a little
single-request speed for throughput and stable tail latency. Set the CPU limit
where the task runs (`cpus: "2.0"` in compose, or an ECS task `cpu: 2048`) and
leave `CPU_BUDGET` unset; the process reads the quota itself.

**Rate limiting.** Each request takes one token from a per-user, per-tenant and
(optionally) per-IP bucket in a single Redis Lua script; the clock comes from
Redis `TIME`, so replicas with skewed clocks agree. A denied request does not
consume a token and the 429 carries the exact `Retry-After` in seconds. Set
`RATE_LIMIT_IP_REQUESTS` above 0 to add a per-source-address bucket; behind a
load balancer also set `TRUST_FORWARDED_FOR=true` so the bucket keys on the
client, and leave it off when many patients share one NAT address. The IP is
hashed and never logged. Greetings take the cheap reply path *before* the
limiter, and a misspelled greeting ("helllo") is corrected first so it is
exempt too: neither spends a model, retrieval, cache or LLM call.

**Backpressure.** `SEM_GUARD`, `SEM_RAG`, `SEM_RERANK` and `SEM_LLM` cap
concurrent work per stage, `UserSlots` keeps one in-flight question per user
(shared across replicas through a TTL Redis lock), `QUEUE_TIMEOUT` bounds the
wait for a slot, and `UVICORN_LIMIT_CONCURRENCY` caps connections. Under
overload callers get a 503 busy event instead of unbounded queueing.

**Measuring it.** Each request log line carries stage timings:
`spell_ms`, `expand_ms`, `embed_ms`, `cache_ms`, `retrieve_ms`, `rerank_ms`,
`guard_ms`, `es_ms`, `pre_llm_ms` and `ttft_ms`, plus `guard` and `cache` result
labels and, when a follower was served, `coalesce=hit`. CPU tasks that waited in
the executor queue appear as `cpu_queue_ms=<n> cpu_task=<name>`. `rerank_ms` is
wall time and overlaps `guard_ms`, so on a miss expect
`pre_llm_ms` to be closer to `embed_ms + max(guard_ms, rerank_ms) + expand_ms`
than to their sum. On a cache hit expect `pre_llm_ms` to be close to `embed_ms` +
`cache_ms`, with the embedding usually dominating; on a provider query expect the
total to track `es_ms` (two searches when the strict pass is empty). Use these to
decide whether to tune thread counts or raise a semaphore.

**Scaling out.** Replicas share Qdrant, Redis, MySQL and Elasticsearch, and
nothing that must stay local is kept in the web process. The per-user slot is
a TTL counter in Redis (`USER_LOCK_TTL_SECONDS`), so a second question is
rejected whichever replica receives it; the slot is renewed while the answer
streams and expires if a task dies. A follow-up is written to MySQL before the
slot is released, so the next question on another replica sees it. The
semantic cache is tenant- and audience-scoped and follows the corpus
fingerprint, which is re-read every `CORPUS_RELOAD_SECONDS`. The in-flight
marker and peer answer used for coalescing live in Redis too, so an identical
question is answered once even when the two callers land on different replicas.
If Redis is unreachable the lock degrades to per-replica slots, coalescing is
skipped, and the shared rate limit fails open unless `RATE_LIMIT_FAIL_OPEN=false`.

## Configuration

Every variable is read once in `src/chat_search/config.py`. Blank counts as
unset.

### Required in production

| Variable | Notes |
| --- | --- |
| `JWT_SECRET` | HS* signing secret; use `JWT_PUBLIC_KEY` instead for RS*/ES* |
| `LLM_API_KEY` | OpenAI-compatible provider key |
| `ES_API_KEY` | read-only Elasticsearch key (enable provider search) |
| `ES_SYNC_API_KEY` | write key used only by the sync job |
| `CORS_ORIGINS` | explicit `http(s)://host[:port]` list; never `*` |
| `DEV_BYPASS_ROLE` | must be empty; the bypass only logs a warning at startup |

### Application and stores

| Variable | Default |
| --- | --- |
| `APP_NAME` | `Platform FAQ Assistant` |
| `PORT` | `8000` |
| `DATA_DIR` | `data` (read only by the index job) |
| `ENABLE_DOCS` | `false` (no `/docs` or `/openapi.json`) |
| `LOG_LEVEL` | `INFO` |
| `LOG_TRACEBACKS` | `false` |
| `AUDIT_LOG_ENABLED` | `true` (one audit record per chat request) |
| `AUDIT_LOG_CONTENT` | `true` (include question and answer; `false` keeps ids and sources) |
| `AUDIT_MAX_CHARS` | `4000` (per text field, after redaction) |
| `HEALTH_DETAILS` | `false` (public `/health` returns only the verdict) |
| `MAX_MESSAGE_CHARS` | `1000` |
| `QDRANT_URL` | `http://localhost:6333` |
| `QDRANT_API_KEY` | unset |
| `QDRANT_COLLECTION` | `faq_documents_v2` |
| `REDIS_URL` | `redis://localhost:6379/0` |
| `MYSQL_URL` | empty (conversation history disabled) |

### Models and retrieval

| Variable | Default |
| --- | --- |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` |
| `EMBED_QUERY_PREFIX` | auto for bge models |
| `SPARSE_MODEL` | `Qdrant/bm25` |
| `RERANK_ENABLED` | `true` |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `RERANK_MAX_LEN` | `256` (tokens per chunk seen by the reranker) |
| `RERANK_MIN_SCORE` | `-10.0` (safety floor, not a relevance cut) |
| `RERANK_RELATIVE_MARGIN` | `6.0` (logits below best − margin are dropped) |
| `GUARDRAIL_MODEL` | `protectai/deberta-v3-base-prompt-injection-v2` |
| `GUARD_THRESHOLD` | `0.92` |
| `GUARD_BLOCK_LABELS` | `INJECTION,JAILBREAK,UNSAFE` |
| `PII_ENABLED` | `true` |
| `PII_SPACY_MODEL` | `en_core_web_sm` (installed in the image) |
| `PII_SCORE_THRESHOLD` | `0.5` |
| `PII_ENTITIES` | email, phone, card, SSN/ITIN/passport/driver licence, bank, medical licence, IBAN, crypto, NHS, self-identifier, long number |
| `TOP_K` | `5` |
| `PREFETCH_LIMIT` | `20` per leg |
| `HYBRID_LIMIT` | `12` after RRF |

### Query understanding

| Variable | Default |
| --- | --- |
| `SPELLCHECK_ENABLED` | `true` |
| `SPELL_MAX_EDIT_DISTANCE` | `2` (words under 8 chars still use 1; capped at 2; unknown long clinical words may use one extra edit) |
| `SPELL_MIN_FREQUENCY` | `20` (candidate frequency floor) |
| `SPELL_MAX_CORRECTIONS` | `4` per query |
| `SPELL_DOMAIN_FREQUENCY` | `1000` (catalogue terms injected into the dictionary) |
| `SYNONYMS_ENABLED` | `true` |
| `CONCEPTS_ENABLED` | `true` |
| `CONCEPTS_FILE` | `concepts.json` (routing/Qdrant; provider rules are installed by the sync job) |
| `CONCEPTS_JSON` | empty; inline JSON override |
| `CONCEPTS_RELOAD_SECONDS` | `60` (0 checks on every query) |
| `UNDERSTANDING_ENABLED` | `true` (structured intent/concept output) |
| `UNDERSTANDING_SPELLING` | `true` (standalone calls; the chat pipeline corrects first) |
| `SCISPACY_ENABLED` | `true` (degrades to catalogue concepts if the model is absent) |
| `SCISPACY_MODEL` | `en_core_sci_sm` |
| `UMLS_ENABLED` | `false` (needs a local UMLS database) |
| `UMLS_DATA_PATH` | empty; directory containing the UMLS linker data |
| `SYNONYMS_FILE` | unset; `synonyms.json` when the setting is absent (legacy shape; explicit value overrides `CONCEPTS_FILE`) |
| `SYNONYMS_JSON` | empty; inline JSON override |
| `SYNONYMS_RELOAD_SECONDS` | `60` (0 checks on every query) |

### Agent routing

| Variable | Default |
| --- | --- |
| `AGENT_ROUTER_ENABLED` | `true` (the router node asks the model; false uses the rules) |
| `AGENT_ROUTER_BACKEND` | `laya` (`llm` for the label-only prompt, `rules` for no model) |
| `AGENT_ROUTER_MODEL` | `convaiinnovations/laya` (baked into the image) |
| `AGENT_ROUTER_SUBFOLDER` | empty, the English checkpoint; `multilingual` for the smaller one |
| `AGENT_ROUTER_DEVICE` | `cpu` |
| `AGENT_ROUTER_TIMEOUT` | `4` seconds for the LLM label call |
| `AGENT_ROUTER_LAYA_TIMEOUT` | `8` seconds; a slow forward pass falls back to the rules |
| `AGENT_ROUTER_MIN_CONFIDENCE` | `0` (off; Laya's temperatures are uncalibrated) |

### MCP

| Variable | Default |
| --- | --- |
| `MCP_ENABLED` | `true` (mounts the streamable HTTP tools at `/mcp/`) |
| `MCP_PATH` | `/mcp` |
| `MCP_ALLOWED_HOSTS` | empty; host check off (a bearer token is required instead) |
| `MCP_ROLE` | `patient` (stdio transport only) |
| `MCP_TENANT_ID` | `default` (stdio transport only) |
| `SYNONYM_MAX_EXPANSIONS` | `4` canonical terms per query |
| `CONDENSE_ENABLED` | `true` (rewrites a follow-up into a standalone search query) |
| `CONDENSE_TIMEOUT` | `4` seconds; a slow rewrite falls back to heuristic text |

Airflow's document index DAG also reads the Variables `qdrant_url`,
`qdrant_api_key`, `redis_url`, `qdrant_collection`, `embedding_model`,
`sparse_model` and the chunk sizes, plus the `DOCUMENTS_PATH` environment
variable. The provider DAG reads `es_url`, `es_sync_api_key`,
`es_providers_index`, the `providers_*` Variables and `provider_sync_min_ratio`.
The compose stack fills all of them from the same `.env` keys that serving uses.

### JWT and RBAC

| Variable | Default |
| --- | --- |
| `JWT_ALGORITHM` | `HS256` (pinned; never read from the token header) |
| `JWT_ISSUER` / `JWT_AUDIENCE` | unset (verified only when set) |
| `JWT_LEEWAY_SECONDS` | `10` |
| `REQUIRED_PERMISSION` | `docs:read` |
| `DEV_BYPASS_ROLE` | unset; `patient` or `provider` disables auth for local dev |

### LLM

| Variable | Default |
| --- | --- |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` |
| `LLM_MODEL` | `openai/gpt-oss-120b` |
| `LLM_MAX_TOKENS` | `700` |
| `LLM_REASONING_EFFORT` | `low` (empty omits the field) |
| `LLM_CONNECT_TIMEOUT` / `LLM_READ_TIMEOUT` / `LLM_TOTAL_TIMEOUT` | `5` / `30` / `90` s |
| `LLM_MAX_CONNECTIONS` | `50` |

### Cache, rate limits and concurrency

| Variable | Default |
| --- | --- |
| `CACHE_ENABLED` | `true` (needs Redis Stack) |
| `CACHE_SIMILARITY` | `0.92` |
| `CACHE_TTL_SECONDS` | `86400` |
| `CACHE_PER_USER` | `false` |
| `CACHE_WRITE_TIMEOUT` | `0.25` s (the entry is stored before `done`; this bounds the delay) |
| `CACHE_COALESCE_WAIT_SECONDS` | `5` (identical questions wait this long for the leader; `0` disables) |
| `CACHE_COALESCE_TTL_SECONDS` | `120` (in-flight marker safety net if a leader dies) |
| `CACHE_PEER_TTL_SECONDS` | `60` (in-flight answer channel between replicas) |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` |
| `RATE_LIMIT_USER_REQUESTS` | `20` (burst capacity, refilled evenly over the window) |
| `RATE_LIMIT_TENANT_REQUESTS` | `300` |
| `RATE_LIMIT_IP_REQUESTS` | `0` (off; per-source-address bucket when > 0) |
| `TRUST_FORWARDED_FOR` | `false` (set behind a load balancer; last `X-Forwarded-For` entry) |
| `RATE_LIMIT_FAIL_OPEN` | `true` |
| `CPU_BUDGET` | cgroup CPU quota, else host cores |
| `EXECUTOR_WORKERS` | `CPU_BUDGET - 1` (min 2; guard, rerank and expansion work) |
| `EMBED_WORKERS` | `CPU_BUDGET // 4` (min 1; dedicated so query embeddings never queue behind rerank) |
| `SEM_GUARD` / `SEM_RAG` / `SEM_RERANK` / `SEM_LLM` | `max(CPU_BUDGET, 2)` / `max(2*CPU_BUDGET, 4)` / `max(CPU_BUDGET//2, 1)` / `16` |
| `CPU_QUEUE_WARN_MS` | `50` (log `cpu_queue_ms` when a CPU task waited this long) |
| `TORCH_NUM_THREADS` | `0` (torch default; set `2` on a 4-core host serving many users) |
| `USER_CONCURRENCY` | `1` in-flight question per user |
| `USER_WAIT_TIMEOUT` | `0` (reject the second question at once) |
| `USER_LOCK_TTL_SECONDS` | `180` (shared slot TTL; `0` = in-process slots only) |
| `QUEUE_TIMEOUT` | `5` (then 503 busy) |
| `UVICORN_LIMIT_CONCURRENCY` | `200` |
| `REDIS_MAX_CONNECTIONS` | `100` per task (rate limits, cache, user slots) |
| `CORPUS_RELOAD_SECONDS` | `60` (`0` = read the fingerprint once at startup) |

The per-user limit is enforced in-process and, when Redis is reachable, by a
counter under `ulock:<tenant>:<subject>` shared by every replica. The counter
is renewed every third of `USER_LOCK_TTL_SECONDS` while the answer streams and
expires after a crash; a Redis failure falls back to per-replica slots without
failing the request. If a client disconnect interrupts the release itself, the
decrement is finished by a short background task, so a dropped stream cannot
lock that caller out until the TTL. Set the TTL above the longest possible
answer (pre-LLM work plus `LLM_TOTAL_TIMEOUT`).

Rate limits are a token bucket per user (`rl:u:`), tenant (`rl:t:`) and,
optionally, source address (`rl:i:`, hashed). Each request runs one Lua script
that reads Redis `TIME`, refills the bucket, and returns allow/deny; a denied
request is not charged and the 429 carries the exact `Retry-After`. Set
`RATE_LIMIT_IP_REQUESTS>0` to enable the IP bucket and `TRUST_FORWARDED_FOR=true`
when a proxy terminates TLS, so the bucket sees the client address rather than
the proxy. The counters use `REDIS_MAX_CONNECTIONS` like the cache and slots;
size it for all three.

### Conversation state

| Variable | Default |
| --- | --- |
| `MYSQL_URL` | empty (history disabled; `docker compose` sets it to the `mysql` service) |
| `MYSQL_POOL_SIZE` | `20` connections per task (each in-flight history read takes one) |
| `MYSQL_TIMEOUT` | `2` s (a slow store degrades, it never blocks the answer) |
| `HISTORY_ENABLED` | `true` |
| `HISTORY_TURNS` | `4` (messages kept: two exchanges) |
| `HISTORY_MAX_CHARS` | `400` (per stored turn, bounds the prompt) |
| `HISTORY_TTL_SECONDS` | `1800` (idle conversation expires) |

`MYSQL_URL` is a standard DSN: `mysql://user:password@host:3306/database`.
Percent-encode reserved characters in the password (for example `@` → `%40`).

### Elasticsearch and provider sync

| Variable | Default |
| --- | --- |
| `ES_URL` | empty (feature disabled) |
| `ES_API_KEY` / `ES_USER` + `ES_PASSWORD` | unset |
| `ES_CA_CERTS` | unset |
| `ES_PROVIDERS_INDEX` | `providers` (created by the sync job if missing) |
| `ES_TIMEOUT` | `3` s |
| `PROVIDER_RESULTS` | `5` |
| `PROVIDER_SEARCH_PERMISSION` | `providers:search` (empty disables the check) |
| `PROVIDERS_API_URL` / `PROVIDERS_API_TOKEN` | unset |
| `PROVIDERS_API_TENANT` | unset; set it when records have no `tenant_id` — must match the JWT tenant |
| `PROVIDERS_API_PAGE_SIZE` | `100` |
| `PROVIDERS_API_ITEMS_KEY` | `data` |
| `PROVIDERS_API_PARAMS` | `{}` JSON |
| `PROVIDERS_FIELD_MAP` | `{}` JSON, e.g. `{"provider_id": "id", "name": "full_name"}` |
| `PROVIDER_SYNC_MIN_RATIO` | `0.5` |

### Chunking and context

| Variable | Default |
| --- | --- |
| `CHUNK_TOKENS` | `300` target |
| `CHUNK_MAX_TOKENS` | `380` hard limit |
| `CHUNK_OVERLAP_TOKENS` | `40` |
| `MIN_CHUNK_TOKENS` | `60` (smaller chunks merge) |
| `WHOLE_DOC_TOKENS` | `350` (short documents stay whole) |
| `EXPAND_DOC_TOKENS` | `1000` |
| `EXPAND_DOCS` | `2` |
| `EXPAND_SCORE_MARGIN` | `4.0` |
| `MAX_CONTEXT_TOKENS` | `1600` |

## Security notes

- `.env` holds real secrets. It is git-ignored and excluded from the Docker
  build context; never commit it, print it, or copy values into examples.
- `DEV_BYPASS_ROLE` authenticates every caller. Use it only for local
  development. Startup logs a warning if it is set.
- JWT algorithms are pinned in configuration, not taken from the token header.
- `CORS_ORIGINS` is an allow-list; wildcards are rejected at startup.
- Logs contain method, path, status, latency, request ID and metric labels.
  Query strings, messages, prompts and retrieved text are never in the request
  line. Conversation text appears only in the separate audit record, which
  carries the caller, question, answer and sources and must be routed and
  retained as PHI (see [Audit log](#audit-log)).
- Every log record, including tracebacks, passes through a redacting formatter:
  configured secret values, URL userinfo, `Authorization`/bearer tokens and
  credential-shaped pairs (`api_key=`, `token=`, `password=`) are replaced.
  Log URLs with `sanitize_url()` so query strings and credentials never reach a
  handler.
- Callers see stable error codes and messages only. Unexpected failures return
  `internal_error` with a generic message and a request ID; exception text,
  stack traces and upstream responses stay in the process. Failure logs record
  the exception *type* and request ID, never `str(exc)`.
- Background tasks (cache writes, warmups) report failures through the same
  helper, so an exception cannot surface as an unretrieved-task traceback.
- The provider normalizer copies only public directory fields; emails, phones,
  notes and other API fields never leave the normalizer.
- Conversation state is keyed by tenant and user, is only reachable with that
  caller's credentials, and expires (`HISTORY_TTL_SECONDS`). Turns are user text
  and answers, never credentials; they are bounded per turn and per
  conversation so the store cannot be used as unbounded storage. Give the MySQL
  user access only to the `chat_search` database and keep the volume encrypted
  if it holds patient-facing text.
- Guardrails are risk reduction, not a guarantee. The injection model fails
  closed if it cannot run; PII detection fails closed for sharing and storage
  (the text is treated as personal) while the answer is still produced.

## Operations

- Set `LOG_TRACEBACKS=true` only when debugging; default logs omit tracebacks.
  When enabled, tracebacks are attached for diagnosis and still redacted, but
  they may contain upstream response detail — treat such logs as sensitive.
- Startup and shutdown log lifecycle events once. Shutdown closes each store
  independently and logs failures instead of aborting the rest of cleanup.
- The cache degrades to "no cache" if Redis is not Redis Stack; startup logs
  the reason and the service keeps serving.
- Rate limiting fails open by default. Set `RATE_LIMIT_FAIL_OPEN=false` to
  reject with 503 when Redis is down.
- `/health` checks stores, not model quality. Readiness is also gated by the
  container healthcheck and its 180 s start period.
- Deploy document indexing and provider sync as separate one-shot containers,
  never in the web process. Re-run `index-documents` after every `data/` change
  and `sync-providers` on its schedule.

### Audit log

Every chat request writes one structured record to the `chat_search.audit`
logger at INFO, after the answer is complete (both endpoints use the same
pipeline):

| Field | Meaning |
| --- | --- |
| `ts` | ISO 8601 UTC timestamp with milliseconds |
| `event` | `chat` |
| `request_id` | The ID also returned in the `X-Request-ID` header |
| `tenant_id`, `role`, `subject` | Verified JWT identity |
| `patient_id` / `provider_id` | Exactly one identifies the caller |
| `session_id`, `mode`, `node` | Conversation and routing (`faq`, `providers`, `greeting`, `rate_limit`) |
| `input` / `output` | Caller message and final answer, redacted and bounded |
| `sources` | `document` + `page` for document answers |
| `providers` | `name`, `specialty`, `city` for directory answers |
| `cached`, `outcome`, `error` | `ok`, `error`, `cancelled` or `rate_limited`; error is a code, never exception text |
| `history_turns`, `latency_ms` | Conversation depth and wall time |

This is the only line that carries conversation text, and that text may be
PHI. Route the `chat_search.audit` logger to its own sink with access control,
encryption at rest and a retention policy; do not mix it into the metric-filter
stream. The audit logger stays at INFO when `LOG_LEVEL` is raised, so lowering
verbosity cannot silently drop the trail. `AUDIT_LOG_CONTENT=false` keeps the
ids, sources and timing without the text (useful where content logging is not
approved), `AUDIT_LOG_ENABLED=false` turns the record off, and `AUDIT_MAX_CHARS`
bounds each text field after secret redaction. The shared redacting formatter
still removes configured secrets, bearer tokens and credential-shaped strings.
A client disconnect records `outcome=cancelled` with the partial answer; an
upstream failure records `outcome=error` and the code (for example `busy`).
Unauthenticated requests are rejected before an identity exists and appear only
in the metadata-only request line.

## Production readiness

Short answer: the service is hardened, the deployment in this repository is not
turnkey. `docker-compose.yml` is a development stack (single-node
Elasticsearch, MySQL and Redis containers, Docker-socket Airflow) and nothing
here provisions AWS infrastructure, secrets or CI/CD. The list below is what is
already in place and what must be closed before a production launch.

Already in place:

- JWT verification with pinned algorithms; tenant, role and audience RBAC
  derived from the token; per-role cache scopes and conversation keys.
- Deterministic guardrails plus the injection model, Redis rate limits,
  per-user slots shared across replicas, timeouts and semaphores on every heavy
  stage.
- Graceful degradation: Redis, MySQL, Elasticsearch and the semantic cache can
  be down without stopping chat (each logs the consequence).
- A read-only web process; indexing and provider sync are separate one-shot
  jobs.
- Secrets redacted from every log record; request lines stay metadata-only;
  the separate audit record carries caller ids, question, answer and sources
  and is documented as PHI; stable error envelopes; `/docs` disabled by default.
- 416 unit tests and `ruff check` clean in CI-shaped runs.

Close before calling it production-ready:

1. **No infrastructure as code, CI/CD or secrets wiring.** The compose file and
   the local Airflow stack must be replaced by provisioned infrastructure (see
   below). `DEV_BYPASS_ROLE` is only warned about at startup, not refused, so
   the deploy pipeline must guarantee it is unset.
2. **Redis must be Redis Stack.** The semantic cache uses RediSearch
   (`FT.CREATE`/`FT.SEARCH` vectors). Amazon ElastiCache for Redis OSS does not
   provide those commands; the service logs "Semantic cache disabled" and serves
   without it. Run Redis Stack yourself (with AOF and `volatile-lru`) or accept
   the cache being off. ElastiCache for Valkey's vector search is not a
   drop-in replacement for the redis-py `ft()` API without testing.
3. **Elasticsearch, not OpenSearch, with the current client.** The code uses
   `elasticsearch[async]` 8.x, which requires the `X-Elastic-Product:
   Elasticsearch` response header. Amazon OpenSearch Service does not send it,
   so `make_es_client` fails. Use a self-managed Elasticsearch 8.x cluster, or
   migrate `providers/client.py` and `providers/search.py` to `opensearch-py`.
4. **MySQL TLS is not wired.** `state.py` builds an aiomysql DSN without an SSL
   context, so RDS must allow unencrypted connections inside the VPC, or
   `_dsn()` must pass `ssl=` before "require TLS" is enabled on RDS.
5. **Verify the shared per-user slot under load.** `UserSlots` now enforces the
   limit across replicas with a TTL counter in Redis, renewed while the answer
   streams, and the conversation turn is stored before the slot is released.
   Confirm `USER_LOCK_TTL_SECONDS` exceeds the longest answer, size
   `REDIS_MAX_CONNECTIONS` for lock + cache + rate limits, and decide whether a
   Redis outage may fall back to per-replica slots (it currently does, without
   failing requests).
6. **Health is shallow.** `/health` returns 200/503 from Qdrant and Redis
   (probed in parallel, each capped at one second) and does not check MySQL or
   the LLM. Per-dependency booleans require `HEALTH_DETAILS=true`. Alarm on the
   HTTP status and the startup log line (`history=True providers=True`).
7. **No metrics exporter or tracing.** Metric fields are written to the
   structured request log (`guard=`, `cache=`, `ttft_ms=`, …); there is no
   `/metrics`, Prometheus, OpenTelemetry or CloudWatch EMF. Use CloudWatch
   metric filters or run a log agent until an exporter is added.
8. **The document index reads a local directory.** `DATA_DIR` is a filesystem
   path; S3 documents need a staging step (an `aws s3 sync` sidecar into an EFS
   volume, or a wrapper command) before `index-documents` runs.
9. **Airflow here is local-only.** Both DAGs use `DockerOperator` against the
   host Docker socket; MWAA has no Docker daemon. On AWS, schedule the two
   commands with EventBridge Scheduler + ECS `RunTask` instead (or use MWAA with
   an ECS operator).
10. **Warm start and CPU-only inference.** Models are baked into the image
    (6.2 GB) and load in ~60–90 s per task; there is no GPU path. Keep warm
    capacity and use a long target-group grace period and scale-in cooldown.
11. **No backup/DR automation or load test.** RDS automated backups, Qdrant
    snapshots and Elasticsearch snapshots are not scripted here, and the
    concurrency limits have not been load-tested beyond the defaults. Pool and
    semaphore defaults follow the cgroup CPU quota, so confirm the task CPU
    limit is visible in the container (`cpu.max`, logged at startup as
    `cpu_budget=`) and set `CPU_BUDGET` explicitly if your platform sizes
    differently; watch `cpu_queue_ms` in the request log while load testing.

A controlled pilot is reasonable once the AWS section below is applied; a
regulated launch also needs items 1–11 closed.

## Deploying on AWS

### Reference architecture

| Piece | AWS service | Notes |
| --- | --- | --- |
| Edge | ALB (TLS, WAF optional) | SSE-friendly; idle timeout ≥ 120 s |
| Web | ECS Fargate service | same image, `python -m chat_search`; scale by tasks |
| Index/sync jobs | ECS one-off tasks + EventBridge Scheduler | replaces Airflow |
| Conversation state | RDS for MySQL 8.4 | table is created by the app; see TLS caveat |
| Cache + rate limits | Self-managed Redis Stack on ECS/EC2 | RediSearch required; or `CACHE_ENABLED=false` |
| Vectors | Qdrant on ECS + EFS, or Qdrant Cloud | collection `faq_documents_v2` |
| Directory | Self-managed Elasticsearch 8.x | OpenSearch is not drop-in |
| Documents | S3 | stage to EFS/local disk for the index job |
| Secrets | Secrets Manager / SSM | inject with task-definition `secrets` |
| Logs/alarms | CloudWatch Logs + metric filters | JSON log lines, no query text |

### 1. Build and push the image

The image is also the job runner; build it once per release and tag it by
digest.

```bash
aws ecr get-login-password --region <region> | docker login --username AWS \
  --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker build -t <repo>:<tag> .
docker tag <repo>:<tag> <account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>
docker push <account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>
```

Build on `linux/amd64` for Fargate; the build installs its own dependencies, so
no runtime internet access is needed for Python or the models.

### 2. Store secrets and configuration

Put `LLM_API_KEY`, `PROVIDERS_API_TOKEN`, `ES_API_KEY`, `ES_SYNC_API_KEY`,
`QDRANT_API_KEY`, `JWT_PUBLIC_KEY` and the MySQL password in Secrets Manager.
Reference them from the task definition's `secrets` block; keep non-secret
values in plain environment variables.

`JWT_PUBLIC_KEY` is a PEM. Paste it with literal `\n` escapes (the config
replaces `\\n` with newlines) or store it base64 and decode in the entrypoint.
Prefer `JWT_ALGORITHM=RS256` over a shared HS256 secret so the Laravel issuer
keeps the private key.

### 3. Create the web task definition

- CPU/memory: at least 2 vCPU / 4 GB (models plus torch); 4 vCPU / 8 GB for
  throughput. Set `TORCH_NUM_THREADS=2` on a 4 vCPU task so guard and rerank
  inference do not oversubscribe the host.
- Container port 8000, health check path `/health`, start period 180 s.
- Command: `python -m chat_search` (the image default). Run one uvicorn process
  per task; add tasks to scale, not workers.
- Ephemeral storage: the default 20 GB is enough for the 6.2 GB image.
- Run as the image's non-root `app` user; no Docker socket, no EFS needed for
  the web service.

Minimum environment (values that differ from the compose defaults):

| Variable | Production value | Why |
| --- | --- | --- |
| `DEV_BYPASS_ROLE` | unset | the bypass authenticates everyone |
| `JWT_ALGORITHM` | `RS256` | verify with a public key, not a shared secret |
| `JWT_PUBLIC_KEY` | PEM from Secrets Manager | required when `JWT_ALGORITHM` is RS*/ES* |
| `JWT_ISSUER`, `JWT_AUDIENCE` | the Laravel issuer/audience | verified only when set; set both |
| `REQUIRED_PERMISSION` | `docs:read` | keeps the existing token contract |
| `CORS_ORIGINS` | `https://portal.example.com` | explicit origins; wildcards are rejected |
| `ENABLE_DOCS` | `false` | no `/docs` or `/openapi.json` |
| `HEALTH_DETAILS` | `false` | `/health` exposes only the verdict; 503 when Qdrant or Redis is down |
| `MYSQL_URL` | `mysql://user:pass@<rds-endpoint>:3306/chat_search` | percent-encode the password; no TLS in the DSN |
| `REDIS_URL` | `redis://<redis-stack-endpoint>:6379/0` | `rediss://` if you terminate TLS yourself |
| `QDRANT_URL` | your Qdrant endpoint | HTTPS + `QDRANT_API_KEY` if managed |
| `QDRANT_COLLECTION` | `faq_documents_v2` | must match the index job |
| `ES_URL` | `https://<es-endpoint>:9200` | self-managed ES 8.x |
| `ES_API_KEY` | index-scoped read key | serving never writes |
| `ES_PROVIDERS_INDEX` | `providers` | must match the sync job |
| `PROVIDERS_API_URL`, `PROVIDERS_API_TOKEN` | real provider API | sync job only |
| `PROVIDERS_API_PARAMS`, `PROVIDERS_FIELD_MAP` | JSON per API | sync job only |
| `PROVIDER_SYNC_MIN_RATIO` | `0.5` | refuse a sync that would delete most providers |
| `LOG_LEVEL`, `LOG_TRACEBACKS` | `INFO`, `false` | tracebacks may carry upstream detail |
| `RATE_LIMIT_USER_REQUESTS`, `RATE_LIMIT_TENANT_REQUESTS` | per plan | shared across tasks via Redis |
| `USER_LOCK_TTL_SECONDS` | `180` | the per-user slot across tasks; keep it above the longest answer |
| `MYSQL_POOL_SIZE`, `REDIS_MAX_CONNECTIONS` | `20`, `100` per task | tasks × pool must stay under the RDS and Redis limits |

Everything else keeps its default; see [Configuration](#configuration).

### 4. ALB and autoscaling

- Target group health check: `/health`, 15 s interval, 5 s timeout, 3 failures,
  deregistration delay 30 s, health-check grace period 180 s (model warmup).
- Idle timeout at least 120 s; the LLM can stay silent for 30 s between tokens
  and 90 s in total. Do not put a buffering CDN in front of
  `/v1/chat/stream` unless it forwards chunked responses.
- Autoscale on `ALBRequestCountPerTarget` and 60 % CPU with a scale-in cooldown
  that covers warmup (~3–5 min). Keep at least two tasks in two AZs; do not
  scale to zero.
- Roll out with the ECS deployment circuit breaker, 100 % minimum healthy and
  200 % maximum (one spare task during a deploy). Long-lived SSE connections
  finish on the old tasks before they drain.

### 5. Provision the stateful services

- **RDS MySQL**: enable automated backups and encryption; create the
  `chat_search` database and a user scoped to it. The app creates
  `conversation_turn` on startup (`CREATE TABLE IF NOT EXISTS`), so the user
  needs DDL on that schema. Point `MYSQL_URL` at the endpoint and keep the
  security group limited to the app tasks. Size `MYSQL_POOL_SIZE` per task so
  task count × pool stays well below the instance `max_connections`; a pool
  that cannot keep up degrades turns to stateless instead of failing.
- **Redis Stack**: run it as its own ECS service with an EFS or EBS volume,
  AOF enabled, `--maxmemory` set and `--maxmemory-policy volatile-lru` (only
  keys with a TTL may be evicted). It carries the rate limits, the semantic
  cache and the shared per-user slot, so keep it reachable from every task;
  if it is down, the slot and rate limit fall back to per-task behavior.
  If you use ElastiCache without RediSearch, set `CACHE_ENABLED=false` and
  accept a cache-less deployment.
- **Qdrant**: a single node with a persistent volume is enough for a pilot; use
  Qdrant Cloud if you want backups and HA managed. Set `QDRANT_API_KEY` and
  terminate TLS in front of it.
- **Elasticsearch**: self-managed 8.x with security enabled, TLS and an
  index-scoped read key for `ES_API_KEY` plus a write key for
  `ES_SYNC_API_KEY`. Take snapshots to S3.

### 6. Run indexing and provider sync

Use the same image with a command override on an ECS task (no load balancer, no
service):

| Job | Command | Schedule (example) |
| --- | --- | --- |
| Document index | `python -m chat_search index-documents` | weekly, EventBridge cron `0 3 ? * SUN *` |
| Provider sync | `python -m chat_search sync-providers` | weekly, `0 2 ? * SUN *` |

The document job needs the PDFs on `DATA_DIR`. With documents in S3, add an
`amazon/aws-cli` sidecar that runs `aws s3 sync s3://<bucket>/data /mnt/data`
into a shared EFS volume, and set `DATA_DIR=/mnt/data` on the job container.
Run both jobs once before the first deployment; the index job publishes the
corpus fingerprint that invalidates the semantic cache, and the sync job creates
the `providers` index when missing.

Neither job belongs in the web service.

### 7. Networking and security

- Private subnets for tasks and data stores; NAT gateway egress for
  `api.groq.com` and the providers API. VPC endpoints for ECR, S3, Secrets
  Manager, CloudWatch Logs, STS and EFS.
- No public endpoints for MySQL, Redis, Qdrant or Elasticsearch; allow only the
  app and job security groups.
- TLS terminates at the ALB. Keep `ENABLE_DOCS=false`, an explicit
  `CORS_ORIGINS` and the default redacting logs. Add WAF if the API is
  internet-facing.
- The bundled page at `/` is a demo; serve the real portal from Laravel and let
  it call the API with the user's JWT.

### 8. Logs and alarms

Logs are single-line JSON on stdout, prefixed by timestamp, level and logger.
Build CloudWatch metric filters for:
5xx responses, `guard=` blocks, `rate_limit=blocked`, `provider_search=denied`
or `error`, `cache_error`, and the `/health` booleans. Alarm on p95 latency and
on tasks being replaced for failed health checks. Keep `LOG_TRACEBACKS=false`
unless debugging, and treat a stack-trace window as sensitive. Route the
`chat_search.audit` logger to a separate, access-controlled log group treated
as PHI (see [Audit log](#audit-log)); do not build metric filters that copy
`input` or `output` into alarm text.

### 9. First-deploy checks

1. `/health` returns 200 `{"status":"ok"}`; with `HEALTH_DETAILS=true` the
   booleans `qdrant`, `redis`, `cache` and `elasticsearch` are true (or false
   where you deliberately disabled the feature), one slow dependency does not
   delay the others, and stopping Redis flips the status to 503 `degraded`.
2. The startup line reads `Ready corpus=… cache=True history=True guard=True
   rerank=True providers=True … audit=True … cpu_budget=N`; check `cpu_budget`
   matches the task's vCPU limit.
3. `index-documents` and `sync-providers` ran; a patient question returns
   sources and a provider query returns `data.providers`.
4. A provider-role token gets the directory refusal, and
   `DEV_BYPASS_ROLE` is absent from the task environment.
5. Two simultaneous questions for the same user are rejected on whichever task
   receives the second one (`request_in_progress`), and a follow-up right after
   the first answer still sees the previous turn.
6. Two calls with the same question and token at the same moment produce one
   answer (the second replays the leader's stream); during it
   `semcache:inflight:*` and `semcache:peer:*` keys exist in Redis and clear
   afterwards. Asking again once the stream ends is a cache hit.
7. Exceeding `RATE_LIMIT_USER_REQUESTS` within the window returns 429 with a
   numeric `Retry-After`, and the next request after that many seconds
   succeeds.
8. A chat request writes one `chat_search.audit` JSON line with `ts`, the
   caller's `patient_id` or `provider_id`, `input`, `output` and `sources`;
   confirm its log group has restricted access, encryption and a retention
   policy.

## Testing and quality

```bash
pip install -e ".[dev,stores]"
ruff check .
ruff format --check .
pytest -q
```

The unit suite needs the `stores` extra (clients only), but no running Qdrant,
Redis, MySQL, Elasticsearch or ML models. The spelling tests use `symspellpy` from the
base dependencies and skip if its dictionary is unavailable. The retrieval tests
skip when `qdrant-client` is absent. `test_separation.py` imports every serving
module in a clean subprocess and fails if any indexing module is pulled in. CI
runs the same three commands on Python 3.11.

`make test`, `make lint`, `make format` and `make docker-build` wrap the common
tasks.

## Migration notes from the single-file version

- All root commands keep working through `main.py`; prefer
  `python -m chat_search <command>`.
- Behavior preserved: routes, event shapes, chunking, cache identity, rate
  limits, error messages, health payload and the embedded UI.
- Fixed: the non-streaming `/v1/chat` path no longer crashes while converting a
  generic error event (it falls back to HTTP 500).
- Fixed: `/health` now reports a dependency as down when its probe returns
  `false`, not only when the probe raises.
- Changed: `/health` returns 503 `{"status":"degraded"}` when Qdrant or Redis
  is down and hides the dependency inventory unless `HEALTH_DETAILS=true`.
  The container healthcheck already relies on the status code, so it now
  reports an unusable app instead of always "healthy".
- Fixed: provider documents now carry `tenant_id` and the ES query filters on
  it. Re-run the provider sync after deploying.
- Changed: provider search applies specialties with an Elasticsearch
  `synonym_graph` search analyzer instead of expanding queries in the app.
  `best gynecologist`, `best ob-gyn`, `best eye specialist` and the misspelling
  `gyneacologist` reach the same rows. Run `sync-providers` (or
  `sync-provider-synonyms`) after deploying to upgrade the index analysis.
- Changed: provider answers stream line by line instead of arriving as one
  event, and the directory path no longer runs the prompt-injection classifier
  (deterministic rules still apply, and there is no LLM prompt to protect).
- Added: conservative query spelling correction and a hot-reloaded concept
  catalogue. Both are on by default and need no re-index.
- Added: `dags/document_index_dag.py` schedules the Qdrant index weekly and on
  demand; `provider_sync_dag.py` remains the Elasticsearch job.
- Replaced: the old unreachable provider spell-correction code with the tested
  query-understanding modules (`spelling.py`, `synonyms.py`).
- Changed: startup no longer ingests PDFs or creates the Elasticsearch index.
  Run `python -m chat_search index-documents` once and after every `data/`
  change; `ES_CREATE_INDEX` is gone because `sync-providers` always creates a
  missing index.
- Dependencies moved from `requirements.txt` to `pyproject.toml`; the old file
  is kept as a pointer for tooling that still reads it. Airflow is an optional
  extra because only the scheduler image needs it.
