FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models/hf \
    FASTEMBED_CACHE_PATH=/opt/models/fastembed

WORKDIR /app

# CPU-only torch keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# pip cannot resume a truncated wheel: a slow link ends in
# `Expected sha256 ... Got e3b0c442...` (the digest of an empty file). PyPI's CDN
# is throttled on some networks, so the index is a build arg:
#   PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple docker compose build app
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_RETRIES=20 \
    PIP_TIMEOUT=1000

COPY pyproject.toml README.md ./
# scispaCy pins spaCy to 3.7.x, so the English model must come from the same
# line: spaCy refuses a model trained on a newer minor version.
ARG SPACY_MODEL_URL=https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
# Dependencies install against an empty package so this layer depends on
# pyproject.toml alone: editing src/ no longer re-downloads the ML stack. The
# real package replaces the stub in the next layer. The cache mount survives an
# interrupted build and stays out of the image; `env -u` re-enables pip's cache,
# which PIP_NO_CACHE_DIR above would otherwise keep off.
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p src/chat_search && touch src/chat_search/__init__.py \
 && env -u PIP_NO_CACHE_DIR pip install --index-url "$PIP_INDEX_URL" ".[all]" \
 && pip install "$SPACY_MODEL_URL"

# Clinical NER model (scispaCy). Optional at build time: without it the service
# logs one warning and uses the structured catalogue concepts only.
ARG SCISPACY_MODEL_URL=https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
RUN pip install "$SCISPACY_MODEL_URL" \
 || echo "en_core_sci_sm not installed; clinical NER degrades to catalogue concepts"

COPY src ./src
# The dependency layer installed an empty stub package; uninstall it first so no
# stub file (notably the empty __init__.py) can survive next to the real one.
RUN pip uninstall -y chat-search \
 && rm -rf /app/build /app/src/chat_search.egg-info /app/src/*.egg-info /app/*.egg-info \
 && pip install --no-cache-dir --no-build-isolation --no-deps --no-cache-dir .

# Operator-editable concept catalogue (mounted or replaced to reload at runtime).
COPY concepts.json ./

# Bake every HuggingFace model into the image (dense, BM25, reranker, injection
# guardrail, System 1 router). The Presidio spaCy model is installed by the
# dependency layer. If you change a model name in .env, add it here too or it
# downloads at startup.
# The downloads go through a cache mount, so an interrupted build resumes
# instead of re-fetching ~1 GB, and are then copied into the image: the cache
# itself never ships, but startup stays fully offline.
# The router checkpoint is selected by AGENT_ROUTER_MODEL / AGENT_ROUTER_SUBFOLDER;
# a different one there is not baked and is fetched on startup instead.
ARG LAYA_MODEL=convaiinnovations/laya
ARG LAYA_SUBFOLDER=
RUN --mount=type=cache,target=/hf-cache \
    HF_HOME=/hf-cache python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')" \
 && HF_HOME=/hf-cache python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" \
 && HF_HOME=/hf-cache python -c "from transformers import pipeline; pipeline('text-classification', model='protectai/deberta-v3-base-prompt-injection-v2')" \
 && HF_HOME=/hf-cache python -c "from fastembed import SparseTextEmbedding; list(SparseTextEmbedding('Qdrant/bm25').embed(['warmup']))" \
 && HF_HOME=/hf-cache LAYA_MODEL="$LAYA_MODEL" LAYA_SUBFOLDER="$LAYA_SUBFOLDER" python -c "import os, laya; laya.load(os.environ['LAYA_MODEL'], device='cpu', subfolder=os.environ['LAYA_SUBFOLDER'] or None)" \
 && mkdir -p /opt/models/hf && cp -a /hf-cache/. /opt/models/hf/

# Legacy launcher: `python main.py ...` still works inside the container.
COPY main.py ./

RUN useradd --create-home app && chown -R app:app /opt/models
USER app

EXPOSE 8000
CMD ["python", "-m", "chat_search"]
