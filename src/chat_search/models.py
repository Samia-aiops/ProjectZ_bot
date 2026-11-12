"""Model loading.

Imports are intentionally inside ``load_models`` so the rest of the package can
be imported (and unit-tested) without pulling in torch/transformers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from . import config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Models:
    embedder: Any
    sparse: Any
    reranker: Any
    guard: Any


@dataclass(frozen=True)
class IndexModels:
    embedder: Any
    sparse: Any


def load_index_models() -> IndexModels:
    """Models the document index job needs: no reranker or guardrail."""
    from fastembed import SparseTextEmbedding
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    sparse = SparseTextEmbedding(config.SPARSE_MODEL)

    embedder.encode("warmup")
    list(sparse.embed("warmup"))

    log.info("index models loaded embedding=%s sparse=%s", config.EMBEDDING_MODEL, config.SPARSE_MODEL)
    return IndexModels(embedder=embedder, sparse=sparse)


def load_models() -> Models:
    if config.TORCH_NUM_THREADS > 0:
        import torch

        torch.set_num_threads(config.TORCH_NUM_THREADS)

    from fastembed import SparseTextEmbedding
    from sentence_transformers import CrossEncoder, SentenceTransformer

    embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    sparse = SparseTextEmbedding(config.SPARSE_MODEL)
    reranker = CrossEncoder(config.RERANK_MODEL, max_length=config.RERANK_MAX_LEN) if config.RERANK_ENABLED else None
    guard = None
    if config.GUARDRAIL_MODEL:
        from transformers import pipeline

        guard = pipeline("text-classification", model=config.GUARDRAIL_MODEL, device=-1)

    # Warm up so the first real request does not pay the initialisation cost.
    embedder.encode("warmup")
    list(sparse.query_embed("warmup"))
    if reranker:
        reranker.predict([("warmup", "warmup")])
    if guard:
        guard("warmup", truncation=True, max_length=512, return_token_type_ids=False)

    log.info(
        "models loaded embedding=%s sparse=%s rerank=%s guard=%s",
        config.EMBEDDING_MODEL,
        config.SPARSE_MODEL,
        bool(reranker),
        bool(guard),
    )
    return Models(embedder=embedder, sparse=sparse, reranker=reranker, guard=guard)
