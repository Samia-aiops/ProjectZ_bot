"""Qdrant collection setup and PDF indexing."""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from .. import config
from ..corpus import settings_salt
from ..text import embed_text
from .chunking import Chunk, build_chunks, doc_info
from .pdf import extract_pages

log = logging.getLogger(__name__)


def eq_condition(key: str, value: str) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def ensure_collection(client: QdrantClient, dim: int) -> None:
    if not client.collection_exists(config.QDRANT_COLLECTION):
        client.create_collection(
            config.QDRANT_COLLECTION,
            vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            # IDF is computed by Qdrant -> BM25 weighting.
            sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
    for name in ("tenant_id", "audience", "source", "file_hash"):
        client.create_payload_index(config.QDRANT_COLLECTION, name, models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(config.QDRANT_COLLECTION, "order", models.PayloadSchemaType.INTEGER)


def classify_path(relative: Path) -> tuple[str, str, str] | None:
    """Return (tenant_id, audience, display_name) or None if the location is not recognised."""
    parts = relative.parts
    if parts[0] in config.FOLDER_AUDIENCE:  # data/<audience>/...
        return config.GLOBAL_TENANT, config.FOLDER_AUDIENCE[parts[0]], relative.as_posix()
    if len(parts) >= 3 and config.TENANT_RE.match(parts[0]) and parts[1] in config.FOLDER_AUDIENCE:
        return parts[0], config.FOLDER_AUDIENCE[parts[1]], "/".join(parts[1:])  # data/<tenant>/<audience>/...
    return None


def index_documents(embedder: Any, sparse: Any) -> str:
    """Create/update the collection from DATA_DIR and return the corpus fingerprint."""
    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        timeout=max(int(config.QDRANT_TIMEOUT), 60),
    )
    try:
        ensure_collection(client, embedder.get_sentence_embedding_dimension())
        return ingest_documents(client, embedder, sparse)
    finally:
        client.close()


def ingest_documents(client: QdrantClient, embedder: Any, sparse: Any) -> str:
    """Index new/changed PDFs, drop removed ones. Return a corpus version hash."""
    data_dir = config.DATA_DIR
    if not data_dir.is_dir():
        raise RuntimeError(f"DATA_DIR not found: {data_dir.resolve()}")

    sources: list[str] = []
    versions: list[str] = []
    salt = settings_salt().encode()

    for path in sorted(data_dir.rglob("*.pdf")):
        relative = path.relative_to(data_dir)
        info = classify_path(relative)
        if info is None:
            log.warning("Skipping %s: use data/<shared|patient|provider>/ or data/<tenant>/<...>/", relative)
            continue
        tenant, audience, _ = info

        source = relative.as_posix()
        digest = hashlib.sha256(path.read_bytes() + salt).hexdigest()
        sources.append(source)
        versions.append(f"{source}:{digest}")

        already, _ = client.scroll(
            config.QDRANT_COLLECTION,
            scroll_filter=models.Filter(must=[eq_condition("source", source), eq_condition("file_hash", digest)]),
            limit=1,
        )
        if already:
            continue

        doc_type, title = doc_info(path.stem)
        try:
            chunks = build_chunks(extract_pages(path), doc_type, title)
        except Exception as exc:
            log.error("Could not read %s (%s)", source, type(exc).__name__)
            continue

        # Replace any older version of this file.
        client.delete(
            config.QDRANT_COLLECTION,
            points_selector=models.FilterSelector(filter=models.Filter(must=[eq_condition("source", source)])),
        )
        if not chunks:
            log.warning("No extractable text in %s", source)
            continue

        texts = [embed_text(title, chunk) for chunk in chunks]
        dense = embedder.encode(texts, normalize_embeddings=True, batch_size=32)
        sparse_vectors = list(sparse.embed(texts))
        client.upsert(
            config.QDRANT_COLLECTION,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{digest}:{chunk['order']}")),
                    vector={
                        "dense": vector.tolist(),
                        "sparse": models.SparseVector(
                            indices=sparse_vector.indices.tolist(),
                            values=sparse_vector.values.tolist(),
                        ),
                    },
                    payload=_payload(
                        chunk,
                        source=source,
                        title=title,
                        doc_type=doc_type,
                        tenant=tenant,
                        audience=audience,
                        digest=digest,
                    ),
                )
                for chunk, vector, sparse_vector in zip(chunks, dense, sparse_vectors, strict=True)
            ],
        )
        log.info(
            "Indexed %s (%s, %d chunks, tenant=%s, audience=%s)",
            source,
            doc_type,
            len(chunks),
            tenant,
            audience,
        )

    if sources:  # never wipe the collection because of an empty/misconfigured folder
        client.delete(
            config.QDRANT_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(must_not=[models.FieldCondition(key="source", match=models.MatchAny(any=sources))])
            ),
        )
    else:
        log.warning("No PDFs found in %s", data_dir.resolve())

    return hashlib.sha256(("|".join(versions) + "|" + settings_salt()).encode()).hexdigest()[:16]


def _payload(
    chunk: Chunk,
    *,
    source: str,
    title: str,
    doc_type: str,
    tenant: str,
    audience: str,
    digest: str,
) -> dict[str, Any]:
    return {
        "text": chunk["text"],  # clean text (no title/section prefix)
        "source": source,
        "doc": title,
        "doc_type": doc_type,
        "kind": chunk["kind"],  # text | table | faq
        "section": chunk["section"],
        "page": chunk["page_start"],
        "page_end": chunk["page_end"],
        "order": chunk["order"],  # position inside the document (for neighbours)
        "doc_tokens": chunk["doc_tokens"],  # size of the whole document
        "tenant_id": tenant,
        "audience": audience,
        "file_hash": digest,
    }
