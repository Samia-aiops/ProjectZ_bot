from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("qdrant_client")

from chat_search import config  # noqa: E402
from chat_search.auth import Principal  # noqa: E402
from chat_search.retrieval import document_entry, expand_context, rbac_filter, rerank  # noqa: E402


def principal(role: str = "patient") -> Principal:
    return Principal(
        subject="user-1",
        role=role,
        tenant_id="acme",
        patient_id="patient-1" if role == "patient" else None,
        provider_id="provider-1" if role == "provider" else None,
        permissions=frozenset({"docs:read"}),
    )


def test_rbac_filter_uses_tenant_and_role_audience():
    query_filter = rbac_filter(principal())

    assert query_filter.must is not None
    assert len(query_filter.must) == 2
    tenant, audience = query_filter.must
    assert tenant.key == "tenant_id"
    assert set(tenant.match.any) == {"acme", config.GLOBAL_TENANT}
    assert audience.key == "audience"
    assert set(audience.match.any) == {"patient", "shared"}


def test_rbac_filter_for_providers():
    audience = rbac_filter(principal("provider")).must[1]

    assert set(audience.match.any) == {"provider", "shared"}


def test_document_entry_keeps_reading_order_and_page_range():
    chunks = [
        {
            "doc": "Policy",
            "section": "Eligibility",
            "text": "Members only.",
            "page": 1,
            "page_end": 1,
            "order": 0,
        },
        {
            "doc": "Policy",
            "section": "Eligibility",
            "text": "Active account.",
            "page": 1,
            "page_end": 2,
            "order": 1,
        },
        {
            "doc": "Policy",
            "section": "",
            "text": "Footer.",
            "page": 2,
            "page_end": 2,
            "order": 2,
        },
    ]

    entry = document_entry(chunks)

    assert entry == {
        "doc": "Policy",
        "page": "1-2",
        "text": "[Eligibility]\nMembers only.\nActive account.\nFooter.",
    }


async def test_rerank_without_a_model_returns_the_top_k(monkeypatch):
    monkeypatch.setattr(
        "chat_search.retrieval.get_runtime",
        lambda: SimpleNamespace(reranker=None),
    )
    chunks = [
        {"doc": "D", "section": "", "text": str(index), "page": 1, "page_end": 1} for index in range(config.TOP_K + 3)
    ]

    ranked = await rerank("question", chunks)

    assert ranked == chunks[: config.TOP_K]


async def test_expand_context_reads_documents_concurrently(monkeypatch):
    started = asyncio.Event()
    active = 0
    peak = 0

    async def fake_fetch(principal, conditions, max_points):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 2:
            started.set()
        await asyncio.wait_for(started.wait(), 0.5)  # both reads must be in flight
        active -= 1
        return []

    def chunk(source: str, score: float) -> dict:
        return {
            "source": source,
            "doc": source,
            "section": "",
            "text": "text",
            "page": 1,
            "page_end": 1,
            "order": 0,
            "doc_tokens": 500,
            "kind": "doc",
            "_score": score,
        }

    monkeypatch.setattr("chat_search.retrieval.fetch_chunks", fake_fetch)

    documents = await expand_context(principal(), [chunk("A", 5.0), chunk("B", 5.0)])

    assert peak == 2
    assert [document["doc"] for document in documents] == ["A", "B"]
