from __future__ import annotations

import pytest

from chat_search import config
from chat_search.indexing.chunking import (
    build_chunks,
    chunk_faq,
    chunk_sectioned,
    clean_lines,
    collapse_whole,
    doc_info,
    merge_small,
    parse_heading,
    question_of,
    split_table,
    split_text,
    to_blocks,
    word_windows,
)
from chat_search.text import embed_text, est_tokens


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("refund-policy", ("policy", "Refund Policy")),
        ("member-faqs", ("faq", "Member FAQs")),
        ("member-faq", ("faq", "Member FAQs")),
        ("ops-manual", ("manual", "Ops Manual")),
        ("onboarding", ("policy", "Onboarding")),
    ],
)
def test_doc_info(stem, expected):
    assert doc_info(stem) == expected


def test_clean_lines_drops_pages_figures_and_repeated_headers():
    pages = [
        (1, "Acme Handbook\nPage 1\nIntro line\nFigure 1: Architecture"),
        (2, "Acme Handbook\nPage 2\nBody line\n==> picture"),
        (3, "Acme Handbook\nPage 3\nMore body"),
    ]

    lines = clean_lines(pages)
    texts = [text.strip() for _, text in lines]

    assert "Acme Handbook" not in texts
    assert "Page 1" not in texts
    assert "Page 2" not in texts
    assert "Figure 1: Architecture" not in texts
    assert "==> picture" not in texts
    assert {"Intro line", "Body line", "More body"} <= set(texts)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("## 2. Eligibility", (2, "2. Eligibility")),
        ("**Note**", (4, "Note")),
        ("4.2 Data retention", (3, "4.2 Data retention")),
        ("plain text", None),
        ("## ", None),
    ],
)
def test_parse_heading(line, expected):
    assert parse_heading(line) == expected


def test_to_blocks_stitches_a_sentence_across_a_page_break():
    blocks = to_blocks([(1, "The policy applies to all"), (1, ""), (2, "members of the platform.")])

    assert len(blocks) == 1
    assert blocks[0].text == "The policy applies to all members of the platform."
    assert blocks[0].page_end == 2


def test_word_windows_hard_splits_a_giant_unit(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_MAX_TOKENS", 30)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_TOKENS", 5)

    windows = word_windows(" ".join(f"w{index}" for index in range(100)))

    assert len(windows) > 1
    assert all(len(window.split()) <= 23 for window in windows)
    assert windows[0].startswith("w0")


def test_split_text_keeps_small_blocks_intact():
    text = "A short policy sentence."

    assert split_text(text) == [text]


def test_split_text_splits_long_blocks(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_MAX_TOKENS", 60)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_TOKENS", 10)

    sentence = "This is one policy sentence with several words."
    text = " ".join([sentence] * 40)

    pieces = split_text(text)

    assert len(pieces) > 1
    assert pieces[0].startswith("This is one policy")
    assert config.CHUNK_MAX_TOKENS + config.CHUNK_OVERLAP_TOKENS + 10 >= max(est_tokens(piece) for piece in pieces)


def test_split_table_repeats_the_header(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_MAX_TOKENS", 40)
    lines = ["| Name | City |", "| --- | --- |"] + [f"| Person {index} | City {index} |" for index in range(60)]

    pieces = split_table("\n".join(lines))

    assert len(pieces) > 1
    for piece in pieces:
        assert piece.splitlines()[0] == "| Name | City |"
        assert piece.splitlines()[1] == "| --- | --- |"


@pytest.mark.parametrize(
    ("line", "previous_blank", "expected"),
    [
        ("Q: How do I pay?", True, "How do I pay?"),
        ("Question 3. Can I get a refund?", False, "Can I get a refund?"),
        ("2. Where is my invoice?", False, "Where is my invoice?"),
        ("How do I reset my password?", True, "How do I reset my password?"),
        ("Just text.", True, None),
        ("", True, None),
        ("| table |", True, None),
    ],
)
def test_question_of(line, previous_blank, expected):
    assert question_of(line, previous_blank) == expected


def test_chunk_faq_builds_one_chunk_per_question():
    lines = [
        (1, "## Billing"),
        (1, "Q: How do I pay?"),
        (1, "A: Use the portal."),
        (1, ""),
        (1, "Q: Can I get a refund?"),
        (1, "A: Yes, within 14 days."),
    ]

    chunks = chunk_faq(lines, "Member FAQs")

    assert len(chunks) == 2
    assert chunks[0] == {
        "text": "Q: How do I pay?\nA: Use the portal.",
        "section": "Billing",
        "page_start": 1,
        "page_end": 1,
        "kind": "faq",
    }
    assert chunks[1]["text"] == "Q: Can I get a refund?\nA: Yes, within 14 days."


def test_build_chunks_assigns_order_and_document_size():
    pages = [
        (
            1,
            "Q: How do I pay?\nA: Use the portal.\n\nQ: Can I get a refund?\nA: Yes, within 14 days.",
        )
    ]

    chunks = build_chunks(pages, "faq", "Member FAQs")

    assert [chunk["order"] for chunk in chunks] == [0, 1]
    assert all(chunk["doc_tokens"] > 0 for chunk in chunks)
    assert all(chunk["kind"] == "faq" for chunk in chunks)


def test_chunk_sectioned_builds_heading_paths():
    pages = [
        (
            1,
            "## Purpose\nWe help members use the platform.\n"
            "## Eligibility\nMembers need an active account and a verified email address.",
        )
    ]

    chunks = chunk_sectioned(to_blocks(clean_lines(pages)), "Policy")

    assert [chunk["section"] for chunk in chunks] == ["Purpose", "Eligibility"]
    assert all(chunk["kind"] == "text" for chunk in chunks)


def test_merge_small_folds_a_tiny_section_into_the_previous_chunk(monkeypatch):
    monkeypatch.setattr(config, "MIN_CHUNK_TOKENS", 50)
    monkeypatch.setattr(config, "CHUNK_MAX_TOKENS", 200)
    chunks = [
        {
            "text": "Parent section text with enough words to stay small.",
            "section": "Parent > First",
            "page_start": 1,
            "page_end": 1,
            "kind": "text",
        },
        {
            "text": "Second section text that is also small.",
            "section": "Parent > Second",
            "page_start": 1,
            "page_end": 1,
            "kind": "text",
        },
    ]

    merged = merge_small(chunks)

    assert len(merged) == 1
    assert merged[0]["section"] == "Parent"
    assert "Second" in merged[0]["text"]


def test_collapse_whole_keeps_a_short_document_in_one_chunk(monkeypatch):
    monkeypatch.setattr(config, "WHOLE_DOC_TOKENS", 1000)
    chunks = [
        {
            "text": "First part.",
            "section": "Purpose",
            "page_start": 1,
            "page_end": 1,
            "kind": "text",
        },
        {
            "text": "Second part.",
            "section": "Scope",
            "page_start": 2,
            "page_end": 2,
            "kind": "text",
        },
    ]

    collapsed = collapse_whole(chunks)

    assert len(collapsed) == 1
    assert collapsed[0]["section"] == ""
    assert collapsed[0]["page_start"] == 1
    assert collapsed[0]["page_end"] == 2
    assert "Purpose" in collapsed[0]["text"]
    assert "Scope" in collapsed[0]["text"]


def test_build_chunks_collapses_a_short_policy():
    pages = [
        (
            1,
            "## Purpose\nWe help members use the platform.\n## Eligibility\nMembers need an active account.",
        )
    ]

    chunks = build_chunks(pages, "policy", "Policy")

    assert len(chunks) == 1
    assert "We help members" in chunks[0]["text"]
    assert "Eligibility" in chunks[0]["text"]


def test_embed_text_prefixes_title_and_section():
    chunk = {"text": "Members only.", "section": "Eligibility"}

    assert embed_text("Policy", chunk) == "Policy > Eligibility\nMembers only."
    assert embed_text("Policy", {"text": "Body"}) == "Policy\nBody"
