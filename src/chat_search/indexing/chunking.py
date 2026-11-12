"""Document chunking.

The file-name convention decides the strategy:

* ``*-faqs.pdf``   one chunk per question + answer
* ``*-manual.pdf`` sections; each table is its own chunk (header row repeated if split)
* ``*-policy.pdf`` (and anything else) sections; a short policy becomes one chunk

Pages are joined into one stream, so a section that crosses a page break or sits
around a figure stays together. Figures are dropped; chunks carry page ranges.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import TypedDict

from .. import config
from ..text import est_tokens

log = logging.getLogger(__name__)


class Chunk(TypedDict, total=False):
    text: str
    section: str
    page_start: int
    page_end: int
    kind: str
    order: int
    doc_tokens: int


PAGE_NUM_RE = re.compile(r"^\W*(page\s+)?\d+(\s*(of|/)\s*\d+)?\W*$", re.I)
FIGURE_RE = re.compile(r"^(figure|fig\.?|image|screenshot|diagram|illustration)\s*\d+\w*\s*[.:)\-\u2013\u2014]?", re.I)
PICTURE_RE = re.compile(r"==>\s*picture|intentionally omitted", re.I)
MD_HEAD_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
BOLD_HEAD_RE = re.compile(r"^\*\*(.{3,100}?)\*\*\s*:?$")
NUM_HEAD_RE = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+([A-Z][^\n]{1,99})$")  # "4.2 Data retention"
Q_EXPLICIT_RE = re.compile(r"^Q(?:uestion)?\s*\d*\s*[:.)\-]\s*", re.I)
NUM_PREFIX_RE = re.compile(r"^\d+\s*[.)]\s+")
A_PREFIX_RE = re.compile(r"^A(?:nswer)?\s*\d*\s*[:.)\-]\s*", re.I)
BULLET_RE = re.compile(r"^[-*\u2022]\s+")


def strip_md(text: str) -> str:
    return re.sub(r"[*_`#]+", "", text).strip()


def doc_info(stem: str) -> tuple[str, str]:
    """Return (doc_type, title) from the file name, e.g. 'refund-policy' -> ('policy', 'Refund Policy')."""
    lowered = stem.lower()
    doc_type = "faq" if lowered.endswith(("-faqs", "-faq")) else "manual" if lowered.endswith("-manual") else "policy"
    title = " ".join(word.capitalize() for word in re.split(r"[-_\s]+", stem) if word)
    return doc_type, re.sub(r"\bFaqs?\b", "FAQs", title)


def clean_lines(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Drop page numbers, figure captions/placeholders and headers/footers repeated on most pages."""
    page_count = len(pages)
    counts: Counter[str] = Counter()
    if page_count >= 3:
        for _, text in pages:
            counts.update(
                {
                    line.strip()
                    for line in text.splitlines()
                    if 0 < len(line.strip()) <= 80 and not line.strip().startswith("|")
                }
            )
    repeated = {line for line, count in counts.items() if count >= max(3, int(page_count * 0.6))}

    lines: list[tuple[int, str]] = []
    for page_number, text in pages:
        for raw in text.splitlines():
            line = raw.rstrip()
            stripped = line.strip()
            plain = strip_md(stripped)
            if stripped in repeated or PAGE_NUM_RE.match(plain) or PICTURE_RE.search(stripped):
                continue
            if len(plain) <= 150 and FIGURE_RE.match(plain):  # caption of a dropped figure
                continue
            lines.append((page_number, line))
    return lines


@dataclass
class Block:
    kind: str  # heading | text | table
    text: str
    page_start: int
    page_end: int
    level: int = 0


def parse_heading(line: str) -> tuple[int, str] | None:
    stripped = line.strip()
    if match := MD_HEAD_RE.match(stripped):
        text = strip_md(match[2])
        return (len(match[1]), text) if 0 < len(text) <= 150 else None
    if match := BOLD_HEAD_RE.match(stripped):
        return 4, strip_md(match[1])
    if (match := NUM_HEAD_RE.match(stripped)) and not stripped.endswith((".", ":", ";", ",")):
        return min(match[1].count(".") + 2, 6), f"{match[1]} {match[2]}".strip()
    return None


def to_blocks(lines: list[tuple[int, str]]) -> list[Block]:
    blocks: list[Block] = []
    current: Block | None = None

    def close() -> None:
        nonlocal current
        if current:
            blocks.append(current)
            current = None

    for page, line in lines:
        stripped = line.strip()
        if not stripped:
            close()
            continue
        if heading := parse_heading(line):
            close()
            blocks.append(Block("heading", heading[1], page, page, heading[0]))
            continue
        kind = "table" if stripped.startswith("|") else "text"
        if current and current.kind != kind:
            close()
        if current is None:
            last = blocks[-1] if blocks else None
            # A sentence cut by a page break continues on the next page: stitch it back together.
            if (
                kind == "text"
                and last
                and last.kind == "text"
                and last.page_end < page
                and not last.text.rstrip().endswith((".", "!", "?", ":", ";", ")"))
                and stripped[:1].islower()
            ):
                current = blocks.pop()
                current.text += " " + stripped
                current.page_end = page
                continue
            current = Block(kind, stripped, page, page)
        else:
            current.text += "\n" + stripped
            current.page_end = page
    close()
    return blocks


def word_windows(text: str) -> list[str]:
    words = text.split()
    size = max(int(config.CHUNK_MAX_TOKENS / 1.3), 20)
    step = max(size - int(config.CHUNK_OVERLAP_TOKENS / 1.3), 1)
    out: list[str] = []
    index = 0
    while index < len(words):
        out.append(" ".join(words[index : index + size]))
        if index + size >= len(words):
            break
        index += step
    return out


def split_text(text: str) -> list[str]:
    """Split an oversized block on lines (lists/procedures) or sentences, with a small overlap."""
    if est_tokens(text) <= config.CHUNK_MAX_TOKENS:
        return [text]
    separator = "\n" if "\n" in text else " "
    units = text.split("\n") if separator == "\n" else re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    current: list[str] = []
    for unit in units:
        if est_tokens(unit) > config.CHUNK_MAX_TOKENS:  # one giant line/sentence: hard split by words
            if current:
                pieces.append(separator.join(current))
                current = []
            pieces.extend(word_windows(unit))
            continue
        if current and est_tokens(separator.join(current + [unit])) > config.CHUNK_MAX_TOKENS:
            pieces.append(separator.join(current))
            tail = current[-1]
            current = [tail] if est_tokens(tail) <= config.CHUNK_OVERLAP_TOKENS else []
        current.append(unit)
    if current:
        pieces.append(separator.join(current))
    return pieces


def split_table(text: str) -> list[str]:
    """A table is one chunk; if it is too big, split by rows and repeat the header row."""
    if est_tokens(text) <= config.CHUNK_MAX_TOKENS:
        return [text]
    lines = text.split("\n")
    head = lines[:2] if len(lines) > 2 and re.match(r"^\|?\s*:?-{2,}", lines[1].strip()) else lines[:1]
    pieces: list[str] = []
    current = list(head)
    for row in lines[len(head) :]:
        if len(current) > len(head) and est_tokens("\n".join(current + [row])) > config.CHUNK_MAX_TOKENS:
            pieces.append("\n".join(current))
            current = list(head)
        current.append(row)
    if len(current) > len(head):
        pieces.append("\n".join(current))
    return pieces


def chunk_sectioned(blocks: list[Block], title: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = []  # heading path, e.g. [(2, "2. Eligibility")]
    buffer: list[Block] = []
    buffer_tokens = 0
    buffer_path = ""

    def path() -> str:
        return " > ".join(text for _, text in stack)

    def flush() -> None:
        nonlocal buffer, buffer_tokens
        if buffer:
            chunks.append(
                {
                    "text": "\n\n".join(block.text for block in buffer),
                    "section": buffer_path,
                    "page_start": buffer[0].page_start,
                    "page_end": buffer[-1].page_end,
                    "kind": "text",
                }
            )
        buffer, buffer_tokens = [], 0

    for block in blocks:
        if block.kind == "heading":
            flush()
            if not stack and block.text.lower() == title.lower():
                continue  # the document's own title is not a section
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, block.text))
        elif block.kind == "table":
            flush()
            for piece in split_table(block.text):
                chunks.append(
                    {
                        "text": piece,
                        "section": path(),
                        "page_start": block.page_start,
                        "page_end": block.page_end,
                        "kind": "table",
                    }
                )
        else:
            for piece in split_text(block.text):
                tokens = est_tokens(piece)
                if buffer and buffer_tokens + tokens > config.CHUNK_MAX_TOKENS:
                    flush()
                if not buffer:
                    buffer_path = path()
                buffer.append(Block("text", piece, block.page_start, block.page_end))
                buffer_tokens += tokens
                if buffer_tokens >= config.CHUNK_TOKENS:
                    flush()
    flush()
    return chunks


def common_path(first: str, second: str) -> str:
    """Shared parent heading of two section paths ('' if they only share the document)."""
    keep: list[str] = []
    for left, right in zip(first.split(" > "), second.split(" > "), strict=False):
        if left != right:
            break
        keep.append(left)
    return " > ".join(part for part in keep if part)


def merge_small(chunks: list[Chunk]) -> list[Chunk]:
    """Fold tiny text chunks into the previous one (keeping their heading as a label)."""
    out: list[Chunk] = []
    for chunk in chunks:
        previous = out[-1] if out else None
        if (
            previous
            and chunk["kind"] == "text"
            and previous["kind"] == "text"
            and est_tokens(chunk["text"]) < config.MIN_CHUNK_TOKENS
            and est_tokens(previous["text"]) + est_tokens(chunk["text"]) <= config.CHUNK_MAX_TOKENS + 60
        ):
            label = (
                chunk["section"].split(" > ")[-1]
                if chunk["section"] and chunk["section"] != previous["section"]
                else ""
            )
            previous["text"] += "\n\n" + (label + "\n" if label else "") + chunk["text"]
            previous["page_end"] = max(previous["page_end"], chunk["page_end"])
            previous["section"] = common_path(previous["section"], chunk["section"])
        else:
            out.append(chunk)
    if len(out) > 1:  # a tiny opening section (e.g. "Purpose") has no previous chunk: fold it forward
        first, second = out[0], out[1]
        if (
            first["kind"] == second["kind"] == "text"
            and est_tokens(first["text"]) < config.MIN_CHUNK_TOKENS
            and est_tokens(first["text"]) + est_tokens(second["text"]) <= config.CHUNK_MAX_TOKENS + 60
        ):
            label = first["section"].split(" > ")[-1] if first["section"] else ""
            second["text"] = (label + "\n" if label else "") + first["text"] + "\n\n" + second["text"]
            second["page_start"] = first["page_start"]
            second["section"] = common_path(first["section"], second["section"])
            out.pop(0)
    return out


def collapse_whole(chunks: list[Chunk]) -> list[Chunk]:
    """A very short document stays one chunk."""
    if len(chunks) < 2 or sum(est_tokens(chunk["text"]) for chunk in chunks) > config.WHOLE_DOC_TOKENS:
        return chunks
    parts: list[str] = []
    last_label = ""
    for chunk in chunks:
        label = chunk["section"].split(" > ")[-1] if chunk["section"] else ""
        parts.append((label + "\n" if label and label != last_label else "") + chunk["text"])
        last_label = label
    return [
        {
            "text": "\n\n".join(parts),
            "section": "",
            "page_start": chunks[0]["page_start"],
            "page_end": chunks[-1]["page_end"],
            "kind": "text",
        }
    ]


def question_of(line: str, previous_blank: bool) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("|"):
        return None
    plain = strip_md(stripped)
    if len(plain) > 300:
        return None
    if match := Q_EXPLICIT_RE.match(plain):  # "Q: ...", "Question 3. ..."
        return plain[match.end() :].strip() or None
    numbered = bool(NUM_PREFIX_RE.match(plain))
    heading = bool(MD_HEAD_RE.match(stripped) or BOLD_HEAD_RE.match(stripped))
    if plain.endswith("?") and (numbered or heading or previous_blank):
        return NUM_PREFIX_RE.sub("", plain).strip()
    return None


def chunk_faq(lines: list[tuple[int, str]], title: str) -> list[Chunk]:
    """One chunk per question + answer. Detects 'Q:', numbered, bold/heading and standalone '?' lines."""
    items: list[dict] = []
    group = ""
    current: dict | None = None
    previous_blank = True
    for page, line in lines:
        stripped = line.strip()
        if not stripped:
            previous_blank = True
            continue
        question = question_of(line, previous_blank)
        previous_blank = False
        if question:
            current = {"q": question, "ans": [], "page_start": page, "page_end": page, "group": group}
            items.append(current)
            continue
        if match := MD_HEAD_RE.match(stripped):  # a heading that is not a question starts a new FAQ group
            heading = strip_md(match[2])
            group, current = ("" if heading.lower() == title.lower() else heading), None
            continue
        if current is None:
            continue  # intro text before the first question is ignored
        if not current["ans"]:
            stripped = A_PREFIX_RE.sub("", BULLET_RE.sub("", stripped).lstrip("*_ "), count=1)
        current["ans"].append(stripped)
        current["page_end"] = page

    chunks: list[Chunk] = []
    for item in items:
        answer = "\n".join(item["ans"]).strip()
        if not answer:
            continue
        head = f"Q: {item['q']}\nA: "
        pieces = split_text(answer) if est_tokens(head + answer) > config.CHUNK_MAX_TOKENS else [answer]
        for piece in pieces:
            chunks.append(
                {
                    "text": head + piece,
                    "section": item["group"],
                    "page_start": item["page_start"],
                    "page_end": item["page_end"],
                    "kind": "faq",
                }
            )
    return chunks


def build_chunks(pages: list[tuple[int, str]], doc_type: str, title: str) -> list[Chunk]:
    lines = clean_lines(pages)
    chunks = chunk_faq(lines, title) if doc_type == "faq" else []
    if doc_type == "faq" and not chunks:
        log.warning("No question/answer pairs detected in '%s'; using section chunking", title)
    if not chunks:
        chunks = collapse_whole(merge_small(chunk_sectioned(to_blocks(lines), title)))
    total = sum(est_tokens(chunk["text"]) for chunk in chunks)
    for index, chunk in enumerate(chunks):
        chunk["order"] = index
        chunk["doc_tokens"] = total
    return chunks
