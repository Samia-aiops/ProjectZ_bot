"""PDF text extraction."""

from __future__ import annotations

from pathlib import Path


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Digital PDF -> markdown per page. Headers/footers, images and OCR are off."""
    import pymupdf4llm

    parts = pymupdf4llm.to_markdown(
        str(path),
        page_chunks=True,
        write_images=False,
        ignore_images=True,
        header=False,
        footer=False,
        use_ocr=False,
    )
    return [(number, part["text"]) for number, part in enumerate(parts, start=1)]
