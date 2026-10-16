"""Paragraph-aware chunking that retains source page metadata."""
from __future__ import annotations

from dataclasses import dataclass
from .pdf_processor import PageText


@dataclass(frozen=True)
class TextChunk:
    text: str
    page: int
    chunk_index: int


def chunk_pages(pages: tuple[PageText, ...] | list[PageText], chunk_size: int = 1000, overlap: int = 150) -> list[TextChunk]:
    if chunk_size < 100 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be >= 100 and overlap must be between 0 and chunk_size")
    output: list[TextChunk] = []
    for page in pages:
        text = "\n\n".join(part.strip() for part in page.text.split("\n\n") if part.strip())
        if not text:
            continue
        # Prefer paragraph breaks near the target end; character overlap keeps
        # retrieval context connected even when a paragraph crosses a boundary.
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind("\n\n", start + chunk_size // 2, end)
                if boundary > start:
                    end = boundary
            window = text[start:end].strip()
            if window:
                output.append(TextChunk(window, page.page, len(output)))
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return output
