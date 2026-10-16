"""Safe, page-aware PDF text extraction."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import BinaryIO

import fitz


class PDFProcessingError(ValueError):
    """Raised when a PDF cannot be used for text retrieval."""


@dataclass(frozen=True)
class PageText:
    page: int  # original, one-based PDF page number
    text: str


@dataclass(frozen=True)
class ProcessedPDF:
    document_id: str
    filename: str
    pages: tuple[PageText, ...]


def process_pdf(data: bytes, filename: str, max_pages: int = 5000) -> ProcessedPDF:
    if not filename.lower().endswith(".pdf"):
        raise PDFProcessingError("Please upload a PDF file.")
    if not data:
        raise PDFProcessingError("The uploaded file is empty.")
    doc_id = hashlib.sha256(data).hexdigest()
    try:
        with fitz.open(stream=data, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise PDFProcessingError("This PDF is password-protected. Please upload an unlocked copy.")
            if pdf.page_count == 0:
                raise PDFProcessingError("This PDF has no pages.")
            if pdf.page_count > max_pages:
                raise PDFProcessingError(f"This PDF has too many pages (limit: {max_pages}).")
            pages = tuple(PageText(i + 1, pdf[i].get_text("text").strip()) for i in range(pdf.page_count))
    except PDFProcessingError:
        raise
    except (fitz.FileDataError, RuntimeError, ValueError) as exc:
        raise PDFProcessingError("Could not read this PDF. It may be malformed or encrypted.") from exc
    if not any(page.text for page in pages):
        raise PDFProcessingError("No selectable text was found. This may be a scanned PDF; OCR is not included, so run OCR and upload a searchable PDF.")
    return ProcessedPDF(doc_id, filename, pages)
