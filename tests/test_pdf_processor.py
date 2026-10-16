import fitz
import pytest
from rag.pdf_processor import PDFProcessingError, process_pdf


def make_pdf():
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "A test paper about a model.")
    doc.new_page().insert_text((72, 72), "Results are on this page.")
    return doc.tobytes()


def test_extraction_preserves_one_based_pages():
    pdf = process_pdf(make_pdf(), "paper.pdf")
    assert [p.page for p in pdf.pages] == [1, 2]
    assert "Results" in pdf.pages[1].text
    assert len(pdf.document_id) == 64


def test_invalid_and_empty_scanned_pdf():
    with pytest.raises(PDFProcessingError):
        process_pdf(b"not a pdf", "bad.pdf")
    blank = fitz.open(); blank.new_page()
    with pytest.raises(PDFProcessingError, match="OCR"):
        process_pdf(blank.tobytes(), "scan.pdf")


def test_reject_non_pdf_extension():
    with pytest.raises(PDFProcessingError, match="PDF"):
        process_pdf(make_pdf(), "paper.txt")
