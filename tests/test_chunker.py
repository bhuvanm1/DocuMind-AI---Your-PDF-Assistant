from rag.chunker import chunk_pages
from rag.pdf_processor import PageText


def test_chunking_has_overlap_and_pages():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_pages([PageText(7, text)], chunk_size=300, overlap=60)
    assert len(chunks) > 1
    assert all(c.page == 7 and len(c.text) <= 300 for c in chunks)
    assert chunks[0].text[-30:] in chunks[1].text


def test_empty_page_skipped():
    assert chunk_pages([PageText(1, "  ")]) == []
