"""Application workflow and evidence-only query retrieval."""
from .chunker import chunk_pages
from .llm import answer
from .pdf_processor import process_pdf
from .vector_store import VectorStore


def index_pdf(store: VectorStore, data: bytes, filename: str) -> tuple[str, bool]:
    pdf = process_pdf(data, filename)
    if store.has_document(pdf.document_id):
        return pdf.document_id, False
    store.add_document(pdf.document_id, filename, chunk_pages(pdf.pages))
    return pdf.document_id, True


def ask(store: VectorStore, question: str, document_ids: list[str], history: list[dict] | None = None, count: int = 5) -> tuple[str, list[dict]]:
    # Simple query rewrite for follow-ups keeps original question intact for generation.
    search_query = question
    if history and len(question.split()) < 7:
        prior = next((x.get("content", "") for x in reversed(history) if x.get("role") == "user"), "")
        if prior:
            search_query = f"{prior}\nFollow-up: {question}"
    evidence = store.search(search_query, document_ids, count)
    return answer(question, evidence, history), evidence
