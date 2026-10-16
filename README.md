# DocuMind AI — Your PDF Research Assistant

A local, beginner-friendly RAG chatbot for asking questions about research papers and other PDFs. It extracts page-aware text with PyMuPDF, indexes chunks in persistent ChromaDB using its local default embedding function, retrieves evidence only from selected documents, and asks a local Ollama `qwen2.5:3b` model to draft a cited answer. No PDF text is sent to an external service and no paid API key is needed.

## Architecture and workflow

`app.py` provides the Streamlit interface. `rag/pdf_processor.py` validates and extracts page text; `rag/chunker.py` creates overlapping paragraph-aware chunks; `rag/vector_store.py` stores and searches local embeddings; `rag/pipeline.py` ties indexing and retrieval together; `rag/llm.py` communicates with Ollama on localhost. Chroma persists in `.chroma/` beside the app. PDFs are identified by a SHA-256 content hash, and query filters restrict retrieval to selected IDs. The generator receives retrieved passages and the question, with instructions to treat PDF text as untrusted and cite only the supplied source IDs.

## Prerequisites (Windows)

- Python 3.11 (64-bit) or newer, installed from [python.org](https://www.python.org/downloads/windows/). Select “Add Python to PATH”.
- [Ollama for Windows](https://ollama.com/download/windows). Install and start Ollama.

Open PowerShell in this project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull qwen2.5:3b
python -m streamlit run app.py
```

If script activation is blocked, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` for that PowerShell session, then activate again. Ollama must remain running locally. On first run ChromaDB may download its default embedding model; that model runs locally after setup.

## Use

Upload PDFs from the sidebar, select which documents to search, then enter a question. Use the retrieved evidence expander to inspect actual passages and page numbers. The retrieval count is configurable. Delete a document from the library controls or clear the conversation with the sidebar button.

Example questions:

- Which model or algorithm is used?
- What dataset was used?
- What accuracy was achieved?
- Explain the proposed methodology.
- What are the main findings?
- What limitations are mentioned?

## Tests

```powershell
pytest -q
```

The tests mock Ollama and do not need a running model. They cover page extraction, page numbering, chunk overlap, malformed/scanned input, selected-document filtering, citation metadata, and missing evidence.

## Troubleshooting

- **Ollama unavailable:** install Ollama, start the Ollama app/service, then reload Streamlit.
- **Model missing:** run `ollama pull qwen2.5:3b` and reload.
- **Slow answer or timeout:** wait for the model to load, close other GPU-heavy tasks, or retry.
- **No text found:** the PDF is likely scanned. Run OCR with a trusted local OCR tool to make it searchable, then upload the result.
- **Dependency setup fails:** use 64-bit Python 3.11 and recreate `.venv`, then run the install commands again.
- **Reset local index:** stop Streamlit and remove the project’s `.chroma` folder; this permanently removes its saved document index.

## Limitations and privacy

Scanned PDFs need OCR, which is not built in. Very large documents are limited to 20 MB and 5,000 pages. Local models can misread evidence or hallucinate; verify answers against cited excerpts. ChromaDB's default local embedder may need a one-time model download. PDF content is kept in the local Chroma index and sent only to the Ollama service on this computer. No paid API key or cloud LLM is required.
