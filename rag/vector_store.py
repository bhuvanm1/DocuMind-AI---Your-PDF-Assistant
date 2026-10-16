"""Persistent Chroma storage using its local default embedding function."""
from __future__ import annotations

import chromadb
from chromadb.config import Settings
from .chunker import TextChunk


class VectorStore:
    def __init__(self, path: str = "./.chroma") -> None:
        self.client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection("documind_chunks", metadata={"hnsw:space": "cosine"})

    def has_document(self, document_id: str) -> bool:
        return bool(self.collection.get(where={"document_id": document_id}, limit=1)["ids"])

    def add_document(self, document_id: str, filename: str, chunks: list[TextChunk]) -> None:
        if not chunks:
            raise ValueError("No text chunks were produced from this PDF.")
        # Content hash makes repeated uploads idempotent, while IDs remain document scoped.
        ids = [f"{document_id}:{i}" for i in range(len(chunks))]
        self.collection.upsert(
            ids=ids,
            documents=[chunk.text for chunk in chunks],
            metadatas=[{"document_id": document_id, "filename": filename, "page": chunk.page, "chunk_index": i} for i, chunk in enumerate(chunks)],
        )

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})

    def list_documents(self) -> list[dict]:
        result = self.collection.get(include=["metadatas"])
        docs: dict[str, dict] = {}
        for meta in result.get("metadatas", []):
            docs.setdefault(meta["document_id"], {"document_id": meta["document_id"], "filename": meta["filename"]})
        return list(docs.values())

    def search(self, question: str, document_ids: list[str], count: int = 5) -> list[dict]:
        if not document_ids:
            return []
        result = self.collection.query(query_texts=[question], n_results=count, where={"document_id": {"$in": document_ids}})
        hits = []
        for i, text in enumerate(result["documents"][0]):
            hits.append({"id": result["ids"][0][i], "text": text, **result["metadatas"][0][i], "distance": result["distances"][0][i]})
        return hits
