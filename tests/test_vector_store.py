from rag.vector_store import VectorStore


class RecordingCollection:
    def query(self, **kwargs):
        self.kwargs = kwargs
        return {"ids": [["doc-a:0"]], "documents": [["evidence"]], "metadatas": [[{"document_id": "doc-a", "filename": "a.pdf", "page": 2, "chunk_index": 0}]], "distances": [[0.1]]}


def test_retrieval_filter_and_metadata_preservation():
    store = VectorStore.__new__(VectorStore)
    store.collection = RecordingCollection()
    hits = store.search("question", ["doc-a", "doc-b"], count=4)
    assert store.collection.kwargs["where"] == {"document_id": {"$in": ["doc-a", "doc-b"]}}
    assert store.collection.kwargs["n_results"] == 4
    assert hits[0]["page"] == 2 and hits[0]["filename"] == "a.pdf"


class RecordingUpsertCollection:
    def upsert(self, **kwargs):
        self.kwargs = kwargs


def test_distinct_content_ids_are_isolated():
    from rag.chunker import TextChunk
    store = VectorStore.__new__(VectorStore)
    store.collection = RecordingUpsertCollection()
    store.add_document("hash-a", "paper.pdf", [TextChunk("A", 1, 0)])
    first = list(store.collection.kwargs["ids"])
    store.add_document("hash-b", "paper.pdf", [TextChunk("B", 1, 0)])
    second = list(store.collection.kwargs["ids"])
    assert first == ["hash-a:0"] and second == ["hash-b:0"]
