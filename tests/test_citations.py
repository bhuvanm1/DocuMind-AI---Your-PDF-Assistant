from unittest.mock import patch
from rag.llm import answer
from rag.pipeline import ask


class FakeStore:
    def search(self, question, document_ids, count):
        assert document_ids == ["only-this"]
        return [{"id": "only-this:0", "text": "The paper reports 94% accuracy.", "page": 3, "filename": "paper.pdf", "distance": .1}]


def test_retrieval_is_limited_to_selected_docs_and_source_metadata():
    with patch("rag.pipeline.answer", return_value="94% [only-this:0]") as generator:
        text, evidence = ask(FakeStore(), "What accuracy?", ["only-this"])
    assert text == "94% [only-this:0]"
    assert evidence[0]["page"] == 3 and evidence[0]["filename"] == "paper.pdf"
    generator.assert_called_once()


def test_missing_evidence_behavior():
    assert "could not find sufficient evidence" in answer("Unknown?", [])


def test_prompt_carries_exact_citation_id_and_evidence():
    evidence = [{"id": "doc:2", "page": 9, "filename": "study.pdf", "text": "CNN was used."}]
    with patch("rag.llm.requests.post") as post:
        post.return_value.raise_for_status.return_value = None
        post.return_value.json.return_value = {"message": {"content": "CNN [doc:2]"}}
        result = answer("Which model?", evidence)
    prompt = post.call_args.kwargs["json"]["messages"][-1]["content"]
    assert "[doc:2] (page 9; study.pdf)" in prompt
    assert "CNN was used." in prompt
    assert result == "CNN [doc:2]"
