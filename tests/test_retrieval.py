import pytest
from retrieval.retrieve import classify_intent, is_confident, heuristic_is_code


class TestClassifyIntent:
    def test_code_intent(self):
        assert classify_intent("show me code for softmax") == "code"

    def test_text_intent(self):
        assert classify_intent("why is softmax used?") == "text"

    def test_mixed_defaults_text(self):
        assert classify_intent("what is this?") == "text"

    def test_implement_is_code(self):
        assert classify_intent("implement a neural network") == "code"


class TestIsConfident:
    def test_confident_with_low_distance(self):
        results = [{"distance": 0.3, "chunk": {}}]
        assert is_confident(results) is True

    def test_not_confident_with_high_distance(self):
        results = [{"distance": 1.5, "chunk": {}}]
        assert is_confident(results) is False

    def test_not_confident_empty(self):
        assert is_confident([]) is False

    def test_custom_threshold(self):
        results = [{"distance": 0.8, "chunk": {}}]
        assert is_confident(results, threshold=0.5) is False
        assert is_confident(results, threshold=1.0) is True


class TestHeuristicIsCode:
    def test_detects_code(self):
        assert heuristic_is_code("def foo():\n    return x") is True

    def test_rejects_plain_text(self):
        assert heuristic_is_code("This is a normal sentence.") is False

    def test_pytorch_code(self):
        assert heuristic_is_code("model = nn.Linear(10, 5)\ntorch.relu(x)") is True


class TestReranking:
    """Integration-level tests that verify reranking behavior."""

    @pytest.fixture
    def small_store(self):
        from embeddings.embed_chunks import embed_chunks
        from embeddings.vector_store import VectorStore

        chunks = [
            {
                "id": "text_0",
                "text": "Softmax converts logits into probabilities for classification.",
                "metadata": {"type": "text", "post_title": "nn", "section": "intro"},
            },
            {
                "id": "code_0",
                "text": "import torch\nmodel = torch.nn.Sequential(torch.nn.Linear(784, 10), torch.nn.Softmax(dim=1))",
                "metadata": {"type": "code", "post_title": "nn", "section": "Code"},
            },
        ]
        embedded = embed_chunks(chunks)
        store = VectorStore(dim=len(embedded[0]["embedding"]))
        store.add_embeddings(embedded)
        return store

    @pytest.mark.slow
    def test_reranking_boosts_matching_intent(self, small_store):
        from retrieval.retrieve import retrieve

        results = retrieve("show me code for softmax", small_store, k=2)
        assert len(results) > 0
        # Code query should rank the code chunk higher
        assert results[0]["chunk"]["metadata"]["type"] == "code"

    @pytest.mark.slow
    def test_deduplication(self, small_store):
        from retrieval.retrieve import retrieve

        results = retrieve("softmax", small_store, k=5)
        seen_keys = set()
        for r in results:
            key = f"{r['chunk']['metadata']['post_title']}:{r['chunk']['metadata']['section']}"
            assert key not in seen_keys
            seen_keys.add(key)
