import pytest
import numpy as np

pytestmark = pytest.mark.slow


class TestEmbeddings:
    def test_same_text_same_embedding(self):
        from embeddings.embed_chunks import embed_chunks

        chunk = {
            "id": "test_0",
            "text": "Hello world, this is a test.",
            "metadata": {"type": "text"},
        }
        result_a = embed_chunks([chunk])
        result_b = embed_chunks([chunk])
        np.testing.assert_array_almost_equal(
            result_a[0]["embedding"], result_b[0]["embedding"]
        )

    def test_embedding_dimension(self):
        from embeddings.embed_chunks import embed_chunks

        chunk = {
            "id": "dim_test",
            "text": "Checking embedding dimensions.",
            "metadata": {"type": "text"},
        }
        result = embed_chunks([chunk])
        assert len(result) == 1
        assert len(result[0]["embedding"]) == 384

    def test_empty_text_skipped(self):
        from embeddings.embed_chunks import embed_chunks

        chunks = [
            {"id": "empty", "text": "", "metadata": {"type": "text"}},
            {"id": "valid", "text": "Valid text content.", "metadata": {"type": "text"}},
        ]
        result = embed_chunks(chunks)
        assert len(result) == 1
        assert result[0]["id"] == "valid"

    def test_code_and_text_different_embeddings(self):
        from embeddings.embed_chunks import embed_chunks

        same_text = "x = torch.nn.Linear(10, 5)"
        chunks = [
            {"id": "as_text", "text": same_text, "metadata": {"type": "text"}},
            {"id": "as_code", "text": same_text, "metadata": {"type": "code"}},
        ]
        result = embed_chunks(chunks)
        assert len(result) == 2
        # Different prefixes should produce different embeddings
        assert not np.array_equal(result[0]["embedding"], result[1]["embedding"])
