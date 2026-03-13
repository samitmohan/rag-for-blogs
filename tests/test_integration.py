"""
Integration test: ingest synthetic docs, build index, query, verify results.
No LLM call required.
"""

import pytest

pytestmark = pytest.mark.slow


class TestRetrievePipeline:
    def test_end_to_end_retrieve(self):
        """Ingest synthetic docs, build FAISS index, query, verify relevant results."""
        from embeddings.embed_chunks import embed_chunks
        from embeddings.vector_store import VectorStore
        from ingestion.chunker import chunk_parsed_cleaned_document
        from retrieval.retrieve import retrieve

        # Synthetic documents
        docs = [
            {
                "metadata": {
                    "post_title": "Understanding Backpropagation",
                    "date": "2025-01-01",
                    "url": "/test/backprop",
                },
                "sections": [
                    {
                        "section": "intro",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Backpropagation is the algorithm used to compute "
                                    "gradients in neural networks. It applies the chain "
                                    "rule of calculus to propagate errors backward through "
                                    "the network layers."
                                ),
                            },
                        ],
                    },
                ],
            },
            {
                "metadata": {
                    "post_title": "Python Tips and Tricks",
                    "date": "2025-02-01",
                    "url": "/test/python",
                },
                "sections": [
                    {
                        "section": "intro",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Python list comprehensions provide a concise way "
                                    "to create lists. They are often faster than "
                                    "traditional for loops."
                                ),
                            },
                        ],
                    },
                ],
            },
            {
                "metadata": {
                    "post_title": "Sorting Algorithms",
                    "date": "2025-03-01",
                    "url": "/test/sorting",
                },
                "sections": [
                    {
                        "section": "quicksort",
                        "content": [
                            {
                                "type": "code",
                                "content": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    return quicksort([x for x in arr[1:] if x < pivot]) + [pivot] + quicksort([x for x in arr[1:] if x >= pivot])",
                                "language": "python",
                            },
                        ],
                    },
                ],
            },
        ]

        # Chunk all docs
        all_chunks = []
        for doc in docs:
            all_chunks.extend(chunk_parsed_cleaned_document(doc))

        assert len(all_chunks) >= 3

        # Embed and build index
        embedded = embed_chunks(all_chunks)
        store = VectorStore(dim=len(embedded[0]["embedding"]))
        store.add_embeddings(embedded)

        # Query: should find backpropagation doc
        results = retrieve("How does backpropagation work?", store, k=3)
        assert len(results) > 0
        top_title = results[0]["chunk"]["metadata"]["post_title"]
        assert top_title == "Understanding Backpropagation"

        # Query: should find Python doc
        results = retrieve("Python list comprehensions", store, k=3)
        assert len(results) > 0
        top_title = results[0]["chunk"]["metadata"]["post_title"]
        assert top_title == "Python Tips and Tricks"

    def test_confidence_filtering(self):
        """Queries far from any document content should yield low confidence."""
        from embeddings.embed_chunks import embed_chunks
        from embeddings.vector_store import VectorStore
        from retrieval.retrieve import retrieve, is_confident

        chunks = [
            {
                "id": "test_0",
                "text": "[Section: intro] Neural networks learn through gradient descent optimization.",
                "metadata": {
                    "type": "text",
                    "post_title": "ML Basics",
                    "section": "intro",
                },
            },
        ]

        embedded = embed_chunks(chunks)
        store = VectorStore(dim=len(embedded[0]["embedding"]))
        store.add_embeddings(embedded)

        # Very unrelated query
        results = retrieve("How to bake chocolate chip cookies?", store, k=1)
        # With a single very unrelated chunk, confidence should be low
        # (distance will be high since normalized L2)
        if results:
            assert results[0]["distance"] > 0.3
