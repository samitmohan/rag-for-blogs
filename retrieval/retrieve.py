"""
Multi-stage retrieval pipeline:
1. Embed query
2. FAISS top-K search
3. Intent-aware reranking with lexical boosting
4. Deduplication and confidence check
"""

import logging

from config import CONFIDENCE_THRESHOLD
from embeddings.embed_chunks import embed_chunks
from embeddings.vector_store import VectorStore

logger = logging.getLogger(__name__)


def is_confident(retrieved, threshold=None):
    if threshold is None:
        threshold = CONFIDENCE_THRESHOLD
    if not retrieved:
        return False
    best_distance = retrieved[0]["distance"]
    return best_distance < threshold


def classify_intent(query):
    query_lower = query.lower()
    code_indicators = ["code", "implement", "example", "sample", "snippet"]
    text_indicators = ["why", "explain", "intuition", "understand", "describe"]

    code_score = sum(1 for word in code_indicators if word in query_lower)
    text_score = sum(1 for word in text_indicators if word in query_lower)

    if code_score > text_score:
        return "code"
    return "text"


def heuristic_is_code(text):
    indicators = [
        " = ", "def ", "class ", "import ", "return ", "print(",
        "torch.", "nn.", "->", "{", "}", "plt.", "np.",
    ]
    count = sum(1 for i in indicators if i in text)
    return count >= 2


def retrieve(query: str, store: VectorStore, k: int = 5):
    intent = classify_intent(query)
    query_chunk = {
        "id": "query_0",
        "text": query,
        "metadata": {"type": "text"},
    }
    query_embedding = embed_chunks([query_chunk])[0]["embedding"]

    candidates = store.search(query_embedding, top_k=10)

    reranked = []
    for cand in candidates:
        distance = cand["distance"]
        chunk = cand["chunk"]

        chunk_type = chunk["metadata"].get("type", "text")
        text = chunk["text"]

        # Heuristic: treat code-like text as code
        effective_chunk_type = chunk_type
        if chunk_type == "text" and heuristic_is_code(text):
            effective_chunk_type = "code"

        if effective_chunk_type == intent:
            distance *= 0.5
        else:
            distance *= 2.0

        # Section-based boosting for code
        section_name = chunk["metadata"].get("section", "").lower()
        if intent == "code" and "code" in section_name:
            distance *= 0.8

        # Hybrid Search: Lexical Overlap Boosting
        query_words = set(w.lower() for w in query.split() if len(w) > 2)
        if query_words:
            chunk_text_lower = text.lower()
            matched_words = sum(1 for word in query_words if word in chunk_text_lower)
            coverage_ratio = matched_words / len(query_words)

            if coverage_ratio > 0:
                lexical_boost = 1.0 - (0.3 * coverage_ratio)
                distance *= lexical_boost

                if query.lower() in chunk_text_lower:
                    distance *= 0.8

        if len(text.split()) < 5:
            continue

        reranked.append({"distance": distance, "chunk": chunk})

    reranked.sort(key=lambda x: x["distance"])

    # Deduplicate: keep only first occurrence of (post_title + section)
    seen = set()
    final_results = []
    for item in reranked:
        c = item["chunk"]
        key = f"{c['metadata'].get('post_title')}:{c['metadata'].get('section')}"
        if key not in seen:
            seen.add(key)
            final_results.append(item)

    return final_results[:k]
