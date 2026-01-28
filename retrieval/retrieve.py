"""
Docstring for retrieval.retrieve
Responsibilities:

- Embed query
- Retrieve top-K from FAISS
- Re-rank using metadata
- Return a small, intentional context set

Step A: Embed query text 

contains "code", "implement", "example" → prefer type=code
contains "why", "explain", "intuition" → prefer type=text

Step B: Retrieve top-K from FAISS
candidates = faiss.search(query_embedding, k=10)

Step C: re-rank by intent

boost score if chunk.metadata["type"] matches intent
penalize mismatches
drop chunks that don't stand alone

def retrieve(query: str, store: FaissVectorStore, k: int = 5):
    intent = classify_intent(query)
    query_embedding = embed(query)

    candidates = store.search(query_embedding, k=10)

    reranked = rerank(candidates, intent)

    return reranked[:k]
"""

import sys
import os

from embeddings.embed_chunks import embed_chunks
from embeddings.vector_store import VectorStore

def is_confident(retrieved, threshold=1.0):
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
    indicators = [" = ", "def ", "class ", "import ", "return ", "print(", "torch.", "nn.", "->", "{", "}", "plt.", "np."]
    count = sum(1 for i in indicators if i in text)
    return count >= 2
    
def retrieve(query: str, store: VectorStore, k: int = 5):
    intent = classify_intent(query)
    query_chunk = {
        "id": "query_0",
        "text": query,
        "metadata": {"type": "text"}
    }
    query_embedding = embed_chunks([query_chunk])[0]["embedding"]

    candidates = store.search(query_embedding, top_k=10)

    reranked = []
    for cand in candidates:
        distance = cand["distance"]  # LOWER is better
        chunk = cand["chunk"]

        chunk_type = chunk["metadata"].get("type", "text")
        text = chunk["text"]

        # Heuristic: treat code-like text as code
        effective_chunk_type = chunk_type
        if chunk_type == "text" and heuristic_is_code(text):
            effective_chunk_type = "code"

        if effective_chunk_type == intent:
            distance *= 0.5   # strong boost (closer)
        else:
            distance *= 2.0   # strong penalty (farther)
        
        # Section-based boosting for code
        section_name = chunk["metadata"].get("section", "").lower()
        if intent == "code" and "code" in section_name:
             distance *= 0.8

        # Hybrid Search: Lexical Overlap Boosting
        # We calculate how many unique query terms are present in the chunk.
        query_words = set([w.lower() for w in query.split() if len(w) > 2])
        if query_words:
            chunk_text_lower = text.lower()
            matched_words = sum(1 for word in query_words if word in chunk_text_lower)
            
            # Coverage ratio: what % of query words were found?
            coverage_ratio = matched_words / len(query_words)
            
            if coverage_ratio > 0:
                # Stronger boost for higher coverage
                # If coverage is 100%, distance is multiplied by 0.7
                # If coverage is 50%, distance is multiplied by 0.85
                lexical_boost = 1.0 - (0.3 * coverage_ratio)
                distance *= lexical_boost
                
                # Bonus: Exact phrase match boost
                if query.lower() in chunk_text_lower:
                    distance *= 0.8

        if len(text.split()) < 5: continue

        reranked.append({ "distance": distance, "chunk": chunk })

    # Sort by best (lowest) distance
    reranked.sort(key=lambda x: x["distance"])

    # Deduplicate: Keep only first occurrence of (post_title + section)
    seen = set()
    final_results = []
    
    for item in reranked:
        c = item["chunk"]
        # unique key based on file/title + section
        key = f"{c['metadata'].get('post_title')}:{c['metadata'].get('section')}"
        if key not in seen:
            seen.add(key)
            final_results.append(item)

    return final_results[:k]

if __name__ == "__main__":
    vector_store = VectorStore(dim=384)

    chunks = [
        {
            "id": "intro_0",
            "text": "Softmax is used in MNIST classification to convert logits into probabilities.",
            "metadata": {"type": "text"}
        },
        {
            "id": "code_0",
            "text": "model = nn.Sequential(nn.Linear(784, 10), nn.Softmax(dim=1))",
            "metadata": {"type": "code"}
        }
    ]

    embedded = embed_chunks(chunks)
    vector_store.add_embeddings(embedded)

    query = "Why do we use softmax in MNIST?"
    results = retrieve(query, vector_store, k=2)

    for r in results:
        print(
            f"Distance: {r['distance']:.4f}, "
            f"ID: {r['chunk']['id']}, "
            f"Type: {r['chunk']['metadata']['type']}"
        )