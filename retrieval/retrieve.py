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

# Add the project root to sys.path to allow importing from 'embeddings'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

        if chunk_type == intent:
            distance *= 0.85   # boost (closer)
        else:
            distance *= 1.15   # penalize (farther)

        # Keyword boosting
        query_terms = [w for w in query.lower().split() if len(w) > 3]
        chunk_lower = text.lower()
        for term in query_terms:
            if term in chunk_lower:
                distance *= 0.95  # slightly boost for every keyword match

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