"""
FAISS vector store with persistence.
"""

import json
import logging
import os

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    @staticmethod
    def _normalize(vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def add_embeddings(self, embedded_chunks):
        vectors = []
        for chunk in embedded_chunks:
            vector = np.array(chunk["embedding"], dtype="float32")
            vector = self._normalize(vector)
            vectors.append(vector)
            self.metadata.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
            })
        vectors_np = np.vstack(vectors)
        self.index.add(vectors_np)

    def search(self, query_embedding, top_k=5):
        query_vector = np.array(query_embedding, dtype="float32")
        query_vector = self._normalize(query_vector).reshape(1, -1)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "distance": float(dist),
                "chunk": self.metadata[idx],
            })
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(self.metadata, f)
        logger.info("Saved vector store to %s (%d vectors)", path, self.index.ntotal)

    @classmethod
    def load(cls, path: str):
        index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        store = cls(dim=index.d)
        store.index = index
        store.metadata = metadata
        return store
