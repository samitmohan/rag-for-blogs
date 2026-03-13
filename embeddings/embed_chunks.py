"""
Load the embedding model (once)
Embed chunk text only
Preserve metadata untouched
Return embedding-ready records
"""

import logging

from fastembed import TextEmbedding

from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_CACHED_MODEL = None


def _get_model():
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        _CACHED_MODEL = TextEmbedding(model_name=EMBEDDING_MODEL)
        logger.info("Loaded embedding model: %s", EMBEDDING_MODEL)
    return _CACHED_MODEL


def _prepare_embedding_text(chunk):
    text = chunk.get("text", "").strip()
    chunk_type = chunk.get("metadata", {}).get("type", "text")
    if not text:
        return ""
    if chunk_type == "code":
        return f"Code example:\n{text}"
    return f"Explain the following text:\n{text}"


def embed_chunks(chunks):
    embedding_model = _get_model()

    text_to_embed = []
    valid_chunks = []
    for chunk in chunks:
        prepared_text = _prepare_embedding_text(chunk)
        if prepared_text:
            text_to_embed.append(prepared_text)
            valid_chunks.append(chunk)

    if not text_to_embed:
        return []

    embeddings_gen = embedding_model.embed(text_to_embed)
    embeddings = list(embeddings_gen)

    embedded_chunks = []
    for chunk, embedding_vector in zip(valid_chunks, embeddings):
        embedded_chunks.append({
            "id": chunk.get("id"),
            "text": chunk.get("text"),
            "embedding": embedding_vector,
            "metadata": chunk.get("metadata", {}),
        })

    return embedded_chunks
