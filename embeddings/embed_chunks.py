'''
Load the embedding model (once)
Embed chunk text only
Preserve metadata untouched
Return embedding-ready records
'''

from fastembed import TextEmbedding
from tqdm import tqdm
import numpy as np

_CACHED_MODEL = None

def _get_model():
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        # BAAI/bge-small-en-v1.5 is default, light and good.
        # It's an ONNX model, very low memory usage (~200MB or less)
        _CACHED_MODEL = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _CACHED_MODEL

def _prepare_embedding_text(chunk):
    text = chunk.get("text", "").strip()
    chunk_type = chunk.get("metadata", {}).get("type", "text")
    if not text: return ""
    if chunk_type == "code":
        return f"Code example:\n{text}"
    else:
        # fastembed / BGE models are often fine with raw text, 
        # but maintaining the format for now.
        return f"Explain the following text:\n{text}"
    

def embed_chunks(chunks):
    # Load embedding model
    embedding_model = _get_model()

    embedded_chunks = []
    # batching can be added here for efficiency
    text_to_embed = []
    valid_chunks = []
    for chunk in chunks:
        prepared_text = _prepare_embedding_text(chunk)
        if prepared_text:
            text_to_embed.append(prepared_text)
            valid_chunks.append(chunk)

    if not text_to_embed:
        return []
    
    # FastEmbed returns a generator of numpy arrays
    embeddings_gen = embedding_model.embed(text_to_embed)
    embeddings = list(embeddings_gen)

    # reassemble records
    for chunk, embedding_vector in zip(valid_chunks, embeddings):
        # fastembed returns numpy array, convert to list if needed for consistency,
        # but vector_store expects numpy or list.
        embedded_chunks.append({
            "id": chunk.get("id"),
            "text": chunk.get("text"),
            "embedding": embedding_vector, # Keep as numpy array or .tolist()
            "metadata": chunk.get("metadata", {})
        })

    return embedded_chunks

if __name__ == "__main__":
    input_chunks = [
        {
            "id": "test_intro_0",
            "text": "Softmax converts logits into probabilities.",
            "metadata": {"type": "text"}
        },
        {
            "id": "test_intro_1",
            "text": "model = nn.Linear(784, 10)",
            "metadata": {"type": "code"}
        }
    ]

    embedded = embed_chunks(input_chunks)
    for record in embedded:
        print(record["id"], len(record["embedding"])) # Should be 384 for BGE-small