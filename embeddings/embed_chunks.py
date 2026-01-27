'''
Load the embedding model (once)
Embed chunk text only
Preserve metadata untouched
Return embedding-ready records
'''

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

def _prepare_embedding_text(chunk):
    text = chunk.get("text", "").strip()
    chunk_type = chunk.get("metadata", {}).get("type", "text")
    if not text: return ""
    if chunk_type == "code":
        return f"Code example:\n{text}"
    else:
        return f"Explain the following text:\n{text}"
    

def embed_chunks(chunks):
    # Load embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
    
    embeddings = embedding_model.embed_documents(text_to_embed)

    # reassemble records
    for chunk, embedding_vector in zip(valid_chunks, embeddings):
        embedded_chunks.append({
            "id": chunk.get("id"),
            "text": chunk.get("text"),
            "embedding": embedding_vector,
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
        print(record["id"], len(record["embedding"])) # test_intro_0 384 test_intro_1 384