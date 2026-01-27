# backend/app.py
import os
from typing import List, Dict
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ingestion.load_all_blogs import fetch_all_posts
from ingestion.parse_markdown import parse_github_markdown
from ingestion.clean_text import clean_parsed_content
from ingestion.chunker import chunk_parsed_cleaned_document

from embeddings.embed_chunks import embed_chunks
from embeddings.vector_store import VectorStore

from retrieval.retrieve import retrieve, classify_intent, is_confident
from generation.answer_with_citations import answer_with_citations

DATA_DIR = "data"  # where FAISS + metadata will be stored
FAISS_DIR = os.path.join(DATA_DIR, "faiss")

app = FastAPI(title="RAG backend for samitmohan.github.io")

# allow the GitHub Pages UI to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down to your site URL in production
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# global store (in memory for server lifetime)
vector_store: VectorStore = None

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5

class GenerateRequest(BaseModel):
    query: str
    chunks: List[Dict]

@app.post("/ingest")
def ingest(force_reindex: bool = False):
    """
    Ingest all blog posts, parse, clean, chunk, embed and build persistent index.
    Set GITHUB_TOKEN env if needed.
    """
    global vector_store

    # fetch all posts
    posts = fetch_all_posts()  # [(filename, raw_text), ...]
    all_chunks = []
    metadata_for_chunks = {
        "source_repo": f"https://github.com/{os.environ.get('GITHUB_REPO', 'samitmohan/samitmohan.github.io')}"
    }

    for filename, raw in posts:
        parsed = parse_github_markdown(raw)
        cleaned_sections = clean_parsed_content(parsed.get("sections", []))

        chunk_input = {
            "metadata": {
                "post_title": parsed.get("metadata", {}).get("title", filename),
                "date": parsed.get("metadata", {}).get("date"),
                "url": f"/{filename.replace('.md','.html')}"  # adjust if you have slug logic
            },
            "sections": cleaned_sections
        }
        chunks = chunk_parsed_cleaned_document(chunk_input)
        # attach origin filename to chunk metadata
        for c in chunks:
            c["metadata"].setdefault("source_file", filename)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No chunks produced")

    # embed all chunks (this may take time)
    embedded = embed_chunks(all_chunks)

    # build vector store
    dim = len(embedded[0]["embedding"])
    store = VectorStore(dim=dim)
    store.add_embeddings(embedded)
    store.save(FAISS_DIR)

    vector_store = store
    return {"status": "ingested", "n_chunks": len(embedded)}

@app.on_event("startup")
def load_index_on_start():
    """
    When the backend starts, try to load the persisted index if present.
    """
    global vector_store
    if os.path.exists(os.path.join(FAISS_DIR, "faiss.index")):
        vector_store = VectorStore.load(FAISS_DIR)
        print("Loaded vector store from disk")
    else:
        print("No persisted index found; call /ingest to build it")

@app.post("/retrieve")
def retrieve_endpoint(req: RetrieveRequest):
    global vector_store
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not ready. Call /ingest first.")

    # run retrieval
    retrieved = retrieve(req.query, vector_store, k=req.top_k)

    if retrieved:
        print(f"Query: '{req.query}' | Best Distance: {retrieved[0]['distance']:.4f}")

    # confidence guard
    if not is_confident(retrieved):
         return {"chunks": []}

    # return the chunks (the retriever returns distance+chunk)
    retrieved_chunks = [r["chunk"] for r in retrieved]
    
    return {"chunks": retrieved_chunks}

@app.post("/generate")
def generate_endpoint(req: GenerateRequest):
    # generate answer with citations using provided chunks
    result = answer_with_citations(req.query, req.chunks)
    return {"answer": result["answer"], "citations": result["citations"]}

@app.post("/query")
def query_endpoint(req: RetrieveRequest):
    """
    Legacy/All-in-one endpoint.
    """
    # 1. Retrieve
    retrieval_resp = retrieve_endpoint(req)
    chunks = retrieval_resp["chunks"]
    
    if not chunks:
        return {"answer": "I don't know based on the provided information.", "citations": []}

    # 2. Generate
    gen_req = GenerateRequest(query=req.query, chunks=chunks)
    return generate_endpoint(gen_req)

@app.get("/status")
def status():
    return {"ready": vector_store is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
