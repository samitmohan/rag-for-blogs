import logging
import os
import re
from contextlib import asynccontextmanager
from typing import List, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from config import FAISS_DIR, LOG_LEVEL

from ingestion.load_all_blogs import fetch_all_posts
from ingestion.parse_markdown import parse_github_markdown
from ingestion.clean_text import clean_parsed_content
from ingestion.chunker import chunk_parsed_cleaned_document

from embeddings.embed_chunks import embed_chunks
from embeddings.vector_store import VectorStore

from retrieval.retrieve import retrieve, is_confident
from generation.answer_with_citations import answer_with_citations, answer_with_citations_stream

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

vector_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store
    try:
        index_path = os.path.join(FAISS_DIR, "faiss.index")
        if os.path.exists(index_path):
            vector_store = VectorStore.load(FAISS_DIR)
            logger.info("Loaded vector store from disk")
        else:
            logger.warning("No persisted index found; call /ingest to build it")
    except Exception as e:
        logger.error("Failed to load vector store: %s", e)
        vector_store = None
    yield


app = FastAPI(title="RAG backend for samitmohan.github.io", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_index():
    return FileResponse("index.html")


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "rag-backend"}


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


class GenerateRequest(BaseModel):
    query: str
    chunks: List[Dict]


@app.post("/ingest")
def ingest(force_reindex: bool = False):
    global vector_store

    posts = fetch_all_posts()
    all_chunks = []

    for filename, raw in posts:
        parsed = parse_github_markdown(raw)
        cleaned_sections = clean_parsed_content(parsed.get("sections", []))

        categories = parsed.get("metadata", {}).get("categories", "").strip()
        if "," in categories:
            categories = categories.split(",")[0].strip()

        url = f"/{filename.replace('.md', '.html')}"
        match = re.match(r"^(\d{4})-(\d{2})-(\d{2})-(.*)\.md$", filename)
        if match:
            year, month, day, slug = match.groups()
            if categories:
                url = f"/{categories}/{year}/{month}/{day}/{slug}.html"
            else:
                url = f"/{year}/{month}/{day}/{slug}.html"

        chunk_input = {
            "metadata": {
                "post_title": parsed.get("metadata", {}).get("title", filename),
                "date": parsed.get("metadata", {}).get("date"),
                "url": url,
            },
            "sections": cleaned_sections,
        }
        chunks = chunk_parsed_cleaned_document(chunk_input)
        for c in chunks:
            c["metadata"].setdefault("source_file", filename)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No chunks produced")

    embedded = embed_chunks(all_chunks)

    dim = len(embedded[0]["embedding"])
    store = VectorStore(dim=dim)
    store.add_embeddings(embedded)
    store.save(FAISS_DIR)

    vector_store = store
    logger.info("Ingested %d chunks", len(embedded))
    return {"status": "ingested", "n_chunks": len(embedded)}


@app.post("/retrieve")
def retrieve_endpoint(req: RetrieveRequest):
    global vector_store
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not ready. Call /ingest first.")

    retrieved = retrieve(req.query, vector_store, k=req.top_k)

    if retrieved:
        logger.debug("Query: '%s' | Best Distance: %.4f", req.query, retrieved[0]["distance"])

    if not is_confident(retrieved):
        return {"chunks": []}

    retrieved_chunks = [r["chunk"] for r in retrieved]
    return {"chunks": retrieved_chunks}


@app.post("/generate")
def generate_endpoint(req: GenerateRequest):
    result = answer_with_citations(req.query, req.chunks)
    return {"answer": result["answer"], "citations": result["citations"]}


@app.post("/query")
def query_endpoint(req: RetrieveRequest):
    retrieval_resp = retrieve_endpoint(req)
    chunks = retrieval_resp["chunks"]

    if not chunks:
        return {"answer": "No relevant results found for this query.", "citations": []}

    gen_req = GenerateRequest(query=req.query, chunks=chunks)
    return generate_endpoint(gen_req)


@app.post("/query/stream")
def query_stream_endpoint(req: RetrieveRequest):
    """Streaming endpoint: retrieves context then streams LLM tokens via SSE."""
    global vector_store
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not ready.")

    retrieved = retrieve(req.query, vector_store, k=req.top_k)
    if not is_confident(retrieved):
        import json

        def empty_stream():
            yield f"data: {json.dumps({'done': True, 'citations': [], 'no_results': True})}\n\n"

        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    chunks = [r["chunk"] for r in retrieved]
    return StreamingResponse(
        answer_with_citations_stream(req.query, chunks),
        media_type="text/event-stream",
    )


@app.get("/status")
def status():
    return {"ready": vector_store is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
