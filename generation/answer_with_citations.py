"""
Responsibilities:
- Generate answer from LLM
- Attach deterministic citations from retrieved chunks
- Support both blocking and streaming modes
"""

import json
import logging
from typing import Generator

from generation.answer import generate
from generation.groq_client import generate_stream
from generation.prompt_builder import build_prompt
from generation.formatter import format_answer

logger = logging.getLogger(__name__)


def _extract_citations(chunks):
    citations = []
    for chunk in chunks:
        citations.append({
            "chunk_id": chunk["id"],
            "section": chunk["metadata"].get("section"),
            "type": chunk["metadata"].get("type"),
        })
    return citations


def answer_with_citations(query: str, retrieved_chunks):
    prompt = build_prompt(query, retrieved_chunks)
    answer = generate(prompt)
    citations = _extract_citations(retrieved_chunks)
    formatted_answer = format_answer(answer, citations)
    return {
        "answer": formatted_answer,
        "citations": citations,
    }


def answer_with_citations_stream(
    query: str, retrieved_chunks
) -> Generator[str, None, None]:
    """Yields SSE-formatted events: tokens then a final done event with citations."""
    prompt = build_prompt(query, retrieved_chunks)
    citations = _extract_citations(retrieved_chunks)

    try:
        for token in generate_stream(prompt):
            yield f"data: {json.dumps({'token': token})}\n\n"
    except Exception as e:
        logger.error("Streaming generation failed: %s", e)
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

    yield f"data: {json.dumps({'done': True, 'citations': citations})}\n\n"
