"""
Responsibilities:
- Generate answer from LLM
- Attach deterministic citations from retrieved chunks
"""

from typing import List, Dict
from generation.answer import generate
from generation.prompt_builder import build_prompt
from generation.formatter import format_answer

def answer_with_citations(query: str, retrieved_chunks):
    prompt = build_prompt(query, retrieved_chunks)
    answer = generate(prompt)

    citations = []
    for chunk in retrieved_chunks:
        citations.append({
            "chunk_id": chunk["id"],
            "section": chunk["metadata"].get("section"),
            "type": chunk["metadata"].get("type"),
        })

    formatted_answer = format_answer(answer, citations)

    return {
        "answer": formatted_answer,
        "citations": citations
    }
