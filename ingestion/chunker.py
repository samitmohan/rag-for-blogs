"""
Semantic chunking with overlap.

- Merges consecutive text items within a section into a single buffer
- Splits by approximate token count (~200 tokens target, 50 token overlap)
- Keeps code blocks intact (never split)
- Prepends [Section: {name}] to each chunk for embedding context
"""

import logging
from typing import List, Dict

from config import CHUNK_TARGET_TOKENS, CHUNK_OVERLAP_TOKENS

logger = logging.getLogger(__name__)


def _approx_tokens(text: str) -> int:
    return len(text.split())


def _split_text_with_overlap(text: str, target: int, overlap: int) -> List[str]:
    """Split text into chunks of ~target words with ~overlap word overlap."""
    words = text.split()
    if len(words) <= target:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + target
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        # Advance by (target - overlap) so the next chunk overlaps
        start += target - overlap
        if start >= len(words):
            break

    return chunks


def chunk_parsed_cleaned_document(parsed_cleaned_doc: Dict) -> List[Dict]:
    chunks: List[Dict] = []

    metadata_base = parsed_cleaned_doc.get("metadata", {})
    post_title = metadata_base.get("post_title", "doc")
    post_date = metadata_base.get("date")
    post_url = metadata_base.get("url")

    for section in parsed_cleaned_doc.get("sections", []):
        section_name = section.get("section", "unknown")
        section_chunk_index = 0

        # Merge consecutive text items into a single buffer
        text_buffer = []
        items = section.get("content", [])

        def flush_text_buffer():
            nonlocal section_chunk_index
            if not text_buffer:
                return

            merged = " ".join(text_buffer)
            text_buffer.clear()

            # Split with overlap if needed
            sub_chunks = _split_text_with_overlap(
                merged, CHUNK_TARGET_TOKENS, CHUNK_OVERLAP_TOKENS
            )

            for sub in sub_chunks:
                sub = sub.strip()
                if not sub:
                    continue
                chunk_id = f"{post_title}_{section_name}_{section_chunk_index}"
                prefixed_text = f"[Section: {section_name}] {sub}"
                chunks.append({
                    "id": chunk_id,
                    "text": prefixed_text,
                    "metadata": {
                        "post_title": post_title,
                        "section": section_name,
                        "date": post_date,
                        "url": post_url,
                        "type": "text",
                    },
                })
                section_chunk_index += 1

        for item in items:
            item_type = item.get("type")

            if item_type == "text":
                text = item.get("text", "").strip()
                if text:
                    text_buffer.append(text)

            elif item_type == "code":
                # Flush any accumulated text before emitting a code chunk
                flush_text_buffer()

                code_content = item.get("content", "").strip()
                if not code_content:
                    continue

                chunk_id = f"{post_title}_{section_name}_{section_chunk_index}"
                prefixed_text = f"[Section: {section_name}]\n{code_content}"
                chunks.append({
                    "id": chunk_id,
                    "text": prefixed_text,
                    "metadata": {
                        "post_title": post_title,
                        "section": section_name,
                        "date": post_date,
                        "url": post_url,
                        "type": "code",
                        "language": item.get("language"),
                    },
                })
                section_chunk_index += 1

        # Flush remaining text at end of section
        flush_text_buffer()

    return chunks
