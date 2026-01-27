"""
Responsibilities:
- Merge cleaned content atoms into semantic retrieval units
- Produce paragraph-level chunks (per section)
- Embed links into text
- Emit standalone code chunks
- Attach stable, rich metadata
"""

from typing import List, Dict


def chunk_parsed_cleaned_document(parsed_cleaned_doc: Dict) -> List[Dict]:
    chunks: List[Dict] = []

    metadata_base = parsed_cleaned_doc.get("metadata", {})
    post_title = metadata_base.get("post_title", "doc")
    post_date = metadata_base.get("date")
    post_url = metadata_base.get("url")

    for section in parsed_cleaned_doc.get("sections", []):
        section_name = section.get("section", "unknown")
        section_chunk_index = 0  # reset per section

        for item in section.get("content", []):
            item_type = item.get("type")

            # text (now guaranteed to be a full paragraph)
            if item_type == "text":
                text = item.get("text", "").strip()
                if not text: 
                    continue

                chunk_id = f"{post_title}_{section_name}_{section_chunk_index}"
                chunks.append({
                    "id": chunk_id,
                    "text": text,
                    "metadata": {
                        "post_title": post_title,
                        "section": section_name,
                        "date": post_date,
                        "url": post_url,
                        "type": "text"
                    }
                })
                section_chunk_index += 1

            # code
            elif item_type == "code":
                code_content = item.get("content", "").strip()
                if not code_content:
                    continue

                chunk_id = f"{post_title}_{section_name}_{section_chunk_index}"

                chunks.append({
                    "id": chunk_id,
                    "text": code_content,
                    "metadata": {
                        "post_title": post_title,
                        "section": section_name,
                        "date": post_date,
                        "url": post_url,
                        "type": "code",
                        "language": item.get("language")
                    }
                })

                section_chunk_index += 1

    return chunks
