# Responsibilities:
# - parse structure (sections, paragraphs, links, code blocks)
# - extract frontmatter as metadata
# - DO NOT enforce semantic policy (no deletion here, that happens in clean_text.py)

import re


def parse_github_markdown(raw_text: str) -> dict:
    """
    Returns:
    {
        "metadata": {...},
        "sections": [
            {
                "section": "intro",
                "content": [
                    { "type": "text", "text": "..." },
                    { "type": "link", "text": "...", "url": "..." },
                    { "type": "code", "language": "python", "content": "..." }
                ]
            }
        ]
    }
    """

    # frontmatter
    metadata = {}
    content_text = raw_text

    fm_match = re.match(r'^\s*---\s*\n(.*?)\n---\s*\n', raw_text, re.DOTALL)
    if fm_match:
        fm_content = fm_match.group(1)
        content_text = raw_text[fm_match.end():]

        for line in fm_content.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                metadata[key.strip()] = val.strip().strip('"')

    # parse
    sections = []
    current_section = {"section": "intro", "content": []}

    lines = content_text.splitlines()
    text_buffer = []

    code_block = False
    code_language = None
    code_content = []

    def flush_text_buffer():
        if not text_buffer:
            return

        paragraph = " ".join(text_buffer).strip()
        text_buffer.clear()

        if paragraph:
            current_section["content"].append({
                "type": "text",
                "text": paragraph
            })

    for line in lines:
        stripped = line.strip()

        # code blocks
        if stripped.startswith("```"):
            if code_block:
                current_section["content"].append({
                    "type": "code",
                    "language": code_language,
                    "content": "\n".join(code_content)
                })
                code_block = False
                code_language = None
                code_content = []
            else:
                flush_text_buffer()
                code_block = True
                code_language = stripped[3:].strip() or None
            continue

        if code_block:
            code_content.append(line)
            continue

        # heading
        heading_match = re.match(r'^(#{1,6})\s+(.*)', line)
        if heading_match:
            flush_text_buffer()

            if current_section["content"]:
                sections.append(current_section)

            current_section = {
                "section": heading_match.group(2).strip(),
                "content": []
            }
            continue

        # text
        if stripped:
            text_buffer.append(stripped)
        else:
            flush_text_buffer()

    flush_text_buffer()
    if current_section["content"]:
        sections.append(current_section)

    return {
        "metadata": metadata,
        "sections": sections
    }


if __name__ == "__main__":
    sample_text = """---
layout: post
title: "600+ leetcode questions: lessons"
date: 2025-10-25 14:06:04 +0530
categories: tech
---
![Image](https://i.ibb.co/LXzF50zN/Screenshot.png)

nothing. spend your time doing something more meaningful than cramming algo problems.

[repository](https://github.com/samitmohan/interviews)
"""

    import json
    parsed = parse_github_markdown(sample_text)
    print(json.dumps(parsed, indent=2))
