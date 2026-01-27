# Responsibilities:
# - normalize whitespace
# - remove non-semantic text nodes
# - drop image artifacts
# - DO NOT summarize or rewrite
# - input and output structure must be identical

import re
from copy import deepcopy


def _clean_text_string(text: str) -> str:
    # normalise internal whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_parsed_content(parsed_sections: list[dict]) -> list[dict]:
    """
    Clean parsed markdown output.

    Input format:
    [
      {
        "section": "...",
        "content": [
          { "type": "text", "text": "..." },
          { "type": "link", "text": "...", "url": "..." }
        ]
      }
    ]
    """
    cleaned_sections = []

    for section in parsed_sections:
        new_section = {
            "section": section["section"],
            "content": []
        }

        for item in section["content"]:
            item_type = item.get("type")

            if item_type == "text":
                cleaned_text = _clean_text_string(item.get("text", ""))
                
                # Remove markdown images: ![alt](url)
                cleaned_text = re.sub(r'!\[.*?\]\(.*?\)', '', cleaned_text)

                # Drop punctuation-only or empty text
                if not cleaned_text:
                    continue
                if all(ch in "!-_*`" for ch in cleaned_text):
                    continue

                new_section["content"].append({
                    "type": "text",
                    "text": cleaned_text
                })

            elif item_type == "code":
                # Pass code blocks through
                new_section["content"].append(item)

            # unknown content types are ignored
            else:
                continue

        # Only keep section if it still has meaningful content
        if new_section["content"]:
            cleaned_sections.append(new_section)

    return cleaned_sections


if __name__ == "__main__":
    # Minimal sanity test
    raw_text = """ 
layout: post
title: "600+ leetcode questions: lessons"
date: 2025-10-25 14:06:04 +0530
categories: tech
---
![Image](...)

nothing. spend your time...

[repository](...)
"""
    parsed = parse_github_markdown(raw_text)
    cleaned = clean_parsed_content(parsed["sections"])
