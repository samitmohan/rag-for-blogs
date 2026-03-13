import pytest
from ingestion.chunker import chunk_parsed_cleaned_document, _split_text_with_overlap


def _make_doc(sections, title="Test Post"):
    return {
        "metadata": {
            "post_title": title,
            "date": "2025-01-01",
            "url": "/test",
        },
        "sections": sections,
    }


def _text_item(text):
    return {"type": "text", "text": text}


def _code_item(content, language="python"):
    return {"type": "code", "content": content, "language": language}


class TestChunkMetadata:
    def test_chunks_have_correct_metadata(self):
        doc = _make_doc([
            {"section": "intro", "content": [_text_item("Some meaningful text content here.")]},
        ])
        chunks = chunk_parsed_cleaned_document(doc)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "id" in chunk
            assert "text" in chunk
            assert "metadata" in chunk
            meta = chunk["metadata"]
            assert meta["post_title"] == "Test Post"
            assert meta["section"] == "intro"
            assert meta["date"] == "2025-01-01"
            assert meta["url"] == "/test"
            assert meta["type"] in ("text", "code")

    def test_code_chunk_has_language(self):
        doc = _make_doc([
            {"section": "code", "content": [_code_item("print('hello')", "python")]},
        ])
        chunks = chunk_parsed_cleaned_document(doc)
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["type"] == "code"
        assert chunks[0]["metadata"]["language"] == "python"

    def test_section_prefix_in_text(self):
        doc = _make_doc([
            {"section": "Gradient Descent", "content": [_text_item("SGD is a variant.")]},
        ])
        chunks = chunk_parsed_cleaned_document(doc)
        assert "[Section: Gradient Descent]" in chunks[0]["text"]


class TestChunkSizing:
    def test_chunk_size_within_bounds(self):
        """No text chunk should exceed 2x the target token count."""
        from config import CHUNK_TARGET_TOKENS
        long_text = " ".join(["word"] * 500)
        doc = _make_doc([
            {"section": "long", "content": [_text_item(long_text)]},
        ])
        chunks = chunk_parsed_cleaned_document(doc)
        for chunk in chunks:
            if chunk["metadata"]["type"] == "text":
                word_count = len(chunk["text"].split())
                # Allow for section prefix overhead
                assert word_count < CHUNK_TARGET_TOKENS * 2 + 10

    def test_short_text_not_split(self):
        doc = _make_doc([
            {"section": "short", "content": [_text_item("Just a few words here.")]},
        ])
        chunks = chunk_parsed_cleaned_document(doc)
        text_chunks = [c for c in chunks if c["metadata"]["type"] == "text"]
        assert len(text_chunks) == 1


class TestOverlap:
    def test_overlap_exists(self):
        """Consecutive text chunks from the same section should share words."""
        long_text = " ".join([f"word{i}" for i in range(400)])
        doc = _make_doc([
            {"section": "overlap", "content": [_text_item(long_text)]},
        ])
        chunks = chunk_parsed_cleaned_document(doc)
        text_chunks = [c for c in chunks if c["metadata"]["type"] == "text"]
        assert len(text_chunks) >= 2

        # Check overlap between consecutive chunks
        for i in range(len(text_chunks) - 1):
            words_a = set(text_chunks[i]["text"].split())
            words_b = set(text_chunks[i + 1]["text"].split())
            overlap = words_a & words_b
            # Should have more than just the section prefix in common
            assert len(overlap) > 5

    def test_split_with_overlap_basic(self):
        chunks = _split_text_with_overlap("a b c d e f g h i j", target=5, overlap=2)
        assert len(chunks) >= 2
        # First chunk should have 5 words
        assert len(chunks[0].split()) == 5


class TestCodeBlockIntegrity:
    def test_code_blocks_intact(self):
        """Code blocks should never be split regardless of size."""
        long_code = "\n".join([f"line_{i} = {i}" for i in range(100)])
        doc = _make_doc([
            {"section": "code", "content": [_code_item(long_code)]},
        ])
        chunks = chunk_parsed_cleaned_document(doc)
        code_chunks = [c for c in chunks if c["metadata"]["type"] == "code"]
        assert len(code_chunks) == 1
        assert "line_99" in code_chunks[0]["text"]


class TestEdgeCases:
    def test_empty_input(self):
        doc = _make_doc([])
        chunks = chunk_parsed_cleaned_document(doc)
        assert chunks == []

    def test_single_paragraph(self):
        doc = _make_doc([
            {"section": "only", "content": [_text_item("A single paragraph.")]},
        ])
        chunks = chunk_parsed_cleaned_document(doc)
        assert len(chunks) == 1

    def test_empty_text_items_skipped(self):
        doc = _make_doc([
            {"section": "mixed", "content": [
                _text_item(""),
                _text_item("Real content here."),
                _text_item("   "),
            ]},
        ])
        chunks = chunk_parsed_cleaned_document(doc)
        assert len(chunks) == 1

    def test_consecutive_text_merged(self):
        """Multiple consecutive text items should be merged before chunking."""
        doc = _make_doc([
            {"section": "merged", "content": [
                _text_item("First paragraph."),
                _text_item("Second paragraph."),
                _text_item("Third paragraph."),
            ]},
        ])
        chunks = chunk_parsed_cleaned_document(doc)
        text_chunks = [c for c in chunks if c["metadata"]["type"] == "text"]
        # Should be merged into one chunk (total is very short)
        assert len(text_chunks) == 1
        assert "First paragraph." in text_chunks[0]["text"]
        assert "Third paragraph." in text_chunks[0]["text"]

    def test_text_code_text_sequence(self):
        """Text before and after code should be separate chunks."""
        doc = _make_doc([
            {"section": "mixed", "content": [
                _text_item("Before code."),
                _code_item("x = 1"),
                _text_item("After code."),
            ]},
        ])
        chunks = chunk_parsed_cleaned_document(doc)
        assert len(chunks) == 3
        assert chunks[0]["metadata"]["type"] == "text"
        assert chunks[1]["metadata"]["type"] == "code"
        assert chunks[2]["metadata"]["type"] == "text"
