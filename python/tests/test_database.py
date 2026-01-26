"""
Unit tests for the legacy `python.data.database` module.

`python.data.db` is the canonical database layer (external schema at
`data/schema.sql`) and is covered by `python/tests/test_db.py` +
`python/tests/test_db_schema.py`.

These tests remain to document/lock down legacy behavior, but new code should not
import or depend on `python.data.database`.
"""

import tempfile
from pathlib import Path

import pytest

from python.data.database import (
    AsciiArtDatabase,
    compute_content_hash,
    compute_metrics,
    validate_art,
)


class TestComputeContentHash:
    """Tests for content hash computation."""

    def test_same_content_same_hash(self):
        """Same content should produce same hash."""
        text = "hello world"
        assert compute_content_hash(text) == compute_content_hash(text)

    def test_different_content_different_hash(self):
        """Different content should produce different hash."""
        assert compute_content_hash("hello") != compute_content_hash("world")

    def test_whitespace_matters(self):
        """Whitespace differences should produce different hashes."""
        assert compute_content_hash("hello ") != compute_content_hash("hello")

    def test_hash_is_hex_string(self):
        """Hash should be a 64-character hex string (SHA256)."""
        h = compute_content_hash("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestComputeMetrics:
    """Tests for metrics computation."""

    def test_basic_metrics(self):
        """Test basic width/height/char counts."""
        text = "abc\ndef\nghi"
        metrics = compute_metrics(text)
        assert metrics["width"] == 3
        assert metrics["height"] == 3
        assert metrics["total_chars"] == 11  # 9 chars + 2 newlines
        assert metrics["non_space_chars"] == 9

    def test_variable_width_lines(self):
        """Width should be max line width."""
        text = "a\nab\nabc\nab\na"
        metrics = compute_metrics(text)
        assert metrics["width"] == 3  # max line is "abc"
        assert metrics["height"] == 5

    def test_empty_string(self):
        """Empty string should have zero dimensions."""
        metrics = compute_metrics("")
        assert metrics["width"] == 0
        assert metrics["height"] == 1  # split produces ['']
        assert metrics["total_chars"] == 0

    def test_ascii_charset(self):
        """Normal ASCII text should be marked as 'ascii'."""
        text = "Hello World!\n@#$%^&*()"
        metrics = compute_metrics(text)
        assert metrics["charset"] == "ascii"

    def test_unicode_charset(self):
        """Unicode characters should be marked as 'unicode'."""
        # Use explicit Unicode escape for a smiley face (U+263A)
        text = "Hello \u263a World!"
        metrics = compute_metrics(text)
        assert metrics["charset"] == "unicode"

    def test_box_drawing_detection(self):
        """Should detect box drawing characters."""
        # Box drawing characters: U+2500 (horizontal line), U+2502 (vertical line)
        text = "\u2500\u2502\u250c\u2510"
        metrics = compute_metrics(text)
        assert metrics["uses_box_drawing"] is True

    def test_block_element_detection(self):
        """Should detect block element characters."""
        # Block elements: U+2588 (full block), U+2584 (lower half block)
        text = "\u2588\u2584\u2580"
        metrics = compute_metrics(text)
        assert metrics["uses_block_chars"] is True

    def test_ansi_escape_detection(self):
        """Should detect ANSI escape sequences."""
        text = "\x1b[31mRed Text\x1b[0m"
        metrics = compute_metrics(text)
        assert metrics["has_ansi_codes"] is True

    def test_no_ansi_in_normal_text(self):
        """Normal text should not be marked as having ANSI codes."""
        text = "Normal text without escapes"
        metrics = compute_metrics(text)
        assert metrics["has_ansi_codes"] is False

    def test_char_histogram(self):
        """Character histogram should count correctly."""
        text = "aabbc"
        metrics = compute_metrics(text)
        assert metrics["char_histogram"]["a"] == 2
        assert metrics["char_histogram"]["b"] == 2
        assert metrics["char_histogram"]["c"] == 1

    def test_char_density(self):
        """Char density should be ratio of non-space to area."""
        text = "ab\ncd"  # 4 non-space chars, 2x2 area
        metrics = compute_metrics(text)
        # density = 4 / (2 * 2) = 1.0
        assert metrics["char_density"] == 1.0


class TestValidateArt:
    """Tests for art validation."""

    def test_valid_art(self):
        """Normal art should be valid."""
        text = "line1\nline2\nline3\nline4"
        metrics = compute_metrics(text)
        assert validate_art(text, metrics) is True

    def test_empty_string_invalid(self):
        """Empty string should be invalid."""
        text = ""
        metrics = compute_metrics(text)
        assert validate_art(text, metrics) is False

    def test_whitespace_only_invalid(self):
        """Whitespace-only string should be invalid."""
        text = "   \n   \n   "
        metrics = compute_metrics(text)
        assert validate_art(text, metrics) is False

    def test_too_short_invalid(self):
        """Art with fewer than 3 lines should be invalid."""
        text = "line1\nline2"
        metrics = compute_metrics(text)
        assert validate_art(text, metrics) is False

    def test_exactly_3_lines_valid(self):
        """Art with exactly 3 lines should be valid."""
        text = "line1\nline2\nline3"
        metrics = compute_metrics(text)
        assert validate_art(text, metrics) is True


class TestAsciiArtDatabase:
    """Tests for the database class."""

    @pytest.fixture
    def db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        db = AsciiArtDatabase(db_path)
        yield db
        db.close()
        # Intentionally do not delete temp files from tests; project policy forbids deletion.

    def test_insert_and_retrieve(self, db):
        """Should insert and retrieve art correctly."""
        test_art = """
 /\\_/\\
( o.o )
 > ^ <
"""
        art_id = db.insert(
            raw_text=test_art,
            source="test",
            title="Cat",
            description="A cat face",
            category="animal",
        )
        assert art_id is not None

        art = db.get_by_id(art_id)
        assert art is not None
        assert art.title == "Cat"
        assert art.source == "test"
        assert art.category == "animal"
        assert art.raw_text == test_art

    def test_deduplication(self, db):
        """Same content should not be inserted twice."""
        test_art = "line1\nline2\nline3\nline4"

        id1 = db.insert(raw_text=test_art, source="test1")
        id2 = db.insert(raw_text=test_art, source="test2")

        assert id1 is not None
        assert id2 is None  # Duplicate returns None
        assert db.count() == 1

    def test_get_by_hash(self, db):
        """Should retrieve by content hash."""
        test_art = "unique\ncontent\nhere\ntest"
        db.insert(raw_text=test_art, source="test")

        content_hash = compute_content_hash(test_art)
        art = db.get_by_hash(content_hash)
        assert art is not None
        assert art.raw_text == test_art

    def test_tags_stored_as_json(self, db):
        """Tags should be stored and retrieved correctly."""
        test_art = "line1\nline2\nline3\nline4"
        tags = ["cute", "simple", "test"]

        art_id = db.insert(raw_text=test_art, source="test", tags=tags)
        art = db.get_by_id(art_id)

        assert art.tags == tags

    def test_full_text_search(self, db):
        """FTS should find matching art."""
        art1 = "cat\nart\nhere\ntest"
        art2 = "dog\nart\nhere\ntest"

        db.insert(
            raw_text=art1, source="test", title="Cute Cat", description="A fluffy cat"
        )
        db.insert(
            raw_text=art2, source="test", title="Happy Dog", description="A happy dog"
        )

        # Search for cat
        results = db.search_fts("cat")
        assert len(results) == 1
        assert results[0].title == "Cute Cat"

        # Search for fluffy
        results = db.search_fts("fluffy")
        assert len(results) == 1
        assert results[0].title == "Cute Cat"

    def test_search_by_size(self, db):
        """Should filter by size constraints."""
        small_art = "ab\ncd\nef\ngh"  # 2x4
        large_art = "abcdefghij\n" * 10  # 10x10

        db.insert(raw_text=small_art, source="test", title="Small")
        db.insert(raw_text=large_art, source="test", title="Large")

        # Find small art
        results = db.search_by_size(max_width=5)
        assert len(results) == 1
        assert results[0].title == "Small"

        # Find large art
        results = db.search_by_size(min_width=8)
        assert len(results) == 1
        assert results[0].title == "Large"

    def test_search_by_category(self, db):
        """Should filter by category."""
        db.insert(
            raw_text="cat\nart\nhere\ntest",
            source="test",
            category="animal",
            subcategory="cat",
        )
        db.insert(
            raw_text="tree\nart\nhere\ntest",
            source="test",
            category="nature",
            subcategory="tree",
        )

        results = db.search_by_category("animal")
        assert len(results) == 1
        assert results[0].category == "animal"

        results = db.search_by_category("animal", subcategory="cat")
        assert len(results) == 1

    def test_count(self, db):
        """Should count records correctly."""
        assert db.count() == 0

        db.insert(raw_text="line1\nline2\nline3\nline4", source="test")
        assert db.count() == 1

        db.insert(raw_text="diff1\ndiff2\ndiff3\ndiff4", source="test")
        assert db.count() == 2

    def test_get_stats(self, db):
        """Should return comprehensive statistics."""
        db.insert(raw_text="cat1\ncat2\ncat3\ncat4", source="hf", category="animal")
        db.insert(raw_text="dog1\ndog2\ndog3\ndog4", source="hf", category="animal")
        db.insert(raw_text="tree\ntree\ntree\ntree", source="web", category="nature")

        stats = db.get_stats()
        assert stats["total"] == 3
        assert stats["valid"] == 3
        assert stats["sources"]["hf"] == 2
        assert stats["sources"]["web"] == 1
        assert stats["categories"]["animal"] == 2
        assert stats["categories"]["nature"] == 1

    def test_iterate_valid(self, db):
        """Should iterate over all valid records."""
        for i in range(5):
            db.insert(raw_text=f"art{i}\nline2\nline3\nline4", source="test")

        count = 0
        for art in db.iterate_valid(batch_size=2):
            count += 1
            assert art.is_valid

        assert count == 5

    def test_metrics_computed_on_insert(self, db):
        """Metrics should be computed automatically on insert."""
        test_art = "123456\n123456\n123456\n123456"
        art_id = db.insert(raw_text=test_art, source="test")
        art = db.get_by_id(art_id)

        assert art.width == 6
        assert art.height == 4
        assert art.charset == "ascii"

    def test_context_manager(self):
        """Database should work as context manager."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        with AsciiArtDatabase(db_path) as db:
            db.insert(raw_text="test\nart\nhere\ndata", source="test")
            assert db.count() == 1

        # File should still exist after closing
        assert Path(db_path).exists()
        # Intentionally do not delete temp files from tests; project policy forbids deletion.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
