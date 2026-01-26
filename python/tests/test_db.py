import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from python.data.db import (
    compute_metadata,
    connect,
    initialize,
    insert_ascii_art,
    normalize_newlines,
    search_ascii_art,
    update_ascii_art,
    upsert_ascii_art,
)


class TestDatabaseSchemaAndCrud(unittest.TestCase):
    def _open_temp_db(self) -> sqlite3.Connection:
        handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = Path(handle.name)
        handle.close()

        conn = connect(db_path)
        initialize(conn, schema_path=_schema_path())
        self.addCleanup(conn.close)
        return conn

    def test_schema_creates_tables_and_fts(self) -> None:
        conn = self._open_temp_db()
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view');"
            )
        }
        self.assertIn("ascii_art", tables)
        self.assertIn("ascii_art_fts", tables)

    def test_insert_roundtrip_and_dedupe(self) -> None:
        conn = self._open_temp_db()

        art = " /\\_/\\\n( o.o )\n > ^ <\n"
        inserted_id = insert_ascii_art(
            conn,
            raw_text=art,
            source="unit_test",
            title="cat",
            description="a cute cat",
            category="animal",
            subcategory="cat",
            tags=["animal", "cat"],
        )
        self.assertIsNotNone(inserted_id)

        inserted_id2 = insert_ascii_art(conn, raw_text=art, source="unit_test")
        self.assertIsNone(inserted_id2)

        meta = compute_metadata(art)
        rows = conn.execute(
            "SELECT raw_text, content_hash, width, height FROM ascii_art WHERE content_hash = ?;",
            (meta.content_hash,),
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], art.replace("\r\n", "\n").replace("\r", "\n"))
        self.assertEqual(rows[0][1], meta.content_hash)
        self.assertGreaterEqual(rows[0][2], 1)
        self.assertGreaterEqual(rows[0][3], 1)

    def test_fts_search_matches_description(self) -> None:
        conn = self._open_temp_db()

        insert_ascii_art(
            conn,
            raw_text="hello\n",
            source="unit_test",
            title="greeting",
            description="bright sunshine",
            category="text",
        )

        matches = search_ascii_art(conn, "sunshine")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].title, "greeting")

    def test_insert_skips_empty_text(self) -> None:
        """Empty or whitespace-only text should be skipped by default."""
        conn = self._open_temp_db()

        # Empty string
        result = insert_ascii_art(conn, raw_text="", source="unit_test")
        self.assertIsNone(result)

        # Whitespace only
        result = insert_ascii_art(conn, raw_text="   \n\t\n   ", source="unit_test")
        self.assertIsNone(result)

        # Only newlines
        result = insert_ascii_art(conn, raw_text="\n\n\n", source="unit_test")
        self.assertIsNone(result)

        # Verify nothing was inserted
        count = conn.execute("SELECT COUNT(*) FROM ascii_art").fetchone()[0]
        self.assertEqual(count, 0)

    def test_insert_skip_empty_can_be_disabled(self) -> None:
        """skip_empty=False allows inserting empty/whitespace text."""
        conn = self._open_temp_db()

        result = insert_ascii_art(
            conn, raw_text="   ", source="unit_test", skip_empty=False
        )
        self.assertIsNotNone(result)

        count = conn.execute("SELECT COUNT(*) FROM ascii_art").fetchone()[0]
        self.assertEqual(count, 1)

    def test_newline_normalization(self) -> None:
        """Windows (CRLF) and old Mac (CR) newlines should be normalized to LF."""
        conn = self._open_temp_db()

        # Windows CRLF
        art_crlf = "line1\r\nline2\r\nline3"
        insert_ascii_art(conn, raw_text=art_crlf, source="unit_test")

        row = conn.execute("SELECT raw_text FROM ascii_art").fetchone()
        self.assertEqual(row[0], "line1\nline2\nline3")

    def test_cr_only_normalization(self) -> None:
        """Old Mac CR-only newlines should be normalized to LF."""
        conn = self._open_temp_db()

        art_cr = "line1\rline2\rline3"
        insert_ascii_art(conn, raw_text=art_cr, source="unit_test")

        row = conn.execute("SELECT raw_text FROM ascii_art").fetchone()
        self.assertEqual(row[0], "line1\nline2\nline3")

    def test_upsert_skip_empty_true_skips_whitespace(self) -> None:
        conn = self._open_temp_db()

        result = upsert_ascii_art(
            conn, raw_text="   \n\t\n   ", source="unit_test", skip_empty=True
        )
        self.assertIsNone(result)

        count = conn.execute("SELECT COUNT(*) FROM ascii_art").fetchone()[0]
        self.assertEqual(count, 0)

    def test_upsert_dedup_returns_inserted_false(self) -> None:
        conn = self._open_temp_db()

        art = "line1\nline2\nline3\nline4"
        meta = compute_metadata(art)

        res1 = upsert_ascii_art(conn, raw_text=art, source="unit_test")
        self.assertIsNotNone(res1)
        if res1 is None:
            self.fail("Expected upsert_ascii_art to return UpsertResult")
        self.assertTrue(res1.inserted)
        self.assertEqual(res1.content_hash, meta.content_hash)

        res2 = upsert_ascii_art(conn, raw_text=art, source="unit_test_2")
        self.assertIsNotNone(res2)
        if res2 is None:
            self.fail("Expected upsert_ascii_art to return UpsertResult")
        self.assertFalse(res2.inserted)
        self.assertEqual(res2.id, res1.id)
        self.assertEqual(res2.content_hash, meta.content_hash)

    def test_upsert_json_serialization_tags_and_histogram(self) -> None:
        conn = self._open_temp_db()

        art = "ab\nba"
        tags = ["animal", "cat"]
        res = upsert_ascii_art(conn, raw_text=art, source="unit_test", tags=tags)
        self.assertIsNotNone(res)
        if res is None:
            self.fail("Expected upsert_ascii_art to return UpsertResult")

        row = conn.execute(
            "SELECT tags, char_histogram, content_hash FROM ascii_art WHERE id = ?;",
            (res.id,),
        ).fetchone()
        self.assertIsNotNone(row)
        if row is None:
            self.fail("Expected row to exist after upsert")

        expected_tags = json.dumps(tags, ensure_ascii=False)
        expected_hist = json.dumps(
            dict(compute_metadata(art).char_histogram),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.assertEqual(row["tags"], expected_tags)
        self.assertEqual(row["char_histogram"], expected_hist)
        self.assertEqual(row["content_hash"], res.content_hash)


class TestUpdateSemantics(unittest.TestCase):
    def _open_temp_db(self) -> sqlite3.Connection:
        handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = Path(handle.name)
        handle.close()

        conn = connect(db_path)
        initialize(conn, schema_path=_schema_path())
        self.addCleanup(conn.close)
        return conn

    def test_rejects_unsupported_columns(self) -> None:
        conn = self._open_temp_db()
        art_id = insert_ascii_art(conn, raw_text="hello\n", source="unit_test")
        self.assertIsNotNone(art_id)
        if art_id is None:
            self.fail("Expected insert_ascii_art to return an id")

        with self.assertRaises(ValueError):
            update_ascii_art(conn, art_id=art_id, fields={"raw_text": "nope"})

    def test_tags_list_converts_to_json_string(self) -> None:
        conn = self._open_temp_db()
        art_id = insert_ascii_art(conn, raw_text="hello\n", source="unit_test")
        self.assertIsNotNone(art_id)
        if art_id is None:
            self.fail("Expected insert_ascii_art to return an id")

        update_ascii_art(conn, art_id=art_id, fields={"tags": ["a", "b"]})
        row = conn.execute(
            "SELECT tags FROM ascii_art WHERE id = ?;", (art_id,)
        ).fetchone()
        self.assertIsNotNone(row)
        if row is None:
            self.fail("Expected row to exist after update")

        self.assertEqual(row["tags"], json.dumps(["a", "b"], ensure_ascii=False))

    def test_is_valid_bool_converts_to_int(self) -> None:
        conn = self._open_temp_db()
        art_id = insert_ascii_art(conn, raw_text="hello\n", source="unit_test")
        self.assertIsNotNone(art_id)
        if art_id is None:
            self.fail("Expected insert_ascii_art to return an id")

        update_ascii_art(conn, art_id=art_id, fields={"is_valid": False})
        row = conn.execute(
            "SELECT is_valid FROM ascii_art WHERE id = ?;", (art_id,)
        ).fetchone()
        self.assertIsNotNone(row)
        if row is None:
            self.fail("Expected row to exist after update")

        self.assertEqual(row["is_valid"], 0)

    def test_updated_at_changes_after_update(self) -> None:
        conn = self._open_temp_db()
        art_id = insert_ascii_art(conn, raw_text="hello\n", source="unit_test")
        self.assertIsNotNone(art_id)
        if art_id is None:
            self.fail("Expected insert_ascii_art to return an id")

        sentinel = "2000-01-01 00:00:00"
        conn.execute(
            "UPDATE ascii_art SET updated_at = ? WHERE id = ?;",
            (sentinel, art_id),
        )
        update_ascii_art(conn, art_id=art_id, fields={"title": "new title"})

        row = conn.execute(
            "SELECT updated_at FROM ascii_art WHERE id = ?;",
            (art_id,),
        ).fetchone()
        self.assertIsNotNone(row)
        if row is None:
            self.fail("Expected row to exist after update")

        self.assertIsNotNone(row["updated_at"])
        self.assertNotEqual(row["updated_at"], sentinel)


class TestMetadataEdgeCases(unittest.TestCase):
    def test_normalize_newlines_handles_crlf_and_cr(self) -> None:
        self.assertEqual(normalize_newlines("a\r\nb\r\nc"), "a\nb\nc")
        self.assertEqual(normalize_newlines("a\rb\rc"), "a\nb\nc")

    def test_trailing_newline_treated_as_terminator(self) -> None:
        meta_no_trailing = compute_metadata("ab\ncd")
        meta_trailing = compute_metadata("ab\ncd\n")
        self.assertEqual(meta_trailing.height, meta_no_trailing.height)
        self.assertEqual(meta_trailing.width, meta_no_trailing.width)

        # But an additional blank row (double newline) should count towards height.
        meta_double = compute_metadata("ab\ncd\n\n")
        self.assertEqual(meta_double.width, 2)
        self.assertEqual(meta_double.height, 3)

    def test_ansi_stripping_affects_metrics_but_sets_has_ansi(self) -> None:
        art = "\x1b[31mXX\x1b[0m\n"
        meta = compute_metadata(art)

        self.assertTrue(meta.has_ansi_codes)
        self.assertEqual(meta.width, 2)
        self.assertEqual(meta.height, 1)
        self.assertEqual(meta.non_space_chars, 2)
        self.assertAlmostEqual(meta.char_density, 1.0)
        self.assertNotIn("\x1b", meta.char_histogram)

    def test_charset_classification_boundaries(self) -> None:
        self.assertEqual(compute_metadata("\x7f").charset, "ascii")
        self.assertEqual(compute_metadata("\x80").charset, "extended")
        self.assertEqual(compute_metadata("\u00ff").charset, "extended")
        self.assertEqual(compute_metadata("\u0100").charset, "unicode")

    def test_box_drawing_and_block_element_detection_boundaries(self) -> None:
        box_end = chr(0x257F)
        block_start = chr(0x2580)
        block_end = chr(0x259F)
        block_excluded = chr(0x25A0)

        meta_box_end = compute_metadata(box_end)
        self.assertTrue(meta_box_end.uses_box_drawing)
        self.assertFalse(meta_box_end.uses_block_chars)

        meta_block_start = compute_metadata(block_start)
        self.assertFalse(meta_block_start.uses_box_drawing)
        self.assertTrue(meta_block_start.uses_block_chars)

        meta_block_end = compute_metadata(block_end)
        self.assertFalse(meta_block_end.uses_box_drawing)
        self.assertTrue(meta_block_end.uses_block_chars)

        meta_block_excluded = compute_metadata(block_excluded)
        self.assertFalse(meta_block_excluded.uses_box_drawing)
        self.assertFalse(meta_block_excluded.uses_block_chars)

    def test_density_is_zero_when_width_times_height_is_zero(self) -> None:
        meta_empty = compute_metadata("")
        self.assertEqual(meta_empty.width, 0)
        self.assertEqual(meta_empty.char_density, 0.0)

        meta_newline = compute_metadata("\n")
        self.assertEqual(meta_newline.width, 0)
        self.assertEqual(meta_newline.char_density, 0.0)


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "schema.sql"


if __name__ == "__main__":
    unittest.main()
