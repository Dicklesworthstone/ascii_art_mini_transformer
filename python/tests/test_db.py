import sqlite3
import unittest
from pathlib import Path

from python.data.db import compute_metadata, initialize, insert_ascii_art, search_ascii_art


class TestDatabaseSchemaAndCrud(unittest.TestCase):
    def test_schema_creates_tables_and_fts(self) -> None:
        conn = sqlite3.connect(":memory:")
        initialize(conn, schema_path=_schema_path())
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view');"
            )
        }
        self.assertIn("ascii_art", tables)
        self.assertIn("ascii_art_fts", tables)

    def test_insert_roundtrip_and_dedupe(self) -> None:
        conn = sqlite3.connect(":memory:")
        initialize(conn, schema_path=_schema_path())

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
        conn = sqlite3.connect(":memory:")
        initialize(conn, schema_path=_schema_path())

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
        conn = sqlite3.connect(":memory:")
        initialize(conn, schema_path=_schema_path())

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
        conn = sqlite3.connect(":memory:")
        initialize(conn, schema_path=_schema_path())

        result = insert_ascii_art(
            conn, raw_text="   ", source="unit_test", skip_empty=False
        )
        self.assertIsNotNone(result)

        count = conn.execute("SELECT COUNT(*) FROM ascii_art").fetchone()[0]
        self.assertEqual(count, 1)

    def test_newline_normalization(self) -> None:
        """Windows (CRLF) and old Mac (CR) newlines should be normalized to LF."""
        conn = sqlite3.connect(":memory:")
        initialize(conn, schema_path=_schema_path())

        # Windows CRLF
        art_crlf = "line1\r\nline2\r\nline3"
        insert_ascii_art(conn, raw_text=art_crlf, source="unit_test")

        row = conn.execute("SELECT raw_text FROM ascii_art").fetchone()
        self.assertEqual(row[0], "line1\nline2\nline3")

    def test_cr_only_normalization(self) -> None:
        """Old Mac CR-only newlines should be normalized to LF."""
        conn = sqlite3.connect(":memory:")
        initialize(conn, schema_path=_schema_path())

        art_cr = "line1\rline2\rline3"
        insert_ascii_art(conn, raw_text=art_cr, source="unit_test")

        row = conn.execute("SELECT raw_text FROM ascii_art").fetchone()
        self.assertEqual(row[0], "line1\nline2\nline3")


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "schema.sql"


if __name__ == "__main__":
    unittest.main()
