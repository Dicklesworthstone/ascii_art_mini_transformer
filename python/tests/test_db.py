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


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "schema.sql"


if __name__ == "__main__":
    unittest.main()
