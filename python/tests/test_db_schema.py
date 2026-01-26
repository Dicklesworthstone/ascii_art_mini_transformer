from __future__ import annotations

import sqlite3
from pathlib import Path

from python.data.db import (
    delete_ascii_art,
    get_ascii_art_by_id,
    initialize,
    insert_ascii_art,
    rebuild_fts,
    search_ascii_art,
    update_ascii_art,
)


def test_schema_round_trip_search_update_delete() -> None:
    conn = sqlite3.connect(":memory:")
    initialize(conn, schema_path=_schema_path())

    art = "X  \n  X\n"
    art_id = insert_ascii_art(
        conn,
        raw_text=art,
        source="test",
        title="diag",
        description="a tiny cat diagram",
        category="animal",
        subcategory="cat",
        tags=["cat", "test"],
    )
    assert art_id is not None

    row = get_ascii_art_by_id(conn, art_id)
    assert row is not None
    assert row.raw_text == art
    assert row.width == 3
    assert row.height == 2
    assert row.non_space_chars == 2
    assert row.charset == "ascii"

    hits = search_ascii_art(conn, "cat")
    assert any(h.id == art_id for h in hits)

    update_ascii_art(conn, art_id=art_id, fields={"description": "a tiny dog diagram"})
    hits2 = search_ascii_art(conn, "dog")
    assert any(h.id == art_id for h in hits2)

    delete_ascii_art(conn, art_id=art_id)
    assert get_ascii_art_by_id(conn, art_id) is None


def test_fts_triggers_insert_update_delete_and_rebuild() -> None:
    conn = sqlite3.connect(":memory:")
    initialize(conn, schema_path=_schema_path())

    art_id = insert_ascii_art(
        conn,
        raw_text="X\nX\n",
        source="test",
        title="cat diag",
        description="a tiny cat diagram",
        category="animal",
        subcategory="cat",
    )
    assert art_id is not None

    # Insert should populate FTS and search should find it.
    fts_rowids = [
        row[0]
        for row in conn.execute(
            "SELECT rowid FROM ascii_art_fts WHERE ascii_art_fts MATCH ?;",
            ("cat",),
        )
    ]
    assert art_id in fts_rowids
    assert any(hit.id == art_id for hit in search_ascii_art(conn, "cat"))

    # Update should propagate into FTS (e.g. title/category changes).
    update_ascii_art(
        conn,
        art_id=art_id,
        fields={
            "title": "dog diag",
            "description": "a tiny dog diagram",
            "category": "canine",
            "subcategory": "dog",
        },
    )
    fts_rowids_dog = [
        row[0]
        for row in conn.execute(
            "SELECT rowid FROM ascii_art_fts WHERE ascii_art_fts MATCH ?;",
            ("dog",),
        )
    ]
    assert art_id in fts_rowids_dog

    fts_rowids_cat = [
        row[0]
        for row in conn.execute(
            "SELECT rowid FROM ascii_art_fts WHERE ascii_art_fts MATCH ?;",
            ("cat",),
        )
    ]
    assert art_id not in fts_rowids_cat

    # Delete should remove from FTS (not just from the content table).
    delete_ascii_art(conn, art_id=art_id)
    fts_rowids_dog_after_delete = [
        row[0]
        for row in conn.execute(
            "SELECT rowid FROM ascii_art_fts WHERE ascii_art_fts MATCH ?;",
            ("dog",),
        )
    ]
    assert art_id not in fts_rowids_dog_after_delete
    assert not any(hit.id == art_id for hit in search_ascii_art(conn, "dog"))

    # Rebuild runs without error (smoke only).
    rebuild_fts(conn)


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "schema.sql"
