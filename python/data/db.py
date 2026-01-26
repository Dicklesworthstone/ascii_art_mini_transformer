from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_BOX_DRAWING_RANGE = range(0x2500, 0x2580)
_BLOCK_ELEMENT_RANGE = range(0x2580, 0x25A0)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_schema_path() -> Path:
    return repo_root() / "data" / "schema.sql"


def default_db_path() -> Path:
    return repo_root() / "data" / "ascii_art.db"


def normalize_newlines(text: str) -> str:
    # Canonicalize line endings for consistent hashing/metrics and DB storage.
    return text.replace("\r\n", "\n").replace("\r", "\n")


@dataclass(frozen=True, slots=True)
class AsciiArtMetadata:
    content_hash: str
    width: int
    height: int
    total_chars: int
    non_space_chars: int
    char_density: float
    charset: str
    char_histogram: Mapping[str, int]
    uses_box_drawing: bool
    uses_block_chars: bool
    has_ansi_codes: bool


@dataclass(frozen=True, slots=True)
class AsciiArtRow:
    id: int
    content_hash: str
    raw_text: str
    source: str
    source_id: str | None
    title: str | None
    description: str | None
    category: str | None
    subcategory: str | None
    tags: Sequence[str] | None
    artist: str | None
    width: int
    height: int
    total_chars: int
    non_space_chars: int
    char_density: float
    charset: str
    char_histogram: Mapping[str, int] | None
    uses_box_drawing: bool
    uses_block_chars: bool
    has_ansi_codes: bool
    is_valid: bool
    quality_score: float | None


@dataclass(frozen=True, slots=True)
class UpsertResult:
    id: int
    inserted: bool
    content_hash: str


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def initialize(conn: sqlite3.Connection, schema_path: Path | str | None = None) -> None:
    conn.row_factory = sqlite3.Row
    path = Path(schema_path) if schema_path is not None else default_schema_path()
    schema_sql = path.read_text(encoding="utf-8")
    conn.executescript(schema_sql)


def open_db(db_path: Path | str | None = None) -> sqlite3.Connection:
    return connect(db_path)


def init_db(conn: sqlite3.Connection, schema_path: Path | str | None = None) -> None:
    initialize(conn, schema_path=schema_path)


def rebuild_fts(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT INTO ascii_art_fts(ascii_art_fts) VALUES('rebuild');")


def compute_metadata(raw_text: str) -> AsciiArtMetadata:
    normalized = normalize_newlines(raw_text)
    content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    has_ansi_codes = (
        _ANSI_ESCAPE_RE.search(normalized) is not None or "\x1b" in normalized
    )
    text_for_metrics = (
        _ANSI_ESCAPE_RE.sub("", normalized) if has_ansi_codes else normalized
    )

    lines = text_for_metrics.split("\n")
    # Treat a trailing newline as a line terminator, not an extra blank row.
    if text_for_metrics.endswith("\n") and lines and lines[-1] == "":
        lines = lines[:-1]
    height = len(lines)
    width = max((len(line) for line in lines), default=0)

    total_chars = len(text_for_metrics)
    non_space_chars = sum(1 for ch in text_for_metrics if not ch.isspace())

    area = width * height
    char_density = (non_space_chars / area) if area > 0 else 0.0

    max_codepoint = max((ord(ch) for ch in text_for_metrics), default=0)
    if max_codepoint <= 0x7F:
        charset = "ascii"
    elif max_codepoint <= 0xFF:
        charset = "extended"
    else:
        charset = "unicode"

    uses_box_drawing = any(ord(ch) in _BOX_DRAWING_RANGE for ch in text_for_metrics)
    uses_block_chars = any(ord(ch) in _BLOCK_ELEMENT_RANGE for ch in text_for_metrics)

    histogram = Counter(text_for_metrics)
    histogram_dict: dict[str, int] = dict(histogram)

    return AsciiArtMetadata(
        content_hash=content_hash,
        width=width,
        height=height,
        total_chars=total_chars,
        non_space_chars=non_space_chars,
        char_density=char_density,
        charset=charset,
        char_histogram=histogram_dict,
        uses_box_drawing=uses_box_drawing,
        uses_block_chars=uses_block_chars,
        has_ansi_codes=has_ansi_codes,
    )


def insert_ascii_art(
    conn: sqlite3.Connection,
    *,
    raw_text: str,
    source: str,
    source_id: str | None = None,
    title: str | None = None,
    description: str | None = None,
    category: str | None = None,
    subcategory: str | None = None,
    tags: Sequence[str] | None = None,
    artist: str | None = None,
    is_valid: bool = True,
    quality_score: float | None = None,
    skip_empty: bool = True,
) -> int | None:
    """
    Insert ASCII art, returning the row ID if inserted, or None if duplicate/skipped.

    Args:
        skip_empty: If True (default), return None for empty/whitespace-only text.
    """
    res = upsert_ascii_art(
        conn,
        raw_text=raw_text,
        source=source,
        source_id=source_id,
        title=title,
        description=description,
        category=category,
        subcategory=subcategory,
        tags=tags,
        artist=artist,
        is_valid=is_valid,
        quality_score=quality_score,
        skip_empty=skip_empty,
    )
    if res is None:
        return None  # Skipped (empty content)
    return res.id if res.inserted else None


def upsert_ascii_art(
    conn: sqlite3.Connection,
    *,
    raw_text: str,
    source: str,
    source_id: str | None = None,
    title: str | None = None,
    description: str | None = None,
    category: str | None = None,
    subcategory: str | None = None,
    tags: Sequence[str] | None = None,
    artist: str | None = None,
    is_valid: bool = True,
    quality_score: float | None = None,
    skip_empty: bool = True,
) -> UpsertResult | None:
    """
    Insert or ignore an ASCII art entry.

    Args:
        skip_empty: If True (default), return None for empty/whitespace-only text
                    instead of inserting an empty row.

    Returns:
        UpsertResult on success, or None if skipped due to empty content.
    """
    normalized = normalize_newlines(raw_text)

    # Skip empty or whitespace-only content
    if skip_empty and not normalized.strip():
        return None

    meta = compute_metadata(normalized)

    tags_json = json.dumps(list(tags), ensure_ascii=False) if tags is not None else None
    hist_json = json.dumps(
        dict(meta.char_histogram), ensure_ascii=False, separators=(",", ":")
    )

    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO ascii_art (
            content_hash,
            raw_text,
            source,
            source_id,
            title,
            description,
            category,
            subcategory,
            tags,
            artist,
            width,
            height,
            total_chars,
            non_space_chars,
            char_density,
            charset,
            char_histogram,
            uses_box_drawing,
            uses_block_chars,
            has_ansi_codes,
            is_valid,
            quality_score
        ) VALUES (
            :content_hash,
            :raw_text,
            :source,
            :source_id,
            :title,
            :description,
            :category,
            :subcategory,
            :tags,
            :artist,
            :width,
            :height,
            :total_chars,
            :non_space_chars,
            :char_density,
            :charset,
            :char_histogram,
            :uses_box_drawing,
            :uses_block_chars,
            :has_ansi_codes,
            :is_valid,
            :quality_score
        );
        """,
        {
            "content_hash": meta.content_hash,
            "raw_text": normalized,
            "source": source,
            "source_id": source_id,
            "title": title,
            "description": description,
            "category": category,
            "subcategory": subcategory,
            "tags": tags_json,
            "artist": artist,
            "width": meta.width,
            "height": meta.height,
            "total_chars": meta.total_chars,
            "non_space_chars": meta.non_space_chars,
            "char_density": meta.char_density,
            "charset": meta.charset,
            "char_histogram": hist_json,
            "uses_box_drawing": 1 if meta.uses_box_drawing else 0,
            "uses_block_chars": 1 if meta.uses_block_chars else 0,
            "has_ansi_codes": 1 if meta.has_ansi_codes else 0,
            "is_valid": 1 if is_valid else 0,
            "quality_score": quality_score,
        },
    )

    row = conn.execute(
        "SELECT id FROM ascii_art WHERE content_hash = ?;", (meta.content_hash,)
    ).fetchone()
    if row is None:
        raise RuntimeError("upsert failed: row not found after insert/ignore")

    return UpsertResult(
        id=int(row["id"]),
        inserted=cursor.rowcount != 0,
        content_hash=meta.content_hash,
    )


def get_ascii_art_by_id(conn: sqlite3.Connection, art_id: int) -> AsciiArtRow | None:
    row = conn.execute("SELECT * FROM ascii_art WHERE id = ?;", (art_id,)).fetchone()
    return _row_to_ascii_art_row(row) if row is not None else None


def get_ascii_art_by_hash(
    conn: sqlite3.Connection, content_hash: str
) -> AsciiArtRow | None:
    row = conn.execute(
        "SELECT * FROM ascii_art WHERE content_hash = ?;", (content_hash,)
    ).fetchone()
    return _row_to_ascii_art_row(row) if row is not None else None


_ALLOWED_UPDATE_COLUMNS: set[str] = {
    "source",
    "source_id",
    "title",
    "description",
    "category",
    "subcategory",
    "tags",
    "artist",
    "is_valid",
    "quality_score",
}


def update_ascii_art(
    conn: sqlite3.Connection, *, art_id: int, fields: Mapping[str, Any]
) -> None:
    if not fields:
        return

    bad = set(fields) - _ALLOWED_UPDATE_COLUMNS
    if bad:
        raise ValueError(f"unsupported update columns: {sorted(bad)!r}")

    set_parts: list[str] = []
    params: list[Any] = []
    for key, value in fields.items():
        if key == "tags" and value is not None and not isinstance(value, str):
            value = json.dumps(list(value), ensure_ascii=False)
        if key == "is_valid" and isinstance(value, bool):
            value = 1 if value else 0
        set_parts.append(f"{key} = ?")
        params.append(value)

    set_parts.append("updated_at = CURRENT_TIMESTAMP")
    params.append(art_id)

    sql = f"UPDATE ascii_art SET {', '.join(set_parts)} WHERE id = ?;"
    conn.execute(sql, params)


def delete_ascii_art(conn: sqlite3.Connection, *, art_id: int) -> None:
    conn.execute("DELETE FROM ascii_art WHERE id = ?;", (art_id,))


def search_ascii_art(
    conn: sqlite3.Connection, query: str, *, limit: int = 20
) -> list[AsciiArtRow]:
    rows = conn.execute(
        """
        SELECT a.*
        FROM ascii_art_fts
        JOIN ascii_art a ON a.id = ascii_art_fts.rowid
        WHERE ascii_art_fts MATCH ?
        ORDER BY bm25(ascii_art_fts)
        LIMIT ?;
        """,
        (query, limit),
    ).fetchall()
    return [_row_to_ascii_art_row(row) for row in rows]


def _row_to_ascii_art_row(row: sqlite3.Row) -> AsciiArtRow:
    tags_val = row["tags"]
    tags: Sequence[str] | None
    if tags_val is None:
        tags = None
    else:
        parsed = json.loads(tags_val)
        tags = list(parsed) if isinstance(parsed, list) else None

    hist_val = row["char_histogram"]
    hist: Mapping[str, int] | None
    if hist_val is None:
        hist = None
    else:
        parsed_hist = json.loads(hist_val)
        hist = dict(parsed_hist) if isinstance(parsed_hist, dict) else None

    return AsciiArtRow(
        id=int(row["id"]),
        content_hash=str(row["content_hash"]),
        raw_text=str(row["raw_text"]),
        source=str(row["source"]),
        source_id=row["source_id"],
        title=row["title"],
        description=row["description"],
        category=row["category"],
        subcategory=row["subcategory"],
        tags=tags,
        artist=row["artist"],
        width=int(row["width"]),
        height=int(row["height"]),
        total_chars=int(row["total_chars"]),
        non_space_chars=int(row["non_space_chars"]),
        char_density=float(row["char_density"])
        if row["char_density"] is not None
        else 0.0,
        charset=str(row["charset"]),
        char_histogram=hist,
        uses_box_drawing=bool(row["uses_box_drawing"]),
        uses_block_chars=bool(row["uses_block_chars"]),
        has_ansi_codes=bool(row["has_ansi_codes"]),
        is_valid=bool(row["is_valid"]),
        quality_score=float(row["quality_score"])
        if row["quality_score"] is not None
        else None,
    )


def iter_ascii_art_ids(conn: sqlite3.Connection) -> Iterable[int]:
    for row in conn.execute("SELECT id FROM ascii_art ORDER BY id;"):
        yield int(row["id"])


def execute(
    conn: sqlite3.Connection, sql: str, params: Mapping[str, Any] | Sequence[Any] = ()
) -> None:
    conn.execute(sql, params)
