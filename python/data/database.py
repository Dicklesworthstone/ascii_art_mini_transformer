"""
SQLite database module for ASCII art storage and retrieval.

.. deprecated::
    This module is DEPRECATED. Use ``python.data.db`` instead, which is the
    canonical database layer and uses the external schema at ``data/schema.sql``.

    This module remains for backward compatibility with ``test_database.py``
    tests but should not be used in new code. See bead bd-1xt for context.

This module implements the database schema and utility functions for:
- Storing ASCII art with rich metadata
- Content-based deduplication via SHA256 hashing
- Full-text search using FTS5
- Computing metrics at ingest time
"""

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Character set classification patterns
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*[mGKHF]")
BOX_DRAWING_RANGE = range(0x2500, 0x2580)
BLOCK_ELEMENT_RANGE = range(0x2580, 0x25A0)


@dataclass
class AsciiArt:
    """Data class representing an ASCII art piece with metadata."""

    id: Optional[int]
    content_hash: str
    raw_text: str
    source: str
    source_id: Optional[str]
    title: Optional[str]
    description: Optional[str]
    category: Optional[str]
    subcategory: Optional[str]
    tags: Optional[list[str]]
    artist: Optional[str]
    width: int
    height: int
    total_chars: int
    non_space_chars: int
    char_density: float
    charset: str
    char_histogram: dict[str, int]
    uses_box_drawing: bool
    uses_block_chars: bool
    has_ansi_codes: bool
    is_valid: bool
    quality_score: Optional[float]


def compute_content_hash(text: str) -> str:
    """Compute SHA256 hash of content for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_metrics(text: str) -> dict:
    """
    Compute all metrics for an ASCII art piece.

    Returns dict with: width, height, total_chars, non_space_chars,
    char_density, charset, char_histogram, uses_box_drawing,
    uses_block_chars, has_ansi_codes
    """
    lines = text.split("\n")
    height = len(lines)
    width = max(len(line) for line in lines) if lines else 0
    total_chars = len(text)
    non_space_chars = sum(1 for c in text if not c.isspace())

    # Avoid division by zero
    area = width * height if width * height > 0 else 1
    char_density = non_space_chars / area

    # Character histogram
    char_histogram: dict[str, int] = {}
    for char in text:
        char_histogram[char] = char_histogram.get(char, 0) + 1

    # Charset detection
    has_ansi = bool(ANSI_ESCAPE_PATTERN.search(text))
    uses_box_drawing = False
    uses_block_chars = False
    charset = "ascii"

    for char in text:
        code = ord(char)
        if code > 127:
            charset = "unicode"
            if code in BOX_DRAWING_RANGE:
                uses_box_drawing = True
            elif code in BLOCK_ELEMENT_RANGE:
                uses_block_chars = True
        elif code > 126 or (code < 32 and code not in (9, 10, 13)):
            # Extended ASCII (excluding tab, newline, carriage return)
            if charset == "ascii":
                charset = "extended"

    return {
        "width": width,
        "height": height,
        "total_chars": total_chars,
        "non_space_chars": non_space_chars,
        "char_density": char_density,
        "charset": charset,
        "char_histogram": char_histogram,
        "uses_box_drawing": uses_box_drawing,
        "uses_block_chars": uses_block_chars,
        "has_ansi_codes": has_ansi,
    }


def validate_art(text: str, metrics: dict) -> bool:
    """
    Validate ASCII art quality.

    Rejects:
    - Empty or whitespace-only content
    - Extremely small (<3 lines)
    - Extremely large (>500 lines)
    """
    if not text or not text.strip():
        return False
    if metrics["height"] < 3:
        return False
    if metrics["height"] > 500:
        return False
    return True


class AsciiArtDatabase:
    """SQLite database interface for ASCII art storage."""

    SCHEMA = """
    -- Main table for ASCII art pieces
    CREATE TABLE IF NOT EXISTS ascii_art (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content_hash TEXT UNIQUE NOT NULL,
        raw_text TEXT NOT NULL,

        -- Descriptive metadata
        source TEXT NOT NULL,
        source_id TEXT,
        title TEXT,
        description TEXT,
        category TEXT,
        subcategory TEXT,
        tags TEXT,  -- JSON array
        artist TEXT,

        -- Computed metrics
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        total_chars INTEGER NOT NULL,
        non_space_chars INTEGER NOT NULL,
        char_density REAL,

        -- Character analysis
        charset TEXT NOT NULL,
        char_histogram TEXT,  -- JSON map
        uses_box_drawing INTEGER DEFAULT 0,
        uses_block_chars INTEGER DEFAULT 0,

        -- Quality indicators
        has_ansi_codes INTEGER DEFAULT 0,
        is_valid INTEGER DEFAULT 1,
        quality_score REAL,

        -- Timestamps
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Indexes for common queries
    CREATE INDEX IF NOT EXISTS idx_source ON ascii_art(source);
    CREATE INDEX IF NOT EXISTS idx_category ON ascii_art(category);
    CREATE INDEX IF NOT EXISTS idx_width ON ascii_art(width);
    CREATE INDEX IF NOT EXISTS idx_height ON ascii_art(height);
    CREATE INDEX IF NOT EXISTS idx_charset ON ascii_art(charset);
    CREATE INDEX IF NOT EXISTS idx_is_valid ON ascii_art(is_valid);

    -- FTS5 full-text search on text fields
    CREATE VIRTUAL TABLE IF NOT EXISTS ascii_art_fts USING fts5(
        title,
        description,
        tags,
        category,
        subcategory,
        content=ascii_art,
        content_rowid=id
    );

    -- Triggers to keep FTS in sync
    CREATE TRIGGER IF NOT EXISTS ascii_art_ai AFTER INSERT ON ascii_art BEGIN
        INSERT INTO ascii_art_fts(rowid, title, description, tags, category, subcategory)
        VALUES (new.id, new.title, new.description, new.tags, new.category, new.subcategory);
    END;

    CREATE TRIGGER IF NOT EXISTS ascii_art_ad AFTER DELETE ON ascii_art BEGIN
        INSERT INTO ascii_art_fts(ascii_art_fts, rowid, title, description, tags, category, subcategory)
        VALUES ('delete', old.id, old.title, old.description, old.tags, old.category, old.subcategory);
    END;

    CREATE TRIGGER IF NOT EXISTS ascii_art_au AFTER UPDATE ON ascii_art BEGIN
        INSERT INTO ascii_art_fts(ascii_art_fts, rowid, title, description, tags, category, subcategory)
        VALUES ('delete', old.id, old.title, old.description, old.tags, old.category, old.subcategory);
        INSERT INTO ascii_art_fts(rowid, title, description, tags, category, subcategory)
        VALUES (new.id, new.title, new.description, new.tags, new.category, new.subcategory);
    END;
    """

    def __init__(self, db_path: str | Path):
        """Initialize database connection and create schema if needed."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def insert(
        self,
        raw_text: str,
        source: str,
        source_id: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        subcategory: Optional[str] = None,
        tags: Optional[list[str]] = None,
        artist: Optional[str] = None,
        quality_score: Optional[float] = None,
    ) -> Optional[int]:
        """
        Insert an ASCII art piece into the database.

        Computes content hash and metrics automatically.
        Returns the ID of the inserted row, or None if duplicate.
        """
        content_hash = compute_content_hash(raw_text)

        # Check for duplicate
        existing = self.conn.execute(
            "SELECT id FROM ascii_art WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        if existing:
            return None  # Duplicate

        metrics = compute_metrics(raw_text)
        is_valid = validate_art(raw_text, metrics)

        cursor = self.conn.execute(
            """
            INSERT INTO ascii_art (
                content_hash, raw_text, source, source_id, title, description,
                category, subcategory, tags, artist, width, height, total_chars,
                non_space_chars, char_density, charset, char_histogram,
                uses_box_drawing, uses_block_chars, has_ansi_codes, is_valid,
                quality_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                content_hash,
                raw_text,
                source,
                source_id,
                title,
                description,
                category,
                subcategory,
                json.dumps(tags) if tags else None,
                artist,
                metrics["width"],
                metrics["height"],
                metrics["total_chars"],
                metrics["non_space_chars"],
                metrics["char_density"],
                metrics["charset"],
                json.dumps(metrics["char_histogram"]),
                int(metrics["uses_box_drawing"]),
                int(metrics["uses_block_chars"]),
                int(metrics["has_ansi_codes"]),
                int(is_valid),
                quality_score,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_by_id(self, art_id: int) -> Optional[AsciiArt]:
        """Retrieve an ASCII art piece by ID."""
        row = self.conn.execute(
            "SELECT * FROM ascii_art WHERE id = ?", (art_id,)
        ).fetchone()
        return self._row_to_ascii_art(row) if row else None

    def get_by_hash(self, content_hash: str) -> Optional[AsciiArt]:
        """Retrieve an ASCII art piece by content hash."""
        row = self.conn.execute(
            "SELECT * FROM ascii_art WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return self._row_to_ascii_art(row) if row else None

    def search_fts(self, query: str, limit: int = 100) -> list[AsciiArt]:
        """
        Full-text search across title, description, tags, category.

        Supports FTS5 query syntax (AND, OR, NOT, phrases, etc.)
        """
        rows = self.conn.execute(
            """
            SELECT ascii_art.* FROM ascii_art
            JOIN ascii_art_fts ON ascii_art.id = ascii_art_fts.rowid
            WHERE ascii_art_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        return [self._row_to_ascii_art(row) for row in rows]

    def search_by_size(
        self,
        min_width: Optional[int] = None,
        max_width: Optional[int] = None,
        min_height: Optional[int] = None,
        max_height: Optional[int] = None,
        limit: int = 100,
    ) -> list[AsciiArt]:
        """Search for art within size constraints."""
        conditions = ["is_valid = 1"]
        params: list = []

        if min_width is not None:
            conditions.append("width >= ?")
            params.append(min_width)
        if max_width is not None:
            conditions.append("width <= ?")
            params.append(max_width)
        if min_height is not None:
            conditions.append("height >= ?")
            params.append(min_height)
        if max_height is not None:
            conditions.append("height <= ?")
            params.append(max_height)

        where_clause = " AND ".join(conditions)
        rows = self.conn.execute(
            f"SELECT * FROM ascii_art WHERE {where_clause} LIMIT ?", params + [limit]
        ).fetchall()
        return [self._row_to_ascii_art(row) for row in rows]

    def search_by_category(
        self,
        category: str,
        subcategory: Optional[str] = None,
        limit: int = 100,
    ) -> list[AsciiArt]:
        """Search for art by category and optional subcategory."""
        if subcategory:
            rows = self.conn.execute(
                """
                SELECT * FROM ascii_art
                WHERE category = ? AND subcategory = ? AND is_valid = 1
                LIMIT ?
                """,
                (category, subcategory, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT * FROM ascii_art
                WHERE category = ? AND is_valid = 1
                LIMIT ?
                """,
                (category, limit),
            ).fetchall()
        return [self._row_to_ascii_art(row) for row in rows]

    def count(self, valid_only: bool = True) -> int:
        """Count total records in database."""
        if valid_only:
            return self.conn.execute(
                "SELECT COUNT(*) FROM ascii_art WHERE is_valid = 1"
            ).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM ascii_art").fetchone()[0]

    def get_stats(self) -> dict:
        """Get database statistics."""
        total = self.count(valid_only=False)
        valid = self.count(valid_only=True)

        # Source distribution
        sources = dict(
            self.conn.execute(
                "SELECT source, COUNT(*) FROM ascii_art GROUP BY source"
            ).fetchall()
        )

        # Category distribution
        categories = dict(
            self.conn.execute(
                "SELECT category, COUNT(*) FROM ascii_art WHERE category IS NOT NULL GROUP BY category"
            ).fetchall()
        )

        # Charset distribution
        charsets = dict(
            self.conn.execute(
                "SELECT charset, COUNT(*) FROM ascii_art GROUP BY charset"
            ).fetchall()
        )

        return {
            "total": total,
            "valid": valid,
            "invalid": total - valid,
            "sources": sources,
            "categories": categories,
            "charsets": charsets,
        }

    def iterate_valid(self, batch_size: int = 1000):
        """
        Iterate over all valid ASCII art in batches.

        Yields AsciiArt objects one at a time, fetching in batches for efficiency.
        """
        offset = 0
        while True:
            rows = self.conn.execute(
                """
                SELECT * FROM ascii_art
                WHERE is_valid = 1
                ORDER BY id
                LIMIT ? OFFSET ?
                """,
                (batch_size, offset),
            ).fetchall()

            if not rows:
                break

            for row in rows:
                yield self._row_to_ascii_art(row)

            offset += batch_size

    def _row_to_ascii_art(self, row: sqlite3.Row) -> AsciiArt:
        """Convert a database row to AsciiArt dataclass."""
        return AsciiArt(
            id=row["id"],
            content_hash=row["content_hash"],
            raw_text=row["raw_text"],
            source=row["source"],
            source_id=row["source_id"],
            title=row["title"],
            description=row["description"],
            category=row["category"],
            subcategory=row["subcategory"],
            tags=json.loads(row["tags"]) if row["tags"] else None,
            artist=row["artist"],
            width=row["width"],
            height=row["height"],
            total_chars=row["total_chars"],
            non_space_chars=row["non_space_chars"],
            char_density=row["char_density"],
            charset=row["charset"],
            char_histogram=json.loads(row["char_histogram"])
            if row["char_histogram"]
            else {},
            uses_box_drawing=bool(row["uses_box_drawing"]),
            uses_block_chars=bool(row["uses_block_chars"]),
            has_ansi_codes=bool(row["has_ansi_codes"]),
            is_valid=bool(row["is_valid"]),
            quality_score=row["quality_score"],
        )


def create_database(db_path: str | Path) -> AsciiArtDatabase:
    """Create a new database at the specified path."""
    return AsciiArtDatabase(db_path)


if __name__ == "__main__":
    # Quick test
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    print(f"Testing database at {db_path}")

    with AsciiArtDatabase(db_path) as db:
        # Insert test art
        test_art = """
 /\\_/\\
( o.o )
 > ^ <
"""
        art_id = db.insert(
            raw_text=test_art,
            source="test",
            title="Cute Cat",
            description="A simple ASCII cat face",
            category="animal",
            subcategory="cat",
            tags=["cute", "simple", "face"],
        )
        print(f"Inserted art with ID: {art_id}")

        # Retrieve and verify
        art = db.get_by_id(art_id)
        print(
            f"Retrieved: {art.title}, {art.width}x{art.height}, charset={art.charset}"
        )

        # Test deduplication
        dup_id = db.insert(raw_text=test_art, source="test")
        print(f"Duplicate insert returned: {dup_id}")

        # Stats
        print(f"Stats: {db.get_stats()}")

    print("Database test passed!")
