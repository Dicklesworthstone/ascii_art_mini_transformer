-- SQLite schema for ASCII art storage.
--
-- Design goals:
-- - Preserve whitespace and line breaks; line endings are normalized to '\n'.
-- - Deduplicate by SHA256 content_hash.
-- - Store computed metadata at ingest time for fast filtering/training selection.
-- - Provide FTS5 search over descriptive fields.

BEGIN;

CREATE TABLE IF NOT EXISTS ascii_art (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT UNIQUE NOT NULL,      -- SHA256 of raw_text (with normalized line endings) for deduplication
    raw_text TEXT NOT NULL,                 -- The actual ASCII art (whitespace preserved; line endings normalized)

    -- Descriptive metadata
    source TEXT NOT NULL,                   -- 'huggingface', 'asciiart.eu', 'figlet', etc.
    source_id TEXT,                         -- Original ID from source if available
    title TEXT,                             -- Title/name if available
    description TEXT,                       -- Full text description for training
    category TEXT,                          -- 'animal', 'banner', 'object', 'scene', etc.
    subcategory TEXT,                       -- 'snake', 'christmas', 'house', etc.
    tags TEXT,                              -- JSON array of tags
    artist TEXT,                            -- Original artist if known

    -- Computed metrics
    width INTEGER NOT NULL,                 -- Max line width in characters
    height INTEGER NOT NULL,                -- Number of lines
    total_chars INTEGER NOT NULL,           -- Total character count (including newlines)
    non_space_chars INTEGER NOT NULL,       -- Non-whitespace count
    char_density REAL,                      -- non_space_chars / (width * height)

    -- Character analysis
    charset TEXT NOT NULL,                  -- 'ascii', 'extended', 'unicode'
    char_histogram TEXT,                    -- JSON map of char -> count
    uses_box_drawing INTEGER DEFAULT 0,     -- Uses box-drawing characters
    uses_block_chars INTEGER DEFAULT 0,     -- Uses block element characters

    -- Quality indicators
    has_ansi_codes INTEGER DEFAULT 0,       -- Contains ANSI escape sequences
    is_valid INTEGER DEFAULT 1,             -- Passes basic validation
    quality_score REAL,                     -- Computed quality score (optional)

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_ascii_art_source ON ascii_art(source);
CREATE INDEX IF NOT EXISTS idx_ascii_art_category ON ascii_art(category);
CREATE INDEX IF NOT EXISTS idx_ascii_art_width ON ascii_art(width);
CREATE INDEX IF NOT EXISTS idx_ascii_art_height ON ascii_art(height);
CREATE INDEX IF NOT EXISTS idx_ascii_art_charset ON ascii_art(charset);
-- Note: UNIQUE constraint on content_hash already creates an index, but we keep an
-- explicit name for clarity and compatibility with the design doc.
CREATE UNIQUE INDEX IF NOT EXISTS idx_ascii_art_content_hash ON ascii_art(content_hash);

-- Full-text search (external content table).
CREATE VIRTUAL TABLE IF NOT EXISTS ascii_art_fts USING fts5(
    title,
    description,
    tags,
    category,
    subcategory,
    content='ascii_art',
    content_rowid='id'
);

-- Keep FTS index up to date.
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

COMMIT;
