"""
HuggingFace dataset ingestion script for ASCII art database.

Ingests four pre-built datasets:
1. Csplk/THE.ASCII.ART.EMPORIUM (3.1M rows)
2. mrzjy/ascii_art_generation_140k (138k instruction-response pairs)
3. apehex/ascii-art (47k with rich metadata)
4. jdpressman/retro-ascii-art-v1 (6k+ synthetic)

Features:
- Streaming for memory-efficient processing of large datasets
- Progress tracking with resumability
- Proper metadata mapping to schema
- Deduplication via content hash
- Comprehensive logging
"""

import itertools
import json
import logging
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, TypeVar

import pyarrow as pa
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from data.db import connect, initialize, insert_ascii_art  # noqa: E402

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _configure_logging(*, verbose: bool) -> None:
    # Avoid configuring logging on import; configure only when running as a CLI.
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("ingestion.log"),
        ],
    )


def retry_on_lock(
    func: Callable[[], T],
    *,
    max_retries: int = 5,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
) -> T:
    """Retry SQLite operations with exponential backoff on lock errors."""
    for attempt in range(max_retries):
        try:
            return func()
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt >= max_retries - 1:
                raise
            delay = min(base_delay * (2**attempt), max_delay)
            logger.debug(
                "SQLite locked; retrying in %.2fs (attempt %d/%d)",
                delay,
                attempt + 1,
                max_retries,
            )
            time.sleep(delay)
    raise RuntimeError("Retry logic error")


@dataclass
class IngestionStats:
    """Statistics for a single dataset ingestion."""

    dataset_name: str
    total_processed: int = 0
    inserted: int = 0
    duplicates: int = 0
    skipped_invalid: int = 0
    errors: int = 0


class ProgressTracker:
    """Track and persist ingestion progress for resumability."""

    def __init__(self, progress_file: str = "ingestion_progress.json") -> None:
        self.progress_file = Path(progress_file)
        self.state: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                return loaded
        return {"completed_datasets": [], "last_row": {}, "stats": {}}

    def save(self) -> None:
        with open(self.progress_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def is_completed(self, dataset_name: str) -> bool:
        return dataset_name in self.state["completed_datasets"]

    def mark_completed(self, dataset_name: str, stats: IngestionStats) -> None:
        if dataset_name not in self.state["completed_datasets"]:
            self.state["completed_datasets"].append(dataset_name)
        self.state["stats"][dataset_name] = {
            "total_processed": stats.total_processed,
            "inserted": stats.inserted,
            "duplicates": stats.duplicates,
            "skipped_invalid": stats.skipped_invalid,
            "errors": stats.errors,
        }
        self.save()

    def update_progress(self, dataset_name: str, row_idx: int) -> None:
        self.state["last_row"][dataset_name] = row_idx
        self.save()

    def get_start_row(self, dataset_name: str) -> int:
        raw = self.state.get("last_row", {}).get(dataset_name, 0)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    def reset_dataset(self, dataset_name: str, *, start_row: int = 0) -> None:
        """Clear completion/progress for a dataset so it can be re-processed."""
        if dataset_name in self.state["completed_datasets"]:
            self.state["completed_datasets"].remove(dataset_name)
        self.state["last_row"][dataset_name] = start_row
        self.state["stats"].pop(dataset_name, None)
        self.save()


def _csplk_is_file_marker(line: str) -> bool:
    return line.startswith("File: ")


def _csplk_parse_file_path(line: str) -> str:
    # Format: "File: some/path.txt"
    return line[len("File: ") :].strip()


def _csplk_metadata_from_path(file_path: str) -> dict[str, Any]:
    parts = [p for p in file_path.split("/") if p]
    category = parts[0] if parts else None
    subcategory = parts[1] if len(parts) > 1 else None

    # Use the file stem as a weak title/description fallback.
    stem = Path(parts[-1]).stem if parts else "ascii_art"
    title = stem.replace("_", " ").replace("-", " ").strip() or None

    tags: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p.lower().endswith((".txt", ".gmi", ".asc", ".ans")):
            p = Path(p).stem
        tags.append(p.replace("_", " ").replace("-", " ").strip())

    # De-dupe while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for t in tags:
        key = t.lower()
        if not t or key in seen:
            continue
        seen.add(key)
        deduped.append(t)

    return {
        "category": category,
        "subcategory": subcategory,
        "title": title,
        "description": title,
        "tags": deduped or None,
    }


def _csplk_non_alnum_ratio(text: str) -> float:
    nonspace = [ch for ch in text if not ch.isspace()]
    if not nonspace:
        return 0.0
    non_alnum = sum(1 for ch in nonspace if not ch.isalnum())
    return non_alnum / len(nonspace)


def _csplk_digit_ratio(text: str) -> float:
    nonspace = [ch for ch in text if not ch.isspace()]
    if not nonspace:
        return 0.0
    digits = sum(1 for ch in nonspace if ch.isdigit())
    return digits / len(nonspace)


def _csplk_max_run_len(text: str) -> int:
    # Long runs of the same char are common in ASCII art (e.g. "XXXXXX", "======").
    longest = 0
    current = 0
    prev: str | None = None
    for ch in text:
        if ch.isspace():
            prev = None
            current = 0
            continue
        if prev == ch:
            current += 1
        else:
            prev = ch
            current = 1
        if current > longest:
            longest = current
    return longest


def _csplk_is_separator_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 3:
        return False
    if stripped.startswith("File: "):
        return False
    # Common separators: "-----", "=====", "*****", "_____", etc.
    if len(set(stripped)) == 1 and stripped[0] in "-=_*~_":
        return True
    return False


def _csplk_is_title_line(line: str) -> bool:
    if not line or line != line.lstrip():
        return False

    text = line.strip()
    if not text:
        return False
    if len(text) > 80:
        return False
    if text.startswith(("http://", "https://")):
        return False
    if text in {"```", "```text"}:
        return False
    if text.startswith("File: "):
        return False
    if _csplk_is_separator_line(text):
        return False
    if not any(ch.isalpha() for ch in text):
        return False

    # Avoid mis-classifying ASCII art lines that start at column 0 (e.g. "Y8888888b...").
    if _csplk_max_run_len(text) >= 8:
        return False
    if _csplk_digit_ratio(text) > 0.2:
        return False

    return _csplk_non_alnum_ratio(text) <= 0.3


def _csplk_looks_like_art_line(line: str) -> bool:
    text = line.rstrip()
    if not text.strip():
        return False
    if text.startswith("File: "):
        return False
    if _csplk_is_separator_line(text):
        return False
    if any(
        ch in text
        for ch in ("\\", "/", "|", "_", "(", ")", "[", "]", "{", "}", "<", ">")
    ):
        return True
    # Lines with very low charset variety (after stripping) are often part of big ASCII banners,
    # even if they contain only alnum + spaces (e.g. "            XX").
    nonspace = [ch for ch in text if not ch.isspace()]
    if len(nonspace) >= 2 and len(set(nonspace)) <= 3 and len(text) >= 16:
        return True
    # Banner-like art often uses long runs of the same character (e.g. "XXXXXX", "OOOOOO").
    if len(text) >= 12 and _csplk_max_run_len(text) >= 6:
        return True
    return _csplk_non_alnum_ratio(text) >= 0.5


def _finalize_block(lines: list[str]) -> str | None:
    # Trim leading/trailing blank lines but preserve internal whitespace.
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return None
    return "\n".join(lines)


def _iter_csplk_blocks(
    ds: Iterable[dict[str, Any]],
    *,
    start_row: int,
    split_blocks: bool,
) -> Iterable[tuple[int, str, int, dict[str, Any], str]]:
    """
    Yield reconstructed Csplk blocks from the line-stream dataset.

    Yields tuples: (line_idx, file_path, block_idx, metadata, block_text)
    """
    current_path: str | None = None
    base_meta: dict[str, Any] | None = None
    current_title: str | None = None
    block_lines: list[str] = []
    pending_blanks: list[str] = []
    block_idx = 0

    def flush(idx: int) -> Optional[tuple[int, str, int, dict[str, Any], str]]:
        nonlocal block_idx
        text = _finalize_block(block_lines)
        block_lines.clear()
        pending_blanks.clear()
        if text is None or current_path is None or base_meta is None:
            return None
        block_idx += 1
        meta = dict(base_meta)
        if current_title:
            meta["title"] = current_title
            meta["description"] = current_title
        return (idx, current_path, block_idx, meta, text)

    last_idx = start_row
    for idx, row in enumerate(ds):
        if idx < start_row:
            continue
        last_idx = idx

        line = row.get("text")
        if not isinstance(line, str):
            continue

        if _csplk_is_file_marker(line):
            # New file begins; flush any pending block from previous file.
            flushed = flush(idx)
            if flushed is not None:
                yield flushed

            current_path = _csplk_parse_file_path(line)
            base_meta = _csplk_metadata_from_path(current_path)
            current_title = None
            block_idx = 0
            continue

        if current_path is None or base_meta is None:
            # Skip preamble before first file marker.
            continue

        if not split_blocks:
            block_lines.append(line)
            continue

        if not line.strip():
            # Defer deciding whether blanks are internal whitespace or separators until we see the
            # next non-empty line.
            if block_lines:
                pending_blanks.append(line)
            continue

        if pending_blanks and block_lines:
            # Heuristic split: treat blank lines as separators when either:
            # - there are 2+ consecutive blanks, or
            # - the next line is a title/separator, or
            # - we see "art -> blank -> art" and the current block already has multiple lines.
            next_is_title = _csplk_is_title_line(line)
            next_is_sep = _csplk_is_separator_line(line)
            art_to_art = (
                len(pending_blanks) >= 1
                and len(block_lines) >= 3
                and _csplk_looks_like_art_line(block_lines[-1])
                and _csplk_looks_like_art_line(line)
            )
            if len(pending_blanks) >= 2 or next_is_title or next_is_sep or art_to_art:
                flushed = flush(idx)
                if flushed is not None:
                    yield flushed
                current_title = None
            else:
                block_lines.extend(pending_blanks)
                pending_blanks.clear()

        if _csplk_is_title_line(line):
            if block_lines:
                flushed = flush(idx)
                if flushed is not None:
                    yield flushed
            current_title = line.strip()
            continue

        if _csplk_is_separator_line(line):
            if block_lines:
                flushed = flush(idx)
                if flushed is not None:
                    yield flushed
            current_title = None
            continue

        if (
            not block_lines
            and current_title is None
            and not _csplk_looks_like_art_line(line)
        ):
            # Preamble text before the first art block in a file.
            continue

        block_lines.append(line)

    # Flush final block at EOF.
    flushed = flush(last_idx)
    if flushed is not None:
        yield flushed


def load_mrzjy_dataset() -> Iterable[dict[str, Any]]:
    """
    Load mrzjy/ascii_art_generation_140k using pyarrow directly.

    The HuggingFace datasets library has issues with this dataset's format,
    so we read the Arrow file directly.
    """
    path = hf_hub_download(
        repo_id="mrzjy/ascii_art_generation_140k",
        filename="train/data-00000-of-00001.arrow",
        repo_type="dataset",
    )

    with open(path, "rb") as source:
        reader = pa.ipc.open_stream(source)
        table = reader.read_all()

    # Convert to list of dicts
    for i in range(table.num_rows):
        row = {}
        for col in table.column_names:
            val = table.column(col)[i].as_py()
            row[col] = val
        yield row


def extract_art_text(row: dict[str, Any], source: str) -> Optional[str]:
    """
    Extract ASCII art text from a dataset row.

    Different datasets have different column names and formats.
    """

    def _decode_newline_escapes(text: str) -> str:
        # Some dataset rows store newline escapes literally (e.g. "\\n") instead of real newlines.
        # Decode newline escapes and escaped backslashes, but avoid interpreting other escape sequences.
        return (
            text.replace("\\r\\n", "\n")
            .replace("\\r", "\n")
            .replace("\\n", "\n")
            .replace("\\\\", "\\")
        )

    # Handle mrzjy conversation format
    if source == "mrzjy/ascii_art_generation_140k":
        if "conversations" in row and row["conversations"]:
            convs = row["conversations"]
            # Find assistant response (usually the second message)
            for conv in convs:
                if conv.get("role") == "assistant":
                    content = conv.get("content", "")
                    # Extract ASCII art from between triple backticks
                    if "```" in content:
                        parts = content.split("```")
                        if len(parts) >= 2:
                            art = _decode_newline_escapes(parts[1]).strip("\r\n")
                            if art.strip():
                                return art
                    # If no backticks, use content directly
                    else:
                        candidate = _decode_newline_escapes(content).strip()
                        if candidate:
                            return candidate
        return None

    # Handle jdpressman dataset with art_aic/art_i2a columns
    if source == "jdpressman/retro-ascii-art-v1":
        # Prefer art_aic (ASCII Image Converter), fallback to art_i2a
        for col in ["art_aic", "art_i2a"]:
            if col in row and row[col]:
                text = row[col]
                if isinstance(text, str) and text.strip():
                    return text
        return None

    # Try common column names in order of preference
    text_columns = ["text", "content", "response", "ascii_art", "art"]
    for col in text_columns:
        if col in row and row[col]:
            text = row[col]
            if isinstance(text, str) and text.strip():
                return text
    return None


def extract_description(row: dict[str, Any], source: str) -> Optional[str]:
    """Extract description/instruction from a dataset row."""
    desc_columns = ["instruction", "description", "caption", "prompt", "title"]
    for col in desc_columns:
        if col in row and row[col]:
            desc = row[col]
            if isinstance(desc, str) and desc.strip():
                return desc.strip()
    return None


def extract_metadata(row: dict[str, Any], source: str) -> dict[str, Any]:
    """Extract all available metadata from a dataset row."""
    metadata: dict[str, Any] = {
        "title": None,
        "description": None,
        "category": None,
        "subcategory": None,
        "tags": None,
        "artist": None,
    }

    # Extract description
    metadata["description"] = extract_description(row, source)

    # Source-specific metadata extraction
    if source == "apehex/ascii-art":
        # apehex has rich metadata
        if "labels" in row and row["labels"]:
            labels = row["labels"]
            if isinstance(labels, list) and labels:
                metadata["category"] = labels[0] if labels else None
                metadata["tags"] = labels
        if "caption" in row:
            caption = row.get("caption")
            metadata["title"] = caption[:200] if caption else None

    elif source == "mrzjy/ascii_art_generation_140k":
        # Extract instruction from conversation format
        if "conversations" in row and row["conversations"]:
            convs = row["conversations"]
            for conv in convs:
                if conv.get("role") == "user":
                    metadata["description"] = conv.get("content", "")
                    break
        elif "instruction" in row:
            metadata["description"] = row["instruction"]

    elif source == "jdpressman/retro-ascii-art-v1":
        # Use prompt as description, style as category
        if "prompt" in row:
            metadata["description"] = row["prompt"]
        if "style" in row:
            metadata["category"] = row["style"]
        if "subject" in row:
            metadata["subcategory"] = row["subject"]

    return metadata


def validate_art(text: str) -> tuple[bool, str]:
    """
    Validate ASCII art before insertion.

    Returns (is_valid, reason).
    """
    if not text or not text.strip():
        return False, "empty"

    # Size limits
    if len(text) > 500_000:  # 500KB limit
        return False, "too_large"

    if len(text) < 10:  # Minimum size
        return False, "too_small"

    # Check for binary data
    if "\x00" in text:
        return False, "binary_data"

    # Check line count
    lines = text.split("\n")
    if len(lines) < 2:
        return False, "too_few_lines"
    if len(lines) > 1000:
        return False, "too_many_lines"

    return True, "ok"


def ingest_dataset(
    conn: sqlite3.Connection,
    dataset_name: str,
    config: Optional[str],
    tracker: ProgressTracker,
    checkpoint_every: int = 10000,
    *,
    max_inserts: Optional[int] = None,
    stop_at_total_rows: Optional[int] = None,
    split_blocks: bool = True,
    force: bool = False,
    start_row_override: Optional[int] = None,
) -> IngestionStats:
    """
    Ingest a single HuggingFace dataset into the database.

    Args:
        conn: SQLite database connection
        dataset_name: HuggingFace dataset identifier
        config: Dataset configuration/subset name
        tracker: Progress tracker for resumability
        checkpoint_every: Save progress every N rows
    """
    full_name = f"{dataset_name}/{config}" if config else dataset_name
    stats = IngestionStats(dataset_name=full_name)

    if force:
        start_row = 0 if start_row_override is None else start_row_override
        tracker.reset_dataset(full_name, start_row=start_row)
    else:
        if tracker.is_completed(full_name):
            logger.info(f"Skipping {full_name} (already completed)")
            return stats
        start_row = (
            start_row_override
            if start_row_override is not None
            else tracker.get_start_row(full_name)
        )
    logger.info(f"Starting ingestion of {full_name}")
    if start_row > 0:
        logger.info(f"Resuming from row {start_row}")

    try:
        # Use explicit transactions for bulk ingestion (python/data/db.py uses autocommit).
        conn.execute("BEGIN")
        # Give SQLite a chance to wait briefly on transient writer locks before raising.
        retry_on_lock(lambda: conn.execute("PRAGMA busy_timeout = 5000;"))

        if dataset_name == "Csplk/THE.ASCII.ART.EMPORIUM":
            ds = load_dataset(dataset_name, config, split="train", streaming=True)
            progress_bar = tqdm(
                total=None, desc=f"Ingesting {dataset_name}", initial=start_row
            )
            last_progress_idx = start_row
            last_checkpoint_idx = start_row

            for line_idx, file_path, block_idx, meta, block_text in _iter_csplk_blocks(
                ds,
                start_row=start_row,
                split_blocks=split_blocks,
            ):
                if line_idx > last_progress_idx:
                    progress_bar.update(line_idx - last_progress_idx)
                    last_progress_idx = line_idx

                stats.total_processed += 1

                # Validate
                is_valid, _reason = validate_art(block_text)
                if not is_valid:
                    stats.skipped_invalid += 1
                    continue

                try:
                    row_id = retry_on_lock(
                        lambda: insert_ascii_art(
                            conn,
                            raw_text=block_text,
                            source=dataset_name,
                            source_id=f"{file_path}#{block_idx}",
                            title=meta.get("title"),
                            description=meta.get("description"),
                            category=meta.get("category"),
                            subcategory=meta.get("subcategory"),
                            tags=meta.get("tags"),
                            artist=meta.get("artist"),
                        )
                    )
                    if row_id is not None:
                        stats.inserted += 1
                    else:
                        stats.duplicates += 1
                except Exception as e:
                    stats.errors += 1
                    if stats.errors <= 10:
                        logger.error(f"Insert error at line {line_idx}: {e}")

                if max_inserts is not None and stats.inserted >= max_inserts:
                    logger.info(f"Reached max_inserts={max_inserts}, stopping early.")
                    retry_on_lock(conn.commit)
                    tracker.update_progress(full_name, line_idx)
                    progress_bar.close()
                    return stats

                if (
                    line_idx > last_checkpoint_idx
                    and (line_idx - last_checkpoint_idx) >= checkpoint_every
                ):
                    retry_on_lock(conn.commit)
                    tracker.update_progress(full_name, line_idx)
                    last_checkpoint_idx = line_idx
                    progress_bar.set_postfix(
                        inserted=stats.inserted,
                        dups=stats.duplicates,
                        skip=stats.skipped_invalid,
                    )

                    if stop_at_total_rows is not None:
                        total = retry_on_lock(
                            lambda: conn.execute(
                                "SELECT COUNT(*) FROM ascii_art"
                            ).fetchone()[0]
                        )
                        if total >= stop_at_total_rows:
                            logger.info(
                                f"Reached stop_at_total_rows={stop_at_total_rows} (total={total}), stopping early."
                            )
                            tracker.update_progress(full_name, line_idx)
                            progress_bar.close()
                            return stats
                    retry_on_lock(lambda: conn.execute("BEGIN"))

            progress_bar.close()

            retry_on_lock(conn.commit)
            tracker.update_progress(full_name, last_progress_idx)
            tracker.mark_completed(full_name, stats)
            logger.info(
                f"Completed {full_name}: {stats.total_processed} processed, "
                f"{stats.inserted} inserted, {stats.duplicates} duplicates, "
                f"{stats.skipped_invalid} skipped, {stats.errors} errors"
            )
            return stats

        # Special handling for mrzjy dataset which has issues with HF datasets library
        if dataset_name == "mrzjy/ascii_art_generation_140k":
            logger.info(f"Loading {dataset_name} with pyarrow...")
            ds = load_mrzjy_dataset()
            total_rows = 138941  # Known size
        else:
            # Load dataset with streaming for memory efficiency
            ds = load_dataset(dataset_name, config, split="train", streaming=True)

            # Get approximate size for progress bar if available
            total_rows = None
            try:
                ds_info = load_dataset(
                    dataset_name, config, split="train", streaming=False
                )
                total_rows = len(ds_info)
                del ds_info  # Free memory
            except Exception:
                pass

        progress_bar = tqdm(
            enumerate(
                itertools.islice(ds, start_row, None) if start_row > 0 else ds,
                start=start_row,
            ),
            total=total_rows,
            desc=f"Ingesting {dataset_name}",
            initial=start_row,
        )

        last_progress_idx = start_row
        for idx, row in progress_bar:
            last_progress_idx = idx

            stats.total_processed += 1

            # Extract art text
            art_text = extract_art_text(row, dataset_name)
            if not art_text:
                stats.skipped_invalid += 1
                continue

            # Validate
            is_valid, reason = validate_art(art_text)
            if not is_valid:
                stats.skipped_invalid += 1
                continue

            # Extract metadata
            metadata = extract_metadata(row, dataset_name)

            # Insert into database
            try:
                row_id = retry_on_lock(
                    lambda: insert_ascii_art(
                        conn,
                        raw_text=art_text,
                        source=dataset_name,
                        source_id=str(idx),
                        title=metadata["title"],
                        description=metadata["description"],
                        category=metadata["category"],
                        subcategory=metadata["subcategory"],
                        tags=metadata["tags"],
                        artist=metadata["artist"],
                    )
                )
                if row_id is not None:
                    stats.inserted += 1
                else:
                    stats.duplicates += 1
            except Exception as e:
                stats.errors += 1
                if stats.errors <= 10:  # Log first 10 errors
                    logger.error(f"Insert error at row {idx}: {e}")

            if max_inserts is not None and stats.inserted >= max_inserts:
                logger.info(f"Reached max_inserts={max_inserts}, stopping early.")
                retry_on_lock(conn.commit)
                tracker.update_progress(full_name, idx)
                progress_bar.close()
                return stats

            # Checkpoint and commit periodically
            if idx > 0 and idx % checkpoint_every == 0:
                retry_on_lock(conn.commit)
                tracker.update_progress(full_name, idx)
                progress_bar.set_postfix(
                    inserted=stats.inserted,
                    dups=stats.duplicates,
                    skip=stats.skipped_invalid,
                )
                if stop_at_total_rows is not None:
                    total = retry_on_lock(
                        lambda: conn.execute(
                            "SELECT COUNT(*) FROM ascii_art"
                        ).fetchone()[0]
                    )
                    if total >= stop_at_total_rows:
                        logger.info(
                            f"Reached stop_at_total_rows={stop_at_total_rows} (total={total}), stopping early."
                        )
                        tracker.update_progress(full_name, idx)
                        progress_bar.close()
                        return stats
                retry_on_lock(lambda: conn.execute("BEGIN"))

        # Final commit and mark as completed
        retry_on_lock(conn.commit)
        tracker.update_progress(full_name, last_progress_idx)
        tracker.mark_completed(full_name, stats)
        logger.info(
            f"Completed {full_name}: {stats.total_processed} processed, "
            f"{stats.inserted} inserted, {stats.duplicates} duplicates, "
            f"{stats.skipped_invalid} skipped, {stats.errors} errors"
        )

    except Exception as e:
        logger.error(f"Error ingesting {full_name}: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        tracker.save()  # Save progress even on failure
        raise

    return stats


def get_db_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """Get database statistics."""
    total = conn.execute("SELECT COUNT(*) FROM ascii_art").fetchone()[0]
    valid = conn.execute(
        "SELECT COUNT(*) FROM ascii_art WHERE is_valid = 1"
    ).fetchone()[0]

    sources = dict(
        conn.execute(
            "SELECT source, COUNT(*) FROM ascii_art GROUP BY source"
        ).fetchall()
    )

    return {"total": total, "valid": valid, "sources": sources}


def ingest_all_datasets(
    db_path: str = "data/ascii_art.db",
    *,
    progress_file: str = "ingestion_progress.json",
) -> dict[str, IngestionStats]:
    """
    Ingest all configured HuggingFace datasets.

    Returns dict of dataset name -> stats.
    """
    # Dataset configurations: (name, config/subset)
    # NOTE: Csplk/THE.ASCII.ART.EMPORIUM is line-by-line format, needs special reconstruction
    datasets_to_ingest = [
        # Priority 1: Instruction-response pairs with good metadata
        ("mrzjy/ascii_art_generation_140k", None),
        # Priority 2: Rich metadata from apehex
        ("apehex/ascii-art", "asciiart"),
        ("apehex/ascii-art", "copypasta"),
        ("apehex/ascii-art", "graffiti"),
        ("apehex/ascii-art", "images"),
        # Priority 3: Synthetic with art_aic/art_i2a columns
        ("jdpressman/retro-ascii-art-v1", None),
    ]

    tracker = ProgressTracker(progress_file=progress_file)
    all_stats: dict[str, IngestionStats] = {}

    # Ensure database directory exists
    db_path_obj = Path(db_path)
    db_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Open database and initialize schema
    conn = connect(db_path)
    try:
        initialize(conn)

        for dataset_name, config in datasets_to_ingest:
            full_name = f"{dataset_name}/{config}" if config else dataset_name

            try:
                stats = ingest_dataset(conn, dataset_name, config, tracker)
                all_stats[full_name] = stats
            except Exception as e:
                logger.error(f"Failed to ingest {full_name}: {e}")
                # Continue with next dataset
                continue

            # Small delay between datasets
            time.sleep(1)

        # Print final summary
        logger.info("\n" + "=" * 60)
        logger.info("INGESTION COMPLETE - SUMMARY")
        logger.info("=" * 60)

        total_inserted = 0
        total_duplicates = 0
        total_processed = 0

        for name, stats in all_stats.items():
            logger.info(
                f"{name}: "
                f"{stats.inserted} inserted, "
                f"{stats.duplicates} dups, "
                f"{stats.skipped_invalid} skipped"
            )
            total_inserted += stats.inserted
            total_duplicates += stats.duplicates
            total_processed += stats.total_processed

        logger.info("-" * 60)
        logger.info(
            f"TOTAL: {total_processed} processed, {total_inserted} inserted, "
            f"{total_duplicates} duplicates"
        )

        # Print database stats
        db_stats = get_db_stats(conn)
        logger.info(
            f"\nDatabase now contains {db_stats['total']} total rows "
            f"({db_stats['valid']} valid)"
        )

    finally:
        conn.close()

    return all_stats


def main() -> None:
    """Main entry point for HuggingFace ingestion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest HuggingFace ASCII art datasets"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--db-path",
        default="data/ascii_art.db",
        help="Path to SQLite database (default: data/ascii_art.db)",
    )
    parser.add_argument(
        "--progress-file",
        default="ingestion_progress.json",
        help="Path to ingestion progress JSON (default: ingestion_progress.json)",
    )
    parser.add_argument(
        "--dataset",
        help="Ingest only a specific dataset (e.g., 'Csplk/THE.ASCII.ART.EMPORIUM')",
    )
    parser.add_argument(
        "--config",
        help="Dataset config/subset (e.g., 'asciiart' for apehex/ascii-art)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10000,
        help="Save progress and commit every N rows/lines (default: 10000)",
    )
    parser.add_argument(
        "--max-inserts",
        type=int,
        default=None,
        help="Stop early after inserting this many new rows (default: no limit)",
    )
    parser.add_argument(
        "--stop-at-total-rows",
        type=int,
        default=None,
        help="Stop early once total ascii_art rows reaches this threshold (checked at checkpoints)",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=None,
        help="Override resume row/line index (default: use progress file)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process the dataset even if marked completed (resets saved progress for that dataset)",
    )
    parser.add_argument(
        "--no-split-blocks",
        action="store_true",
        help="For Csplk ingestion: treat each file as a single block (no double-blank splitting)",
    )
    args = parser.parse_args()

    _configure_logging(verbose=bool(args.verbose))

    if args.dataset:
        # Ingest single dataset
        db_path_obj = Path(args.db_path)
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)

        tracker = ProgressTracker(progress_file=args.progress_file)
        conn = connect(args.db_path)
        try:
            initialize(conn)
            stats = ingest_dataset(
                conn,
                args.dataset,
                args.config,
                tracker,
                checkpoint_every=args.checkpoint_every,
                max_inserts=args.max_inserts,
                stop_at_total_rows=args.stop_at_total_rows,
                split_blocks=not args.no_split_blocks,
                force=args.force,
                start_row_override=args.start_row,
            )
            logger.info(f"Completed: {stats}")
        finally:
            conn.close()
    else:
        # Ingest all datasets
        ingest_all_datasets(args.db_path, progress_file=args.progress_file)


if __name__ == "__main__":
    main()
