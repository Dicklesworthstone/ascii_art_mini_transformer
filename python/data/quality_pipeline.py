"""
Data Quality Pipeline for ASCII Art Database.

Performs validation, deduplication, and quality analysis on ingested ASCII art.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.db import connect, get_ascii_art_by_id, iter_ascii_art_ids, update_ascii_art  # noqa: E402

# Optional: rapidfuzz for near-duplicate detection
try:
    from rapidfuzz import fuzz

    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    logging.warning("rapidfuzz not installed - near-duplicate detection disabled")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("quality_pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)


def retry_on_lock(
    func: Callable[[], T],
    max_retries: int = 5,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
) -> T:
    """Retry a function with exponential backoff on database lock errors."""
    for attempt in range(max_retries):
        try:
            return func()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < max_retries - 1:
                delay = min(base_delay * (2**attempt), max_delay)
                logger.debug(
                    f"Database locked, retrying in {delay:.1f}s (attempt {attempt + 1})"
                )
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("Retry logic error")


@dataclass
class QualityReport:
    """Summary of quality pipeline run."""

    total_checked: int = 0
    valid: int = 0
    invalid: int = 0
    issues_by_type: dict[str, int] = field(default_factory=Counter)
    charset_distribution: dict[str, int] = field(default_factory=Counter)
    source_distribution: dict[str, int] = field(default_factory=Counter)
    size_buckets: dict[str, int] = field(default_factory=Counter)
    near_duplicates_found: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_checked": self.total_checked,
            "valid": self.valid,
            "invalid": self.invalid,
            "valid_pct": round(self.valid / max(1, self.total_checked) * 100, 2),
            "issues_by_type": dict(self.issues_by_type),
            "charset_distribution": dict(self.charset_distribution),
            "source_distribution": dict(self.source_distribution),
            "size_buckets": dict(self.size_buckets),
            "near_duplicates_found": self.near_duplicates_found,
        }


# Validation thresholds
MIN_NON_SPACE_CHARS = 10
MIN_LINES = 2
MAX_LINES = 500
MAX_TOTAL_CHARS = 50000
MAX_WIDTH = 500


def validate_art(raw_text: str, width: int, height: int) -> list[str]:
    """
    Validate an ASCII art piece for structural issues.

    Returns list of issue codes (empty = valid).
    """
    issues = []

    # Check for empty or near-empty
    if not raw_text or not raw_text.strip():
        issues.append("empty")
        return issues  # No point checking further

    non_space = sum(1 for c in raw_text if not c.isspace())
    if non_space < MIN_NON_SPACE_CHARS:
        issues.append("too_sparse")

    # Check dimensions (use pre-computed height from database for consistency)
    if height < MIN_LINES:
        issues.append("too_few_lines")

    if height > MAX_LINES:
        issues.append("too_many_lines")

    if width > MAX_WIDTH:
        issues.append("too_wide")

    if len(raw_text) > MAX_TOTAL_CHARS:
        issues.append("too_large")

    # Check for control characters (except valid whitespace)
    for char in raw_text:
        code = ord(char)
        if code < 32 and char not in "\n\r\t":
            issues.append("control_chars")
            break

    # Check for replacement character (indicates encoding issues)
    if "\ufffd" in raw_text:
        issues.append("encoding_error")

    # Check for null bytes (binary data)
    if "\x00" in raw_text:
        issues.append("null_bytes")

    return issues


def get_size_bucket(width: int, height: int) -> str:
    """Categorize art by size for distribution analysis."""
    area = width * height
    if area < 100:
        return "tiny"
    elif area < 1000:
        return "small"
    elif area < 5000:
        return "medium"
    elif area < 20000:
        return "large"
    else:
        return "huge"


def run_validation(
    conn: sqlite3.Connection,
    dry_run: bool = False,
    limit: int | None = None,
) -> QualityReport:
    """
    Run validation on all art in the database.

    Args:
        conn: Database connection
        dry_run: If True, don't update database
        limit: Optional limit on records to check

    Returns:
        QualityReport with statistics
    """
    report = QualityReport()

    # Get all art IDs (with retry for lock)
    all_ids = retry_on_lock(lambda: list(iter_ascii_art_ids(conn)))
    if limit:
        all_ids = all_ids[:limit]

    logger.info(f"Validating {len(all_ids)} art pieces...")

    for i, art_id in enumerate(all_ids):
        # Use retry for database reads
        aid = art_id
        art = retry_on_lock(lambda: get_ascii_art_by_id(conn, aid))
        if art is None:
            continue

        report.total_checked += 1

        # Validate
        issues = validate_art(art.raw_text, art.width, art.height)
        is_valid = len(issues) == 0

        if is_valid:
            report.valid += 1
        else:
            report.invalid += 1
            for issue in issues:
                report.issues_by_type[issue] += 1

        # Update database if needed (with retry)
        if not dry_run and is_valid != art.is_valid:
            iv = is_valid
            retry_on_lock(lambda: update_ascii_art(conn, art_id=aid, fields={"is_valid": iv}))

        # Collect statistics
        report.charset_distribution[art.charset] += 1
        report.source_distribution[art.source] += 1
        report.size_buckets[get_size_bucket(art.width, art.height)] += 1

        # Progress logging
        if (i + 1) % 10000 == 0:
            logger.info(
                f"Progress: {i + 1}/{len(all_ids)}, "
                f"valid: {report.valid}, invalid: {report.invalid}"
            )

    return report


def find_near_duplicates(
    conn: sqlite3.Connection,
    threshold: int = 95,
    sample_size: int = 1000,
    batch_size: int = 100,
) -> list[tuple[int, int, float]]:
    """
    Find near-duplicate art pieces using fuzzy matching.

    This is expensive, so we sample and compare within batches.

    Args:
        conn: Database connection
        threshold: Minimum similarity score (0-100)
        sample_size: Number of pieces to sample
        batch_size: Compare within batches of this size

    Returns:
        List of (id1, id2, similarity_score) tuples
    """
    if not HAS_RAPIDFUZZ:
        logger.warning("rapidfuzz not available, skipping near-duplicate detection")
        return []

    logger.info(
        f"Finding near-duplicates (sample={sample_size}, threshold={threshold}%)..."
    )

    # Sample valid art pieces
    cursor = conn.execute(
        """
        SELECT id, raw_text FROM ascii_art
        WHERE is_valid = 1
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (sample_size,),
    )
    samples = [(row[0], row[1]) for row in cursor.fetchall()]

    duplicates = []

    # Compare within batches to limit O(n^2) complexity
    for batch_start in range(0, len(samples), batch_size):
        batch = samples[batch_start : batch_start + batch_size]

        for i, (id1, text1) in enumerate(batch):
            for j, (id2, text2) in enumerate(batch[i + 1 :], i + 1):
                # Quick length check to avoid expensive comparison
                len_ratio = len(text1) / max(1, len(text2))
                if len_ratio < 0.5 or len_ratio > 2.0:
                    continue

                # Fuzzy comparison
                score = fuzz.ratio(text1, text2)
                if score >= threshold:
                    duplicates.append((id1, id2, score))

        if batch_start > 0 and batch_start % 500 == 0:
            logger.info(
                f"Duplicate search: {batch_start}/{len(samples)}, found {len(duplicates)}"
            )

    logger.info(f"Found {len(duplicates)} potential near-duplicates")
    return duplicates


def generate_report(
    conn: sqlite3.Connection,
    output_path: Path,
    run_dedup: bool = True,
    dry_run: bool = False,
    limit: int | None = None,
) -> QualityReport:
    """
    Run full quality pipeline and generate report.
    """
    # Run validation
    report = run_validation(conn, dry_run=dry_run, limit=limit)

    # Find near-duplicates (optional, expensive)
    if run_dedup and HAS_RAPIDFUZZ:
        duplicates = find_near_duplicates(
            conn, sample_size=min(1000, report.total_checked)
        )
        report.near_duplicates_found = len(duplicates)

        # Save duplicates to separate file
        if duplicates:
            dup_path = output_path.with_suffix(".duplicates.json")
            with open(dup_path, "w") as f:
                json.dump(
                    [{"id1": d[0], "id2": d[1], "score": d[2]} for d in duplicates],
                    f,
                    indent=2,
                )
            logger.info(f"Duplicates saved to: {dup_path}")

    # Save main report
    with open(output_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info(f"Report saved to: {output_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info("Quality Pipeline Complete")
    logger.info("=" * 60)
    logger.info(f"Total checked:    {report.total_checked}")
    logger.info(
        f"Valid:            {report.valid} ({report.valid / max(1, report.total_checked) * 100:.1f}%)"
    )
    logger.info(f"Invalid:          {report.invalid}")
    logger.info(f"Near-duplicates:  {report.near_duplicates_found}")
    logger.info("-" * 40)
    logger.info("Issues by type:")
    for issue, count in sorted(report.issues_by_type.items(), key=lambda x: -x[1]):
        logger.info(f"  {issue}: {count}")
    logger.info("-" * 40)
    logger.info("Charset distribution:")
    for charset, count in sorted(
        report.charset_distribution.items(), key=lambda x: -x[1]
    ):
        logger.info(f"  {charset}: {count}")
    logger.info("-" * 40)
    logger.info("Source distribution:")
    for source, count in sorted(
        report.source_distribution.items(), key=lambda x: -x[1]
    )[:10]:
        logger.info(f"  {source}: {count}")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run data quality pipeline on ASCII art database"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/ascii_art.db"),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("quality_report.json"),
        help="Output path for quality report",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't update database, just report",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Skip near-duplicate detection (faster)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of records to check (for testing)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.db_path.exists():
        logger.error(f"Database not found: {args.db_path}")
        return 1

    conn = connect(args.db_path)

    try:
        report = generate_report(
            conn,
            output_path=args.output,
            run_dedup=not args.no_dedup,
            dry_run=args.dry_run,
            limit=args.limit,
        )
        return 0 if report.invalid / max(1, report.total_checked) < 0.05 else 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
