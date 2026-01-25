"""
FIGlet Banner Dataset Generation.

Generates ASCII art banners from FIGlet fonts for training data.
This produces high-quality, consistent text banners across many font styles.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import subprocess
import sys
import time
from shutil import which
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, TypeVar, Callable

T = TypeVar("T")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.db import connect, initialize, upsert_ascii_art  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("figlet_generation.log"),
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
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.debug(f"Database locked, retrying in {delay:.1f}s (attempt {attempt + 1})")
                time.sleep(delay)
            else:
                raise
    # This should never be reached, but for type safety
    raise RuntimeError("Retry logic error")


# Word lists for generation
ALPHABET_UPPER = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ALPHABET_LOWER = list("abcdefghijklmnopqrstuvwxyz")
DIGITS = list("0123456789")

COMMON_WORDS = [
    # Greetings and basic
    "HELLO", "WORLD", "WELCOME", "GOODBYE", "THANKS",
    # Tech/Programming
    "ERROR", "DEBUG", "TEST", "TODO", "FIXME", "README",
    "CODE", "DATA", "FILE", "MAIN", "INIT", "EXIT",
    "START", "STOP", "RUN", "BUILD", "DONE", "PASS", "FAIL",
    # Actions
    "HELP", "INFO", "WARN", "ALERT", "STATUS", "LOADING",
    # Common nouns
    "SYSTEM", "USER", "ADMIN", "SERVER", "CLIENT", "GAME",
    "MUSIC", "VIDEO", "IMAGE", "AUDIO", "DEMO", "BETA",
    # Short combinations
    "OK", "YES", "NO", "GO", "END", "NEW", "OLD", "TOP",
    # Numbers as words
    "ONE", "TWO", "TEN", "ZERO", "NULL",
    # ASCII art related
    "ASCII", "ART", "TEXT", "BANNER", "LOGO", "TITLE",
    # Fun words
    "COOL", "EPIC", "MEGA", "ULTRA", "SUPER", "HYPER",
    # Time
    "NOW", "TODAY", "YEAR", "TIME", "DATE",
    # Misc
    "NAME", "LOVE", "PEACE", "POWER", "MAGIC", "DREAM",
]

EXTRA_WORDS = [
    # Short + common
    "CAT", "DOG", "BIRD", "FISH", "MOUSE", "HORSE", "SHEEP", "GOAT", "COW", "PIG",
    "LION", "TIGER", "BEAR", "WOLF", "FOX", "DRAGON", "UNICORN",
    "SUN", "MOON", "STAR", "CLOUD", "RAIN", "SNOW", "WIND", "FIRE", "WATER", "EARTH",
    "TREE", "FLOWER", "ROSE", "LEAF", "STONE", "ROCK", "MOUNTAIN", "OCEAN", "RIVER", "LAKE",
    "HOUSE", "HOME", "CITY", "TOWN", "ROAD", "BRIDGE", "TOWER", "CASTLE",
    "CAR", "TRUCK", "TRAIN", "PLANE", "SHIP", "BOAT", "ROCKET",
    # Colors
    "RED", "GREEN", "BLUE", "CYAN", "MAGENTA", "YELLOW", "BLACK", "WHITE", "GRAY",
    # Tech / CLI
    "API", "CLI", "JSON", "YAML", "TOML", "SQL", "HTTP", "HTTPS", "TCP", "UDP", "SSH", "TLS",
    "GIT", "COMMIT", "PUSH", "PULL", "MERGE", "REBASE", "BRANCH", "TAG", "DIFF", "PATCH",
    "TOKEN", "MODEL", "TRAIN", "EXPORT", "WEIGHTS", "QUANT", "INT8", "INT4",
    "LINUX", "RUST", "PYTHON", "CUDA", "CPU", "GPU",
    # Status-ish
    "READY", "BUSY", "WAIT", "SLEEP", "WAKE", "RETRY", "SUCCESS", "FAILURE",
    "ONLINE", "OFFLINE", "CONNECTED", "DISCONNECTED",
    # Time
    "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY",
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
    # Fun
    "WOW", "NICE", "BRAVO", "CHEERS", "PARTY", "HYPE", "VIBES", "RAD", "NEAT", "GREAT",
    # More short words for incremental runs (kept relatively short to avoid width limits).
    "ROBOT", "ALIEN", "SPACE", "LASER", "PIXEL", "RETRO", "ANSI", "SCENE", "PACK", "GROUP",
    "BBS", "NET", "NODE", "PIPE", "PORT", "SOCK", "QUEUE", "STACK", "HEAP", "CACHE",
    "TRACE", "PANIC", "CRASH", "RECOVER", "ROLLBACK", "RELEASE", "DEPLOY",
    "SHELL", "BASH", "ZSH", "FISH", "CARGO", "MAKE", "TESTS", "UNIT", "E2E",
    "LINT", "FORMAT", "CHECK", "BUILD", "CLEAN",
    "ALPHA", "BETA2", "GAMMA", "DELTA", "OMEGA",
    "NORTH", "SOUTH", "EAST", "WEST",
    "HAPPY", "SMILE", "LAUGH", "CRY", "ANGRY",
    "COFFEE", "TEA", "PIZZA", "TACO", "SUSHI",
    "MUSIC2", "VIDEO2", "PHOTO", "CAMERA", "RADIO",
    "BOOK", "PAPER", "PENCIL", "BRUSH", "CANVAS",
    "BUNNY", "PANDA", "KOALA", "EAGLE", "SHARK",
    "SNAIL", "FROG", "WHALE", "ZEBRA", "OTTER",
    "SWORD", "SHIELD", "QUEST", "BATTLE", "LEVEL",
    "AURORA", "COMET", "GALAXY", "ORBIT", "METEOR",
    "SIGNAL", "NOISE", "ECHO", "VOICE", "SILENCE",
]


def _word_list(kind: str) -> list[str]:
    if kind == "base":
        return COMMON_WORDS
    if kind == "extra":
        return EXTRA_WORDS
    if kind == "all":
        return [*COMMON_WORDS, *EXTRA_WORDS]
    raise ValueError(f"Unsupported word set: {kind!r}")


@dataclass
class FontTestResult:
    """Result of testing a font."""
    font_path: Path
    font_name: str
    works: bool
    reason: str = ""


@dataclass
class GenerationStats:
    """Statistics for generation run."""
    fonts_tested: int = 0
    fonts_working: int = 0
    fonts_broken: int = 0
    total_generated: int = 0
    inserted: int = 0
    duplicates: int = 0
    errors: int = 0
    broken_fonts: list[tuple[str, str]] = field(default_factory=list)


def test_font(font_path: Path, timeout: float = 5.0) -> FontTestResult:
    """Test if a font produces valid output."""
    font_name = font_path.stem

    try:
        result = subprocess.run(
            ["figlet", "-f", str(font_path), "TEST"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            return FontTestResult(
                font_path, font_name, False, f"Exit code {result.returncode}"
            )

        output = result.stdout
        if not output or not output.strip():
            return FontTestResult(font_path, font_name, False, "Empty output")

        lines = output.split("\n")
        # Filter out empty lines for dimension check
        non_empty = [line for line in lines if line.strip()]

        if len(non_empty) < 1:
            return FontTestResult(font_path, font_name, False, "No content lines")

        # Check for reasonable dimensions
        max_width = max((len(line) for line in lines), default=0)
        if max_width > 500:
            return FontTestResult(font_path, font_name, False, "Output too wide")
        if len(lines) > 100:
            return FontTestResult(font_path, font_name, False, "Output too tall")

        return FontTestResult(font_path, font_name, True, "OK")

    except subprocess.TimeoutExpired:
        return FontTestResult(font_path, font_name, False, "Timeout")
    except Exception as e:
        return FontTestResult(font_path, font_name, False, str(e))


def generate_figlet(text: str, font_path: Path, timeout: float = 5.0) -> str | None:
    """Generate FIGlet banner for given text and font."""
    try:
        result = subprocess.run(
            ["figlet", "-f", str(font_path), text],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            return None

        output = result.stdout
        if not output or not output.strip():
            return None

        return output

    except (subprocess.TimeoutExpired, Exception):
        return None


def is_valid_output(art: str) -> bool:
    """Check if FIGlet output is usable for training."""
    if not art or not art.strip():
        return False

    lines = art.split("\n")
    non_empty = [line for line in lines if line.strip()]

    # Must have at least 1 non-empty line
    if len(non_empty) < 1:
        return False

    # Should not be excessively wide or tall
    max_width = max((len(line) for line in lines), default=0)
    if max_width > 200:
        return False
    if len(lines) > 50:
        return False

    return True


def discover_fonts(fonts_dir: Path) -> list[Path]:
    """Find all .flf font files in directory tree."""
    fonts = []

    # Use pathlib glob since find is aliased on this system
    for flf in fonts_dir.rglob("*.flf"):
        fonts.append(flf)

    logger.info(f"Discovered {len(fonts)} font files in {fonts_dir}")
    return sorted(fonts)


def generate_texts_for_font(
    font_path: Path,
    font_name: str,
    *,
    include_letters: bool,
    include_digits: bool,
    words: list[str],
) -> Iterator[tuple[str, str, dict]]:
    """
    Generate all text variations for a font.

    Yields: (raw_text, description, metadata_dict)
    """
    if include_letters:
        # Generate uppercase alphabet
        for char in ALPHABET_UPPER:
            art = generate_figlet(char, font_path)
            if art and is_valid_output(art):
                yield (
                    art,
                    f"FIGlet letter '{char}' in {font_name} font",
                    {"input_text": char, "font": font_name, "type": "letter"},
                )

        # Generate lowercase alphabet
        for char in ALPHABET_LOWER:
            art = generate_figlet(char, font_path)
            if art and is_valid_output(art):
                yield (
                    art,
                    f"FIGlet letter '{char}' in {font_name} font",
                    {"input_text": char, "font": font_name, "type": "letter"},
                )

    if include_digits:
        # Generate digits
        for digit in DIGITS:
            art = generate_figlet(digit, font_path)
            if art and is_valid_output(art):
                yield (
                    art,
                    f"FIGlet digit '{digit}' in {font_name} font",
                    {"input_text": digit, "font": font_name, "type": "digit"},
                )

    # Generate words
    for word in words:
        art = generate_figlet(word, font_path)
        if art and is_valid_output(art):
            yield (
                art,
                f"FIGlet banner '{word}' in {font_name} font",
                {"input_text": word, "font": font_name, "type": "word"},
            )


def generate_dataset(
    fonts_dir: Path,
    db_path: str = "data/ascii_art.db",
    batch_size: int = 100,
    max_fonts: int | None = None,
    *,
    include_letters: bool = True,
    include_digits: bool = True,
    word_set: str = "base",
    stop_at_total_rows: int | None = None,
) -> GenerationStats:
    """
    Generate FIGlet banner dataset from all fonts.

    Args:
        fonts_dir: Directory containing .flf font files
        db_path: Path to SQLite database
        batch_size: Number of entries per commit
        max_fonts: Limit number of fonts (for testing)

    Returns:
        GenerationStats with counts
    """
    stats = GenerationStats()

    # Discover fonts
    all_fonts = discover_fonts(fonts_dir)
    if max_fonts:
        all_fonts = all_fonts[:max_fonts]

    # Test fonts first
    logger.info("Testing fonts for compatibility...")
    working_fonts: list[Path] = []

    for i, font_path in enumerate(all_fonts):
        result = test_font(font_path)
        stats.fonts_tested += 1

        if result.works:
            working_fonts.append(font_path)
            stats.fonts_working += 1
        else:
            stats.fonts_broken += 1
            stats.broken_fonts.append((result.font_name, result.reason))
            logger.debug(f"Skipping font {result.font_name}: {result.reason}")

        if (i + 1) % 50 == 0:
            logger.info(f"Tested {i + 1}/{len(all_fonts)} fonts...")

    logger.info(
        f"Font testing complete: {stats.fonts_working} working, "
        f"{stats.fonts_broken} broken"
    )

    # Connect to database
    conn = connect(db_path)
    initialize(conn)

    # Generate from working fonts
    words = _word_list(word_set)
    pending = 0
    conn.execute("BEGIN")
    hit_stop = False

    for i, font_path in enumerate(working_fonts):
        font_name = font_path.stem

        for raw_text, description, extra_meta in generate_texts_for_font(
            font_path,
            font_name,
            include_letters=include_letters,
            include_digits=include_digits,
            words=words,
        ):
            try:
                # Insert with upsert (handles dedup and computes metadata)
                # Use retry logic to handle database locks from concurrent processes
                # Capture loop variables via default arguments to avoid closure issues.
                def do_upsert(
                    rt=raw_text,
                    fn=font_name,
                    desc=description,
                    em=extra_meta,
                ):
                    return upsert_ascii_art(
                        conn,
                        raw_text=rt,
                        source="figlet",
                        source_id=f"{fn}:{em['input_text']}",
                        title=f"{em['input_text']} ({fn})",
                        description=desc,
                        category="banner",
                        subcategory=fn,
                        tags=["figlet", "banner", "text", fn, em["type"]],
                        is_valid=True,
                    )

                result = retry_on_lock(do_upsert)

                stats.total_generated += 1
                if result is None:
                    stats.errors += 1
                    continue
                if result.inserted:
                    stats.inserted += 1
                else:
                    stats.duplicates += 1
                pending += 1

                if pending >= batch_size:
                    retry_on_lock(conn.commit)
                    conn.execute("BEGIN")
                    pending = 0
                    if stop_at_total_rows is not None:
                        total = retry_on_lock(
                            lambda: conn.execute("SELECT COUNT(*) FROM ascii_art").fetchone()[0]
                        )
                        if int(total) >= stop_at_total_rows:
                            hit_stop = True
                            break

            except Exception as e:
                stats.errors += 1
                logger.warning(f"Error inserting {font_name}/{extra_meta['input_text']}: {e}")

        if hit_stop:
            break

        if (i + 1) % 10 == 0:
            retry_on_lock(conn.commit)
            conn.execute("BEGIN")
            logger.info(
                f"Processed {i + 1}/{len(working_fonts)} fonts, "
                f"{stats.total_generated} generated, {stats.inserted} inserted"
            )
            if stop_at_total_rows is not None:
                total = retry_on_lock(lambda: conn.execute("SELECT COUNT(*) FROM ascii_art").fetchone()[0])
                if int(total) >= stop_at_total_rows:
                    hit_stop = True
                    break

    # Final commit
    retry_on_lock(conn.commit)
    conn.close()

    return stats


def main():
    """Main entry point."""
    if which("figlet") is None:
        logger.error("Missing dependency: `figlet` binary not found in PATH.")
        logger.error("Install it (or run inside an env that has it) before running generation.")
        return 1

    parser = argparse.ArgumentParser(
        description="Generate FIGlet banner dataset from fonts"
    )
    parser.add_argument(
        "--fonts-dir",
        type=Path,
        default=Path("data/raw/figlet-fonts"),
        help="Directory containing FIGlet font files",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/ascii_art.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Commit every N inserts (default: 100)",
    )
    parser.add_argument(
        "--stop-at-total-rows",
        type=int,
        default=None,
        help="Stop early once total ascii_art rows reaches this threshold (checked at commit points)",
    )
    parser.add_argument(
        "--max-fonts",
        type=int,
        default=None,
        help="Limit number of fonts (for testing)",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "words"],
        default="all",
        help="Generation mode: all (letters+digits+words) or words only (default: all)",
    )
    parser.add_argument(
        "--word-set",
        choices=["base", "extra", "all"],
        default="base",
        help="Which word list to generate (default: base)",
    )
    parser.add_argument(
        "--include-system",
        action="store_true",
        help="Also include system fonts from /usr/share/figlet",
    )

    args = parser.parse_args()

    # Collect font directories
    font_dirs = [args.fonts_dir]
    if args.include_system:
        system_dir = Path("/usr/share/figlet")
        if system_dir.exists():
            font_dirs.append(system_dir)

    total_stats = GenerationStats()

    for font_dir in font_dirs:
        if not font_dir.exists():
            logger.error(f"Font directory not found: {font_dir}")
            continue

        logger.info(f"Processing fonts from: {font_dir}")
        include_letters = args.mode == "all"
        include_digits = args.mode == "all"
        stats = generate_dataset(
            fonts_dir=font_dir,
            db_path=args.db_path,
            batch_size=args.batch_size,
            max_fonts=args.max_fonts,
            include_letters=include_letters,
            include_digits=include_digits,
            word_set=args.word_set,
            stop_at_total_rows=args.stop_at_total_rows,
        )

        # Aggregate stats
        total_stats.fonts_tested += stats.fonts_tested
        total_stats.fonts_working += stats.fonts_working
        total_stats.fonts_broken += stats.fonts_broken
        total_stats.total_generated += stats.total_generated
        total_stats.inserted += stats.inserted
        total_stats.duplicates += stats.duplicates
        total_stats.errors += stats.errors
        total_stats.broken_fonts.extend(stats.broken_fonts)

    # Summary
    logger.info("=" * 60)
    logger.info("FIGlet Generation Complete")
    logger.info("=" * 60)
    logger.info(f"Fonts tested:     {total_stats.fonts_tested}")
    logger.info(f"Fonts working:    {total_stats.fonts_working}")
    logger.info(f"Fonts broken:     {total_stats.fonts_broken}")
    logger.info(f"Total generated:  {total_stats.total_generated}")
    logger.info(f"Inserted:         {total_stats.inserted}")
    logger.info(f"Duplicates:       {total_stats.duplicates}")
    logger.info(f"Errors:           {total_stats.errors}")

    # Save broken fonts report
    if total_stats.broken_fonts:
        report_path = Path(__file__).resolve().parent / "figlet_broken_fonts.json"
        with open(report_path, "w") as outfile:
            json.dump(
                [
                    {"font": font, "reason": reason}
                    for font, reason in total_stats.broken_fonts
                ],
                outfile,
                indent=2,
            )
        logger.info(f"Broken fonts report saved to: {report_path}")

    return 0 if total_stats.errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
