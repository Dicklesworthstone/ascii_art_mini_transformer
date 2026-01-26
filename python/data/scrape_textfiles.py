"""
Scraper for the Textfiles.com artscene ANSI/ASCII archive.

Target (per beads):
    http://artscene.textfiles.com/ansi/

This is a simple directory listing style site (no pagination). The scraper:
- Crawls subdirectories under the base path
- Downloads files and decodes bytes as CP437 (common for ANSI art)
- Preserves whitespace and line breaks (normalizes line endings to '\n')
- Inserts into SQLite via `data.db` (dedup by content hash)
- Optionally writes a JSONL export and a progress file for resumability
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urljoin, urlparse

import requests


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from data.db import (  # noqa: E402
    connect,
    initialize,
    insert_ascii_art,
    normalize_newlines,
)


logger = logging.getLogger(__name__)


DEFAULT_BASE_URL = "http://artscene.textfiles.com/ansi/"


_HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True, slots=True)
class ScrapeConfig:
    base_url: str
    db_path: Path
    output_jsonl: Optional[Path]
    progress_path: Optional[Path]
    delay_seconds: float
    max_files: Optional[int]
    strip_ansi: bool
    insert_into_db: bool
    dry_run: bool


@dataclass
class ProgressState:
    processed_dirs: set[str]
    processed_files: set[str]
    inserted: int = 0
    duplicates: int = 0
    errors: int = 0
    skipped: int = 0

    @classmethod
    def load(cls, path: Path) -> "ProgressState":
        if not path.exists():
            return cls(processed_dirs=set(), processed_files=set())
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            processed_dirs=set(data.get("processed_dirs", [])),
            processed_files=set(data.get("processed_files", [])),
            inserted=int(data.get("inserted", 0)),
            duplicates=int(data.get("duplicates", 0)),
            errors=int(data.get("errors", 0)),
            skipped=int(data.get("skipped", 0)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "processed_dirs": sorted(self.processed_dirs),
            "processed_files": sorted(self.processed_files),
            "inserted": self.inserted,
            "duplicates": self.duplicates,
            "errors": self.errors,
            "skipped": self.skipped,
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.strip()
    if not normalized.endswith("/"):
        normalized += "/"
    return normalized


def _same_site(base_url: str, url: str) -> bool:
    return urlparse(base_url).netloc == urlparse(url).netloc


def _within_base_path(base_url: str, url: str) -> bool:
    base = urlparse(base_url)
    parsed = urlparse(url)
    if base.netloc != parsed.netloc:
        return False
    return parsed.path.startswith(base.path)


def _rate_limit(delay_seconds: float) -> None:
    if delay_seconds > 0:
        time.sleep(delay_seconds)


def _http_get(session: requests.Session, url: str) -> bytes:
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def _extract_listing_hrefs(html: str) -> list[str]:
    hrefs = _HREF_RE.findall(html)
    cleaned: list[str] = []
    for href in hrefs:
        href = href.strip()
        if not href or href.startswith("#"):
            continue
        if href in {"../", "./", "/"}:
            continue
        if href.startswith("?") or href.startswith("./?") or href.startswith("../?"):
            # Apache directory listing sort/query links.
            continue
        parsed = urlparse(href)
        if parsed.scheme and parsed.scheme not in {"http", "https"}:
            # Skip non-http(s) schemes (e.g., mailto:, javascript:).
            continue
        cleaned.append(href)
    # De-dupe preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for href in cleaned:
        if href in seen:
            continue
        seen.add(href)
        ordered.append(href)
    return ordered


def _path_category_parts(
    url: str, base_url: str
) -> tuple[Optional[str], Optional[str]]:
    base = urlparse(base_url)
    parsed = urlparse(url)
    if base.netloc != parsed.netloc or not parsed.path.startswith(base.path):
        return None, None
    rel = parsed.path[len(base.path) :]
    parts = [p for p in rel.split("/") if p]
    if not parts:
        return None, None
    category = parts[0]
    subcategory = parts[1] if len(parts) > 2 else None
    return category, subcategory


def _filename_from_url(url: str) -> Optional[str]:
    parsed = urlparse(url)
    name = parsed.path.rsplit("/", 1)[-1]
    return name or None


def _source_id_from_url(url: str, base_url: str) -> str:
    base = urlparse(base_url)
    parsed = urlparse(url)
    rel = parsed.path
    if parsed.netloc == base.netloc and parsed.path.startswith(base.path):
        rel = parsed.path[len(base.path) :]
    return rel.lstrip("/")


def _tags_for_path(
    category: Optional[str], subcategory: Optional[str], filename: Optional[str]
) -> list[str]:
    tags: list[str] = ["textfiles"]
    if category:
        tags.append(category)
    if subcategory:
        tags.append(subcategory)
    if filename and "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext:
            tags.append(ext)
    # De-dupe preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        ordered.append(tag)
    return ordered


_BINARY_EXTENSIONS: set[str] = {
    "7z",
    "avi",
    "bin",
    "bmp",
    "bz2",
    "com",
    "dsk",
    "exe",
    "gif",
    "gz",
    "ico",
    "iso",
    "jar",
    "jpeg",
    "jpg",
    "m4a",
    "mkv",
    "mov",
    "mp3",
    "mp4",
    "mpeg",
    "mpg",
    "ogg",
    "pdf",
    "png",
    "rar",
    "tar",
    "tgz",
    "wav",
    "webm",
    "webp",
    "wma",
    "wmv",
    "xz",
    "zip",
}


def scrape_textfiles(config: ScrapeConfig) -> ProgressState:
    base_url = _normalize_base_url(config.base_url)
    load_progress = config.progress_path is not None
    persist_progress = config.progress_path is not None and not config.dry_run
    state = (
        ProgressState.load(config.progress_path)
        if load_progress and config.progress_path is not None
        else ProgressState(processed_dirs=set(), processed_files=set())
    )

    session = requests.Session()
    session.headers.update({"User-Agent": "ascii-art-mini-transformer/0.1 (+github)"})

    conn: Optional[sqlite3.Connection] = None
    if config.insert_into_db and not config.dry_run:
        conn = connect(config.db_path)
        initialize(conn)

    jsonl_fp = None
    if config.output_jsonl is not None and not config.dry_run:
        config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_fp = config.output_jsonl.open("a", encoding="utf-8", buffering=1)

    to_visit: deque[str] = deque([base_url])
    queued_dirs: set[str] = {base_url}
    dir_failures: dict[str, int] = {}
    failed_dirs: set[str] = set()
    files_seen = 0

    try:
        while to_visit:
            dir_url = _normalize_base_url(to_visit.popleft())
            queued_dirs.discard(dir_url)

            if dir_url in state.processed_dirs or dir_url in failed_dirs:
                continue
            if not _same_site(base_url, dir_url) or not _within_base_path(
                base_url, dir_url
            ):
                continue

            try:
                logger.info("GET %s", dir_url)
                content = _http_get(session, dir_url)
                html = content.decode("utf-8", errors="replace")
            except Exception:
                logger.exception("Failed to fetch directory listing: %s", dir_url)
                state.errors += 1
                if persist_progress and config.progress_path is not None:
                    state.save(config.progress_path)

                attempts = dir_failures.get(dir_url, 0) + 1
                dir_failures[dir_url] = attempts
                if attempts < 3 and dir_url not in queued_dirs:
                    to_visit.append(dir_url)
                    queued_dirs.add(dir_url)
                else:
                    failed_dirs.add(dir_url)

                _rate_limit(config.delay_seconds)
                continue

            _rate_limit(config.delay_seconds)

            hit_limit = False
            hrefs = _extract_listing_hrefs(html)
            for href in hrefs:
                full = urljoin(dir_url, href)
                if not _same_site(base_url, full) or not _within_base_path(
                    base_url, full
                ):
                    continue

                if full.endswith("/"):
                    subdir = _normalize_base_url(full)
                    if (
                        subdir not in state.processed_dirs
                        and subdir not in queued_dirs
                        and subdir not in failed_dirs
                    ):
                        to_visit.append(subdir)
                        queued_dirs.add(subdir)
                    continue

                if full in state.processed_files:
                    continue

                filename = _filename_from_url(full)
                ext = (
                    filename.rsplit(".", 1)[-1].lower()
                    if filename and "." in filename
                    else ""
                )
                if ext in _BINARY_EXTENSIONS:
                    state.skipped += 1
                    state.processed_files.add(full)
                    if persist_progress and config.progress_path is not None:
                        state.save(config.progress_path)
                    continue

                if config.max_files is not None and files_seen >= config.max_files:
                    hit_limit = True
                    break

                files_seen += 1

                category, subcategory = _path_category_parts(full, base_url)
                tags = _tags_for_path(category, subcategory, filename)

                try:
                    logger.info("GET %s", full)
                    payload = _http_get(session, full)
                except Exception:
                    logger.exception("Failed to fetch file: %s", full)
                    state.errors += 1
                    if persist_progress and config.progress_path is not None:
                        state.save(config.progress_path)
                    _rate_limit(config.delay_seconds)
                    continue

                _rate_limit(config.delay_seconds)

                # CP437 is commonly used for ANSI art; it maps 0x00-0xFF losslessly.
                raw_text = payload.decode("cp437", errors="replace")
                if config.strip_ansi:
                    raw_text = _ANSI_ESCAPE_RE.sub("", raw_text)
                raw_text = normalize_newlines(raw_text)

                if not raw_text.strip():
                    state.skipped += 1
                    state.processed_files.add(full)
                    if persist_progress and config.progress_path is not None:
                        state.save(config.progress_path)
                    continue

                record: dict[str, Any] = {
                    "raw_text": raw_text,
                    "source": "textfiles.com",
                    "source_id": _source_id_from_url(full, base_url),
                    "title": filename,
                    "description": None,
                    "category": category,
                    "subcategory": subcategory,
                    "tags": tags,
                    "artist": None,
                    "url": full,
                }

                if jsonl_fp is not None:
                    jsonl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")

                if conn is not None:
                    inserted_id = insert_ascii_art(
                        conn,
                        raw_text=raw_text,
                        source="textfiles.com",
                        source_id=record["source_id"],
                        title=filename,
                        description=None,
                        category=category,
                        subcategory=subcategory,
                        tags=tags,
                        artist=None,
                    )
                    if inserted_id is None:
                        state.duplicates += 1
                    else:
                        state.inserted += 1

                state.processed_files.add(full)
                if persist_progress and config.progress_path is not None:
                    state.save(config.progress_path)

            if not hit_limit:
                state.processed_dirs.add(dir_url)
                if persist_progress and config.progress_path is not None:
                    state.save(config.progress_path)

            if hit_limit:
                break

    finally:
        session.close()
        if jsonl_fp is not None:
            jsonl_fp.close()
        if conn is not None:
            conn.close()

    return state


def _parse_args(argv: Optional[Iterable[str]] = None) -> ScrapeConfig:
    parser = argparse.ArgumentParser(
        description="Scrape textfiles.com artscene ANSI/ASCII archive"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--db-path", type=Path, default=ROOT / "data" / "ascii_art.db")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument(
        "--progress", type=Path, default=ROOT / "data" / "progress_textfiles.json"
    )
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--strip-ansi",
        action="store_true",
        help="Remove ANSI escape sequences before storing",
    )
    parser.add_argument(
        "--no-db", action="store_true", help="Do not insert into SQLite"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    return ScrapeConfig(
        base_url=str(args.base_url),
        db_path=Path(args.db_path),
        output_jsonl=Path(args.output_jsonl) if args.output_jsonl is not None else None,
        progress_path=Path(args.progress) if args.progress is not None else None,
        delay_seconds=float(args.delay_seconds),
        max_files=int(args.max_files) if args.max_files is not None else None,
        strip_ansi=bool(args.strip_ansi),
        insert_into_db=not bool(args.no_db),
        dry_run=bool(args.dry_run),
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    config = _parse_args(argv)
    logger.info("Starting textfiles scrape: %s", config.base_url)
    state = scrape_textfiles(config)
    logger.info(
        "Done. inserted=%d duplicates=%d skipped=%d errors=%d dirs=%d files=%d",
        state.inserted,
        state.duplicates,
        state.skipped,
        state.errors,
        len(state.processed_dirs),
        len(state.processed_files),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
