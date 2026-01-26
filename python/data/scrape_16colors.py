"""
Scraper for 16colo.rs (Sixteen Colors) ANSI/ASCII demoscene archive.

High-level flow:
- Enumerate release years via `/year/<YYYY>/`
- Enumerate packs within a year (`/pack/<pack>/`)
- Enumerate files within a pack (`/pack/<pack>/<filename>`)
- Download raw file bytes via `/pack/<pack>/raw/<filename>`
- Optionally extract SAUCE metadata and insert into SQLite

Notes:
- Many art files use CP437 encoding and may include ANSI escape sequences.
- This scraper stores the decoded text (CP437) in the DB and relies on DB
  metadata (`has_ansi_codes`, `charset`, etc.) to capture properties.
- SAUCE metadata is extracted and removed from the stored `raw_text`.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import requests


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from data.db import connect, initialize, insert_ascii_art, normalize_newlines  # noqa: E402


logger = logging.getLogger(__name__)


DEFAULT_BASE_URL = "https://16colo.rs/"


TEXT_FILE_EXTENSIONS = {
    "ans",
    "asc",
    "txt",
    "nfo",
    "diz",
    "ice",
    "lit",
    "rip",
    "drk",
}


@dataclass(frozen=True, slots=True)
class SauceMetadata:
    title: Optional[str]
    author: Optional[str]
    group: Optional[str]
    date: Optional[str]
    comments: list[str]


def _parse_sauce_record(record: bytes) -> SauceMetadata:
    # SAUCE record is 128 bytes, starting with b"SAUCE00".
    # Layout (subset):
    #  0-6   : "SAUCE00"
    #  7-41  : Title (35)
    # 42-61  : Author (20)
    # 62-81  : Group (20)
    # 82-89  : Date (8, YYYYMMDD)
    # 104    : Comments count (1)
    def _field(start: int, length: int) -> Optional[str]:
        raw = record[start : start + length]
        text = raw.rstrip(b"\x00 ").decode("cp437", errors="replace").strip()
        return text or None

    title = _field(7, 35)
    author = _field(42, 20)
    group = _field(62, 20)
    date = _field(82, 8)
    comment_count = record[104]

    # Comments are stored *before* the SAUCE record:
    #  "COMNT" (5 bytes) + N*64 bytes of comment text (CP437).
    comments: list[str] = []
    if comment_count:
        comments = [""] * int(comment_count)

    return SauceMetadata(
        title=title, author=author, group=group, date=date, comments=comments
    )


def strip_sauce(data: bytes) -> tuple[bytes, Optional[SauceMetadata]]:
    """
    Strip SAUCE metadata (and optional COMNT block) from file bytes.

    Returns (content_without_sauce, sauce_metadata_or_none).
    """
    if len(data) < 128:
        return data, None

    sauce = data[-128:]
    if sauce[:7] != b"SAUCE00":
        return data, None

    meta = _parse_sauce_record(sauce)
    content = data[:-128]

    # Many files include 0x1A before SAUCE (and sometimes between COMNT and SAUCE).
    content = content.rstrip(b"\x1a")

    # Strip COMNT block if present.
    comment_count = sauce[104]
    if comment_count:
        comnt_size = 5 + (int(comment_count) * 64)
        if (
            len(content) >= comnt_size
            and content[-comnt_size : -comnt_size + 5] == b"COMNT"
        ):
            comnt_block = content[-comnt_size:]
            comment_payload = comnt_block[5:]
            comments: list[str] = []
            for i in range(int(comment_count)):
                line = comment_payload[i * 64 : (i + 1) * 64]
                decoded = (
                    line.rstrip(b"\x00 ").decode("cp437", errors="replace").strip()
                )
                if decoded:
                    comments.append(decoded)
            meta = SauceMetadata(
                title=meta.title,
                author=meta.author,
                group=meta.group,
                date=meta.date,
                comments=comments,
            )
            content = content[:-comnt_size]

    return content, meta


def _soup(html: str):
    try:
        from bs4 import BeautifulSoup  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "beautifulsoup4 is required for HTML parsing; install from python/requirements.txt"
        ) from exc
    return BeautifulSoup(html, "html.parser")


def decode_cp437(data: bytes) -> str:
    # Normalize newlines after decoding for DB consistency.
    text = data.decode("cp437", errors="replace")
    return normalize_newlines(text)


def _rate_limit(delay_seconds: float, jitter_seconds: float) -> None:
    if delay_seconds <= 0:
        return
    sleep_for = delay_seconds + (
        random.random() * jitter_seconds if jitter_seconds > 0 else 0.0
    )
    time.sleep(sleep_for)


def _http_get_bytes(session: requests.Session, url: str) -> bytes:
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def _http_get_text_latin1(session: requests.Session, url: str) -> str:
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content.decode("latin-1", errors="replace")


def _extract_year_links(html: str, base_url: str) -> list[str]:
    soup = _soup(html)
    links: list[str] = []
    for opt in soup.select("select#selecty option[value]"):
        val = opt.get("value")
        if not val:
            continue
        full = urljoin(base_url, val)
        if urlparse(full).netloc == urlparse(base_url).netloc:
            links.append(full)
    return sorted(set(links))


def _extract_pack_links(year_html: str, base_url: str) -> list[str]:
    soup = _soup(year_html)
    links: list[str] = []
    for anchor in soup.select("a.dizname[href]"):
        href = anchor.get("href")
        if not href or not href.startswith("/pack/"):
            continue
        links.append(urljoin(base_url, href))
    # De-dupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for link in links:
        if link in seen:
            continue
        seen.add(link)
        out.append(link)
    return out


def _extract_file_page_links(
    pack_html: str, base_url: str, pack_name: str
) -> list[str]:
    soup = _soup(pack_html)
    links: list[str] = []
    prefix = f"/pack/{pack_name}/"
    for anchor in soup.select("a[href]"):
        href = anchor.get("href")
        if not href or not href.startswith(prefix):
            continue
        # Skip known non-file routes.
        if (
            "/raw/" in href
            or "/data/" in href
            or "/x1/" in href
            or "/x2/" in href
            or "/tn/" in href
        ):
            continue
        # File pages are exactly: /pack/<pack>/<filename>
        parts = [p for p in href.split("/") if p]
        if len(parts) != 3:
            continue
        links.append(urljoin(base_url, href))
    return sorted(set(links))


def _file_page_to_raw_url(
    file_page_url: str, pack_name: str, base_url: str
) -> Optional[str]:
    parsed = urlparse(file_page_url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) != 3 or parts[0] != "pack" or parts[1] != pack_name:
        return None
    filename = parts[2]  # keep percent-encoding as-is
    raw_path = f"/pack/{pack_name}/raw/{filename}"
    return urljoin(base_url, raw_path)


def _extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


@dataclass
class ProgressState:
    processed_raw_urls: set[str]
    inserted: int = 0
    duplicates: int = 0
    errors: int = 0

    @classmethod
    def load(cls, path: Path) -> "ProgressState":
        if not path.exists():
            return cls(processed_raw_urls=set())
        data = json.loads(path.read_text(encoding="utf-8"))
        urls = set(data.get("processed_raw_urls", []))
        return cls(
            processed_raw_urls=urls,
            inserted=int(data.get("inserted", 0)),
            duplicates=int(data.get("duplicates", 0)),
            errors=int(data.get("errors", 0)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "processed_raw_urls": sorted(self.processed_raw_urls),
            "inserted": self.inserted,
            "duplicates": self.duplicates,
            "errors": self.errors,
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def scrape_16colo_rs(
    *,
    base_url: str,
    db_path: Path,
    output_jsonl: Path,
    progress_path: Path,
    delay_seconds: float,
    jitter_seconds: float,
    max_years: Optional[int],
    max_packs: Optional[int],
    max_files: Optional[int],
    insert_into_db: bool,
    dry_run: bool,
) -> None:
    base_url = base_url.rstrip("/") + "/"
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "ascii_art_mini_transformer/0.1 (research; respectful crawler; "
                "https://github.com/openai/codex-cli)"
            ),
        }
    )

    progress = (
        ProgressState.load(progress_path)
        if not dry_run
        else ProgressState(processed_raw_urls=set())
    )
    logger.info("Loaded progress: %d files processed", len(progress.processed_raw_urls))

    conn = None
    if insert_into_db and not dry_run:
        conn = connect(db_path)
        initialize(conn)

    jsonl = None if dry_run else output_jsonl.open("a", encoding="utf-8")

    years_seen = 0
    packs_seen = 0
    files_seen = 0

    try:
        home_html = _http_get_text_latin1(session, base_url)
        year_links = _extract_year_links(home_html, base_url)
        if not year_links:
            raise RuntimeError("Failed to discover year links from homepage")

        for year_url in year_links:
            if max_years is not None and years_seen >= max_years:
                break
            years_seen += 1

            year_html = _http_get_text_latin1(session, year_url)
            pack_links = _extract_pack_links(year_html, base_url)
            logger.info(
                "Year %s: %d packs",
                year_url.rstrip("/").split("/")[-1],
                len(pack_links),
            )

            for pack_url in pack_links:
                if max_packs is not None and packs_seen >= max_packs:
                    break
                packs_seen += 1

                pack_name = urlparse(pack_url).path.rstrip("/").split("/")[-1]
                pack_html = _http_get_text_latin1(session, pack_url)
                file_pages = _extract_file_page_links(pack_html, base_url, pack_name)

                for file_page_url in file_pages:
                    if max_files is not None and files_seen >= max_files:
                        break

                    raw_url = _file_page_to_raw_url(file_page_url, pack_name, base_url)
                    if raw_url is None:
                        continue
                    if raw_url in progress.processed_raw_urls and not dry_run:
                        continue

                    filename = urlparse(raw_url).path.split("/")[-1]
                    ext = _extension(filename)
                    if ext not in TEXT_FILE_EXTENSIONS:
                        continue

                    files_seen += 1

                    try:
                        raw_bytes = _http_get_bytes(session, raw_url)
                        content_bytes, sauce = strip_sauce(raw_bytes)
                        text = decode_cp437(content_bytes)

                        # Basic filter: skip empty-ish text.
                        if not text.strip():
                            continue

                        tags: list[str] = [
                            "16colo.rs",
                            "demoscene",
                            f"year:{year_url.rstrip('/').split('/')[-1]}",
                            f"pack:{pack_name}",
                            f"ext:{ext}",
                        ]
                        if sauce and sauce.group:
                            tags.append(f"group:{sauce.group}")

                        entry: dict[str, Any] = {
                            "source": "16colo.rs",
                            "source_url": raw_url,
                            "source_id": f"{pack_name}/{filename}",
                            "title": sauce.title if sauce else None,
                            "artist": sauce.author if sauce else None,
                            "group": sauce.group if sauce else None,
                            "date": sauce.date if sauce else None,
                            "comments": sauce.comments if sauce else None,
                            "category": "demoscene",
                            "subcategory": pack_name,
                            "tags": tags,
                            "raw_text": text,
                        }

                        inserted = 0
                        duplicates = 0
                        if conn is not None:
                            with conn:
                                row_id = insert_ascii_art(
                                    conn,
                                    raw_text=text,
                                    source="16colo.rs",
                                    source_id=entry["source_id"],
                                    title=entry["title"],
                                    description=None,
                                    category=entry["category"],
                                    subcategory=entry["subcategory"],
                                    tags=entry["tags"],
                                    artist=entry["artist"],
                                )
                            if row_id is None:
                                duplicates = 1
                            else:
                                inserted = 1

                        if jsonl is not None:
                            jsonl.write(json.dumps(entry, ensure_ascii=False) + "\n")

                        if not dry_run:
                            progress.inserted += inserted
                            progress.duplicates += duplicates
                            progress.processed_raw_urls.add(raw_url)
                            progress.save(progress_path)

                    except Exception as exc:
                        if not dry_run:
                            progress.errors += 1
                            progress.save(progress_path)
                        logger.warning("Error downloading/parsing %s: %s", raw_url, exc)

                    _rate_limit(delay_seconds, jitter_seconds)

                _rate_limit(delay_seconds, jitter_seconds)

            _rate_limit(delay_seconds, jitter_seconds)

    finally:
        if jsonl is not None:
            jsonl.close()
        if conn is not None:
            conn.close()

    logger.info(
        "Done. years=%d packs=%d files=%d inserted=%d dup=%d errors=%d output=%s",
        years_seen,
        packs_seen,
        files_seen,
        progress.inserted,
        progress.duplicates,
        progress.errors,
        output_jsonl,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape 16colo.rs ANSI/ASCII archive")
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="Base URL (default: %(default)s)"
    )
    parser.add_argument(
        "--db-path", default=str(ROOT / "data" / "ascii_art.db"), help="SQLite DB path"
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(ROOT / "data" / "raw" / "16colo_rs.jsonl"),
        help="Write raw scraped records as JSONL",
    )
    parser.add_argument(
        "--progress-path",
        default=str(ROOT / "data" / "raw" / "16colo_rs_progress.json"),
        help="Progress JSON path for resumability",
    )
    parser.add_argument(
        "--delay-seconds", type=float, default=1.5, help="Base delay between requests"
    )
    parser.add_argument(
        "--jitter-seconds", type=float, default=0.5, help="Random jitter added to delay"
    )
    parser.add_argument(
        "--max-years", type=int, default=None, help="Stop after N years (debugging)"
    )
    parser.add_argument(
        "--max-packs", type=int, default=None, help="Stop after N packs (debugging)"
    )
    parser.add_argument(
        "--max-files", type=int, default=None, help="Stop after N files (debugging)"
    )
    parser.add_argument(
        "--no-db", action="store_true", help="Do not insert into SQLite (JSONL only)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Crawl/download but do not write JSONL/DB",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = _build_arg_parser().parse_args()

    scrape_16colo_rs(
        base_url=str(args.base_url),
        db_path=Path(args.db_path),
        output_jsonl=Path(args.output_jsonl),
        progress_path=Path(args.progress_path),
        delay_seconds=float(args.delay_seconds),
        jitter_seconds=float(args.jitter_seconds),
        max_years=int(args.max_years) if args.max_years is not None else None,
        max_packs=int(args.max_packs) if args.max_packs is not None else None,
        max_files=int(args.max_files) if args.max_files is not None else None,
        insert_into_db=not bool(args.no_db),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
