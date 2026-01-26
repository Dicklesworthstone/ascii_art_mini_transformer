"""
Scraper for the ASCII Art Archive gallery pages.

Target site (per beads): https://v2.asciiart.eu/

Note: As of 2026-01-25, `v2.asciiart.eu` redirects to a login page in this
environment, while `https://www.asciiart.eu/` is publicly accessible. This
scraper defaults to the public domain but allows overriding `--base-url`.

Features:
- Crawls the gallery category tree starting from `/gallery`
- Extracts per-art metadata (title, artist, category/subcategory)
- Preserves whitespace in ASCII art payloads
- Rate limiting (delay between requests)
- Progress tracking for resumability
- Writes raw results to JSONL and optionally inserts into SQLite
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
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from data.db import connect, initialize, insert_ascii_art  # noqa: E402


logger = logging.getLogger(__name__)


DEFAULT_BASE_URL = "https://www.asciiart.eu/"


@dataclass(frozen=True, slots=True)
class ScrapeConfig:
    base_url: str
    db_path: Path
    output_jsonl: Path
    progress_path: Path
    delay_seconds: float
    jitter_seconds: float
    max_pages: Optional[int]
    max_items: Optional[int]
    insert_into_db: bool
    dry_run: bool


@dataclass
class ProgressState:
    processed_pages: set[str]
    inserted: int = 0
    duplicates: int = 0
    errors: int = 0

    @classmethod
    def load(cls, path: Path) -> "ProgressState":
        if not path.exists():
            return cls(processed_pages=set())
        data = json.loads(path.read_text(encoding="utf-8"))
        pages = set(data.get("processed_pages", []))
        return cls(
            processed_pages=pages,
            inserted=int(data.get("inserted", 0)),
            duplicates=int(data.get("duplicates", 0)),
            errors=int(data.get("errors", 0)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "processed_pages": sorted(self.processed_pages),
            "inserted": self.inserted,
            "duplicates": self.duplicates,
            "errors": self.errors,
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


def _rate_limit(delay_seconds: float, jitter_seconds: float) -> None:
    if delay_seconds <= 0:
        return
    sleep_for = delay_seconds + (
        random.random() * jitter_seconds if jitter_seconds > 0 else 0.0
    )
    time.sleep(sleep_for)


def _http_get(session: requests.Session, url: str) -> str:
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def _extract_gallery_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    links: list[str] = []
    for anchor in soup.select("a.card-gallery[href]"):
        href = anchor.get("href")
        if not href:
            continue
        full = urljoin(base_url, href)
        if _same_site(base_url, full):
            links.append(full)
    # De-dupe while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for link in links:
        if link in seen:
            continue
        seen.add(link)
        ordered.append(link)
    return ordered


def _extract_page_summary(soup: BeautifulSoup) -> Optional[str]:
    # Example: <div class="text-truncate ..."><span>A large collection ...</span></div>
    el = soup.select_one("div.text-truncate span")
    if el is None:
        return None
    text = el.get_text(" ", strip=True)
    return text or None


def _extract_page_tags(soup: BeautifulSoup) -> list[str]:
    # Example: <pre class="... opacity-75">cat, domestic cats, ...</pre>
    pre = soup.select_one("pre.opacity-75")
    if pre is None:
        return []
    raw = pre.get_text(" ", strip=True)
    if not raw:
        return []
    tags = [t.strip() for t in raw.split(",")]
    return [t for t in tags if t]


def _path_category_parts(
    url: str, base_url: str
) -> tuple[Optional[str], Optional[str]]:
    parsed = urlparse(url)
    base = urlparse(base_url)
    if parsed.netloc != base.netloc:
        return None, None
    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        return None, None
    if parts[0] == "gallery":
        # /gallery is just the entry point.
        return None, None
    category = parts[0]
    subcategory = parts[1] if len(parts) > 1 else None
    return category, subcategory


def _extract_art_cards(
    soup: BeautifulSoup,
    *,
    page_url: str,
    base_url: str,
) -> list[dict[str, Any]]:
    category, subcategory = _path_category_parts(page_url, base_url)
    page_summary = _extract_page_summary(soup)
    page_tags = _extract_page_tags(soup)

    cards: list[dict[str, Any]] = []
    for card in soup.select("div.card.art-card"):
        ascii_el = card.select_one("div.art-card__ascii")
        if ascii_el is None:
            continue

        raw_text = ascii_el.get_text("\n", strip=False)
        if not raw_text or not raw_text.strip():
            continue

        source_id = card.get("data-id")

        header_span = card.select_one("div.card-header span.fw-medium")
        title: Optional[str] = None
        artist: Optional[str] = None
        if header_span is not None:
            # The title is usually the first direct text node in the span.
            direct_text = header_span.find(string=True, recursive=False)
            if direct_text is not None:
                candidate = str(direct_text).strip()
                if candidate:
                    title = candidate

            small = header_span.select_one("small")
            if small is not None:
                small_text = small.get_text(" ", strip=True)
                if small_text.lower().startswith("by "):
                    artist = small_text[3:].strip() or None
                else:
                    artist = small_text or None

        tags: list[str] = []
        if category:
            tags.append(category)
        if subcategory:
            tags.append(subcategory)
        tags.extend(page_tags)
        # De-dupe tags
        deduped_tags: list[str] = []
        seen_tags: set[str] = set()
        for tag in tags:
            key = tag.strip()
            if not key:
                continue
            if key.lower() in seen_tags:
                continue
            seen_tags.add(key.lower())
            deduped_tags.append(key)

        description: Optional[str]
        if page_summary:
            description = page_summary
        elif category and subcategory:
            description = f"{category}/{subcategory}"
        elif category:
            description = category
        else:
            description = None

        cards.append(
            {
                "source": "asciiart.eu",
                "source_url": page_url,
                "source_id": source_id,
                "title": title,
                "artist": artist,
                "description": description,
                "category": category,
                "subcategory": subcategory,
                "tags": deduped_tags or None,
                "raw_text": raw_text,
            }
        )

    return cards


def scrape_ascii_art_archive(config: ScrapeConfig) -> None:
    base_url = _normalize_base_url(config.base_url)
    if not config.dry_run:
        config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        progress = ProgressState.load(config.progress_path)
        logger.info(
            "Loaded progress: %d pages processed", len(progress.processed_pages)
        )
    else:
        progress = ProgressState(processed_pages=set())
        logger.info("Dry run: progress tracking disabled.")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "ascii_art_mini_transformer/0.1 (research; respectful crawler; "
                "https://github.com/openai/codex-cli)"
            ),
        }
    )

    conn = None
    if config.insert_into_db and not config.dry_run:
        conn = connect(config.db_path)
        initialize(conn)

    pages_seen = 0
    items_seen = 0

    start = urljoin(base_url, "gallery")
    queue: list[str] = [start]
    seen: set[str] = set()

    jsonl = None
    if not config.dry_run:
        jsonl = config.output_jsonl.open("a", encoding="utf-8")

    try:
        while queue:
            page_url = queue.pop(0)
            if page_url in seen:
                continue
            seen.add(page_url)

            if config.max_pages is not None and pages_seen >= config.max_pages:
                logger.info("Reached --max-pages=%d; stopping.", config.max_pages)
                break
            pages_seen += 1

            try:
                html = _http_get(session, page_url)
                soup = BeautifulSoup(html, "html.parser")

                # Discover additional pages first (even if this page was already processed).
                for link in _extract_gallery_links(soup, base_url):
                    if link not in seen:
                        queue.append(link)

                if not config.dry_run and page_url in progress.processed_pages:
                    _rate_limit(config.delay_seconds, config.jitter_seconds)
                    continue

                cards = _extract_art_cards(soup, page_url=page_url, base_url=base_url)
                if not cards:
                    if not config.dry_run:
                        progress.processed_pages.add(page_url)
                        progress.save(config.progress_path)
                    _rate_limit(config.delay_seconds, config.jitter_seconds)
                    continue

                if config.max_items is not None:
                    remaining = config.max_items - items_seen
                    if remaining <= 0:
                        logger.info(
                            "Reached --max-items=%d; stopping.", config.max_items
                        )
                        break
                    cards = cards[:remaining]

                items_seen += len(cards)

                inserted_this_page = 0
                duplicates_this_page = 0

                if conn is not None:
                    with conn:
                        for entry in cards:
                            row_id = insert_ascii_art(
                                conn,
                                raw_text=str(entry["raw_text"]),
                                source=str(entry["source"]),
                                source_id=str(entry["source_id"])
                                if entry.get("source_id")
                                else None,
                                title=entry.get("title"),
                                description=entry.get("description"),
                                category=entry.get("category"),
                                subcategory=entry.get("subcategory"),
                                tags=entry.get("tags"),
                                artist=entry.get("artist"),
                            )
                            if row_id is None:
                                duplicates_this_page += 1
                            else:
                                inserted_this_page += 1

                if jsonl is not None:
                    for entry in cards:
                        jsonl.write(json.dumps(entry, ensure_ascii=False) + "\n")

                if not config.dry_run:
                    progress.inserted += inserted_this_page
                    progress.duplicates += duplicates_this_page
                    progress.processed_pages.add(page_url)
                    progress.save(config.progress_path)

                logger.info(
                    "Scraped %s: %d items (%d inserted, %d dup)",
                    page_url,
                    len(cards),
                    inserted_this_page,
                    duplicates_this_page,
                )

            except Exception as exc:
                if not config.dry_run:
                    progress.errors += 1
                    progress.save(config.progress_path)
                logger.warning("Error scraping %s: %s", page_url, exc)

            _rate_limit(config.delay_seconds, config.jitter_seconds)

    finally:
        if jsonl is not None:
            jsonl.close()

    if conn is not None:
        conn.close()

    logger.info(
        "Done. Pages=%d, inserted=%d, dup=%d, errors=%d. Output=%s",
        pages_seen,
        progress.inserted,
        progress.duplicates,
        progress.errors,
        config.output_jsonl,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scrape the ASCII Art Archive gallery pages"
    )
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="Base URL (default: %(default)s)"
    )
    parser.add_argument(
        "--db-path", default=str(ROOT / "data" / "ascii_art.db"), help="SQLite DB path"
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(ROOT / "data" / "raw" / "asciiart_eu.jsonl"),
        help="Write raw scraped records as JSONL",
    )
    parser.add_argument(
        "--progress-path",
        default=str(ROOT / "data" / "raw" / "asciiart_eu_progress.json"),
        help="Progress JSON path for resumability",
    )
    parser.add_argument(
        "--delay-seconds", type=float, default=1.5, help="Base delay between requests"
    )
    parser.add_argument(
        "--jitter-seconds", type=float, default=0.5, help="Random jitter added to delay"
    )
    parser.add_argument(
        "--max-pages", type=int, default=None, help="Stop after N pages (debugging)"
    )
    parser.add_argument(
        "--max-items", type=int, default=None, help="Stop after N items (debugging)"
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Only write JSONL (do not insert into the SQLite database)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not write JSONL or DB (just crawl)"
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = _build_arg_parser().parse_args()

    config = ScrapeConfig(
        base_url=str(args.base_url),
        db_path=Path(args.db_path),
        output_jsonl=Path(args.output_jsonl),
        progress_path=Path(args.progress_path),
        delay_seconds=float(args.delay_seconds),
        jitter_seconds=float(args.jitter_seconds),
        max_pages=int(args.max_pages) if args.max_pages is not None else None,
        max_items=int(args.max_items) if args.max_items is not None else None,
        insert_into_db=not bool(args.no_db),
        dry_run=bool(args.dry_run),
    )

    if config.dry_run:
        # Avoid creating output files during dry-run.
        config.progress_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Dry run enabled: crawl only (no JSONL/DB writes).")

    scrape_ascii_art_archive(config)


if __name__ == "__main__":
    main()
