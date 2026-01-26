from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BeautifulSoup = pytest.importorskip("bs4").BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from data.scrape_asciiart import (  # noqa: E402
    ScrapeConfig,
    _extract_art_cards,
    scrape_ascii_art_archive,
)


class _FakeResponse:
    def __init__(self, *, text: str, status_code: int = 200):
        self.text = text
        self.status_code = int(status_code)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    def __init__(self, *, responses: dict[str, _FakeResponse]):
        self._responses = dict(responses)
        self.headers: dict[str, str] = {}

    def get(self, url: str, *args: object, **kwargs: object) -> _FakeResponse:
        _ = (args, kwargs)
        try:
            return self._responses[url]
        except KeyError as exc:
            raise KeyError(f"Unexpected URL fetch: {url}") from exc

    def close(self) -> None:
        # Used only when scrapers own the session.
        return None


def test_extract_art_cards_parses_title_artist_and_ascii() -> None:
    html = """
    <html>
      <body data-subpage="animals/cats">
        <div class="text-truncate"><span>A large collection of cats.</span></div>
        <pre class="opacity-75">cat, kitten, domestic cats</pre>

        <div class="card art-card" data-id="abc123">
          <div class="card-header">
            <span class="fw-medium">Cat<small class="d-block">By Alice</small></span>
          </div>
          <div class="card-body">
            <div class="art-card__ascii">  /\\_/\\\n ( o.o )\n  &gt; ^ &lt;\n</div>
          </div>
        </div>
      </body>
    </html>
    """

    soup = BeautifulSoup(html, "html.parser")
    cards = _extract_art_cards(
        soup,
        page_url="https://www.asciiart.eu/animals/cats",
        base_url="https://www.asciiart.eu/",
    )

    assert len(cards) == 1
    card = cards[0]

    assert card["source"] == "asciiart.eu"
    assert card["source_id"] == "abc123"
    assert card["category"] == "animals"
    assert card["subcategory"] == "cats"
    assert card["title"] == "Cat"
    assert card["artist"] == "Alice"
    assert card["description"] == "A large collection of cats."
    assert "cat" in (card["tags"] or [])
    assert "cats" in (card["tags"] or [])
    assert "animals" in (card["tags"] or [])

    # Ensure HTML entities are decoded and newlines are preserved.
    assert "> ^ <" in card["raw_text"]
    assert "/\\_/\\\n" in card["raw_text"]


def test_scrape_asciiart_archive_crawl_writes_progress_and_jsonl(
    tmp_path: Path,
) -> None:
    base_url = "https://example.com/"
    gallery_url = "https://example.com/gallery"
    cats_url = "https://example.com/animals/cats"

    gallery_html = """
    <html><body>
      <a class="card-gallery" href="/animals/cats">Cats</a>
    </body></html>
    """
    cats_html = """
    <html>
      <body>
        <div class="text-truncate"><span>A large collection of cats.</span></div>
        <pre class="opacity-75">cat, kitten, domestic cats</pre>
        <div class="card art-card" data-id="abc123">
          <div class="card-header">
            <span class="fw-medium">Cat<small class="d-block">By Alice</small></span>
          </div>
          <div class="card-body">
            <div class="art-card__ascii">  /\\_/\\\n ( o.o )\n  &gt; ^ &lt;\n</div>
          </div>
        </div>
      </body>
    </html>
    """

    session = _FakeSession(
        responses={
            gallery_url: _FakeResponse(text=gallery_html),
            cats_url: _FakeResponse(text=cats_html),
        }
    )

    out_jsonl = tmp_path / "asciiart.jsonl"
    progress_path = tmp_path / "progress.json"

    config = ScrapeConfig(
        base_url=base_url,
        db_path=tmp_path / "unused.db",
        output_jsonl=out_jsonl,
        progress_path=progress_path,
        delay_seconds=0.0,
        jitter_seconds=0.0,
        max_pages=None,
        max_items=None,
        insert_into_db=False,
        dry_run=False,
    )

    scrape_ascii_art_archive(config, session=session)

    assert progress_path.exists()
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert set(progress["processed_pages"]) == {gallery_url, cats_url}

    lines = out_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["source"] == "asciiart.eu"
    assert entry["source_id"] == "abc123"
    assert entry["category"] == "animals"
    assert entry["subcategory"] == "cats"


def test_scrape_asciiart_archive_skips_already_processed_pages(tmp_path: Path) -> None:
    base_url = "https://example.com/"
    gallery_url = "https://example.com/gallery"
    cats_url = "https://example.com/animals/cats"

    gallery_html = """
    <html><body>
      <a class="card-gallery" href="/animals/cats">Cats</a>
    </body></html>
    """
    cats_html = """
    <html>
      <body>
        <div class="card art-card" data-id="abc123">
          <div class="card-header">
            <span class="fw-medium">Cat<small class="d-block">By Alice</small></span>
          </div>
          <div class="card-body">
            <div class="art-card__ascii">cat\n</div>
          </div>
        </div>
      </body>
    </html>
    """

    session = _FakeSession(
        responses={
            gallery_url: _FakeResponse(text=gallery_html),
            cats_url: _FakeResponse(text=cats_html),
        }
    )

    out_jsonl = tmp_path / "asciiart.jsonl"
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "processed_pages": [gallery_url],
                "inserted": 0,
                "duplicates": 0,
                "errors": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    config = ScrapeConfig(
        base_url=base_url,
        db_path=tmp_path / "unused.db",
        output_jsonl=out_jsonl,
        progress_path=progress_path,
        delay_seconds=0.0,
        jitter_seconds=0.0,
        max_pages=None,
        max_items=None,
        insert_into_db=False,
        dry_run=False,
    )

    scrape_ascii_art_archive(config, session=session)

    # Only the cats page should have been written (gallery was skipped as processed).
    lines = out_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["source_id"] == "abc123"
