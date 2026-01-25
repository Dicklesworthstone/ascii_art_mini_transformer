from __future__ import annotations

import sys
from pathlib import Path

import pytest

BeautifulSoup = pytest.importorskip("bs4").BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from data.scrape_asciiart import _extract_art_cards  # noqa: E402


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
