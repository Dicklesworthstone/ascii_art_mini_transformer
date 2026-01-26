from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from data.scrape_16colors import scrape_16colo_rs, strip_sauce  # noqa: E402


class _FakeResponse:
    def __init__(self, *, content: bytes, status_code: int = 200):
        self.content = content
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


def _make_sauce_record(
    *,
    title: str = "TITLE",
    author: str = "AUTHOR",
    group: str = "GROUP",
    date: str = "19900101",
    comment_count: int = 0,
) -> bytes:
    record = bytearray(128)
    record[0:7] = b"SAUCE00"

    def _put(start: int, length: int, value: str) -> None:
        data = value.encode("cp437", errors="replace")[:length]
        record[start : start + length] = data.ljust(length, b" ")

    _put(7, 35, title)
    _put(42, 20, author)
    _put(62, 20, group)
    _put(82, 8, date)

    record[104] = comment_count
    return bytes(record)


def test_strip_sauce_removes_record_and_extracts_metadata() -> None:
    body = b"hello\r\nworld\r\n"
    sauce = _make_sauce_record(
        title="ABYSS", author="ansiwave", group="art", date="19901231"
    )
    payload = body + b"\x1a" + sauce

    stripped, meta = strip_sauce(payload)
    assert stripped == body.rstrip(b"\x1a")
    assert meta is not None
    assert meta.title == "ABYSS"
    assert meta.author == "ansiwave"
    assert meta.group == "art"
    assert meta.date == "19901231"


def test_strip_sauce_with_comments_block() -> None:
    body = b"payload\n"
    comments = ["first line", "second line"]
    comment_count = len(comments)

    comnt = bytearray(b"COMNT")
    for line in comments:
        comnt.extend(line.encode("cp437").ljust(64, b" "))

    sauce = _make_sauce_record(comment_count=comment_count)
    payload = body + bytes(comnt) + b"\x1a" + sauce

    stripped, meta = strip_sauce(payload)
    assert stripped == body.rstrip(b"\x1a")
    assert meta is not None
    assert meta.comments == comments


def test_strip_sauce_with_missing_comments_block_keeps_empty_comments() -> None:
    body = b"payload\n"
    sauce = _make_sauce_record(comment_count=2)
    payload = body + b"\x1a" + sauce

    stripped, meta = strip_sauce(payload)
    assert stripped == body.rstrip(b"\x1a")
    assert meta is not None
    assert meta.comments == []


def test_scrape_16colors_crawl_writes_progress_and_jsonl(tmp_path: Path) -> None:
    base_url = "https://16colo.rs/"
    year_url = "https://16colo.rs/year/1990"
    pack_url = "https://16colo.rs/pack/mypack"
    raw_url_ans = "https://16colo.rs/pack/mypack/raw/art.ans"

    homepage = """
    <html><body>
      <select id="selecty">
        <option value="/year/1990">1990</option>
      </select>
    </body></html>
    """
    year_html = """
    <html><body>
      <a class="dizname" href="/pack/mypack">mypack</a>
    </body></html>
    """
    pack_html = """
    <html><body>
      <a href="/pack/mypack/art.ans">art.ans</a>
      <a href="/pack/mypack/archive.zip">archive.zip</a>
    </body></html>
    """

    raw_bytes = b"hello\r\nworld\r\n"

    session = _FakeSession(
        responses={
            base_url: _FakeResponse(content=homepage.encode("latin-1")),
            year_url: _FakeResponse(content=year_html.encode("latin-1")),
            pack_url: _FakeResponse(content=pack_html.encode("latin-1")),
            raw_url_ans: _FakeResponse(content=raw_bytes),
        }
    )

    out_jsonl = tmp_path / "out.jsonl"
    progress_path = tmp_path / "progress.json"

    scrape_16colo_rs(
        base_url=base_url,
        db_path=tmp_path / "unused.db",
        output_jsonl=out_jsonl,
        progress_path=progress_path,
        delay_seconds=0.0,
        jitter_seconds=0.0,
        max_years=1,
        max_packs=1,
        max_files=10,
        insert_into_db=False,
        dry_run=False,
        session=session,
    )

    assert progress_path.exists()
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["processed_raw_urls"] == [raw_url_ans]

    lines = out_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["source"] == "16colo.rs"
    assert entry["source_id"] == "mypack/art.ans"
    assert entry["subcategory"] == "mypack"
    assert "ext:ans" in entry["tags"]
