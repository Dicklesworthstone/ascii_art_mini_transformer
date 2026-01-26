from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from data.scrape_textfiles import (  # noqa: E402
    ScrapeConfig,
    _extract_listing_hrefs,
    _parse_args,
    _rate_limit,
    scrape_textfiles,
)


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


def test_extract_listing_hrefs_filters_parent_and_dedupes() -> None:
    html = """
    <html>
      <body>
        <pre>
          <a href="../">Parent Directory</a>
          <a href="?C=N;O=D">Name</a>
          <a href="mailto:test@example.com">mail</a>
          <a href="javascript:alert(1)">js</a>
          <a href="bbs/">bbs/</a>
          <a href="scene/">scene/</a>
          <a href="bbs/">bbs/</a>
          <a href="logo.ans">logo.ans</a>
          <a href="file.txt">file.txt</a>
          <a href="#ignore">ignore</a>
        </pre>
      </body>
    </html>
    """

    hrefs = _extract_listing_hrefs(html)
    assert hrefs == ["bbs/", "scene/", "logo.ans", "file.txt"]


def test_scrape_textfiles_crawls_local_directory_listing(tmp_path: Path) -> None:
    base_url = "http://example.com/ansi/"

    root_html = """
    <html><body>
      <a href="../">Parent Directory</a>
      <a href="sub/">sub/</a>
      <a href="logo.ans">logo.ans</a>
      <a href="skip.zip">skip.zip</a>
    </body></html>
    """
    sub_html = """
    <html><body>
      <a href="../">Parent Directory</a>
      <a href="inner.ans">inner.ans</a>
    </body></html>
    """

    def _cp437(text: str) -> bytes:
        return text.encode("cp437", errors="replace")

    responses: dict[str, _FakeResponse] = {
        base_url: _FakeResponse(content=root_html.encode("utf-8")),
        f"{base_url}sub/": _FakeResponse(content=sub_html.encode("utf-8")),
        f"{base_url}logo.ans": _FakeResponse(content=_cp437("HELLO\r\nWORLD\r\n")),
        f"{base_url}sub/inner.ans": _FakeResponse(content=_cp437("INNER\r\nLINE\r\n")),
    }

    session = _FakeSession(responses=responses)
    progress_path = tmp_path / "progress.json"
    out_jsonl = tmp_path / "out.jsonl"

    config = ScrapeConfig(
        base_url=base_url,
        db_path=tmp_path / "unused.db",
        output_jsonl=out_jsonl,
        progress_path=progress_path,
        delay_seconds=0.0,
        max_files=None,
        strip_ansi=False,
        insert_into_db=False,
        dry_run=False,
    )

    state = scrape_textfiles(config, session=session)

    assert state.errors == 0
    assert state.skipped == 1  # skip.zip
    assert state.inserted == 0  # no DB inserts in this test
    assert f"{base_url}logo.ans" in state.processed_files
    assert f"{base_url}sub/inner.ans" in state.processed_files
    assert progress_path.exists()

    # JSONL contains both records (order not guaranteed).
    raw_lines = out_jsonl.read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in raw_lines]
    assert {p["title"] for p in payloads} == {"logo.ans", "inner.ans"}
    assert any("HELLO\nWORLD" in p["raw_text"] for p in payloads)

    # Second run should load progress and avoid re-fetching.
    state2 = scrape_textfiles(config, session=session)
    assert state2.errors == 0


def test_scrape_textfiles_strip_ansi_and_hits_max_files(tmp_path: Path) -> None:
    base_url = "http://example.com/ansi/"
    root_html = """
    <html><body>
      <a href="a.ans">a.ans</a>
      <a href="b.ans">b.ans</a>
    </body></html>
    """

    ansi_payload = b"\x1b[31mRED\x1b[0m\r\nLINE\r\n"

    responses: dict[str, _FakeResponse] = {
        base_url: _FakeResponse(content=root_html.encode("utf-8")),
        f"{base_url}a.ans": _FakeResponse(content=ansi_payload),
        f"{base_url}b.ans": _FakeResponse(content=b"SHOULD_NOT_FETCH\r\n"),
    }
    session = _FakeSession(responses=responses)

    out_jsonl = tmp_path / "out.jsonl"
    progress_path = tmp_path / "progress.json"
    config = ScrapeConfig(
        base_url=base_url,
        db_path=tmp_path / "unused.db",
        output_jsonl=out_jsonl,
        progress_path=progress_path,
        delay_seconds=0.0,
        max_files=1,
        strip_ansi=True,
        insert_into_db=False,
        dry_run=False,
    )

    state = scrape_textfiles(config, session=session)
    assert state.errors == 0

    raw_lines = out_jsonl.read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in raw_lines]
    assert len(payloads) == 1
    assert payloads[0]["title"] == "a.ans"
    assert "\x1b" not in payloads[0]["raw_text"]


def test_parse_args_and_rate_limit() -> None:
    cfg = _parse_args(
        [
            "--base-url",
            "http://example.com/ansi",
            "--db-path",
            "data/ascii_art.db",
            "--delay-seconds",
            "0",
            "--no-db",
            "--dry-run",
        ]
    )
    assert cfg.base_url.endswith("/")
    assert cfg.dry_run is True
    assert cfg.insert_into_db is False

    _rate_limit(0.0)
    _rate_limit(0.001)
