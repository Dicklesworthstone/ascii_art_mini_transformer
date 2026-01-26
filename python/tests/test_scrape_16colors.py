from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from data.scrape_16colors import strip_sauce  # noqa: E402


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
