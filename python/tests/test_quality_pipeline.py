from __future__ import annotations

import sqlite3
from pathlib import Path

from python.data import quality_pipeline as qp

# quality_pipeline imports db via sys.path hacks as `data.db`; use the same module in tests to
# avoid duplicated module state.
from data import db  # noqa: E402


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "schema.sql"


def test_validate_art_issue_codes() -> None:
    assert qp.validate_art("", width=0, height=0) == ["empty"]
    assert qp.validate_art("   \n\t", width=0, height=0) == ["empty"]

    sparse = "abc\nabc\n"
    issues = qp.validate_art(sparse, width=3, height=2)
    assert "too_sparse" in issues
    assert "empty" not in issues

    wide = ("x" * (qp.MAX_WIDTH + 1)) + "\n" + ("y" * (qp.MAX_WIDTH + 1))
    issues = qp.validate_art(wide, width=qp.MAX_WIDTH + 1, height=2)
    assert "too_wide" in issues

    too_many_lines = "\n".join(["x"] * (qp.MAX_LINES + 1))
    issues = qp.validate_art(too_many_lines, width=1, height=qp.MAX_LINES + 1)
    assert "too_many_lines" in issues

    has_control = "0123456789\nab\x01cd"
    issues = qp.validate_art(has_control, width=10, height=2)
    assert "control_chars" in issues

    has_encoding_error = "0123456789\nbad\ufffdchar"
    issues = qp.validate_art(has_encoding_error, width=10, height=2)
    assert "encoding_error" in issues

    has_null = "0123456789\nnull\x00byte"
    issues = qp.validate_art(has_null, width=10, height=2)
    assert "null_bytes" in issues


def test_run_validation_updates_is_valid_when_not_dry_run(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "quality.db")
    db.initialize(conn, schema_path=_schema_path())

    valid_id = db.insert_ascii_art(
        conn,
        raw_text="0123456789\n0123456789",
        source="test",
        is_valid=False,
    )
    assert valid_id is not None

    invalid_id = db.insert_ascii_art(
        conn,
        raw_text="abc\nabc\n",  # too_sparse per pipeline thresholds
        source="test",
        is_valid=True,
    )
    assert invalid_id is not None

    report = qp.run_validation(conn, dry_run=False)
    assert report.total_checked == 2
    assert report.valid == 1
    assert report.invalid == 1

    valid_row = db.get_ascii_art_by_id(conn, valid_id)
    invalid_row = db.get_ascii_art_by_id(conn, invalid_id)
    assert valid_row is not None and invalid_row is not None
    assert valid_row.is_valid is True
    assert invalid_row.is_valid is False

    conn.close()


def test_run_validation_does_not_mutate_when_dry_run(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "quality_dry.db")
    db.initialize(conn, schema_path=_schema_path())

    invalid_id = db.insert_ascii_art(
        conn,
        raw_text="abc\nabc\n",  # too_sparse per pipeline thresholds
        source="test",
        is_valid=True,
    )
    assert invalid_id is not None

    before = db.get_ascii_art_by_id(conn, invalid_id)
    assert before is not None
    assert before.is_valid is True

    report = qp.run_validation(conn, dry_run=True)
    assert report.total_checked == 1
    assert report.invalid == 1

    after = db.get_ascii_art_by_id(conn, invalid_id)
    assert after is not None
    assert after.is_valid is True

    conn.close()


def test_retry_on_lock_does_not_swallow_non_lock_operational_errors() -> None:
    def _boom() -> int:
        raise sqlite3.OperationalError("not a lock")

    try:
        qp.retry_on_lock(_boom, max_retries=3)
    except sqlite3.OperationalError as exc:
        assert "not a lock" in str(exc)
    else:
        raise AssertionError("Expected OperationalError to be raised")
