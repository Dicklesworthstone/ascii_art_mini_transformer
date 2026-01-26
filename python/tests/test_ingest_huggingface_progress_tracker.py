from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


pytest.importorskip("datasets")
pytest.importorskip("huggingface_hub")
pytest.importorskip("pyarrow")
pytest.importorskip("tqdm")


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "data"))

import ingest_huggingface as ih  # noqa: E402


def test_progress_tracker_default_state_persists_on_save(tmp_path: Path) -> None:
    progress_file = tmp_path / "progress.json"
    assert not progress_file.exists()

    tracker = ih.ProgressTracker(progress_file=str(progress_file))
    assert tracker.state == {"completed_datasets": [], "last_row": {}, "stats": {}}
    assert not progress_file.exists()

    tracker.save()
    assert progress_file.exists()
    persisted = json.loads(progress_file.read_text(encoding="utf-8"))
    assert persisted == tracker.state


def test_progress_tracker_update_progress_roundtrips(tmp_path: Path) -> None:
    progress_file = tmp_path / "progress.json"
    tracker = ih.ProgressTracker(progress_file=str(progress_file))

    tracker.update_progress("demo", 123)
    assert tracker.get_start_row("demo") == 123

    tracker2 = ih.ProgressTracker(progress_file=str(progress_file))
    assert tracker2.get_start_row("demo") == 123


def test_progress_tracker_mark_completed_persists_stats(tmp_path: Path) -> None:
    progress_file = tmp_path / "progress.json"
    tracker = ih.ProgressTracker(progress_file=str(progress_file))

    stats = ih.IngestionStats(
        dataset_name="demo",
        total_processed=10,
        inserted=7,
        duplicates=2,
        skipped_invalid=1,
        errors=0,
    )
    tracker.mark_completed("demo", stats)
    assert tracker.is_completed("demo") is True

    tracker2 = ih.ProgressTracker(progress_file=str(progress_file))
    assert tracker2.is_completed("demo") is True
    assert tracker2.state["stats"]["demo"] == {
        "total_processed": 10,
        "inserted": 7,
        "duplicates": 2,
        "skipped_invalid": 1,
        "errors": 0,
    }


def test_progress_tracker_reset_dataset_clears_completion_and_stats(
    tmp_path: Path,
) -> None:
    progress_file = tmp_path / "progress.json"
    tracker = ih.ProgressTracker(progress_file=str(progress_file))

    tracker.update_progress("demo", 55)
    tracker.mark_completed(
        "demo", ih.IngestionStats(dataset_name="demo", total_processed=1)
    )
    assert tracker.is_completed("demo") is True
    assert tracker.get_start_row("demo") == 55

    tracker.reset_dataset("demo", start_row=7)
    assert tracker.is_completed("demo") is False
    assert tracker.get_start_row("demo") == 7
    assert "demo" not in tracker.state["stats"]

    tracker2 = ih.ProgressTracker(progress_file=str(progress_file))
    assert tracker2.is_completed("demo") is False
    assert tracker2.get_start_row("demo") == 7
    assert "demo" not in tracker2.state["stats"]
