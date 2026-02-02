from __future__ import annotations

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


def test_csplk_is_separator_line_detection() -> None:
    assert ih._csplk_is_separator_line("-----")
    assert ih._csplk_is_separator_line("=====")
    assert ih._csplk_is_separator_line("_____")
    assert ih._csplk_is_separator_line("*****")
    assert ih._csplk_is_separator_line("~~~")

    assert not ih._csplk_is_separator_line("--")  # too short
    assert not ih._csplk_is_separator_line("File: -----")  # file marker line
    assert not ih._csplk_is_separator_line("abc---")  # mixed chars


def test_csplk_is_title_line_positives_and_negatives() -> None:
    # Positives
    assert ih._csplk_is_title_line("Cute Cats")
    assert ih._csplk_is_title_line("Hello World")

    # Negatives: indentation, URLs, code fences, file markers, separators.
    assert not ih._csplk_is_title_line("  Indented Title")
    assert not ih._csplk_is_title_line("https://example.com")
    assert not ih._csplk_is_title_line("```")
    assert not ih._csplk_is_title_line("File: animals/cats/cat1.txt")
    assert not ih._csplk_is_title_line("-----")

    # Negatives: no alpha, too many digits, and art-like long runs.
    assert not ih._csplk_is_title_line("12345")
    assert not ih._csplk_is_title_line("ASCII 2024")  # digit ratio heuristic
    assert not ih._csplk_is_title_line("Y88888888b")  # long digit run typical of art

    # Negatives: high non-alnum ratio.
    assert not ih._csplk_is_title_line("HELLO!!!")


def test_iter_csplk_blocks_splits_on_double_blank_and_file_markers() -> None:
    ds = [
        {"text": "File: animals/cats/cat1.txt"},
        {"text": "Cute Cats"},
        {"text": " /\\_/\\ "},
        {"text": "( o.o )"},
        {"text": ""},
        {"text": ""},
        {"text": "Another Title"},
        {"text": "(=^.^=)"},
        {"text": "File: animals/dogs/dog1.txt"},
        {"text": "Doggo"},
        {"text": " /\\_/\\ "},
        {"text": "( woof )"},
    ]

    blocks = list(ih._iter_csplk_blocks(ds, start_row=0, split_blocks=True))

    assert len(blocks) == 3

    _idx0, path0, block0_idx, meta0, text0 = blocks[0]
    assert path0 == "animals/cats/cat1.txt"
    assert block0_idx == 1
    assert meta0["category"] == "animals"
    assert meta0["subcategory"] == "cats"
    assert meta0["title"] == "Cute Cats"
    assert "Cute Cats" not in text0
    assert "/\\_/\\ " in text0
    assert "( o.o )" in text0

    _idx1, path1, block1_idx, meta1, text1 = blocks[1]
    assert path1 == "animals/cats/cat1.txt"
    assert block1_idx == 2
    assert meta1["title"] == "Another Title"
    assert "Another Title" not in text1
    assert "(=^.^=)" in text1

    _idx2, path2, block2_idx, meta2, text2 = blocks[2]
    assert path2 == "animals/dogs/dog1.txt"
    assert block2_idx == 1  # resets per-file
    assert meta2["title"] == "Doggo"
    assert "/\\_/\\ " in text2
    assert "( woof )" in text2


def test_iter_csplk_blocks_skips_indented_preamble_before_art() -> None:
    ds = [
        {"text": "File: misc/foo/bar.txt"},
        {"text": "  This is a header line, not art"},
        {"text": "  Another header-ish line"},
        {"text": " /\\_/\\ "},
        {"text": "( o.o )"},
    ]

    blocks = list(ih._iter_csplk_blocks(ds, start_row=0, split_blocks=True))
    assert len(blocks) == 1

    _idx, _path, _block_idx, meta, text = blocks[0]
    assert meta["category"] == "misc"
    assert meta["subcategory"] == "foo"
    assert "header line" not in text
    assert "/\\_/\\ " in text
    assert "( o.o )" in text


def test_iter_csplk_blocks_splits_on_separator_lines() -> None:
    ds = [
        {"text": "File: misc/separators/demo.txt"},
        {"text": "Example"},
        {"text": " /\\_/\\ "},
        {"text": "( o.o )"},
        {"text": ""},
        {"text": "-----"},
        {"text": "Second"},
        {"text": "(=^.^=)"},
    ]

    blocks = list(ih._iter_csplk_blocks(ds, start_row=0, split_blocks=True))
    assert len(blocks) == 2

    _idx0, _path0, block0_idx, meta0, text0 = blocks[0]
    assert block0_idx == 1
    assert meta0["title"] == "Example"
    assert "-----" not in text0

    _idx1, _path1, block1_idx, meta1, text1 = blocks[1]
    assert block1_idx == 2
    assert meta1["title"] == "Second"
    assert "-----" not in text1
    assert "(=^.^=)" in text1


def test_iter_csplk_blocks_keeps_single_blank_as_internal_whitespace() -> None:
    ds = [
        {"text": "File: misc/whitespace/demo.txt"},
        {"text": "Example"},
        {"text": "  line 1 (indented, not a title)"},
        {"text": ""},
        {"text": "  line 2 (indented, not a title)"},
    ]

    blocks = list(ih._iter_csplk_blocks(ds, start_row=0, split_blocks=True))
    assert len(blocks) == 1

    _idx, _path, _block_idx, meta, text = blocks[0]
    assert meta["title"] == "Example"
    assert "line 1 (indented, not a title)" in text
    assert "line 2 (indented, not a title)" in text
    assert "\n\n" in text


def test_iter_csplk_blocks_splits_on_art_blank_art_heuristic() -> None:
    # Single blank line between art-like lines should split when the current block already has
    # multiple art lines, to avoid merging separate pieces.
    ds = [
        {"text": "File: misc/art_split/demo.txt"},
        {"text": "Art A"},
        {"text": " /\\_/\\ "},
        {"text": "( o.o )"},
        {"text": " > ^ <"},
        {"text": ""},
        {"text": " /\\_/\\ "},
        {"text": "( woof )"},
    ]

    blocks = list(ih._iter_csplk_blocks(ds, start_row=0, split_blocks=True))
    assert len(blocks) == 2

    _idx0, path0, block0_idx, meta0, text0 = blocks[0]
    assert path0 == "misc/art_split/demo.txt"
    assert block0_idx == 1
    assert meta0["title"] == "Art A"
    assert " /\\_/\\ " in text0
    assert " > ^ <" in text0
    assert "( woof )" not in text0

    _idx1, path1, block1_idx, meta1, text1 = blocks[1]
    assert path1 == "misc/art_split/demo.txt"
    assert block1_idx == 2
    # Title should reset after the split heuristic (second block uses the file-stem fallback).
    assert meta1["title"] == "demo"
    assert "( woof )" in text1


def test_extract_art_text_mrzjy_prefers_assistant_block() -> None:
    row = {
        "conversations": [
            {"role": "user", "content": "Draw a cat"},
            {"role": "assistant", "content": "```\\n/\\\\_/\\\\\\n( o.o )\\n```"},
        ]
    }
    assert (
        ih.extract_art_text(row, "mrzjy/ascii_art_generation_140k")
        == "/\\_/\\\n( o.o )"
    )


def test_extract_art_text_mrzjy_falls_back_to_assistant_content() -> None:
    row = {
        "conversations": [
            {"role": "user", "content": "Draw a cat"},
            {"role": "assistant", "content": " /\\_/\\ \n( o.o )"},
        ]
    }
    assert (
        ih.extract_art_text(row, "mrzjy/ascii_art_generation_140k")
        == "/\\_/\\ \n( o.o )"
    )


def test_extract_art_text_jdpressman_prefers_aic_then_i2a() -> None:
    row = {"art_aic": "aic", "art_i2a": "i2a"}
    assert ih.extract_art_text(row, "jdpressman/retro-ascii-art-v1") == "aic"

    row = {"art_aic": "   ", "art_i2a": "i2a"}
    assert ih.extract_art_text(row, "jdpressman/retro-ascii-art-v1") == "i2a"


def test_extract_art_text_generic_uses_first_nonempty_text_column() -> None:
    row = {"content": "  ", "text": "ok", "ascii_art": "nope"}
    assert ih.extract_art_text(row, "apehex/ascii-art") == "ok"


def test_extract_metadata_apehex_labels_and_caption() -> None:
    row = {"labels": ["animals", "cats"], "caption": "cute cats\nextra"}
    meta = ih.extract_metadata(row, "apehex/ascii-art")
    assert meta["category"] == "animals"
    assert meta["tags"] == ["animals", "cats"]
    assert meta["title"] == "cute cats\nextra"[:200]


def test_extract_metadata_mrzjy_user_instruction() -> None:
    row = {"conversations": [{"role": "user", "content": "Make a dragon"}]}
    meta = ih.extract_metadata(row, "mrzjy/ascii_art_generation_140k")
    assert meta["description"] == "Make a dragon"


def test_extract_metadata_jdpressman_style_subject_prompt() -> None:
    row = {"prompt": "Draw a cat", "style": "retro", "subject": "cat"}
    meta = ih.extract_metadata(row, "jdpressman/retro-ascii-art-v1")
    assert meta["description"] == "Draw a cat"
    assert meta["category"] == "retro"
    assert meta["subcategory"] == "cat"


def test_validate_art_rejects_and_accepts_expected_cases() -> None:
    ok, reason = ih.validate_art("12345\n12345")
    assert ok, reason
    assert reason == "ok"

    assert ih.validate_art("   ")[1] == "empty"
    assert ih.validate_art("123456789")[1] == "too_small"
    assert ih.validate_art("a" * 500_001 + "\nxx")[1] == "too_large"
    assert ih.validate_art("line 1\x00\nline 2")[1] == "binary_data"
    assert ih.validate_art("one line only")[1] == "too_few_lines"

    too_many = "\n".join(["x"] * 1001)
    assert ih.validate_art(too_many)[1] == "too_many_lines"


def test_ingest_dataset_non_csplk_respects_max_inserts_and_closes_transaction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tqdm import tqdm as real_tqdm

    def quiet_tqdm(*args, **kwargs):
        kwargs.setdefault("disable", True)
        return real_tqdm(*args, **kwargs)

    rows = [
        {"text": "aaaaa\naaaaa"},
        {"text": "bbbbb\nbbbbb"},
        {"text": "ccccc\nccccc"},
    ]

    def fake_load_dataset(*_args, **_kwargs):
        return rows

    monkeypatch.setattr(ih, "tqdm", quiet_tqdm)
    monkeypatch.setattr(ih, "load_dataset", fake_load_dataset)

    conn = ih.connect(tmp_path / "hf_ingest.sqlite")
    ih.initialize(conn)
    tracker = ih.ProgressTracker(progress_file=str(tmp_path / "progress.json"))

    stats = ih.ingest_dataset(
        conn,
        "apehex/ascii-art",
        None,
        tracker,
        checkpoint_every=10_000,
        max_inserts=1,
        force=True,
    )

    assert stats.inserted == 1
    assert conn.in_transaction is False
    assert conn.execute("SELECT COUNT(*) FROM ascii_art").fetchone()[0] == 1
    conn.close()


def test_ingest_dataset_stop_at_total_rows_does_not_leave_open_transaction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tqdm import tqdm as real_tqdm

    def quiet_tqdm(*args, **kwargs):
        kwargs.setdefault("disable", True)
        return real_tqdm(*args, **kwargs)

    # Non-CsPLK dataset path: use start_row_override=1 so the first processed item
    # has idx=1 and triggers the checkpoint block when checkpoint_every=1.
    rows = [
        {"text": " "},  # skipped (empty/whitespace)
        {"text": "ddddd\nddddd"},
        {"text": "eeeee\neeeee"},
    ]

    def fake_load_dataset(*_args, **_kwargs):
        return rows

    monkeypatch.setattr(ih, "tqdm", quiet_tqdm)
    monkeypatch.setattr(ih, "load_dataset", fake_load_dataset)

    conn = ih.connect(tmp_path / "hf_ingest_stop.sqlite")
    ih.initialize(conn)
    tracker = ih.ProgressTracker(progress_file=str(tmp_path / "progress_stop.json"))

    stats = ih.ingest_dataset(
        conn,
        "apehex/ascii-art",
        None,
        tracker,
        checkpoint_every=1,
        stop_at_total_rows=1,
        start_row_override=1,
        force=True,
    )

    assert stats.inserted == 1
    assert conn.in_transaction is False
    conn.close()


def test_ingest_dataset_csplk_stop_at_total_rows_does_not_leave_open_transaction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tqdm import tqdm as real_tqdm

    def quiet_tqdm(*args, **kwargs):
        kwargs.setdefault("disable", True)
        return real_tqdm(*args, **kwargs)

    ds = [
        {"text": "File: animals/cats/cat1.txt"},
        {"text": "Cute Cats"},
        {"text": " /\\\\_/\\\\ "},
        {"text": "( o.o )"},
        {"text": ""},
    ]

    def fake_load_dataset(*_args, **_kwargs):
        return ds

    monkeypatch.setattr(ih, "tqdm", quiet_tqdm)
    monkeypatch.setattr(ih, "load_dataset", fake_load_dataset)

    conn = ih.connect(tmp_path / "hf_ingest_csplk.sqlite")
    ih.initialize(conn)
    tracker = ih.ProgressTracker(progress_file=str(tmp_path / "progress_csplk.json"))

    stats = ih.ingest_dataset(
        conn,
        "Csplk/THE.ASCII.ART.EMPORIUM",
        None,
        tracker,
        checkpoint_every=1,
        stop_at_total_rows=1,
        force=True,
    )

    assert stats.inserted == 1
    assert conn.in_transaction is False
    conn.close()
