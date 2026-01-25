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
