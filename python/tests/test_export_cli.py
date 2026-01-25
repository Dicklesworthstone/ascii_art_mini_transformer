"""
CLI-level smoke tests for `python/train/export.py`.

We keep these tests lightweight: export a tiny fresh model (no checkpoint) and
assert the resulting `config.json` reflects requested hyperparameters.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")


def test_export_cli_fresh_model_overrides_config(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = tmp_path / "exported"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "python")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "train.export",
            "--output-dir",
            str(out_dir),
            "--dtype",
            "float32",
            "--quantize",
            "none",
            "--preset",
            "small",
            "--n-layer",
            "2",
            "--n-head",
            "2",
            "--n-embd",
            "64",
            "--block-size",
            "128",
        ],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    cfg_path = out_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["n_layer"] == 2
    assert cfg["n_head"] == 2
    assert cfg["n_embd"] == 64
    assert cfg["block_size"] == 128

