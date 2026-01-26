"""
Smoke tests for the Python inference CLI.

These focus on load compatibility:
- Training checkpoints produced by python/train/train.py (key "model" + "model_config")
- Float safetensors exports produced by python/train/export.py (mask buffers omitted)
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

# Add parent to path for imports before importing project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors.torch")

from safetensors.torch import save_file  # noqa: E402

from inference import cli as inference_cli  # noqa: E402
from model.transformer import AsciiGPTConfig, create_model  # noqa: E402


def _tiny_config() -> AsciiGPTConfig:
    return AsciiGPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=128, dropout=0.0)


def test_inference_cli_loads_training_checkpoint(tmp_path: Path) -> None:
    config = _tiny_config()
    model = create_model(config)

    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": asdict(config),
        },
        ckpt_path,
    )

    rc = inference_cli.main(
        [
            "cat",
            "--checkpoint",
            str(ckpt_path),
            "--width",
            "20",
            "--height",
            "10",
            "--temperature",
            "0",
            "--top-k",
            "0",
            "--top-p",
            "1",
            "--max-tokens",
            "1",
            "--seed",
            "0",
        ]
    )
    assert rc == 0


def test_inference_cli_loads_float_safetensors_export(tmp_path: Path) -> None:
    config = _tiny_config()
    model = create_model(config)

    export_dir = tmp_path / "exported"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Mimic python/train/export.py: omit attention masks and break tied-weight aliasing.
    state: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if name.endswith(".mask"):
            continue
        state[name] = tensor.detach().to(dtype=torch.float32, device="cpu")

    if "token_embedding.weight" in state and "lm_head.weight" in state:
        state["lm_head.weight"] = state["lm_head.weight"].clone()

    weights_path = export_dir / "model.safetensors"
    save_file(state, weights_path)

    config_path = export_dir / "config.json"
    config_payload = asdict(config)
    config_payload["export_dtype"] = "float32"  # extra key should be ignored by loader
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    rc = inference_cli.main(
        [
            "cat",
            "--model",
            str(weights_path),
            "--width",
            "20",
            "--height",
            "10",
            "--temperature",
            "0",
            "--top-k",
            "0",
            "--top-p",
            "1",
            "--max-tokens",
            "1",
            "--seed",
            "0",
        ]
    )
    assert rc == 0
