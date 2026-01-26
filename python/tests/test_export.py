"""
Unit tests for `python/train/export.py` correctness and compatibility guarantees.

These tests focus on:
- safetensors export behavior (mask buffers excluded; tied weights handled)
- quantized export naming + quant_config.json metadata
- validate_export() enforcing config/weights consistency
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add parent to path for imports before importing project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

torch = pytest.importorskip("torch")

from safetensors.torch import load_file, save_file  # noqa: E402

from model.transformer import AsciiGPTConfig, create_model  # noqa: E402
from model.tokenizer import get_tokenizer  # noqa: E402
from train.export import export_model, export_quantized_weights, validate_export  # noqa: E402


def _make_tiny_model() -> tuple[object, object]:
    torch.manual_seed(0)
    config = AsciiGPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=64, dropout=0.0)
    model = create_model(config)
    tokenizer = get_tokenizer()
    return model, tokenizer


def test_export_model_excludes_mask_and_breaks_tied_weights(tmp_path: Path) -> None:
    model, tokenizer = _make_tiny_model()

    state_dict = model.state_dict()
    assert any(k.endswith(".mask") for k in state_dict), (
        "Expected attention mask buffer"
    )
    assert (
        state_dict["token_embedding.weight"].data_ptr()
        == state_dict["lm_head.weight"].data_ptr()
    ), "Expected tied weights to share storage"

    # Prove why export_model clones lm_head: safetensors refuses shared-storage dicts.
    raw_export = {
        k: v.detach().to(device="cpu")
        for k, v in state_dict.items()
        if not k.endswith(".mask")
    }
    with pytest.raises(Exception):
        save_file(raw_export, tmp_path / "raw.safetensors")

    export_model(model, tokenizer, tmp_path, export_dtype="float32")

    weights = load_file(tmp_path / "model.safetensors")
    assert not any(k.endswith(".mask") for k in weights), (
        "Mask buffers must be excluded"
    )
    assert "token_embedding.weight" in weights
    assert "lm_head.weight" in weights
    assert torch.equal(weights["token_embedding.weight"], weights["lm_head.weight"])


def test_export_quantized_weights_naming_and_quant_config_metadata(
    tmp_path: Path,
) -> None:
    model, tokenizer = _make_tiny_model()

    export_model(model, tokenizer, tmp_path, export_dtype="float32")
    export_quantized_weights(model, tmp_path, "int8")
    export_quantized_weights(model, tmp_path, "int4")

    quant_cfg_path = tmp_path / "quant_config.json"
    quant_cfg = json.loads(quant_cfg_path.read_text(encoding="utf-8"))

    schemes = quant_cfg.get("schemes")
    assert isinstance(schemes, dict)
    assert set(schemes) >= {"int8", "int4"}

    int8 = schemes["int8"]
    int4 = schemes["int4"]

    assert int8["weights_file"] == "model_int8.safetensors"
    assert int8["precision"] == "int8"
    assert int8["format_version"] == 1
    assert isinstance(int8["quantized_layers"], dict)
    assert int8["quantized_layers"], (
        "Expected int8 quantized_layers metadata to be non-empty"
    )

    assert int4["weights_file"] == "model_int4.safetensors"
    assert int4["precision"] == "int4"
    assert int4["format_version"] == 1
    assert isinstance(int4["quantized_layers"], dict)
    assert int4["quantized_layers"], (
        "Expected int4 quantized_layers metadata to be non-empty"
    )

    # Spot-check metadata shapes/fields for one layer entry.
    one_int8 = next(iter(int8["quantized_layers"].values()))
    assert one_int8["scheme"] == "symmetric_per_row"
    assert one_int8["bits"] == 8
    assert isinstance(one_int8["orig_shape"], list) and len(one_int8["orig_shape"]) == 2

    one_int4 = next(iter(int4["quantized_layers"].values()))
    assert one_int4["scheme"] == "symmetric_per_row"
    assert one_int4["bits"] == 4
    assert one_int4["packed"] is True
    assert one_int4["pack_format"] == "u4u4_to_u8"
    assert "orig_in_features" in one_int4

    w8 = load_file(tmp_path / "model_int8.safetensors")
    int8_int_data = [k for k in w8 if k.endswith(".int_data")]
    assert int8_int_data, "Expected .int_data tensors in INT8 export"
    for k in int8_int_data[:10]:  # bound runtime; we only need representative coverage
        base = k[: -len(".int_data")]
        assert base not in w8
        assert f"{base}.scale" in w8
        assert w8[k].dtype == torch.int8
        assert w8[f"{base}.scale"].dtype == torch.float32
    assert not any(k.endswith(".mask") for k in w8)

    w4 = load_file(tmp_path / "model_int4.safetensors")
    int4_int_data = [k for k in w4 if k.endswith(".int_data")]
    assert int4_int_data, "Expected .int_data tensors in INT4 export"
    for k in int4_int_data[:10]:
        base = k[: -len(".int_data")]
        assert base not in w4
        assert f"{base}.scale" in w4
        assert w4[k].dtype == torch.uint8
        assert w4[f"{base}.scale"].dtype == torch.float32
    assert not any(k.endswith(".mask") for k in w4)


def test_validate_export_enforces_tensor_shapes_and_layer_count(tmp_path: Path) -> None:
    model, tokenizer = _make_tiny_model()
    export_model(model, tokenizer, tmp_path, export_dtype="float32")

    assert validate_export(tmp_path) is True

    cfg_path = tmp_path / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # Wrong n_layer should be detected (missing block tensors).
    cfg_bad_layers = dict(cfg)
    cfg_bad_layers["n_layer"] = cfg["n_layer"] + 1
    cfg_path.write_text(json.dumps(cfg_bad_layers), encoding="utf-8")
    with pytest.raises(ValueError, match=r"blocks\.2"):
        validate_export(tmp_path)

    # Restore and then break n_embd shape check.
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    cfg_bad_embd = dict(cfg)
    cfg_bad_embd["n_embd"] = cfg["n_embd"] + 2
    cfg_path.write_text(json.dumps(cfg_bad_embd), encoding="utf-8")
    with pytest.raises(ValueError, match=r"token_embedding\.weight"):
        validate_export(tmp_path)
