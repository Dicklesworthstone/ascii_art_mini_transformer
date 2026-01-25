"""
Unit tests for weight-only quantization helpers.

These tests exercise the pure-torch fallback quantization used for exporting
INT8/INT4 weights to safetensors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add parent to path for imports before importing project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

torch = pytest.importorskip("torch")

from safetensors.torch import save_file  # noqa: E402

from model.transformer import AsciiGPTConfig, create_model  # noqa: E402
from train.export import (  # noqa: E402
    _should_quantize_weight,
    quantize_state_dict_weights,
)


def _unpack_int4_packed(packed: torch.Tensor, orig_in_features: int) -> torch.Tensor:
    """Unpack u4u4_to_u8 format back to signed int8 in [-8, 7]."""
    low = packed & 0x0F
    high = packed >> 4
    q0 = low.to(torch.int16) - 8
    q1 = high.to(torch.int16) - 8
    q = torch.stack([q0, q1], dim=2).reshape(packed.shape[0], -1)
    q = q[:, :orig_in_features]
    return q.to(torch.int8)


def test_quantize_state_dict_int8_produces_int_data_and_scale(tmp_path: Path) -> None:
    torch.manual_seed(0)
    config = AsciiGPTConfig(n_layer=2, n_head=2, n_embd=128, block_size=64, dropout=0.0)
    model = create_model(config)
    state_dict = model.state_dict()

    q_state, meta = quantize_state_dict_weights(state_dict, "int8")

    quantized = [k for k, t in state_dict.items() if _should_quantize_weight(k, t)]
    assert quantized, "Expected at least one weight to be quantized"

    for name in quantized:
        assert name not in q_state
        assert f"{name}.int_data" in q_state
        assert f"{name}.scale" in q_state
        assert q_state[f"{name}.int_data"].dtype == torch.int8
        assert q_state[f"{name}.scale"].dtype == torch.float32

    assert meta["precision"] == "int8"

    # File size should shrink vs float32 export.
    float_path = tmp_path / "float.safetensors"
    int8_path = tmp_path / "int8.safetensors"
    float_export = {
        k: v.detach().to(torch.float32)
        for k, v in state_dict.items()
        if not k.endswith(".mask")
    }
    if "token_embedding.weight" in float_export and "lm_head.weight" in float_export:
        float_export["lm_head.weight"] = float_export["lm_head.weight"].clone()
    save_file(float_export, float_path)
    save_file(q_state, int8_path)
    assert int8_path.stat().st_size < float_path.stat().st_size


def test_quantize_state_dict_int4_packing_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    config = AsciiGPTConfig(n_layer=2, n_head=2, n_embd=128, block_size=64, dropout=0.0)
    model = create_model(config)
    state_dict = model.state_dict()

    q_state, meta = quantize_state_dict_weights(state_dict, "int4")

    quantized = [k for k, t in state_dict.items() if _should_quantize_weight(k, t)]
    assert quantized, "Expected at least one weight to be quantized"

    for name in quantized:
        int_data_key = f"{name}.int_data"
        scale_key = f"{name}.scale"
        assert int_data_key in q_state
        assert scale_key in q_state
        packed = q_state[int_data_key]
        scale = q_state[scale_key]

        layer_meta = meta["quantized_layers"][name]
        assert layer_meta["bits"] == 4
        assert layer_meta["packed"] is True
        orig_in = layer_meta["orig_in_features"]

        assert packed.dtype == torch.uint8
        assert scale.dtype == torch.float32

        # Dequantize and ensure finite outputs.
        q = _unpack_int4_packed(packed, orig_in).to(torch.float32)
        w_approx = q * scale.unsqueeze(1)
        assert torch.isfinite(w_approx).all()

    # File size should shrink vs float32 export.
    float_path = tmp_path / "float.safetensors"
    int4_path = tmp_path / "int4.safetensors"
    float_export = {
        k: v.detach().to(torch.float32)
        for k, v in state_dict.items()
        if not k.endswith(".mask")
    }
    if "token_embedding.weight" in float_export and "lm_head.weight" in float_export:
        float_export["lm_head.weight"] = float_export["lm_head.weight"].clone()
    save_file(float_export, float_path)
    save_file(q_state, int4_path)
    assert int4_path.stat().st_size < float_path.stat().st_size
