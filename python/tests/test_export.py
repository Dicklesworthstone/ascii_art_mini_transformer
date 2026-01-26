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


# Additional tests for config inference and export_from_checkpoint
from train.export import (  # noqa: E402
    _infer_config_from_state_dict,
    _extract_config_from_dict,
    _extract_config_from_object,
    export_from_checkpoint,
    print_export_summary,
    _should_quantize_weight,
    quantize_state_dict_weights,
)


class TestInferConfigFromStateDict:
    """Tests for _infer_config_from_state_dict function."""

    def test_infers_vocab_and_n_embd_from_token_embedding(self) -> None:
        """Should infer vocab_size and n_embd from token embedding weight shape."""
        state_dict = {
            "token_embedding.weight": torch.randn(107, 256),
            "blocks.0.ln_1.weight": torch.randn(256),
            "blocks.1.ln_1.weight": torch.randn(256),
        }
        config = _infer_config_from_state_dict(state_dict)
        assert config.vocab_size == 107
        assert config.n_embd == 256

    def test_infers_n_layer_from_block_indices(self) -> None:
        """Should count transformer blocks from state dict keys."""
        state_dict = {
            "token_embedding.weight": torch.randn(107, 128),
            "blocks.0.attn.c_attn.weight": torch.randn(384, 128),
            "blocks.1.attn.c_attn.weight": torch.randn(384, 128),
            "blocks.2.attn.c_attn.weight": torch.randn(384, 128),
        }
        config = _infer_config_from_state_dict(state_dict)
        assert config.n_layer == 3

    def test_infers_block_size_from_attention_mask(self) -> None:
        """Should infer block_size from attention mask shape."""
        state_dict = {
            "token_embedding.weight": torch.randn(107, 128),
            "blocks.0.attn.mask": torch.ones(1, 1, 512, 512),
        }
        config = _infer_config_from_state_dict(state_dict)
        assert config.block_size == 512

    def test_uses_fallback_when_no_blocks_found(self) -> None:
        """Should use fallback n_layer when no blocks in state dict."""
        state_dict = {
            "token_embedding.weight": torch.randn(107, 64),
        }
        config = _infer_config_from_state_dict(state_dict, fallback_n_layer=4)
        assert config.n_layer == 4

    def test_uses_fallback_block_size_when_no_mask(self) -> None:
        """Should use fallback block_size when no attention mask found."""
        state_dict = {
            "token_embedding.weight": torch.randn(107, 64),
        }
        config = _infer_config_from_state_dict(state_dict, fallback_block_size=1024)
        assert config.block_size == 1024

    def test_raises_on_missing_token_embedding(self) -> None:
        """Should raise ValueError when token_embedding.weight is missing."""
        state_dict = {
            "blocks.0.ln_1.weight": torch.randn(256),
        }
        with pytest.raises(ValueError, match="token_embedding.weight"):
            _infer_config_from_state_dict(state_dict)

    def test_infers_n_head_from_common_head_dims(self) -> None:
        """Should infer n_head from common head dimensions (64, 128, etc.)."""
        # With n_embd=256 and head_dim=64, n_head should be 4
        state_dict = {
            "token_embedding.weight": torch.randn(107, 256),
        }
        config = _infer_config_from_state_dict(state_dict)
        assert config.n_embd == 256
        assert config.n_head == 4  # 256 / 64 = 4


class TestExtractConfigFromDict:
    """Tests for _extract_config_from_dict function."""

    def test_extracts_config_from_valid_dict(self) -> None:
        """Should extract config from dict with required keys."""
        cfg = {
            "n_layer": 4,
            "n_head": 4,
            "n_embd": 256,
            "block_size": 1024,
            "dropout": 0.2,
            "vocab_size": 100,
        }
        result = _extract_config_from_dict(cfg)
        assert result is not None
        assert result.n_layer == 4
        assert result.n_head == 4
        assert result.n_embd == 256
        assert result.block_size == 1024
        assert result.dropout == 0.2
        assert result.vocab_size == 100

    def test_uses_defaults_for_optional_keys(self) -> None:
        """Should use defaults for missing optional keys."""
        cfg = {
            "n_layer": 2,
            "n_head": 2,
            "n_embd": 64,
            "block_size": 256,
        }
        result = _extract_config_from_dict(cfg)
        assert result is not None
        assert result.dropout == 0.1  # default
        assert result.vocab_size == 107  # default

    def test_returns_none_for_non_dict(self) -> None:
        """Should return None for non-dict input."""
        assert _extract_config_from_dict(None) is None
        assert _extract_config_from_dict("string") is None
        assert _extract_config_from_dict([1, 2, 3]) is None

    def test_returns_none_for_missing_required_keys(self) -> None:
        """Should return None when required keys are missing."""
        cfg = {"n_layer": 4, "n_head": 4}  # missing n_embd and block_size
        assert _extract_config_from_dict(cfg) is None


class TestExtractConfigFromObject:
    """Tests for _extract_config_from_object function."""

    def test_extracts_config_from_object_with_attributes(self) -> None:
        """Should extract config from object with required attributes."""

        class MockConfig:
            n_layer = 6
            n_head = 6
            n_embd = 384
            block_size = 2048
            dropout = 0.1

        result = _extract_config_from_object(MockConfig())
        assert result is not None
        assert result.n_layer == 6
        assert result.n_head == 6
        assert result.n_embd == 384
        assert result.block_size == 2048

    def test_returns_none_for_none_input(self) -> None:
        """Should return None for None input."""
        assert _extract_config_from_object(None) is None

    def test_returns_none_for_object_without_n_layer(self) -> None:
        """Should return None for object without n_layer attribute."""

        class IncompleteConfig:
            n_head = 4
            n_embd = 256

        assert _extract_config_from_object(IncompleteConfig()) is None

    def test_uses_default_dropout_when_missing(self) -> None:
        """Should use default dropout when attribute is missing."""

        class ConfigNoDropout:
            n_layer = 2
            n_head = 2
            n_embd = 64
            block_size = 256

        result = _extract_config_from_object(ConfigNoDropout())
        assert result is not None
        assert result.dropout == 0.1  # default


class TestExportFromCheckpoint:
    """Tests for export_from_checkpoint function."""

    def test_exports_from_valid_checkpoint(self, tmp_path: Path) -> None:
        """Should export from a valid checkpoint file."""
        # Create a checkpoint
        model, tokenizer = _make_tiny_model()
        checkpoint_path = tmp_path / "test.pt"
        checkpoint = {
            "model": model.state_dict(),
            "model_config": {
                "n_layer": 2,
                "n_head": 2,
                "n_embd": 64,
                "block_size": 64,
            },
        }
        torch.save(checkpoint, checkpoint_path)

        output_dir = tmp_path / "export"
        result = export_from_checkpoint(checkpoint_path, output_dir)

        assert result == output_dir
        assert (output_dir / "model.safetensors").exists()
        assert (output_dir / "config.json").exists()
        assert (output_dir / "tokenizer.json").exists()

    def test_exports_with_quantization(self, tmp_path: Path) -> None:
        """Should export with quantization when requested."""
        model, tokenizer = _make_tiny_model()
        checkpoint_path = tmp_path / "test.pt"
        checkpoint = {
            "model": model.state_dict(),
            "model_config": {
                "n_layer": 2,
                "n_head": 2,
                "n_embd": 64,
                "block_size": 64,
            },
        }
        torch.save(checkpoint, checkpoint_path)

        output_dir = tmp_path / "export"
        export_from_checkpoint(checkpoint_path, output_dir, quantize="both")

        assert (output_dir / "model_int8.safetensors").exists()
        assert (output_dir / "model_int4.safetensors").exists()
        assert (output_dir / "quant_config.json").exists()

    def test_infers_config_when_not_in_checkpoint(self, tmp_path: Path) -> None:
        """Should infer config from weights when not in checkpoint."""
        model, _ = _make_tiny_model()
        checkpoint_path = tmp_path / "test.pt"
        # Save just the model state dict, no config
        torch.save({"model": model.state_dict()}, checkpoint_path)

        output_dir = tmp_path / "export"
        export_from_checkpoint(
            checkpoint_path,
            output_dir,
            n_layer=2,
            n_head=2,
            n_embd=64,
            block_size=64,
        )

        assert (output_dir / "model.safetensors").exists()
        config = json.loads((output_dir / "config.json").read_text())
        assert config["n_layer"] == 2

    def test_handles_training_config_key(self, tmp_path: Path) -> None:
        """Should extract config from training_config key."""
        model, _ = _make_tiny_model()
        checkpoint_path = tmp_path / "test.pt"
        checkpoint = {
            "model": model.state_dict(),
            "training_config": {
                "n_layer": 2,
                "n_head": 2,
                "n_embd": 64,
                "block_size": 64,
            },
        }
        torch.save(checkpoint, checkpoint_path)

        output_dir = tmp_path / "export"
        export_from_checkpoint(checkpoint_path, output_dir)

        assert (output_dir / "config.json").exists()


class TestValidateExportEdgeCases:
    """Additional edge case tests for validate_export."""

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError for missing required files."""
        # Create partial export (missing tokenizer.json)
        model, tokenizer = _make_tiny_model()
        export_model(model, tokenizer, tmp_path, export_dtype="float32")
        (tmp_path / "tokenizer.json").unlink()

        with pytest.raises(FileNotFoundError, match="tokenizer.json"):
            validate_export(tmp_path)

    def test_raises_on_invalid_config_type(self, tmp_path: Path) -> None:
        """Should raise on non-integer config values."""
        model, tokenizer = _make_tiny_model()
        export_model(model, tokenizer, tmp_path, export_dtype="float32")

        cfg_path = tmp_path / "config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["n_layer"] = "six"  # string instead of int
        cfg_path.write_text(json.dumps(cfg))

        with pytest.raises(ValueError, match="must be an int"):
            validate_export(tmp_path)

    def test_raises_on_zero_config_value(self, tmp_path: Path) -> None:
        """Should raise on zero config values."""
        model, tokenizer = _make_tiny_model()
        export_model(model, tokenizer, tmp_path, export_dtype="float32")

        cfg_path = tmp_path / "config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["n_layer"] = 0
        cfg_path.write_text(json.dumps(cfg))

        with pytest.raises(ValueError, match="must be > 0"):
            validate_export(tmp_path)

    def test_raises_on_nembd_not_divisible_by_nhead(self, tmp_path: Path) -> None:
        """Should raise when n_embd is not divisible by n_head."""
        model, tokenizer = _make_tiny_model()
        export_model(model, tokenizer, tmp_path, export_dtype="float32")

        cfg_path = tmp_path / "config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["n_head"] = 3  # 64 / 3 is not an integer
        cfg_path.write_text(json.dumps(cfg))

        with pytest.raises(ValueError, match="divisible by n_head"):
            validate_export(tmp_path)


class TestPrintExportSummary:
    """Tests for print_export_summary function."""

    def test_prints_summary_without_error(self, tmp_path: Path, capsys) -> None:
        """Should print summary without errors."""
        model, tokenizer = _make_tiny_model()
        export_model(model, tokenizer, tmp_path, export_dtype="float32")

        print_export_summary(tmp_path)

        captured = capsys.readouterr()
        assert "Exported Model Summary" in captured.out
        assert "Layers:" in captured.out
        assert "Parameters:" in captured.out


class TestShouldQuantizeWeight:
    """Tests for _should_quantize_weight helper."""

    def test_returns_true_for_linear_weights(self) -> None:
        """Should return True for 2D weights ending in .weight."""
        tensor = torch.randn(256, 128)
        assert _should_quantize_weight("blocks.0.attn.c_attn.weight", tensor) is True

    def test_returns_false_for_non_2d_tensors(self) -> None:
        """Should return False for non-2D tensors."""
        tensor = torch.randn(256)
        assert _should_quantize_weight("ln_f.weight", tensor) is False

    def test_returns_false_for_non_weight_keys(self) -> None:
        """Should return False for keys not ending in .weight."""
        tensor = torch.randn(256, 128)
        assert _should_quantize_weight("blocks.0.ln_1.bias", tensor) is False

    def test_returns_false_for_embeddings(self) -> None:
        """Should return False for embedding layers."""
        tensor = torch.randn(107, 256)
        assert _should_quantize_weight("token_embedding.weight", tensor) is False
        assert _should_quantize_weight("lm_head.weight", tensor) is False
        assert _should_quantize_weight("pos_encoding.weight", tensor) is False


class TestQuantizeStateDictWeights:
    """Tests for quantize_state_dict_weights function."""

    def test_raises_on_invalid_precision(self) -> None:
        """Should raise ValueError for unsupported precision."""
        state_dict = {"blocks.0.attn.c_attn.weight": torch.randn(256, 128)}
        with pytest.raises(ValueError, match="Unsupported precision"):
            quantize_state_dict_weights(state_dict, "int16")

    def test_quantizes_linear_weights_int8(self) -> None:
        """Should quantize linear weights to int8."""
        state_dict = {
            "blocks.0.attn.c_attn.weight": torch.randn(256, 128),
            "ln_f.weight": torch.randn(256),  # not quantized
        }
        quant_state, meta = quantize_state_dict_weights(state_dict, "int8")

        assert "blocks.0.attn.c_attn.weight.int_data" in quant_state
        assert "blocks.0.attn.c_attn.weight.scale" in quant_state
        assert "ln_f.weight" in quant_state  # kept as float
        assert quant_state["blocks.0.attn.c_attn.weight.int_data"].dtype == torch.int8

    def test_quantizes_linear_weights_int4(self) -> None:
        """Should quantize linear weights to int4 (packed)."""
        state_dict = {
            "blocks.0.attn.c_attn.weight": torch.randn(256, 128),
        }
        quant_state, meta = quantize_state_dict_weights(state_dict, "int4")

        assert "blocks.0.attn.c_attn.weight.int_data" in quant_state
        assert quant_state["blocks.0.attn.c_attn.weight.int_data"].dtype == torch.uint8

    def test_skips_mask_buffers(self) -> None:
        """Should skip attention mask buffers."""
        state_dict = {
            "blocks.0.attn.mask": torch.ones(1, 1, 64, 64),
            "blocks.0.attn.c_attn.weight": torch.randn(256, 128),
        }
        quant_state, meta = quantize_state_dict_weights(state_dict, "int8")

        assert not any(k.endswith(".mask") for k in quant_state)
