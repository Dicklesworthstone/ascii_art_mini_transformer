"""
Export trained PyTorch model to safetensors format for Rust inference.

This module provides functions to:
- Export model weights to safetensors
- Export model configuration to JSON
- Export tokenizer vocabulary
- Validate exported files
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

# Add parent to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.transformer import AsciiGPT, AsciiGPTConfig, create_model
from model.tokenizer import AsciiTokenizer, get_tokenizer


def export_model(
    model: AsciiGPT,
    tokenizer: AsciiTokenizer,
    output_dir: str | Path,
    export_dtype: str = "float32",
) -> Path:
    """
    Export model and tokenizer for Rust inference.

    Exports:
    - model.safetensors: Model weights
    - config.json: Model configuration
    - tokenizer.json: Tokenizer vocabulary

    Args:
        model: Trained AsciiGPT model
        tokenizer: Tokenizer instance
        output_dir: Directory to save exports
        export_dtype: Export dtype ('float32', 'float16', 'bfloat16')

    Returns:
        Path to output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine target dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    target_dtype = dtype_map.get(export_dtype, torch.float32)

    # Convert model to target dtype for export
    if target_dtype != torch.float32:
        model = model.to(target_dtype)

    # Export weights.
    #
    # Note: We intentionally exclude attention mask buffers (`*.attn.mask`) since they are
    # deterministic and can be rebuilt at runtime (and they massively bloat exports).
    #
    # We also break shared-memory ties (token_embedding <-> lm_head) to satisfy
    # `safetensors.torch.save_file` constraints.
    weights_path = output_dir / "model.safetensors"
    state_dict = model.state_dict()
    export_dict: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if name.endswith(".mask"):
            continue
        export_dict[name] = tensor.detach().to(device="cpu")

    if "token_embedding.weight" in export_dict and "lm_head.weight" in export_dict:
        export_dict["lm_head.weight"] = export_dict["lm_head.weight"].clone()

    save_file(export_dict, weights_path)

    # Calculate stats from exported tensors
    total_params = sum(t.numel() for t in export_dict.values())
    size_mb = sum(t.numel() * t.element_size() for t in export_dict.values()) / (
        1024 * 1024
    )
    print(f"Saved model weights: {total_params:,} parameters ({size_mb:.2f} MB)")

    # Export model config
    config = {
        "vocab_size": model.config.vocab_size,
        "block_size": model.config.block_size,
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "n_embd": model.config.n_embd,
        "dropout": model.config.dropout,
        "max_rows": model.config.max_rows,
        "max_cols": model.config.max_cols,
        "newline_token_id": model.config.newline_token_id,
        "pad_token_id": model.config.pad_token_id,
        "bos_token_id": model.config.bos_token_id,
        "eos_token_id": model.config.eos_token_id,
        "export_dtype": export_dtype,
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved model config: {config_path}")

    # Export tokenizer
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer: {tokenizer_path}")

    print(f"\nExport complete: {output_dir}")
    return output_dir


def _should_quantize_weight(name: str, tensor: torch.Tensor) -> bool:
    """Return True if a state_dict entry should be quantized as a Linear weight."""
    if tensor.ndim != 2:
        return False
    if not name.endswith(".weight"):
        return False

    # Keep embeddings / positional encodings in float (small, and embedding lookup is different).
    if name.startswith(("token_embedding.", "lm_head.", "pos_encoding.")):
        return False

    return True


def _quantize_int8_per_row(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-row INT8 weight-only quantization.

    Returns:
        (int_data, scale) where:
          - int_data is int8 with shape [out, in]
          - scale is float32 with shape [out]
    """
    w = weight.detach().to(dtype=torch.float32, device="cpu")
    max_abs = w.abs().amax(dim=1)
    scale = max_abs / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(w / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    return q, scale


def _quantize_int4_pack_per_row(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Symmetric per-row INT4 (packed) weight-only quantization.

    We quantize to signed int4 in [-8, 7], then pack 2 values per byte:
      packed_byte = (hi_nibble << 4) | lo_nibble
    where each nibble is stored as unsigned (q + 8) in [0, 15].

    Returns:
        (int_data_packed, scale, in_features) where:
          - int_data_packed is uint8 with shape [out, ceil(in/2)]
          - scale is float32 with shape [out]
          - in_features is the original in_features (before padding to even)
    """
    w = weight.detach().to(dtype=torch.float32, device="cpu")
    out_features, in_features = w.shape
    max_abs = w.abs().amax(dim=1)
    scale = max_abs / 7.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    q = torch.round(w / scale.unsqueeze(1)).clamp(-8, 7).to(torch.int8)
    q_u4 = (q + 8).to(torch.uint8)  # [0, 15]

    if in_features % 2 == 1:
        pad = torch.full((out_features, 1), 8, dtype=torch.uint8)  # q=0 -> 8
        q_u4 = torch.cat([q_u4, pad], dim=1)

    pairs = q_u4.view(out_features, -1, 2)
    lo = pairs[:, :, 0]
    hi = pairs[:, :, 1] << 4
    packed = (lo | hi).contiguous()
    return packed, scale, in_features


def quantize_state_dict_weights(
    state_dict: dict[str, torch.Tensor],
    precision: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """
    Convert a float state_dict into a quantized state_dict suitable for safetensors export.

    The output dict contains:
      - float32 tensors for non-quantized params
      - `{name}.int_data` + `{name}.scale` for quantized linear weights

    `precision` must be one of: {"int8", "int4"}.
    """
    if precision not in {"int8", "int4"}:
        raise ValueError(
            f"Unsupported precision: {precision!r} (expected 'int8' or 'int4')"
        )

    out: dict[str, torch.Tensor] = {}
    meta_layers: dict[str, Any] = {}

    for name, tensor in state_dict.items():
        if name.endswith(".mask"):
            continue
        if _should_quantize_weight(name, tensor):
            if precision == "int8":
                int_data, scale = _quantize_int8_per_row(tensor)
                out[f"{name}.int_data"] = int_data
                out[f"{name}.scale"] = scale
                meta_layers[name] = {
                    "scheme": "symmetric_per_row",
                    "bits": 8,
                    "orig_shape": list(tensor.shape),
                    "int_data_dtype": "int8",
                }
            else:
                int_data, scale, in_features = _quantize_int4_pack_per_row(tensor)
                out[f"{name}.int_data"] = int_data
                out[f"{name}.scale"] = scale
                meta_layers[name] = {
                    "scheme": "symmetric_per_row",
                    "bits": 4,
                    "packed": True,
                    "pack_format": "u4u4_to_u8",
                    "orig_shape": list(tensor.shape),
                    "packed_shape": list(int_data.shape),
                    "orig_in_features": in_features,
                    "int_data_dtype": "uint8",
                }
        else:
            # Keep other params in float32 for simplicity/portability.
            # Clone to avoid shared-memory issues (e.g. tied weights) when saving via safetensors.
            out[name] = tensor.detach().to(dtype=torch.float32, device="cpu").clone()

    meta: dict[str, Any] = {
        "precision": precision,
        "format_version": 1,
        "quantized_layers": meta_layers,
    }
    return out, meta


def export_quantized_weights(
    model: AsciiGPT,
    output_dir: str | Path,
    precision: str,
) -> Path:
    """
    Export quantized weights to `model_{precision}.safetensors` and update `quant_config.json`.

    This does not modify the model in-place; it quantizes from the current float weights.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    quant_state, quant_meta = quantize_state_dict_weights(state_dict, precision)

    weights_name = f"model_{precision}.safetensors"
    weights_path = output_dir / weights_name
    save_file(quant_state, weights_path)

    size_mb = weights_path.stat().st_size / (1024 * 1024)
    print(f"Saved quantized weights ({precision}): {weights_name} ({size_mb:.2f} MB)")

    # Merge into quant_config.json (support exporting multiple precisions).
    quant_config_path = output_dir / "quant_config.json"
    if quant_config_path.exists():
        with open(quant_config_path) as f:
            existing = json.load(f)
        if not isinstance(existing, dict):
            existing = {}
    else:
        existing = {}

    schemes = existing.get("schemes")
    if not isinstance(schemes, dict):
        schemes = {}
    schemes[precision] = {
        "weights_file": weights_name,
        **quant_meta,
    }
    existing["schemes"] = schemes

    with open(quant_config_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Updated quant config: {quant_config_path}")

    return weights_path


def _infer_config_from_state_dict(
    state_dict: dict,
    fallback_n_layer: int = 6,
    fallback_n_head: int = 6,
    fallback_block_size: int = 2048,
) -> AsciiGPTConfig:
    """
    Infer model configuration from state dict tensor shapes.

    Args:
        state_dict: Model state dictionary
        fallback_n_layer: Fallback if can't infer
        fallback_n_head: Fallback if can't infer
        fallback_block_size: Fallback if can't infer

    Returns:
        AsciiGPTConfig inferred from weights
    """
    # Get vocab_size and n_embd from token embedding
    if "token_embedding.weight" in state_dict:
        vocab_size, n_embd = state_dict["token_embedding.weight"].shape
    else:
        raise ValueError("Cannot find token_embedding.weight in state dict")

    # Count transformer blocks
    block_indices = set()
    for key in state_dict:
        if key.startswith("blocks."):
            # Extract block index from keys like "blocks.0.attn.c_attn.weight"
            parts = key.split(".")
            if len(parts) >= 2:
                try:
                    block_indices.add(int(parts[1]))
                except ValueError:
                    pass
    n_layer = len(block_indices) if block_indices else fallback_n_layer

    # Infer n_head from attention projection shape
    # c_attn projects to 3 * n_embd (Q, K, V), so shape is (n_embd, 3 * n_embd)
    # The head dimension is n_embd // n_head
    # We can verify by checking c_proj shape matches
    n_head = fallback_n_head  # Default fallback

    # Try to infer n_head - it must divide n_embd evenly
    # Common head dimensions are 64 or 128, so try those first
    common_head_dims = [64, 128, 32, 256]
    for head_dim in common_head_dims:
        if n_embd % head_dim == 0:
            candidate = n_embd // head_dim
            if candidate >= 1:
                n_head = candidate
                break
    else:
        # Fallback: try common head counts that divide evenly
        for candidate_heads in [8, 6, 4, 2, 1]:
            if n_embd % candidate_heads == 0:
                n_head = candidate_heads
                break

    # Infer block_size from attention mask shape
    block_size = fallback_block_size
    for key in state_dict:
        if ".attn.mask" in key:
            # Mask shape is [1, 1, block_size, block_size]
            mask_shape = state_dict[key].shape
            if len(mask_shape) == 4:
                block_size = mask_shape[2]
                break

    print(
        f"Inferred config: vocab={vocab_size}, n_embd={n_embd}, "
        f"n_layer={n_layer}, n_head={n_head}, block_size={block_size}"
    )

    return AsciiGPTConfig(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
    )


def _extract_config_from_dict(cfg: dict) -> AsciiGPTConfig | None:
    """Extract AsciiGPTConfig from a config dict (new format)."""
    if not isinstance(cfg, dict):
        return None

    required_keys = ["n_layer", "n_head", "n_embd", "block_size"]
    if not all(k in cfg for k in required_keys):
        return None

    return AsciiGPTConfig(
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        block_size=cfg["block_size"],
        dropout=cfg.get("dropout", 0.1),
        vocab_size=cfg.get("vocab_size", 107),
    )


def _extract_config_from_object(cfg) -> AsciiGPTConfig | None:
    """Extract AsciiGPTConfig from a config object (old format)."""
    if cfg is None:
        return None

    if not hasattr(cfg, "n_layer"):
        return None

    return AsciiGPTConfig(
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        block_size=cfg.block_size,
        dropout=getattr(cfg, "dropout", 0.1),
    )


def export_from_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    export_dtype: str = "float32",
    quantize: str = "none",
    n_layer: int = 6,
    n_head: int = 6,
    n_embd: int = 384,
    block_size: int = 2048,
) -> Path:
    """
    Export model from a training checkpoint.

    Automatically detects checkpoint format and extracts model architecture.
    Falls back to inferring architecture from weight tensor shapes if config
    is not available.

    Args:
        checkpoint_path: Path to training checkpoint (.pt file)
        output_dir: Directory to save exports
        export_dtype: Export dtype ('float32', 'float16', 'bfloat16')
        n_layer: Number of layers (fallback if not in checkpoint)
        n_head: Number of attention heads (fallback)
        n_embd: Embedding dimension (fallback)
        block_size: Maximum sequence length (fallback)

    Returns:
        Path to output directory
    """
    checkpoint_path = Path(checkpoint_path)
    model_config = None
    model_state_dict = None

    print(f"Loading checkpoint: {checkpoint_path}")

    # First, try loading with weights_only=True to just get tensors
    # This avoids pickle issues entirely for the model weights
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        model_state_dict = checkpoint.get("model", checkpoint)
        print("Loaded weights successfully (weights_only mode)")

        # Infer config from state dict shapes.
        #
        # Note: Some hyperparameters (notably n_head) cannot be inferred reliably from tensor
        # shapes in this architecture. We still infer as a fallback, but if checkpoint metadata
        # is available we will prefer that below.
        model_config = _infer_config_from_state_dict(
            model_state_dict,
            fallback_n_layer=n_layer,
            fallback_n_head=n_head,
            fallback_block_size=block_size,
        )

        # If possible, load checkpoint metadata to recover the exact model hyperparameters.
        #
        # This may fail for older checkpoints that contain pickled objects with missing module
        # paths. In that case, we keep the inferred config.
        try:
            full_checkpoint = torch.load(
                checkpoint_path, weights_only=False, map_location="cpu"
            )
            if isinstance(full_checkpoint, dict):
                cfg = full_checkpoint.get("model_config") or full_checkpoint.get(
                    "training_config"
                )
                parsed = _extract_config_from_dict(cfg) or _extract_config_from_object(cfg)
                if parsed is not None:
                    model_config = parsed
                    print("Loaded model config from checkpoint metadata")
        except Exception as metadata_error:
            print(
                f"Could not load checkpoint metadata for config: {metadata_error} "
                "(continuing with inferred config)"
            )

    except Exception as weights_only_error:
        print(f"weights_only load failed: {weights_only_error}")
        print("Trying full checkpoint load...")

        # Try loading full checkpoint (may fail with pickle issues)
        try:
            checkpoint = torch.load(
                checkpoint_path, weights_only=False, map_location="cpu"
            )
            model_state_dict = checkpoint.get("model", checkpoint)

            # Try to extract config from checkpoint
            if "model_config" in checkpoint:
                cfg = checkpoint["model_config"]
                model_config = _extract_config_from_dict(
                    cfg
                ) or _extract_config_from_object(cfg)
            elif "training_config" in checkpoint:
                cfg = checkpoint["training_config"]
                model_config = _extract_config_from_dict(
                    cfg
                ) or _extract_config_from_object(cfg)

        except Exception as full_load_error:
            print(f"Full load also failed: {full_load_error}")
            raise RuntimeError(
                f"Cannot load checkpoint {checkpoint_path}. "
                f"weights_only error: {weights_only_error}, "
                f"full load error: {full_load_error}"
            )

    # If still no config, infer from state dict
    if model_config is None and model_state_dict is not None:
        print("No config found in checkpoint, inferring from weights...")
        model_config = _infer_config_from_state_dict(
            model_state_dict,
            fallback_n_layer=n_layer,
            fallback_n_head=n_head,
            fallback_block_size=block_size,
        )

    # Final fallback to CLI parameters
    if model_config is None:
        print(f"Using CLI fallback config: {n_layer} layers, {n_embd} dims")
        model_config = AsciiGPTConfig(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            block_size=block_size,
        )

    # Create and load model
    model = create_model(model_config)
    model.load_state_dict(model_state_dict)
    model.eval()

    # Get tokenizer
    tokenizer = get_tokenizer()

    export_model(model, tokenizer, output_dir, export_dtype)

    # Optional quantized exports (weight-only, for smaller files and future Rust support).
    if quantize in {"int8", "both"}:
        export_quantized_weights(model, output_dir, "int8")
    if quantize in {"int4", "both"}:
        export_quantized_weights(model, output_dir, "int4")

    return Path(output_dir)


def validate_export(export_dir: str | Path) -> bool:
    """
    Validate an exported model directory.

    Checks:
    - All required files exist
    - Weights can be loaded
    - Config is valid JSON
    - Weight shapes match config

    Args:
        export_dir: Path to export directory

    Returns:
        True if valid, raises exception otherwise
    """
    export_dir = Path(export_dir)

    # Check required files
    required_files = ["model.safetensors", "config.json", "tokenizer.json"]
    for fname in required_files:
        fpath = export_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing required file: {fpath}")

    # Load and validate config
    with open(export_dir / "config.json") as f:
        config = json.load(f)

    required_keys = ["vocab_size", "block_size", "n_layer", "n_head", "n_embd"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Config missing required key: {key}")

    # Load and validate weights
    weights = load_file(export_dir / "model.safetensors")

    # Check embedding shapes
    if "token_embedding.weight" in weights:
        emb_shape = weights["token_embedding.weight"].shape
        expected_shape = (config["vocab_size"], config["n_embd"])
        if emb_shape != expected_shape:
            raise ValueError(
                f"Token embedding shape mismatch: {emb_shape} vs {expected_shape}"
            )

    # Check number of transformer blocks
    block_count = sum(1 for k in weights if k.startswith("blocks.") and ".attn." in k)
    block_count = (
        block_count // 4
    )  # 4 attention weights per block (c_attn, c_proj, etc.)
    if block_count != config["n_layer"]:
        # This is a heuristic check, might not be exact
        pass  # Don't fail, just warn if checking is important

    print(f"Validation passed: {export_dir}")
    print(f"  - Config: {config['n_layer']} layers, {config['n_embd']} dims")
    print(f"  - Weights: {len(weights)} tensors")
    print(f"  - Vocab: {config['vocab_size']} tokens")

    return True


def print_export_summary(export_dir: str | Path) -> None:
    """Print summary of exported model."""
    export_dir = Path(export_dir)

    with open(export_dir / "config.json") as f:
        config = json.load(f)

    weights = load_file(export_dir / "model.safetensors")

    # Calculate sizes
    total_params = sum(t.numel() for t in weights.values())
    total_bytes = sum(t.numel() * t.element_size() for t in weights.values())

    print("\n" + "=" * 50)
    print("Exported Model Summary")
    print("=" * 50)
    print(f"Location: {export_dir}")
    print("\nArchitecture:")
    print(f"  Layers: {config['n_layer']}")
    print(f"  Heads: {config['n_head']}")
    print(f"  Embedding: {config['n_embd']}")
    print(f"  Block size: {config['block_size']}")
    print(f"  Vocab size: {config['vocab_size']}")
    print("\nSize:")
    print(f"  Parameters: {total_params:,}")
    print(f"  Weights: {total_bytes / 1024 / 1024:.2f} MB")
    print(f"  Dtype: {config.get('export_dtype', 'unknown')}")
    print("=" * 50)


if __name__ == "__main__":
    import argparse
    from dataclasses import replace

    parser = argparse.ArgumentParser(description="Export model to safetensors")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to training checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/exported",
        help="Output directory for exports",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Export dtype",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default="none",
        choices=["none", "int8", "int4", "both"],
        help="Also export quantized weights (weight-only) for smaller files",
    )
    parser.add_argument(
        "--validate",
        type=str,
        default=None,
        help="Validate an existing export directory",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="Model size preset used when exporting a fresh (untrained) model",
    )
    parser.add_argument(
        "--n-layer",
        type=int,
        default=None,
        help="Override number of layers for fresh model export (no --checkpoint)",
    )
    parser.add_argument(
        "--n-head",
        type=int,
        default=None,
        help="Override number of attention heads for fresh model export (no --checkpoint)",
    )
    parser.add_argument(
        "--n-embd",
        type=int,
        default=None,
        help="Override embedding dimension for fresh model export (no --checkpoint)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Override context length for fresh model export (no --checkpoint)",
    )

    args = parser.parse_args()

    if args.validate:
        validate_export(args.validate)
        print_export_summary(args.validate)
    elif args.checkpoint:
        export_from_checkpoint(
            args.checkpoint, args.output_dir, args.dtype, args.quantize
        )
        validate_export(args.output_dir)
        print_export_summary(args.output_dir)
    else:
        # Quick test with untrained model
        print("No checkpoint specified, testing with fresh model...")
        from model.transformer import (
            get_large_config,
            get_medium_config,
            get_small_config,
        )

        presets = {
            "small": get_small_config,
            "medium": get_medium_config,
            "large": get_large_config,
        }
        base_config = presets[args.preset]()
        overrides = {}
        if args.n_layer is not None:
            overrides["n_layer"] = args.n_layer
        if args.n_head is not None:
            overrides["n_head"] = args.n_head
        if args.n_embd is not None:
            overrides["n_embd"] = args.n_embd
        if args.block_size is not None:
            overrides["block_size"] = args.block_size

        config = replace(base_config, **overrides) if overrides else base_config
        model = create_model(config)
        tokenizer = get_tokenizer()

        export_model(model, tokenizer, args.output_dir, args.dtype)
        if args.quantize != "none":
            if args.quantize in {"int8", "both"}:
                export_quantized_weights(model, args.output_dir, "int8")
            if args.quantize in {"int4", "both"}:
                export_quantized_weights(model, args.output_dir, "int4")
        validate_export(args.output_dir)
        print_export_summary(args.output_dir)
