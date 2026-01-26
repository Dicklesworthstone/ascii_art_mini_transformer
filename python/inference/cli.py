from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional


def _load_torch_checkpoint(
    path: Path, *, unsafe_load: bool
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("torch is required to load checkpoints") from exc

    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        if not unsafe_load:
            raise RuntimeError(
                "Failed to load checkpoint in safe weights_only mode. "
                "If you trust this checkpoint and need legacy pickle loading, re-run with "
                "`--unsafe-load`. WARNING: unsafe for untrusted files."
            ) from exc
        print(
            "Warning: falling back to unsafe torch.load(weights_only=False). "
            "Do not use this on untrusted checkpoints.",
            file=sys.stderr,
        )
        obj = torch.load(str(path), map_location="cpu", weights_only=False)

    if not isinstance(obj, dict):
        raise TypeError(
            f"Unsupported checkpoint format: {type(obj)}"
        )  # pragma: no cover

    # Preferred format: our training checkpoints saved by `python/train/train.py`.
    if "model" in obj and isinstance(obj["model"], dict):
        cfg = obj.get("model_config")
        return obj["model"], cfg if isinstance(cfg, dict) else None

    # Common alternative: `{"model_state_dict": ..., "config": ...}`.
    if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        cfg = obj.get("config")
        return obj["model_state_dict"], cfg if isinstance(cfg, dict) else None

    # Fallback: assume the dict itself is a state_dict.
    return obj, None


def _load_float_safetensors(weights_path: Path) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "safetensors is required to load .safetensors weights"
        ) from exc
    return dict(load_file(str(weights_path)))


def _load_config_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    obj = json.loads(raw)
    if not isinstance(obj, dict):  # pragma: no cover
        raise TypeError(f"Expected JSON object in {path}, got {type(obj)}")
    return obj


def _filter_model_config(cfg: dict[str, Any], config_type: type[Any]) -> dict[str, Any]:
    # Filter unknown keys (e.g. export_dtype) before passing into AsciiGPTConfig.
    from dataclasses import fields

    allowed = {f.name for f in fields(config_type)}
    return {k: v for k, v in cfg.items() if k in allowed}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate ASCII art (Python inference)"
    )
    parser.add_argument("prompt", type=str, help="Text prompt / subject to generate")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to training checkpoint (.pt) produced by python/train/train.py",
    )
    group.add_argument(
        "--model",
        type=Path,
        help="Path to float safetensors weights (expects config.json next to it)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to config.json (defaults to <model_dir>/config.json when using --model)",
    )
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--height", type=int, default=50)
    parser.add_argument(
        "--style",
        type=str,
        default="art",
        choices=["art", "banner", "simple", "detailed"],
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--unsafe-load",
        action="store_true",
        help="Allow legacy pickle checkpoint loading when using --checkpoint (UNSAFE for untrusted files)",
    )
    args = parser.parse_args(argv)

    from model.tokenizer import get_tokenizer  # noqa: E402
    from model.transformer import AsciiGPTConfig, create_model  # noqa: E402
    from inference.generate import generate  # noqa: E402

    cfg: dict[str, Any] | None
    if args.checkpoint is not None:
        state_dict, cfg = _load_torch_checkpoint(
            args.checkpoint, unsafe_load=args.unsafe_load
        )
    else:
        assert args.model is not None
        state_dict = _load_float_safetensors(args.model)
        cfg_path = (
            args.config
            if args.config is not None
            else args.model.parent / "config.json"
        )
        cfg = _load_config_json(cfg_path) if cfg_path.exists() else None

    config_kwargs = _filter_model_config(cfg or {}, AsciiGPTConfig)
    config = AsciiGPTConfig(**config_kwargs) if config_kwargs else AsciiGPTConfig()

    model = create_model(config)
    strict = args.checkpoint is not None
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        non_mask_missing = [k for k in missing if not k.endswith(".mask")]
        if unexpected or non_mask_missing:
            raise RuntimeError(
                f"Unexpected/missing keys when loading safetensors weights.\n"
                f"unexpected={unexpected}\nmissing={missing}"
            )
    model.eval()

    tokenizer = get_tokenizer()

    art = generate(
        model,
        tokenizer,
        args.prompt,
        width=args.width,
        height=args.height,
        style=args.style,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        device="cpu",
    )

    print(art)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
