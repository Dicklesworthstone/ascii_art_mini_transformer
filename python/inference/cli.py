from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("torch is required to load checkpoints") from exc

    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"], obj.get("config")
    if isinstance(obj, dict):
        # Assume it's already a state dict.
        return obj, None
    raise TypeError(f"Unsupported checkpoint format: {type(obj)}")  # pragma: no cover


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate ASCII art (Python inference)")
    parser.add_argument("prompt", type=str, help="Text prompt / subject to generate")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to torch checkpoint / state_dict")
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--height", type=int, default=50)
    parser.add_argument("--style", type=str, default="art", choices=["art", "banner", "simple", "detailed"])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("torch is required to run inference") from exc

    from python.model.tokenizer import AsciiTokenizer  # noqa: E402
    from python.model.transformer import AsciiGPT, AsciiGPTConfig  # noqa: E402
    from python.inference.generate import generate  # noqa: E402

    state_dict, cfg = _load_checkpoint(args.checkpoint)
    config = AsciiGPTConfig(**cfg) if isinstance(cfg, dict) else AsciiGPTConfig()

    model = AsciiGPT(config)
    model.load_state_dict(state_dict)
    model.eval()

    tokenizer = AsciiTokenizer()

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

    torch.set_printoptions(profile="full")
    print(art)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

