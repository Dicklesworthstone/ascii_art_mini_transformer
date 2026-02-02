from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional, Protocol, Sequence

from .constraints import ConstrainedDecoder, TokenizerLike
from .sampler import sample_next_token

if TYPE_CHECKING:  # pragma: no cover
    import torch


class ModelLike(Protocol):
    def eval(self) -> Any: ...

    def parameters(self, recurse: bool = True) -> Iterable["torch.Tensor"]: ...

    def __call__(self, input_ids: "torch.Tensor") -> tuple["torch.Tensor", Any]: ...

    @property
    def config(self) -> Any: ...


class TokenizerForGenerate(TokenizerLike, Protocol):
    def encode_inference_prompt(
        self, prompt: str, *, width: int, height: int, style: str
    ) -> Sequence[int]: ...

    def decode(
        self, token_ids: Sequence[int], skip_special_tokens: bool = True
    ) -> str: ...


def _infer_device(model: ModelLike) -> str:
    # Best-effort device inference without importing torch at module import time.
    try:
        params = list(model.parameters())
    except Exception:
        return "cpu"
    if not params:
        return "cpu"
    return str(params[0].device)


def generate(
    model: ModelLike,
    tokenizer: TokenizerForGenerate,
    prompt: str,
    *,
    width: int = 80,
    height: int = 50,
    style: str = "art",
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    max_tokens: int = 4096,
    seed: Optional[int] = None,
    device: Optional[str] = None,
) -> str:
    """
    Generate ASCII art from a text prompt.

    This runs an explicit decoding loop so we can apply constraints per-step.

    Notes:
    - `max_tokens` is a hard cap on emitted output tokens (characters) when > 0.
      When `max_tokens <= 0`, the cap is disabled but the loop is still bounded
      by a width/height-derived maximum (or a small fallback) to avoid infinite
      generation if EOS is never sampled.
    """
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "torch is required for python inference generation. "
            "Install it via python/requirements.txt."
        ) from exc

    if seed is not None:
        torch.manual_seed(int(seed))

    model.eval()

    dev = device or _infer_device(model)

    # Build prompt tokens: <BOS> [constraints] <STYLE_*> <prompt> <SEP>
    input_token_ids: Sequence[int] = tokenizer.encode_inference_prompt(
        prompt, width=width, height=height, style=style
    )

    input_ids = torch.tensor([list(input_token_ids)], dtype=torch.long, device=dev)

    # In Rust we treat max_chars=0 as "disabled" while still bounding the decoding loop.
    # Mirror that here: max_tokens<=0 disables the hard char cap, but we still need a
    # deterministic outer-loop bound to prevent infinite generation when EOS never appears.
    max_tokens_int = int(max_tokens)
    if max_tokens_int > 0:
        max_steps = max_tokens_int
    elif width > 0 and height > 0:
        # Max visible output implied by width/height limits (including inter-line newlines).
        max_steps = int(width) * int(height) + max(int(height) - 1, 0)
    else:
        # Fallback: keep bounded even when constraints are disabled/missing.
        max_steps = 500

    decoder = ConstrainedDecoder(
        max_width=width, max_height=height, max_tokens=max_tokens_int
    )
    generated: list[int] = []

    for _ in range(max_steps):
        # Keep within the model context window.
        if hasattr(model, "config") and hasattr(model.config, "block_size"):
            block_size = int(model.config.block_size)
            if input_ids.shape[1] > block_size:
                input_ids = input_ids[:, -block_size:]

        with torch.no_grad():
            logits, _loss = model(input_ids)
            next_logits = logits[0, -1, :]

        next_logits = decoder.apply_constraints_to_logits(next_logits, tokenizer)
        next_token = sample_next_token(
            next_logits,
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p) if top_p is not None else None,
        )

        if next_token == tokenizer.eos_token_id:
            break

        generated.append(int(next_token))
        decoder.update(int(next_token), tokenizer)

        # Append token for next step.
        token_tensor = torch.tensor([[int(next_token)]], dtype=torch.long, device=dev)
        input_ids = torch.cat([input_ids, token_tensor], dim=1)

        if decoder.should_stop(tokenizer):
            break

    return tokenizer.decode(generated)


def generate_greedy(
    model: ModelLike,
    tokenizer: TokenizerForGenerate,
    prompt: str,
    *,
    width: int = 80,
    height: int = 50,
    style: str = "art",
    max_tokens: int = 4096,
    seed: Optional[int] = None,
    device: Optional[str] = None,
) -> str:
    return generate(
        model,
        tokenizer,
        prompt,
        width=width,
        height=height,
        style=style,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        max_tokens=max_tokens,
        seed=seed,
        device=device,
    )


def generate_sample(
    model: ModelLike,
    tokenizer: TokenizerForGenerate,
    prompt: str,
    *,
    width: int = 80,
    height: int = 50,
    style: str = "art",
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    max_tokens: int = 4096,
    seed: Optional[int] = None,
    device: Optional[str] = None,
) -> str:
    return generate(
        model,
        tokenizer,
        prompt,
        width=width,
        height=height,
        style=style,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        device=device,
    )


@dataclass(frozen=True, slots=True)
class GoldenCase:
    prompt: str
    width: int = 40
    height: int = 20
    style: str = "art"
    seed: int = 0


def generate_golden_tests(
    model: ModelLike,
    tokenizer: TokenizerForGenerate,
    output_dir: Path,
    *,
    cases: Optional[Sequence[GoldenCase]] = None,
    device: Optional[str] = None,
) -> list[Path]:
    """
    Emit small JSON "golden" artifacts for later cross-validation (Python vs Rust).
    """
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "torch is required for golden test generation"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    dev = device or _infer_device(model)

    if cases is None:
        cases = (
            GoldenCase(prompt="cat", width=40, height=20, style="art", seed=0),
            GoldenCase(prompt="star", width=20, height=10, style="simple", seed=1),
            GoldenCase(prompt="HELLO", width=80, height=8, style="banner", seed=2),
        )

    out_paths: list[Path] = []
    for i, case in enumerate(cases):
        torch.manual_seed(int(case.seed))

        input_token_ids: Sequence[int] = tokenizer.encode_inference_prompt(
            case.prompt, width=case.width, height=case.height, style=case.style
        )

        input_ids = torch.tensor([list(input_token_ids)], dtype=torch.long, device=dev)
        with torch.no_grad():
            logits, _loss = model(input_ids)
            last = logits[0, -1, :]

        payload: dict[str, Any] = {
            "case": {
                "prompt": case.prompt,
                "width": case.width,
                "height": case.height,
                "style": case.style,
                "seed": case.seed,
            },
            "input_ids": list(input_token_ids),
            "logits_shape": list(logits.shape),
            "logits_sum": float(last.sum().item()),
            "logits_first_10": last[:10].tolist(),
            "argmax_token": int(torch.argmax(last).item()),
        }

        path = output_dir / f"golden_{i}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        out_paths.append(path)

    return out_paths
