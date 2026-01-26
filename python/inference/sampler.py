from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Sampler(Protocol):
    def sample(self, logits) -> int: ...


@dataclass(frozen=True, slots=True)
class TopKSampler:
    """
    Sample from the distribution restricted to the top-k logits.

    If `temperature` is 0, this behaves like greedy argmax.
    """

    k: int
    temperature: float = 1.0

    def sample(self, logits) -> int:
        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError("torch is required for sampling") from exc

        if self.temperature <= 0:
            return int(torch.argmax(logits).item())

        scaled = logits / float(self.temperature)
        if self.k and self.k > 0:
            values, _ = torch.topk(scaled, k=min(self.k, scaled.shape[-1]))
            cutoff = values[..., -1, None]
            scaled = torch.where(
                scaled < cutoff, torch.full_like(scaled, float("-inf")), scaled
            )

        probs = torch.softmax(scaled, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return int(next_token.item())


@dataclass(frozen=True, slots=True)
class TopPSampler:
    """
    Nucleus sampling (top-p).

    If `temperature` is 0, this behaves like greedy argmax.
    """

    p: float
    temperature: float = 1.0

    def sample(self, logits) -> int:
        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError("torch is required for sampling") from exc

        if self.temperature <= 0:
            return int(torch.argmax(logits).item())

        scaled = logits / float(self.temperature)
        probs = torch.softmax(scaled, dim=-1)

        # Match Rust behavior: disable nucleus sampling for top_p < 0 or top_p >= 1.
        if not (0.0 <= float(self.p) < 1.0):
            next_token = torch.multinomial(probs, num_samples=1)
            return int(next_token.item())

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above threshold, keeping the first
        # token above the threshold (match Rust / HF behavior).
        to_remove = cum > float(self.p)
        if to_remove.shape[-1] > 1:
            to_remove[..., 1:] = to_remove[..., :-1].clone()
        to_remove[..., 0] = False

        filtered_probs = torch.where(
            to_remove, torch.zeros_like(sorted_probs), sorted_probs
        )
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

        choice = torch.multinomial(filtered_probs, num_samples=1)
        token_id = sorted_idx[choice]
        return int(token_id.item())


def sample_next_token(
    logits,
    *,
    temperature: float,
    top_k: int = 0,
    top_p: float | None = None,
) -> int:
    """
    Convenience sampling helper supporting both top-k and top-p.
    """
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("torch is required for sampling") from exc

    if temperature <= 0:
        return int(torch.argmax(logits).item())

    scaled = logits / float(temperature)

    if top_k and top_k > 0:
        values, _ = torch.topk(scaled, k=min(int(top_k), scaled.shape[-1]))
        cutoff = values[..., -1, None]
        scaled = torch.where(
            scaled < cutoff, torch.full_like(scaled, float("-inf")), scaled
        )

    # Match Rust behavior: disable nucleus sampling for top_p < 0 or top_p >= 1.
    if top_p is not None and 0.0 <= float(top_p) < 1.0:
        probs = torch.softmax(scaled, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        to_remove = cum > float(top_p)
        if to_remove.shape[-1] > 1:
            to_remove[..., 1:] = to_remove[..., :-1].clone()
        to_remove[..., 0] = False
        filtered = torch.where(to_remove, torch.zeros_like(sorted_probs), sorted_probs)
        filtered = filtered / filtered.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(filtered, num_samples=1)
        token_id = sorted_idx[choice]
        return int(token_id.item())

    probs = torch.softmax(scaled, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())
