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
            cutoff = values[-1]
            scaled = torch.where(scaled < cutoff, torch.full_like(scaled, float("-inf")), scaled)

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

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)

        # Keep at least one token.
        keep = cum <= float(self.p)
        keep[..., 0] = True

        filtered_probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        filtered_probs = filtered_probs / torch.sum(filtered_probs)

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
        cutoff = values[-1]
        scaled = torch.where(scaled < cutoff, torch.full_like(scaled, float("-inf")), scaled)

    if top_p is not None and top_p < 1.0:
        probs = torch.softmax(scaled, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        keep = cum <= float(top_p)
        keep[..., 0] = True
        filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        filtered = filtered / torch.sum(filtered)
        choice = torch.multinomial(filtered, num_samples=1)
        token_id = sorted_idx[choice]
        return int(token_id.item())

    probs = torch.softmax(scaled, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())

