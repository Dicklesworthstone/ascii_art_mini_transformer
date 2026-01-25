from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence


class TokenizerLike(Protocol):
    newline_token_id: int
    eos_token_id: int


@dataclass
class ConstrainedDecoder:
    """
    Enforce width/height limits during autoregressive decoding.

    This tracks the generated *art* region (tokens after <SEP>), not the full
    prefix prompt. Call `update()` only for generated tokens.
    """

    max_width: int
    max_height: int
    max_tokens: int
    current_row: int = 0
    current_col: int = 0
    total_tokens: int = 0

    def forced_token_id(self, tokenizer: TokenizerLike) -> Optional[int]:
        """
        Return a single forced token ID (newline or EOS) when constraints are hit.

        Priority:
        1) EOS if max_tokens exceeded or height exceeded
        2) newline if width exceeded (unless we're already at the last allowed row)
        """
        if self.total_tokens >= self.max_tokens:
            return tokenizer.eos_token_id

        # If we've exceeded height, only EOS is allowed.
        if self.current_row >= self.max_height:
            return tokenizer.eos_token_id

        # If we are on the last allowed row and already hit width, end instead of adding a new empty row.
        if self.current_row >= max(self.max_height - 1, 0) and self.current_col >= self.max_width:
            return tokenizer.eos_token_id

        if self.current_col >= self.max_width:
            return tokenizer.newline_token_id

        return None

    def allowed_token_ids(self, tokenizer: TokenizerLike) -> Optional[Sequence[int]]:
        forced = self.forced_token_id(tokenizer)
        if forced is None:
            return None
        return (forced,)

    def apply_constraints_to_logits(self, logits, tokenizer: TokenizerLike):
        """
        Apply constraints to a 1D logits tensor by masking disallowed tokens to -inf.

        This method requires torch at runtime but is written to avoid importing
        torch at module import time.
        """
        allowed = self.allowed_token_ids(tokenizer)
        if allowed is None:
            return logits

        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError("torch is required to mask logits") from exc

        masked = torch.full_like(logits, float("-inf"))
        for token_id in allowed:
            masked[token_id] = logits[token_id]
        return masked

    def update(self, token_id: int, tokenizer: TokenizerLike) -> None:
        self.total_tokens += 1
        if token_id == tokenizer.newline_token_id:
            self.current_row += 1
            self.current_col = 0
        else:
            self.current_col += 1

    def should_stop(self, tokenizer: TokenizerLike) -> bool:
        return self.forced_token_id(tokenizer) == tokenizer.eos_token_id


def compute_row_col_from_tokens(token_ids: Sequence[int], tokenizer: TokenizerLike) -> tuple[int, int]:
    """
    Compute (row, col) position after consuming `token_ids`, using the same
    semantics as ConstrainedDecoder.update().
    """
    row = 0
    col = 0
    for token_id in token_ids:
        if token_id == tokenizer.newline_token_id:
            row += 1
            col = 0
        else:
            col += 1
    return row, col

