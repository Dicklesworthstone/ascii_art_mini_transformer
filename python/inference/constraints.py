from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Protocol, Sequence, cast

if TYPE_CHECKING:  # pragma: no cover
    import torch


class TokenizerLike(Protocol):
    @property
    def newline_token_id(self) -> int: ...

    @property
    def eos_token_id(self) -> int: ...


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
        if self.max_tokens > 0 and self.total_tokens >= self.max_tokens:
            return tokenizer.eos_token_id

        # If we've exceeded height, only EOS is allowed.
        if self.max_height > 0 and self.current_row >= self.max_height:
            return tokenizer.eos_token_id

        if self.max_width <= 0:
            return None

        # If we are on the last allowed row and already hit width, end instead of adding a new empty row.
        if (
            self.max_height > 0
            and self.current_row >= self.max_height - 1
            and self.current_col >= self.max_width
        ):
            return tokenizer.eos_token_id

        if self.current_col >= self.max_width:
            return tokenizer.newline_token_id

        return None

    def allowed_token_ids(self, tokenizer: TokenizerLike) -> Optional[Sequence[int]]:
        forced = self.forced_token_id(tokenizer)
        if forced is None:
            return None
        return (forced,)

    def apply_constraints_to_logits(
        self, logits: "torch.Tensor", tokenizer: TokenizerLike
    ) -> "torch.Tensor":
        """
        Apply constraints to a 1D logits tensor by masking disallowed tokens to -inf.

        When no hard constraint is active, this also masks non-output special tokens
        (if the tokenizer provides `is_special_token`), matching Rust inference behavior.
        """
        allowed = self.allowed_token_ids(tokenizer)
        if allowed is not None:
            masked = logits.new_full(logits.shape, float("-inf"))
            for token_id in allowed:
                masked[int(token_id)] = logits[int(token_id)]
            return masked

        masked = logits

        # Avoid empty generations: only permit EOS after we've emitted at least one output token.
        if self.total_tokens == 0:
            masked = masked.clone()
            masked[int(tokenizer.eos_token_id)] = float("-inf")

        # Prevent creating an extra row beyond `max_height`. On the last allowed row, disallow
        # newline unless it's already forced to EOS by `forced_token_id()`.
        if self.max_height > 0 and self.current_row >= self.max_height - 1:
            if masked is logits:
                masked = masked.clone()
            masked[int(tokenizer.newline_token_id)] = float("-inf")

        is_special_token = getattr(tokenizer, "is_special_token", None)
        if not callable(is_special_token):
            return masked

        is_special_fn = cast(Callable[[int], bool], is_special_token)
        if masked is logits:
            masked = logits.clone()

        vocab_size = int(masked.shape[0])
        newline_token_id = int(tokenizer.newline_token_id)
        eos_token_id = int(tokenizer.eos_token_id)

        for token_id in range(vocab_size):
            if token_id == newline_token_id or token_id == eos_token_id:
                continue
            if is_special_fn(token_id):
                masked[token_id] = float("-inf")

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


def compute_row_col_from_tokens(
    token_ids: Sequence[int], tokenizer: TokenizerLike
) -> tuple[int, int]:
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
