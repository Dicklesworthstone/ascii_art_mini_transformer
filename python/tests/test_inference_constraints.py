from __future__ import annotations

import torch

from inference.constraints import ConstrainedDecoder
from model.tokenizer import get_tokenizer


def test_eos_disallowed_before_any_output() -> None:
    tok = get_tokenizer()
    dec = ConstrainedDecoder(max_width=40, max_height=20, max_tokens=256)
    logits = torch.zeros(tok.vocab_size, dtype=torch.float32)
    masked = dec.apply_constraints_to_logits(logits, tok)
    assert not torch.isfinite(masked[tok.eos_token_id]).item()


def test_newline_disallowed_on_last_row() -> None:
    tok = get_tokenizer()
    dec = ConstrainedDecoder(max_width=40, max_height=1, max_tokens=256)
    logits = torch.zeros(tok.vocab_size, dtype=torch.float32)
    masked = dec.apply_constraints_to_logits(logits, tok)
    assert not torch.isfinite(masked[tok.newline_token_id]).item()

