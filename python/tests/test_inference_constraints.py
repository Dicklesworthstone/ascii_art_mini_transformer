from __future__ import annotations

import sys
from pathlib import Path

import torch

# Add parent to path for imports before importing project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.constraints import ConstrainedDecoder  # noqa: E402
from inference.generate import generate_greedy  # noqa: E402
from model.tokenizer import get_tokenizer  # noqa: E402


class _ConstantLogitsModel(torch.nn.Module):
    def __init__(
        self, *, vocab_size: int, preferred_token_id: int, fallback_token_id: int
    ):
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.zeros(()))
        self._vocab_size = int(vocab_size)
        self._preferred = int(preferred_token_id)
        self._fallback = int(fallback_token_id)

        class _Config:
            block_size = 4096

        self.config = _Config()

    def forward(self, input_ids):
        batch, seq = input_ids.shape
        logits = torch.full(
            (batch, seq, self._vocab_size),
            -10.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[:, :, self._fallback] = 0.0
        logits[:, :, self._preferred] = 5.0
        return logits, None


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


def test_forced_token_id_prioritizes_eos_over_width() -> None:
    tok = get_tokenizer()
    dec = ConstrainedDecoder(max_width=1, max_height=10, max_tokens=1)
    dec.total_tokens = 1
    dec.current_col = 999
    assert dec.forced_token_id(tok) == tok.eos_token_id


def test_forced_token_id_newline_when_width_exceeded_not_last_row() -> None:
    tok = get_tokenizer()
    dec = ConstrainedDecoder(max_width=3, max_height=2, max_tokens=256)
    dec.current_row = 0
    dec.current_col = 3
    assert dec.forced_token_id(tok) == tok.newline_token_id


def test_forced_token_id_eos_when_width_exceeded_on_last_row() -> None:
    tok = get_tokenizer()
    dec = ConstrainedDecoder(max_width=3, max_height=2, max_tokens=256)
    dec.current_row = 1
    dec.current_col = 3
    assert dec.forced_token_id(tok) == tok.eos_token_id


def test_apply_constraints_masks_special_tokens_except_newline_and_eos() -> None:
    tok = get_tokenizer()
    dec = ConstrainedDecoder(max_width=40, max_height=20, max_tokens=256)
    dec.total_tokens = 1

    logits = torch.zeros(tok.vocab_size, dtype=torch.float32)
    masked = dec.apply_constraints_to_logits(logits, tok)

    for token_id in range(tok.vocab_size):
        if token_id in (tok.newline_token_id, tok.eos_token_id):
            assert torch.isfinite(masked[token_id]).item()
            continue
        if tok.is_special_token(token_id):
            assert not torch.isfinite(masked[token_id]).item()


def test_apply_constraints_forces_newline_when_only_token_allowed() -> None:
    tok = get_tokenizer()
    dec = ConstrainedDecoder(max_width=3, max_height=2, max_tokens=256)
    dec.current_col = 3

    logits = torch.zeros(tok.vocab_size, dtype=torch.float32)
    logits[tok.newline_token_id] = -3.0
    logits[tok.encode("A")[0]] = 10.0

    masked = dec.apply_constraints_to_logits(logits, tok)
    assert torch.isfinite(masked[tok.newline_token_id]).item()
    assert int(torch.argmax(masked).item()) == tok.newline_token_id


def test_generate_greedy_respects_width_height_and_max_tokens() -> None:
    tok = get_tokenizer()
    a_id = tok.encode("A")[0]
    b_id = tok.encode("B")[0]

    model = _ConstantLogitsModel(
        vocab_size=tok.vocab_size, preferred_token_id=a_id, fallback_token_id=b_id
    )
    out = generate_greedy(model, tok, "cat", width=3, height=2, max_tokens=64, seed=0)

    assert out == "AAA\nAAA"
    assert len(out) <= 64
    lines = out.split("\n")
    assert len(lines) <= 2
    assert all(len(line) <= 3 for line in lines)


def test_generate_greedy_respects_max_tokens_cap() -> None:
    tok = get_tokenizer()
    a_id = tok.encode("A")[0]
    b_id = tok.encode("B")[0]

    model = _ConstantLogitsModel(
        vocab_size=tok.vocab_size, preferred_token_id=a_id, fallback_token_id=b_id
    )
    out = generate_greedy(model, tok, "cat", width=80, height=50, max_tokens=5, seed=0)

    assert out == "AAAAA"
    assert len(out) <= 5
