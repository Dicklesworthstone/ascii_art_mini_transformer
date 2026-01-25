from __future__ import annotations

from dataclasses import dataclass

from python.inference.constraints import ConstrainedDecoder, compute_row_col_from_tokens


@dataclass(frozen=True, slots=True)
class _Tok:
    newline_token_id: int = 7
    eos_token_id: int = 2


def test_forces_newline_at_width_limit() -> None:
    tok = _Tok()
    dec = ConstrainedDecoder(max_width=3, max_height=10, max_tokens=100)

    # Generate 3 non-newline tokens -> col == 3 => next must be newline.
    for token_id in (10, 11, 12):
        assert dec.forced_token_id(tok) is None
        dec.update(token_id, tok)

    assert dec.current_col == 3
    assert dec.forced_token_id(tok) == tok.newline_token_id


def test_forces_eos_at_height_limit() -> None:
    tok = _Tok()
    dec = ConstrainedDecoder(max_width=5, max_height=2, max_tokens=100)

    # Line 1: 5 chars then newline (row -> 1)
    for token_id in (10, 11, 12, 13, 14, tok.newline_token_id):
        dec.update(token_id, tok)

    assert dec.current_row == 1
    assert dec.current_col == 0

    # Line 2: 5 chars -> EOS should be forced (don't create a 3rd empty line)
    for token_id in (20, 21, 22, 23, 24):
        dec.update(token_id, tok)

    assert dec.current_row == 1
    assert dec.current_col == 5
    assert dec.forced_token_id(tok) == tok.eos_token_id


def test_compute_row_col_from_tokens_matches_decoder() -> None:
    tok = _Tok()
    seq = [10, 11, tok.newline_token_id, 12, 13, tok.newline_token_id, 14]
    row, col = compute_row_col_from_tokens(seq, tok)
    assert (row, col) == (2, 1)


def test_width_zero_disables_width_constraint() -> None:
    tok = _Tok()
    dec = ConstrainedDecoder(max_width=0, max_height=10, max_tokens=100)

    assert dec.forced_token_id(tok) is None
    for token_id in (10, 11, 12, 13, 14):
        dec.update(token_id, tok)
        assert dec.forced_token_id(tok) is None


def test_height_zero_disables_height_constraint() -> None:
    tok = _Tok()
    dec = ConstrainedDecoder(max_width=3, max_height=0, max_tokens=100)

    # Height=0 should not force EOS, and width should still force newline.
    for token_id in (10, 11, 12):
        dec.update(token_id, tok)
    assert dec.current_col == 3
    assert dec.forced_token_id(tok) == tok.newline_token_id


def test_apply_constraints_masks_special_tokens_when_supported() -> None:
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover
        return

    from python.model.tokenizer import AsciiTokenizer

    tok = AsciiTokenizer()
    dec = ConstrainedDecoder(max_width=10, max_height=10, max_tokens=100)

    vocab = tok.vocab_size
    logits = torch.zeros((vocab,), dtype=torch.float32)

    # Make a special token attractive.
    width_id = tok.get_special_token_id("<WIDTH>")
    logits[width_id] = 100.0

    # Make a printable token less attractive.
    token_a = tok.encode("A")[0]
    logits[token_a] = 1.0

    masked = dec.apply_constraints_to_logits(logits, tok)
    assert not torch.isfinite(masked[width_id]), "special tokens should be masked out"
    assert torch.isfinite(masked[token_a]), "printable tokens should remain allowed"
