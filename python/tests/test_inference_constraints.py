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

