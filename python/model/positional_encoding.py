"""
2D Positional Encoding for ASCII art transformers.

ASCII art is fundamentally a 2D grid, not a 1D sequence. Standard positional
encoding loses critical spatial information. This module implements 2D
encoding that captures both row and column position.

Based on research:
- ViTARC (2024): "Tackling ARC with Vision Transformers: Importance of 2D Representation"
- GridPE (2024): "GridPE: Unifying Positional Encoding with Grid Cell-Inspired Framework"

Key insight: Treating ASCII art as a grid dramatically improves spatial reasoning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


class LearnedPositionalEncoding2D(nn.Module):
    """
    Learned 2D positional encoding using separate row and column embeddings.

    The embedding dimension is split between row and column representations,
    which are then concatenated. This allows the model to learn appropriate
    positional representations for ASCII art grids.

    Args:
        d_model: Total embedding dimension (will be split between row/col)
        max_rows: Maximum number of rows to support
        max_cols: Maximum number of columns to support
    """

    def __init__(
        self,
        d_model: int,
        max_rows: int = 100,
        max_cols: int = 200,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_rows = max_rows
        self.max_cols = max_cols

        # Split dimensions between row and column
        self.row_dim = d_model // 2
        self.col_dim = d_model - self.row_dim

        # Learned embeddings for row and column positions
        self.row_embedding = nn.Embedding(max_rows, self.row_dim)
        self.col_embedding = nn.Embedding(max_cols, self.col_dim)

        # Initialize with small values
        nn.init.normal_(self.row_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.col_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute 2D positional embeddings.

        Args:
            row_indices: Row indices of shape (batch_size, seq_len)
            col_indices: Column indices of shape (batch_size, seq_len)

        Returns:
            Positional embeddings of shape (batch_size, seq_len, d_model)
        """
        # Clamp indices to valid range
        row_indices = row_indices.clamp(0, self.max_rows - 1)
        col_indices = col_indices.clamp(0, self.max_cols - 1)

        # Get embeddings
        row_emb = self.row_embedding(row_indices)  # (batch, seq, row_dim)
        col_emb = self.col_embedding(col_indices)  # (batch, seq, col_dim)

        # Concatenate to form full positional encoding
        return torch.cat([row_emb, col_emb], dim=-1)  # (batch, seq, d_model)


class SinusoidalPositionalEncoding2D(nn.Module):
    """
    Sinusoidal 2D positional encoding (non-learned).

    Uses sine and cosine functions at different frequencies for row and
    column positions. This is similar to the original Transformer positional
    encoding but extended to 2D.

    Args:
        d_model: Total embedding dimension
        max_rows: Maximum number of rows
        max_cols: Maximum number of columns
    """

    def __init__(
        self,
        d_model: int,
        max_rows: int = 100,
        max_cols: int = 200,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_rows = max_rows
        self.max_cols = max_cols

        # Precompute sinusoidal encodings
        row_pe = self._compute_sinusoidal(max_rows, d_model // 2)
        col_pe = self._compute_sinusoidal(max_cols, d_model - d_model // 2)

        self.row_pe: torch.Tensor
        self.col_pe: torch.Tensor
        # Register as buffers (not parameters)
        self.register_buffer("row_pe", row_pe)
        self.register_buffer("col_pe", col_pe)

    def _compute_sinusoidal(self, max_len: int, d: int) -> torch.Tensor:
        """Compute sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d, 2).float() * (-torch.log(torch.tensor(10000.0)) / d)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: d // 2])

        return pe

    def forward(
        self,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute 2D sinusoidal positional embeddings.

        Args:
            row_indices: Row indices of shape (batch_size, seq_len)
            col_indices: Column indices of shape (batch_size, seq_len)

        Returns:
            Positional embeddings of shape (batch_size, seq_len, d_model)
        """
        # Clamp indices
        row_indices = row_indices.clamp(0, self.max_rows - 1)
        col_indices = col_indices.clamp(0, self.max_cols - 1)

        # Gather encodings for each position
        row_emb = self.row_pe[row_indices]  # (batch, seq, d//2)
        col_emb = self.col_pe[col_indices]  # (batch, seq, d//2)

        return torch.cat([row_emb, col_emb], dim=-1)


def compute_2d_positions_vectorized(
    token_ids: torch.Tensor,
    newline_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized computation of 2D positions from 1D token sequence.

    Computes row and column indices for each token, where:
    - Row increments after each newline token
    - Column resets to 0 after each newline token
    - First token starts at (row=0, col=0)

    This implementation is fully vectorized with no Python loops in the
    forward pass, making it efficient for batched training.

    Args:
        token_ids: Token IDs of shape (batch_size, seq_len)
        newline_token_id: The token ID representing newline

    Returns:
        rows: Row indices of shape (batch_size, seq_len)
        cols: Column indices of shape (batch_size, seq_len)
    """
    batch_size, seq_len = token_ids.shape
    device = token_ids.device

    # Find newline positions
    is_newline = (token_ids == newline_token_id).long()

    # Row index = cumulative sum of newlines seen so far
    # We shift by 1 because first token is row 0, row increments AFTER newline
    rows_unshifted = torch.cumsum(is_newline, dim=1)
    rows = torch.cat(
        [
            torch.zeros(batch_size, 1, dtype=torch.long, device=device),
            rows_unshifted[:, :-1],
        ],
        dim=1,
    )

    # For columns: need to reset to 0 after each newline
    # Strategy: compute position within current line using cummax trick

    # Create position indices
    position_in_seq = (
        torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    )

    # Find the most recent newline position BEFORE this token
    # We need to look at the previous position's newline, not the current
    # First, shift the newline positions right by 1 (the previous newline matters)
    newline_positions = torch.where(
        is_newline.bool(),
        position_in_seq,
        torch.tensor(-1, dtype=torch.long, device=device),
    )

    # Shift right: for each position, we want the most recent newline BEFORE it
    # Create shifted version: prepend -1, drop last element
    shifted_newline_positions = torch.cat(
        [
            torch.full((batch_size, 1), -1, dtype=torch.long, device=device),
            newline_positions[:, :-1],
        ],
        dim=1,
    )

    # Propagate the last newline index forward using cummax
    last_newline_before, _ = torch.cummax(shifted_newline_positions, dim=1)

    # Column = current position - last newline position before this token - 1
    # For positions before any newline, last_newline_before is -1, so col = pos - (-1) - 1 = pos
    cols = position_in_seq - last_newline_before - 1

    return rows, cols


def compute_2d_positions_simple(
    text: str,
    newline_char: str = "\n",
) -> Tuple[list[int], list[int]]:
    """
    Simple (non-vectorized) computation of 2D positions.

    This is useful for preprocessing during data loading, where the
    positions can be computed once per example and cached.

    Args:
        text: The text to compute positions for
        newline_char: Character that represents newline

    Returns:
        rows: List of row indices
        cols: List of column indices
    """
    row, col = 0, 0
    rows, cols = [], []

    for char in text:
        rows.append(row)
        cols.append(col)

        if char == newline_char:
            row += 1
            col = 0
        else:
            col += 1

    return rows, cols


class PositionalEncoding2DModule(nn.Module):
    """
    Complete 2D positional encoding module for use in transformer models.

    This module combines position computation and embedding lookup, providing
    a clean interface for adding to token embeddings.

    Args:
        d_model: Embedding dimension
        max_rows: Maximum supported rows
        max_cols: Maximum supported columns
        newline_token_id: Token ID for newline
        learned: If True, use learned embeddings; else use sinusoidal
    """

    def __init__(
        self,
        d_model: int,
        max_rows: int = 100,
        max_cols: int = 200,
        newline_token_id: int = 7,
        learned: bool = True,
    ) -> None:
        super().__init__()
        self.newline_token_id = newline_token_id
        self.pos_encoding: LearnedPositionalEncoding2D | SinusoidalPositionalEncoding2D

        if learned:
            self.pos_encoding = LearnedPositionalEncoding2D(d_model, max_rows, max_cols)
        else:
            self.pos_encoding = SinusoidalPositionalEncoding2D(
                d_model, max_rows, max_cols
            )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute positional embeddings for token sequence.

        Args:
            token_ids: Token IDs of shape (batch_size, seq_len)

        Returns:
            Positional embeddings of shape (batch_size, seq_len, d_model)
        """
        rows, cols = compute_2d_positions_vectorized(token_ids, self.newline_token_id)
        return self.pos_encoding.forward(rows, cols)


# Factory function for creating positional encoding
def create_positional_encoding(
    d_model: int,
    max_rows: int = 100,
    max_cols: int = 200,
    newline_token_id: int = 7,
    learned: bool = True,
) -> PositionalEncoding2DModule:
    """
    Create a 2D positional encoding module.

    Args:
        d_model: Embedding dimension
        max_rows: Maximum supported rows
        max_cols: Maximum supported columns
        newline_token_id: Token ID for newline
        learned: If True, use learned embeddings; else use sinusoidal

    Returns:
        Configured positional encoding module
    """
    return PositionalEncoding2DModule(
        d_model=d_model,
        max_rows=max_rows,
        max_cols=max_cols,
        newline_token_id=newline_token_id,
        learned=learned,
    )


if __name__ == "__main__":  # pragma: no cover
    # Quick test
    print("Testing 2D Positional Encoding")
    print("=" * 50)

    # Test vectorized position computation
    print("\n1. Testing position computation...")
    token_ids = torch.tensor(
        [
            [1, 2, 3, 7, 4, 5, 7, 6],  # 7 is newline
            [1, 7, 2, 3, 4, 7, 5, 6],
        ]
    )
    rows, cols = compute_2d_positions_vectorized(token_ids, newline_token_id=7)
    print(f"Token IDs:\n{token_ids}")
    print(f"Rows:\n{rows}")
    print(f"Cols:\n{cols}")

    # Verify first sequence: 1,2,3,NL,4,5,NL,6
    # Expected rows: 0,0,0,0,1,1,1,2
    # Expected cols: 0,1,2,3,0,1,2,0
    assert rows[0].tolist() == [0, 0, 0, 0, 1, 1, 1, 2], (
        f"Row mismatch: {rows[0].tolist()}"
    )
    assert cols[0].tolist() == [0, 1, 2, 3, 0, 1, 2, 0], (
        f"Col mismatch: {cols[0].tolist()}"
    )
    print("Position computation test passed!")

    # Test learned encoding
    print("\n2. Testing learned encoding...")
    encoding = LearnedPositionalEncoding2D(d_model=64, max_rows=50, max_cols=100)
    pos_emb = encoding(rows, cols)
    print(f"Positional embedding shape: {pos_emb.shape}")
    assert pos_emb.shape == (2, 8, 64)
    print("Learned encoding test passed!")

    # Test sinusoidal encoding
    print("\n3. Testing sinusoidal encoding...")
    encoding_sin = SinusoidalPositionalEncoding2D(d_model=64, max_rows=50, max_cols=100)
    pos_emb_sin = encoding_sin(rows, cols)
    print(f"Sinusoidal embedding shape: {pos_emb_sin.shape}")
    assert pos_emb_sin.shape == (2, 8, 64)
    print("Sinusoidal encoding test passed!")

    # Test complete module
    print("\n4. Testing complete module...")
    module = PositionalEncoding2DModule(d_model=64, newline_token_id=7)
    pos_emb_module = module(token_ids)
    print(f"Module output shape: {pos_emb_module.shape}")
    assert pos_emb_module.shape == (2, 8, 64)
    print("Complete module test passed!")

    # Test simple (non-vectorized) computation
    print("\n5. Testing simple computation...")
    text = "abc\nde\nf"
    rows_simple, cols_simple = compute_2d_positions_simple(text)
    print(f"Text: {repr(text)}")
    print(f"Rows: {rows_simple}")
    print(f"Cols: {cols_simple}")
    # Expected: a(0,0) b(0,1) c(0,2) \n(0,3) d(1,0) e(1,1) \n(1,2) f(2,0)
    assert rows_simple == [0, 0, 0, 0, 1, 1, 1, 2]
    assert cols_simple == [0, 1, 2, 3, 0, 1, 2, 0]
    print("Simple computation test passed!")

    print("\n" + "=" * 50)
    print("All tests passed!")
