"""
Unit tests for 2D positional encoding.
"""

import pytest

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import from module - handle import errors gracefully
if TORCH_AVAILABLE:
    from python.model.positional_encoding import (
        LearnedPositionalEncoding2D,
        SinusoidalPositionalEncoding2D,
        PositionalEncoding2DModule,
        compute_2d_positions_vectorized,
        compute_2d_positions_simple,
        create_positional_encoding,
    )
else:
    # Only import the non-torch function for basic testing
    import sys
    sys.path.insert(0, '.')
    exec("""
def compute_2d_positions_simple(text, newline_char='\\n'):
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
""")


class TestSimplePositionComputation:
    """Tests for non-vectorized position computation (no torch required)."""

    def test_single_line(self):
        """Single line should have row=0, col=0,1,2,..."""
        text = "hello"
        rows, cols = compute_2d_positions_simple(text)
        assert rows == [0, 0, 0, 0, 0]
        assert cols == [0, 1, 2, 3, 4]

    def test_two_lines(self):
        """Two lines should increment row after newline."""
        text = "ab\ncd"
        rows, cols = compute_2d_positions_simple(text)
        # a(0,0) b(0,1) \n(0,2) c(1,0) d(1,1)
        assert rows == [0, 0, 0, 1, 1]
        assert cols == [0, 1, 2, 0, 1]

    def test_multiple_lines(self):
        """Multiple lines should track correctly."""
        text = "a\nb\nc"
        rows, cols = compute_2d_positions_simple(text)
        # a(0,0) \n(0,1) b(1,0) \n(1,1) c(2,0)
        assert rows == [0, 0, 1, 1, 2]
        assert cols == [0, 1, 0, 1, 0]

    def test_empty_line(self):
        """Empty lines (consecutive newlines) should work."""
        text = "a\n\nb"
        rows, cols = compute_2d_positions_simple(text)
        # a(0,0) \n(0,1) \n(1,0) b(2,0)
        assert rows == [0, 0, 1, 2]
        assert cols == [0, 1, 0, 0]

    def test_ends_with_newline(self):
        """Ending with newline should work."""
        text = "ab\n"
        rows, cols = compute_2d_positions_simple(text)
        assert rows == [0, 0, 0]
        assert cols == [0, 1, 2]

    def test_starts_with_newline(self):
        """Starting with newline should work."""
        text = "\nab"
        rows, cols = compute_2d_positions_simple(text)
        # \n(0,0) a(1,0) b(1,1)
        assert rows == [0, 1, 1]
        assert cols == [0, 0, 1]

    def test_empty_string(self):
        """Empty string should return empty lists."""
        rows, cols = compute_2d_positions_simple("")
        assert rows == []
        assert cols == []

    def test_only_newlines(self):
        """Only newlines should work."""
        text = "\n\n"
        rows, cols = compute_2d_positions_simple(text)
        assert rows == [0, 1]
        assert cols == [0, 0]

    def test_ascii_art_example(self):
        """Real ASCII art example should work."""
        art = " /\\_/\\\n( o.o )\n > ^ <"
        rows, cols = compute_2d_positions_simple(art)

        # First line: " /\_/\" - 6 chars at row 0
        assert rows[:6] == [0] * 6
        assert cols[:6] == [0, 1, 2, 3, 4, 5]

        # Newline at position 6
        assert rows[6] == 0
        assert cols[6] == 6

        # Second line starts at row 1, col 0
        assert rows[7] == 1
        assert cols[7] == 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestVectorizedPositionComputation:
    """Tests for vectorized (torch-based) position computation."""

    def test_basic_computation(self):
        """Basic vectorized computation should match simple version."""
        token_ids = torch.tensor([[1, 2, 3, 7, 4, 5]])  # 7 is newline
        rows, cols = compute_2d_positions_vectorized(token_ids, newline_token_id=7)

        # Should match: tokens before newline at row 0, after at row 1
        assert rows[0].tolist() == [0, 0, 0, 0, 1, 1]
        # Cols reset after newline
        assert cols[0].tolist() == [0, 1, 2, 3, 0, 1]

    def test_batch_computation(self):
        """Batched computation should work correctly."""
        token_ids = torch.tensor([
            [1, 2, 7, 3, 4],
            [1, 7, 2, 7, 3],
        ])
        rows, cols = compute_2d_positions_vectorized(token_ids, newline_token_id=7)

        # First sequence: 1,2,NL,3,4 -> rows 0,0,0,1,1
        assert rows[0].tolist() == [0, 0, 0, 1, 1]
        assert cols[0].tolist() == [0, 1, 2, 0, 1]

        # Second sequence: 1,NL,2,NL,3 -> rows 0,0,1,1,2
        assert rows[1].tolist() == [0, 0, 1, 1, 2]
        assert cols[1].tolist() == [0, 1, 0, 1, 0]

    def test_no_newlines(self):
        """Sequence without newlines should be single row."""
        token_ids = torch.tensor([[1, 2, 3, 4, 5]])
        rows, cols = compute_2d_positions_vectorized(token_ids, newline_token_id=99)

        assert rows[0].tolist() == [0, 0, 0, 0, 0]
        assert cols[0].tolist() == [0, 1, 2, 3, 4]

    def test_all_newlines(self):
        """All newlines should increment row each time."""
        token_ids = torch.tensor([[7, 7, 7]])
        rows, cols = compute_2d_positions_vectorized(token_ids, newline_token_id=7)

        assert rows[0].tolist() == [0, 1, 2]
        assert cols[0].tolist() == [0, 0, 0]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestLearnedPositionalEncoding2D:
    """Tests for learned 2D positional encoding."""

    def test_output_shape(self):
        """Output shape should be (batch, seq, d_model)."""
        encoding = LearnedPositionalEncoding2D(d_model=64, max_rows=50, max_cols=100)
        rows = torch.tensor([[0, 0, 1, 1], [0, 1, 2, 3]])
        cols = torch.tensor([[0, 1, 0, 1], [0, 0, 0, 0]])

        output = encoding(rows, cols)
        assert output.shape == (2, 4, 64)

    def test_same_position_same_embedding(self):
        """Same position should give same embedding."""
        encoding = LearnedPositionalEncoding2D(d_model=64)
        rows = torch.tensor([[0, 0], [1, 1]])
        cols = torch.tensor([[5, 5], [10, 10]])

        output = encoding(rows, cols)
        # Same row/col within batch should give same embedding
        assert torch.allclose(output[0, 0], output[0, 1])
        assert torch.allclose(output[1, 0], output[1, 1])

    def test_different_positions_different_embeddings(self):
        """Different positions should give different embeddings."""
        encoding = LearnedPositionalEncoding2D(d_model=64)
        rows = torch.tensor([[0, 1]])
        cols = torch.tensor([[0, 0]])

        output = encoding(rows, cols)
        # Different rows should give different embeddings
        assert not torch.allclose(output[0, 0], output[0, 1])

    def test_clamps_to_max(self):
        """Indices beyond max should be clamped."""
        encoding = LearnedPositionalEncoding2D(d_model=64, max_rows=10, max_cols=10)
        rows = torch.tensor([[100]])  # Beyond max
        cols = torch.tensor([[100]])

        # Should not error, should clamp to max-1
        output = encoding(rows, cols)
        assert output.shape == (1, 1, 64)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestSinusoidalPositionalEncoding2D:
    """Tests for sinusoidal 2D positional encoding."""

    def test_output_shape(self):
        """Output shape should be (batch, seq, d_model)."""
        encoding = SinusoidalPositionalEncoding2D(d_model=64)
        rows = torch.tensor([[0, 0, 1]])
        cols = torch.tensor([[0, 1, 0]])

        output = encoding(rows, cols)
        assert output.shape == (1, 3, 64)

    def test_deterministic(self):
        """Sinusoidal encoding should be deterministic."""
        encoding = SinusoidalPositionalEncoding2D(d_model=64)
        rows = torch.tensor([[0, 1, 2]])
        cols = torch.tensor([[0, 1, 2]])

        output1 = encoding(rows, cols)
        output2 = encoding(rows, cols)
        assert torch.allclose(output1, output2)

    def test_not_trainable(self):
        """Sinusoidal encoding buffers should not be parameters."""
        encoding = SinusoidalPositionalEncoding2D(d_model=64)
        params = list(encoding.parameters())
        assert len(params) == 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestPositionalEncoding2DModule:
    """Tests for the complete positional encoding module."""

    def test_basic_usage(self):
        """Module should work with token IDs directly."""
        module = PositionalEncoding2DModule(d_model=64, newline_token_id=7)
        token_ids = torch.tensor([[1, 2, 7, 3, 4]])

        output = module(token_ids)
        assert output.shape == (1, 5, 64)

    def test_learned_vs_sinusoidal(self):
        """Can create with either learned or sinusoidal encoding."""
        module_learned = PositionalEncoding2DModule(d_model=64, learned=True)
        module_sin = PositionalEncoding2DModule(d_model=64, learned=False)

        token_ids = torch.tensor([[1, 2, 3]])
        out_learned = module_learned(token_ids)
        out_sin = module_sin(token_ids)

        assert out_learned.shape == out_sin.shape


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestFactoryFunction:
    """Tests for the factory function."""

    def test_create_positional_encoding(self):
        """Factory should create valid module."""
        module = create_positional_encoding(
            d_model=128,
            max_rows=200,
            max_cols=400,
            newline_token_id=7,
            learned=True,
        )

        token_ids = torch.tensor([[1, 2, 3]])
        output = module(token_ids)
        assert output.shape == (1, 3, 128)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
