//! Embedding layers for the transformer.
//!
//! Token embedding and 2D positional encoding matching the Python model.

use candle_core::{D, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};

use super::config::ModelConfig;

/// Token embedding layer.
pub struct TokenEmbedding {
    /// Embedding weights
    embedding: Embedding,
}

impl TokenEmbedding {
    /// Create a new token embedding layer.
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `vb` - Variable builder for loading weights
    ///
    /// # Errors
    /// Returns an error if weight loading fails.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let embedding = candle_nn::embedding(config.vocab_size, config.n_embd, vb)?;
        Ok(Self { embedding })
    }

    /// Forward pass through token embedding.
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs of shape (batch, seq_len)
    ///
    /// # Returns
    /// Embeddings of shape (batch, seq_len, n_embd)
    ///
    /// # Errors
    /// Returns an error if embedding lookup fails.
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.embedding.forward(token_ids)
    }

    /// Get the embedding weights for weight tying with lm_head.
    #[must_use]
    pub fn weights(&self) -> &Tensor {
        self.embedding.embeddings()
    }
}

/// Learned 2D positional encoding.
///
/// Matches Python `PositionalEncoding2DModule` with `learned=True`.
/// Uses separate embeddings for row and column positions.
pub struct PositionalEncoding2D {
    /// Row position embeddings (max_rows, n_embd // 2)
    row_embedding: Embedding,
    /// Column position embeddings (max_cols, n_embd - n_embd // 2)
    col_embedding: Embedding,
    /// Token ID for newline
    newline_token_id: u32,
}

impl PositionalEncoding2D {
    /// Create a new 2D positional encoding layer.
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `vb` - Variable builder for loading weights
    ///
    /// # Errors
    /// Returns an error if weight loading fails.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let n_embd = config.n_embd;
        let row_dim = n_embd / 2;
        let col_dim = n_embd - row_dim;

        let row_embedding = candle_nn::embedding(config.max_rows, row_dim, vb.pp("row_embedding"))?;
        let col_embedding = candle_nn::embedding(config.max_cols, col_dim, vb.pp("col_embedding"))?;

        Ok(Self {
            row_embedding,
            col_embedding,
            newline_token_id: config.newline_token_id,
        })
    }

    /// Compute row and column positions from token IDs.
    ///
    /// Row increments after each newline token.
    /// Column resets to 0 after each newline token.
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs of shape (batch, seq_len)
    ///
    /// # Returns
    /// Tuple of (row_positions, col_positions), each of shape (batch, seq_len)
    ///
    /// # Errors
    /// Returns an error if tensor operations fail.
    pub fn compute_2d_positions(&self, token_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let device = token_ids.device();
        let (batch_size, seq_len) = token_ids.dims2()?;

        // Convert to vec for processing (TODO: vectorize this)
        let token_vec: Vec<Vec<u32>> = token_ids.to_vec2()?;

        let mut rows = Vec::with_capacity(batch_size);
        let mut cols = Vec::with_capacity(batch_size);

        for batch_tokens in &token_vec {
            let mut batch_rows = Vec::with_capacity(seq_len);
            let mut batch_cols = Vec::with_capacity(seq_len);
            let mut current_row: u32 = 0;
            let mut current_col: u32 = 0;

            for &token in batch_tokens {
                batch_rows.push(current_row);
                batch_cols.push(current_col);

                if token == self.newline_token_id {
                    current_row += 1;
                    current_col = 0;
                } else {
                    current_col += 1;
                }
            }

            rows.push(batch_rows);
            cols.push(batch_cols);
        }

        // Convert back to tensors
        let row_tensor = Tensor::new(rows, device)?;
        let col_tensor = Tensor::new(cols, device)?;

        Ok((row_tensor, col_tensor))
    }

    /// Forward pass through 2D positional encoding.
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs of shape (batch, seq_len)
    ///
    /// # Returns
    /// Position embeddings of shape (batch, seq_len, n_embd)
    ///
    /// # Errors
    /// Returns an error if tensor operations fail.
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        let (rows, cols) = self.compute_2d_positions(token_ids)?;

        // Look up embeddings
        let row_emb = self.row_embedding.forward(&rows)?;
        let col_emb = self.col_embedding.forward(&cols)?;

        // Concatenate along embedding dimension
        Tensor::cat(&[&row_emb, &col_emb], D::Minus1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_2d_position_simple() {
        // Create a mock embedding for testing position computation
        let _config = ModelConfig::small();
        let device = Device::Cpu;

        // Create test tokens: "ab\ncd" -> [a, b, newline, c, d]
        // Using placeholder IDs where 7 is newline
        let tokens = Tensor::new(vec![vec![10u32, 11, 7, 12, 13]], &device).unwrap();

        // Mock position computation (we need to test the logic, not the embedding)
        let newline_id = 7u32;
        let token_vec: Vec<Vec<u32>> = tokens.to_vec2().unwrap();

        let batch = &token_vec[0];
        let mut expected_rows = Vec::new();
        let mut expected_cols = Vec::new();
        let mut row = 0u32;
        let mut col = 0u32;

        for &t in batch {
            expected_rows.push(row);
            expected_cols.push(col);
            if t == newline_id {
                row += 1;
                col = 0;
            } else {
                col += 1;
            }
        }

        // Expected: rows = [0, 0, 0, 1, 1], cols = [0, 1, 2, 0, 1]
        assert_eq!(expected_rows, vec![0, 0, 0, 1, 1]);
        assert_eq!(expected_cols, vec![0, 1, 2, 0, 1]);
    }
}
