use serde::{Deserialize, Serialize};

/// Model hyperparameters matching the Python AsciiGPTConfig.
///
/// This configuration must match exactly for weight compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Maximum sequence length (block_size in Python)
    pub block_size: usize,
    /// Vocabulary size (12 special + 95 printable ASCII)
    pub vocab_size: usize,
    /// Number of transformer blocks
    pub n_layer: usize,
    /// Number of attention heads
    pub n_head: usize,
    /// Embedding dimension
    pub n_embd: usize,
    /// Maximum supported rows for 2D positional encoding
    pub max_rows: usize,
    /// Maximum supported columns for 2D positional encoding
    pub max_cols: usize,
    /// Token ID for newline (used in 2D position computation)
    pub newline_token_id: u32,
    /// Padding token ID
    pub pad_token_id: u32,
    /// Beginning of sequence token ID
    pub bos_token_id: u32,
    /// End of sequence token ID
    pub eos_token_id: u32,
}

impl ModelConfig {
    /// Dimension per attention head.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// Validate that configuration is consistent.
    ///
    /// # Errors
    /// Returns an error if the configuration is internally inconsistent.
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.n_embd.is_multiple_of(self.n_head) {
            return Err("n_embd must be divisible by n_head");
        }
        if self.vocab_size == 0 {
            return Err("vocab_size must be positive");
        }
        if self.block_size == 0 {
            return Err("block_size must be positive");
        }
        Ok(())
    }

    /// Small config (~10M parameters) for fast iteration.
    #[must_use]
    pub fn small() -> Self {
        Self {
            block_size: 1024,
            vocab_size: 107,
            n_layer: 4,
            n_head: 4,
            n_embd: 256,
            max_rows: 100,
            max_cols: 200,
            newline_token_id: 7,
            pad_token_id: 0,
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }

    /// Medium config (~20M parameters) - default.
    #[must_use]
    pub fn medium() -> Self {
        Self {
            block_size: 2048,
            vocab_size: 107,
            n_layer: 6,
            n_head: 6,
            n_embd: 384,
            max_rows: 100,
            max_cols: 200,
            newline_token_id: 7,
            pad_token_id: 0,
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }

    /// Large config (~30M parameters) for best quality.
    #[must_use]
    pub fn large() -> Self {
        Self {
            block_size: 4096,
            vocab_size: 107,
            n_layer: 8,
            n_head: 8,
            n_embd: 512,
            max_rows: 100,
            max_cols: 200,
            newline_token_id: 7,
            pad_token_id: 0,
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::medium()
    }
}
