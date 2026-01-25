//! ASCII Art GPT Transformer Model.
//!
//! A decoder-only transformer for ASCII art generation.
//! Matches the Python AsciiGPT implementation.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};

use super::attention::create_causal_mask;
use super::config::ModelConfig;
use super::embedding::{PositionalEncoding2D, TokenEmbedding};
use super::layers::TransformerBlock;

/// ASCII Art GPT - A decoder-only transformer for ASCII art generation.
///
/// Features:
/// - Character-level tokenization (~107 tokens)
/// - 2D positional encoding (row + column)
/// - Causal attention for autoregressive generation
/// - Weight tying between embeddings and LM head (via Linear weight reuse)
pub struct AsciiGPT {
    /// Token embedding
    token_embedding: TokenEmbedding,
    /// 2D positional encoding
    pos_encoding: PositionalEncoding2D,
    /// Transformer blocks
    blocks: Vec<TransformerBlock>,
    /// Final layer norm
    ln_f: LayerNorm,
    /// Language model head (uses tied weights from token_embedding)
    lm_head: Linear,
    /// Model configuration
    config: ModelConfig,
    /// Device for tensor operations
    device: Device,
}

impl AsciiGPT {
    /// Create a new AsciiGPT model by loading weights.
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `vb` - Variable builder for loading weights
    ///
    /// # Errors
    /// Returns an error if weight loading fails.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(config: ModelConfig, vb: VarBuilder) -> Result<Self> {
        config
            .validate()
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let device = vb.device().clone();

        // Token embedding
        let token_vb = vb.pp("token_embedding");
        let token_embedding = TokenEmbedding::new(&config, token_vb)?;

        // 2D positional encoding
        // Match Python module nesting: pos_encoding.pos_encoding.{row_embedding,col_embedding}
        let pos_encoding =
            PositionalEncoding2D::new(&config, vb.pp("pos_encoding").pp("pos_encoding"))?;

        // Transformer blocks
        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let block = TransformerBlock::new(&config, vb.pp(format!("blocks.{i}")))?;
            blocks.push(block);
        }

        // Final layer norm
        let ln_f = candle_nn::layer_norm(
            config.n_embd,
            candle_nn::LayerNormConfig::default(),
            vb.pp("ln_f"),
        )?;

        // Language model head: projects n_embd -> vocab_size.
        //
        // Python ties lm_head.weight and token_embedding.weight. Prefer loading lm_head if it
        // exists, otherwise reuse token_embedding weights.
        let lm_head = if vb.pp("lm_head").contains_tensor("weight") {
            candle_nn::linear_no_bias(config.n_embd, config.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::new(token_embedding.weights().clone(), None)
        };

        Ok(Self {
            token_embedding,
            pos_encoding,
            blocks,
            ln_f,
            lm_head,
            config,
            device,
        })
    }

    /// Forward pass through the model.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs of shape (batch, seq_len)
    ///
    /// # Returns
    /// Logits of shape (batch, seq_len, vocab_size)
    ///
    /// # Errors
    /// Returns an error if tensor operations fail.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_, t) = input_ids.dims2()?;

        if t > self.config.block_size {
            return Err(candle_core::Error::Msg(format!(
                "Sequence length {} exceeds block_size {}",
                t, self.config.block_size
            )));
        }

        // Token embeddings
        let tok_emb = self.token_embedding.forward(input_ids)?;

        // 2D positional embeddings
        let pos_emb = self.pos_encoding.forward(input_ids)?;

        // Combine embeddings
        let mut x = (tok_emb + pos_emb)?;

        // Create causal mask
        let mask = create_causal_mask(t, &self.device)?;

        // Apply transformer blocks
        for block in &self.blocks {
            x = block.forward(&x, &mask)?;
        }

        // Final layer norm
        x = self.ln_f.forward(&x)?;

        // LM head: (B, T, n_embd) @ (n_embd, vocab_size) -> (B, T, vocab_size)
        // Note: The lm_head Linear expects (vocab_size, n_embd) weights,
        // but we loaded it as (n_embd, vocab_size) matching token_embedding.
        // We need to transpose or handle this correctly.
        // Actually, Linear.forward does: x @ weight.T, so if weight is (vocab_size, n_embd),
        // then x @ weight.T gives (B, T, vocab_size). We need weight (vocab_size, n_embd).
        // The lm_head should be linear(n_embd, vocab_size) which gives weight (vocab_size, n_embd).
        self.lm_head.forward(&x)
    }

    /// Get the model configuration.
    #[must_use]
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get the device.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get logits for the last token only (for generation).
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs of shape (batch, seq_len)
    ///
    /// # Returns
    /// Logits for last position of shape (batch, vocab_size)
    ///
    /// # Errors
    /// Returns an error if tensor operations fail.
    pub fn forward_last(&self, input_ids: &Tensor) -> Result<Tensor> {
        let logits = self.forward(input_ids)?;
        let (_, t, _) = logits.dims3()?;
        logits.narrow(1, t - 1, 1)?.squeeze(1)
    }
}

/// Load an AsciiGPT model from a safetensors file.
///
/// # Arguments
/// * `path` - Path to the safetensors weights file
/// * `config` - Model configuration
/// * `device` - Device to load model on
///
/// # Errors
/// Returns an error if file loading or model construction fails.
///
/// # Safety
/// Uses memory-mapped file loading for efficiency. This is safe as long as
/// the file is not modified while the model is in use.
pub fn load_model(
    path: &std::path::Path,
    config: ModelConfig,
    device: &Device,
) -> Result<AsciiGPT> {
    let data = std::fs::read(path)?;
    let vb = VarBuilder::from_buffered_safetensors(data, DType::F32, device)?;
    AsciiGPT::new(config, vb)
}

/// Load an AsciiGPT model from in-memory safetensors bytes.
///
/// # Errors
/// Returns an error if the buffer cannot be parsed as safetensors or model construction fails.
pub fn load_model_from_bytes(
    data: &[u8],
    config: ModelConfig,
    device: &Device,
) -> Result<AsciiGPT> {
    let vb = VarBuilder::from_buffered_safetensors(data.to_vec(), DType::F32, device)?;
    AsciiGPT::new(config, vb)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full tests require weight files. These are structural tests only.

    #[test]
    fn test_config_validation() {
        let config = ModelConfig::medium();
        assert!(config.validate().is_ok());

        let bad_config = ModelConfig {
            n_embd: 100,
            n_head: 3, // 100 / 3 is not integer
            ..ModelConfig::medium()
        };
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_causal_mask() {
        let device = Device::Cpu;
        let mask = create_causal_mask(4, &device).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_forward_shapes_with_random_init() {
        let device = Device::Cpu;
        let config = ModelConfig::small();
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model = AsciiGPT::new(config.clone(), vb).unwrap();
        let input = Tensor::zeros((2, 8), DType::U32, &device).unwrap();

        let logits = model.forward(&input).unwrap();
        assert_eq!(logits.dims(), &[2, 8, config.vocab_size]);
    }
}
