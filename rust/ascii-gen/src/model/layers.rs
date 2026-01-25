//! Common neural network layers.
//!
//! MLP and TransformerBlock implementations matching the Python model.

use candle_core::{Result, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};

use super::attention::CausalSelfAttention;
use super::config::ModelConfig;

/// Feed-forward network with GELU activation.
///
/// Standard transformer MLP with 4x hidden dimension expansion.
/// Matches Python: c_fc -> GELU -> c_proj
pub struct MLP {
    /// Up projection (n_embd -> 4 * n_embd)
    c_fc: Linear,
    /// Down projection (4 * n_embd -> n_embd)
    c_proj: Linear,
}

impl MLP {
    /// Create a new MLP layer.
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
        let hidden_dim = 4 * n_embd;

        // No bias, matching Python
        let c_fc = candle_nn::linear_no_bias(n_embd, hidden_dim, vb.pp("c_fc"))?;
        let c_proj = candle_nn::linear_no_bias(hidden_dim, n_embd, vb.pp("c_proj"))?;

        Ok(Self { c_fc, c_proj })
    }

    /// Forward pass through MLP.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape (batch, seq_len, n_embd)
    ///
    /// # Returns
    /// Output tensor of shape (batch, seq_len, n_embd)
    ///
    /// # Errors
    /// Returns an error if tensor operations fail.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.c_fc.forward(x)?;
        let h = h.gelu()?;
        self.c_proj.forward(&h)
    }
}

/// Transformer block with pre-norm architecture.
///
/// Structure: LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual
pub struct TransformerBlock {
    /// First layer norm (before attention)
    ln_1: LayerNorm,
    /// Causal self-attention
    attn: CausalSelfAttention,
    /// Second layer norm (before MLP)
    ln_2: LayerNorm,
    /// Feed-forward network
    mlp: MLP,
}

impl TransformerBlock {
    /// Create a new transformer block.
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

        let ln_1 =
            candle_nn::layer_norm(n_embd, candle_nn::LayerNormConfig::default(), vb.pp("ln_1"))?;
        let attn = CausalSelfAttention::new(config, vb.pp("attn"))?;
        let ln_2 =
            candle_nn::layer_norm(n_embd, candle_nn::LayerNormConfig::default(), vb.pp("ln_2"))?;
        let mlp = MLP::new(config, vb.pp("mlp"))?;

        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }

    /// Forward pass through the transformer block.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape (batch, seq_len, n_embd)
    /// * `mask` - Causal attention mask
    ///
    /// # Returns
    /// Output tensor of shape (batch, seq_len, n_embd)
    ///
    /// # Errors
    /// Returns an error if tensor operations fail.
    pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // Pre-norm attention with residual
        let h = self.ln_1.forward(x)?;
        let h = self.attn.forward(&h, mask)?;
        let x = (x + h)?;

        // Pre-norm MLP with residual
        let h = self.ln_2.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        x + h
    }
}
