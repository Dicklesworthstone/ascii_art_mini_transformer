//! Causal self-attention implementation.
//!
//! Multi-head causal self-attention matching the Python CausalSelfAttention class.

use candle_core::{D, Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::config::ModelConfig;

/// Multi-head causal self-attention.
///
/// Uses combined QKV projection for efficiency.
pub struct CausalSelfAttention {
    /// Combined query, key, value projection (n_embd -> 3 * n_embd)
    c_attn: Linear,
    /// Output projection (n_embd -> n_embd)
    c_proj: Linear,
    /// Number of attention heads
    n_head: usize,
    /// Embedding dimension
    n_embd: usize,
    /// Dimension per head
    head_dim: usize,
}

impl CausalSelfAttention {
    /// Create a new causal self-attention layer.
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
        let n_head = config.n_head;
        let head_dim = config.head_dim();

        // Combined QKV projection (no bias, matching Python)
        let c_attn = candle_nn::linear_no_bias(n_embd, 3 * n_embd, vb.pp("c_attn"))?;

        // Output projection (no bias, matching Python)
        let c_proj = candle_nn::linear_no_bias(n_embd, n_embd, vb.pp("c_proj"))?;

        Ok(Self {
            c_attn,
            c_proj,
            n_head,
            n_embd,
            head_dim,
        })
    }

    /// Forward pass for causal self-attention.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape (batch, seq_len, n_embd)
    /// * `mask` - Causal mask tensor
    ///
    /// # Returns
    /// Output tensor of shape (batch, seq_len, n_embd)
    ///
    /// # Errors
    /// Returns an error if tensor operations fail.
    #[allow(clippy::many_single_char_names)]
    pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, t, _c) = x.dims3()?;

        // Combined QKV projection: (B, T, C) -> (B, T, 3*C)
        let qkv = self.c_attn.forward(x)?;

        // Split into Q, K, V
        let q = qkv.narrow(D::Minus1, 0, self.n_embd)?;
        let k = qkv.narrow(D::Minus1, self.n_embd, self.n_embd)?;
        let v = qkv.narrow(D::Minus1, 2 * self.n_embd, self.n_embd)?;

        // Reshape for multi-head attention: (B, T, C) -> (B, n_head, T, head_dim)
        // Make contiguous after transpose for efficient matmul
        let q = q
            .reshape((b, t, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, t, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, t, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) * V
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let att = (q.matmul(&k_t)? * scale)?;

        // Apply causal mask (adding large negative values to masked positions)
        let att = att.broadcast_add(mask)?;

        // Softmax over last dimension
        let att = candle_nn::ops::softmax_last_dim(&att)?;

        // Apply attention to values: (B, n_head, T, T) @ (B, n_head, T, head_dim)
        let y = att.matmul(&v)?;

        // Reshape back: (B, n_head, T, head_dim) -> (B, T, C)
        let y = y
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, self.n_embd))?;

        // Output projection
        self.c_proj.forward(&y)
    }
}

/// Create a causal attention mask.
///
/// Returns a mask where position (i, j) is -inf if j > i, else 0.
/// This prevents attending to future tokens.
///
/// # Arguments
/// * `seq_len` - Sequence length
/// * `device` - Device to create tensor on
///
/// # Errors
/// Returns an error if tensor creation fails.
pub fn create_causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    // Build the mask manually: mask[i][j] = 0 if j <= i, else -inf
    let neg_inf = f32::NEG_INFINITY;
    let mut mask_data = Vec::with_capacity(seq_len * seq_len);

    for i in 0..seq_len {
        for j in 0..seq_len {
            if j <= i {
                mask_data.push(0.0f32);
            } else {
                mask_data.push(neg_inf);
            }
        }
    }

    let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), device)?;

    // Add batch and head dimensions: (T, T) -> (1, 1, T, T)
    mask.unsqueeze(0)?.unsqueeze(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask_shape() {
        let device = Device::Cpu;
        let mask = create_causal_mask(4, &device).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_causal_mask_values() {
        let device = Device::Cpu;
        let mask = create_causal_mask(3, &device).unwrap();
        let mask = mask.squeeze(0).unwrap().squeeze(0).unwrap();
        let values: Vec<Vec<f32>> = mask.to_vec2().unwrap();

        // Row 0: can only attend to position 0
        assert!(values[0][0].is_finite()); // 0
        assert!(values[0][1].is_infinite()); // -inf
        assert!(values[0][2].is_infinite()); // -inf

        // Row 1: can attend to positions 0, 1
        assert!(values[1][0].is_finite()); // 0
        assert!(values[1][1].is_finite()); // 0
        assert!(values[1][2].is_infinite()); // -inf

        // Row 2: can attend to all positions
        assert!(values[2][0].is_finite()); // 0
        assert!(values[2][1].is_finite()); // 0
        assert!(values[2][2].is_finite()); // 0
    }
}
