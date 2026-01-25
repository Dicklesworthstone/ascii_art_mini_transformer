//! Constrained generation loop.
//!
//! Implements a simple autoregressive loop using `model.forward_last`, applying constraints
//! and sampling each step.

use candle_core::{Result, Tensor};
use rand::SeedableRng;

use crate::inference::constraints::{ConstrainedDecoder, apply_constraints_to_logits};
use crate::inference::sampling::sample_from_logits;
use crate::model::AsciiGPT;
use crate::tokenizer::ascii::{AsciiTokenizer, SEP_ID};

/// Generation settings.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub max_width: usize,
    pub max_lines: usize,
    pub max_chars: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: Option<u64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 500,
            max_width: 80,
            max_lines: 50,
            max_chars: 4_000,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            seed: None,
        }
    }
}

/// Generate tokens with hard width/height/char constraints.
///
/// Returns the full token stream (prompt + generated).
///
/// # Errors
/// Returns an error if model inference or tensor conversions fail.
pub fn generate_constrained(
    model: &AsciiGPT,
    prompt_tokens: &[u32],
    cfg: &GenerationConfig,
    tokenizer: AsciiTokenizer,
) -> Result<Vec<u32>> {
    let mut tokens = prompt_tokens.to_vec();

    // Only start tracking constraints after the last <SEP>.
    let art_start = tokens
        .iter()
        .rposition(|&t| t == SEP_ID)
        .map_or(0, |idx| idx + 1);

    let mut decoder = ConstrainedDecoder::new(cfg.max_width, cfg.max_lines, cfg.max_chars);
    for &t in tokens.iter().skip(art_start) {
        decoder.update(t, tokenizer);
    }

    let mut rng = match cfg.seed {
        Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
        None => rand::rngs::StdRng::from_entropy(),
    };

    for _ in 0..cfg.max_new_tokens {
        // Crop to block size (like Python generate()).
        let block_size = model.config().block_size;
        let ctx = if tokens.len() > block_size {
            &tokens[tokens.len() - block_size..]
        } else {
            &tokens[..]
        };

        // Model expects (batch, seq_len).
        let input = Tensor::new(vec![ctx.to_vec()], model.device())?;
        let logits = model.forward_last(&input)?;
        let logits = logits.squeeze(0)?;
        let mut logits_vec = logits.to_vec1::<f32>()?;

        apply_constraints_to_logits(&mut logits_vec, &decoder, tokenizer);
        let next = sample_from_logits(&logits_vec, cfg.temperature, cfg.top_k, cfg.top_p, &mut rng);

        tokens.push(next);

        if next == tokenizer.eos_id() {
            break;
        }

        decoder.update(next, tokenizer);
        if decoder.should_stop() {
            break;
        }
    }

    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelConfig;

    #[test]
    fn test_generation_respects_max_width() {
        let device = candle_core::Device::Cpu;
        let config = ModelConfig::small();
        let varmap = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let model = AsciiGPT::new(config, vb).unwrap();

        let tok = AsciiTokenizer::new();
        let prompt = vec![tok.bos_id(), SEP_ID];

        let cfg = GenerationConfig {
            max_new_tokens: 32,
            max_width: 1,
            max_lines: 10,
            max_chars: 32,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: Some(0),
        };

        let out = generate_constrained(&model, &prompt, &cfg, tok).unwrap();
        let decoded = tok.decode(&out);
        for line in decoded.split('\n') {
            assert!(line.len() <= 1);
        }
    }
}
