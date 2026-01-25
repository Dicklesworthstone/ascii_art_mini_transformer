//! Constrained decoding state.
//!
//! Tracks width/height/char limits during generation and provides helpers
//! to enforce hard constraints (force newline / EOS) and soft biasing.

use crate::tokenizer::ascii::AsciiTokenizer;

/// Stateful constraint tracker for generation.
#[derive(Debug, Clone)]
pub struct ConstrainedDecoder {
    pub max_width: usize,
    pub max_lines: usize,
    pub max_chars: usize,

    pub current_col: usize,
    pub current_line: usize,
    pub total_chars: usize,
}

impl ConstrainedDecoder {
    #[must_use]
    pub fn new(max_width: usize, max_lines: usize, max_chars: usize) -> Self {
        Self {
            max_width,
            max_lines,
            max_chars,
            current_col: 0,
            current_line: 0,
            total_chars: 0,
        }
    }

    /// Hard width constraint: force newline once we reach max width.
    #[must_use]
    pub fn should_force_newline(&self) -> bool {
        self.max_width > 0 && self.current_col >= self.max_width
    }

    /// Hard stop constraint: force EOS once we exceed line or char limits.
    #[must_use]
    pub fn should_stop(&self) -> bool {
        (self.max_lines > 0 && self.current_line >= self.max_lines)
            || (self.max_chars > 0 && self.total_chars >= self.max_chars)
    }

    /// Update decoder state after emitting a token.
    pub fn update(&mut self, token_id: u32, tokenizer: AsciiTokenizer) {
        if !tokenizer.is_output_token(token_id) {
            return;
        }

        self.total_chars += 1;

        if token_id == tokenizer.newline_id() {
            self.current_line += 1;
            self.current_col = 0;
        } else {
            self.current_col += 1;
        }
    }
}

/// Apply hard/soft constraints in-place to a logits vector.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub fn apply_constraints_to_logits(
    logits: &mut [f32],
    decoder: &ConstrainedDecoder,
    tokenizer: AsciiTokenizer,
) {
    // Only allow printable ASCII + newline + EOS during generation.
    for (idx, logit) in logits.iter_mut().enumerate() {
        let token_id = idx as u32;
        if token_id == tokenizer.eos_id() {
            continue;
        }
        if tokenizer.is_output_token(token_id) {
            continue;
        }
        *logit = f32::NEG_INFINITY;
    }

    if decoder.should_stop() {
        // Force EOS.
        for (idx, logit) in logits.iter_mut().enumerate() {
            let token_id = idx as u32;
            if token_id != tokenizer.eos_id() {
                *logit = f32::NEG_INFINITY;
            }
        }
        return;
    }

    // If we've hit max width on the last allowed line, end instead of emitting a newline
    // that would create an extra empty row.
    let last_allowed_line = decoder.max_lines.saturating_sub(1);
    if decoder.should_force_newline()
        && decoder.max_lines > 0
        && decoder.current_line >= last_allowed_line
    {
        for (idx, logit) in logits.iter_mut().enumerate() {
            let token_id = idx as u32;
            if token_id != tokenizer.eos_id() {
                *logit = f32::NEG_INFINITY;
            }
        }
        return;
    }

    if decoder.should_force_newline() {
        // Force newline.
        let nl = tokenizer.newline_id();
        for (idx, logit) in logits.iter_mut().enumerate() {
            let token_id = idx as u32;
            if token_id != nl {
                *logit = f32::NEG_INFINITY;
            }
        }
        return;
    }

    // Soft bias towards newline as we approach max width.
    if decoder.max_width > 0 {
        let ratio = decoder.current_col as f32 / decoder.max_width as f32;
        if ratio > 0.8 {
            let bias = (ratio - 0.8) * 10.0;
            let nl = tokenizer.newline_id() as usize;
            if let Some(v) = logits.get_mut(nl) {
                *v += bias;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_width_forces_newline() {
        let tok = AsciiTokenizer::new();
        let mut d = ConstrainedDecoder::new(2, 10, 100);
        d.update(tok.encode_char('a'), tok);
        d.update(tok.encode_char('b'), tok);
        assert!(d.should_force_newline());
    }

    #[test]
    fn test_last_line_width_forces_eos() {
        let tok = AsciiTokenizer::new();
        let mut d = ConstrainedDecoder::new(1, 1, 100);
        d.update(tok.encode_char('a'), tok);
        assert!(d.should_force_newline());

        let mut logits = vec![0.0_f32; tok.vocab_size() as usize];
        apply_constraints_to_logits(&mut logits, &d, tok);

        let eos = tok.eos_id() as usize;
        let nl = tok.newline_id() as usize;
        assert!(logits[eos].is_finite());
        assert!(!logits[nl].is_finite());
        assert_eq!(
            logits.iter().filter(|v| v.is_finite()).count(),
            1,
            "only EOS should be allowed at last-line width boundary"
        );
    }
}
