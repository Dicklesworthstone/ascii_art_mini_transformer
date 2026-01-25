//! Sampling utilities for generation.
//!
//! Implements temperature scaling, top-k, top-p (nucleus) filtering, and multinomial sampling.

use std::cmp::Ordering;

use rand::Rng;

/// Sample a token ID from a logits vector.
#[must_use]
pub fn sample_from_logits(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng: &mut impl Rng,
) -> u32 {
    if temperature <= 0.0 {
        return argmax(logits);
    }

    let mut filtered = logits.to_vec();

    if (temperature - 1.0).abs() > 1e-6 {
        for v in &mut filtered {
            *v /= temperature;
        }
    }

    if top_k > 0 && top_k < filtered.len() {
        top_k_filter_in_place(&mut filtered, top_k);
    }

    if (0.0..1.0).contains(&top_p) {
        top_p_filter_in_place(&mut filtered, top_p);
    }

    let probs = softmax(&filtered);
    if probs.iter().all(|p| *p == 0.0) {
        return argmax(logits);
    }

    sample_multinomial(&probs, rng)
}

#[must_use]
#[allow(clippy::cast_possible_truncation)]
fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &v) in logits.iter().enumerate() {
        if v.is_finite() && v > best_val {
            best_val = v;
            best_idx = idx;
        }
    }
    best_idx as u32
}

fn top_k_filter_in_place(logits: &mut [f32], k: usize) {
    let mut sorted = logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();

    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
    let kth = *sorted
        .get(k.saturating_sub(1))
        .unwrap_or(&f32::NEG_INFINITY);

    for v in logits {
        if *v < kth {
            *v = f32::NEG_INFINITY;
        }
    }
}

fn top_p_filter_in_place(logits: &mut [f32], top_p: f32) {
    let mut idx_logits = logits.iter().copied().enumerate().collect::<Vec<_>>();

    idx_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let max_logit = idx_logits
        .iter()
        .find_map(|(_, v)| v.is_finite().then_some(*v))
        .unwrap_or(f32::NEG_INFINITY);

    let mut exps = Vec::with_capacity(idx_logits.len());
    let mut sum = 0.0_f32;
    for &(_, v) in &idx_logits {
        let e = if v.is_finite() {
            (v - max_logit).exp()
        } else {
            0.0
        };
        sum += e;
        exps.push(e);
    }
    if sum == 0.0 {
        return;
    }

    let mut keep = vec![false; logits.len()];
    let mut cum = 0.0_f32;
    for ((idx, _), e) in idx_logits.iter().zip(exps.iter()) {
        cum += e / sum;
        keep[*idx] = true;
        if cum > top_p {
            break;
        }
    }

    if !keep.iter().any(|v| *v) {
        keep[idx_logits[0].0] = true;
    }

    for (i, v) in logits.iter_mut().enumerate() {
        if !keep[i] {
            *v = f32::NEG_INFINITY;
        }
    }
}

#[must_use]
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);

    if !max_logit.is_finite() {
        return vec![0.0; logits.len()];
    }

    let mut exps = Vec::with_capacity(logits.len());
    let mut sum = 0.0_f32;
    for &v in logits {
        let e = if v.is_finite() {
            (v - max_logit).exp()
        } else {
            0.0
        };
        sum += e;
        exps.push(e);
    }

    if sum == 0.0 {
        return vec![0.0; logits.len()];
    }

    exps.into_iter().map(|e| e / sum).collect()
}

#[must_use]
#[allow(clippy::cast_possible_truncation)]
fn sample_multinomial(probs: &[f32], rng: &mut impl Rng) -> u32 {
    let r: f32 = rng.r#gen();
    let mut cum = 0.0_f32;
    for (idx, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return idx as u32;
        }
    }
    // Numerical edge case: return last non-zero prob, else 0.
    probs.iter().rposition(|p| *p > 0.0).unwrap_or(0) as u32
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;

    #[test]
    fn test_temperature_zero_is_greedy() {
        let logits = vec![0.0, 1.0, 0.5];
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let sampled = sample_from_logits(&logits, 0.0, 0, 1.0, &mut rng);
        assert_eq!(sampled, 1);
    }

    #[test]
    fn test_top_k_filters() {
        let logits = vec![0.0, 10.0, 9.0, 1.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        // top_k=1 should always pick argmax (index 1) after filtering.
        let sampled = sample_from_logits(&logits, 1.0, 1, 1.0, &mut rng);
        assert_eq!(sampled, 1);
    }
}
