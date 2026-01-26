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

    // ==================== top_p boundary tests ====================

    #[test]
    fn test_top_p_zero_keeps_only_highest() {
        // top_p=0.0 should keep only the highest probability token
        let logits = vec![0.0, 10.0, 9.0, 1.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        // With top_p=0.0, should keep only highest (index 1)
        let sampled = sample_from_logits(&logits, 1.0, 0, 0.0, &mut rng);
        assert_eq!(sampled, 1, "top_p=0.0 should select argmax");
    }

    #[test]
    fn test_top_p_one_keeps_all() {
        // top_p=1.0 disables nucleus sampling (keeps all tokens)
        let logits = vec![1.0, 1.0, 1.0, 1.0]; // uniform
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        // With top_p=1.0 and uniform logits, any token is possible
        let sampled = sample_from_logits(&logits, 1.0, 0, 1.0, &mut rng);
        assert!(sampled < 4, "should return valid token index");
    }

    #[test]
    fn test_top_p_slightly_above_one_keeps_all() {
        // top_p > 1.0 should also keep all tokens (no filtering)
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let sampled = sample_from_logits(&logits, 1.0, 0, 1.5, &mut rng);
        assert!(sampled < 4, "top_p > 1.0 should return valid token");
    }

    #[test]
    fn test_top_p_negative_keeps_all() {
        // top_p < 0.0 should also keep all tokens (no filtering)
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let sampled = sample_from_logits(&logits, 1.0, 0, -0.5, &mut rng);
        assert!(sampled < 4, "top_p < 0 should return valid token");
    }

    #[test]
    fn test_top_p_point_five_keeps_top_half_probability() {
        // top_p=0.5 should keep tokens until cumulative prob > 0.5
        let logits = vec![10.0, 5.0, 0.0, -5.0]; // descending order of prob
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        // Index 0 has highest prob (~99%), should always be selected with top_p=0.5
        let sampled = sample_from_logits(&logits, 1.0, 0, 0.5, &mut rng);
        assert_eq!(
            sampled, 0,
            "top_p=0.5 with skewed logits should select top token"
        );
    }

    // ==================== all-filtered/all-inf tests ====================

    #[test]
    fn test_all_neg_inf_logits_returns_zero() {
        let logits = vec![f32::NEG_INFINITY; 4];
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let sampled = sample_from_logits(&logits, 1.0, 0, 1.0, &mut rng);
        // argmax of all -inf should return 0 (first index)
        assert_eq!(sampled, 0, "all -inf logits should return index 0");
    }

    #[test]
    fn test_all_nan_logits_returns_zero() {
        let logits = vec![f32::NAN; 4];
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let sampled = sample_from_logits(&logits, 1.0, 0, 1.0, &mut rng);
        // NaN is not finite, so argmax falls back to index 0
        assert_eq!(sampled, 0, "all NaN logits should return index 0");
    }

    #[test]
    fn test_mixed_inf_nan_prefers_finite() {
        let logits = vec![f32::NEG_INFINITY, f32::NAN, 1.0, f32::NEG_INFINITY];
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let sampled = sample_from_logits(&logits, 0.0, 0, 1.0, &mut rng);
        // Only index 2 is finite, so it should be selected
        assert_eq!(sampled, 2, "should select the only finite logit");
    }

    #[test]
    fn test_top_k_filters_all_but_one() {
        // top_k=1 should always return the argmax
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        for seed in 0..10 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let sampled = sample_from_logits(&logits, 1.0, 1, 1.0, &mut rng);
            assert_eq!(sampled, 1, "top_k=1 should always return argmax");
        }
    }

    // ==================== deterministic RNG tests ====================

    #[test]
    fn test_deterministic_sampling_same_seed() {
        let logits = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let seed = 12345u64;

        // Sample multiple times with the same seed
        let mut results = Vec::new();
        for _ in 0..5 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            results.push(sample_from_logits(&logits, 1.0, 0, 1.0, &mut rng));
        }

        // All results should be identical with the same seed
        assert!(
            results.iter().all(|&r| r == results[0]),
            "same seed should produce identical results: {results:?}"
        );
    }

    #[test]
    fn test_deterministic_sampling_different_seeds() {
        let logits = vec![1.0, 1.0, 1.0, 1.0, 1.0]; // uniform probs

        // Sample with different seeds
        let mut results = Vec::new();
        for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            results.push(sample_from_logits(&logits, 1.0, 0, 1.0, &mut rng));
        }

        // With uniform logits and 10 different seeds, we should see some variety
        let unique: std::collections::HashSet<_> = results.iter().collect();
        assert!(
            unique.len() > 1,
            "different seeds should produce varied results with uniform logits: {results:?}"
        );
    }

    #[test]
    fn test_sequential_samples_differ() {
        let logits = vec![1.0, 1.0, 1.0, 1.0, 1.0]; // uniform probs
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Take multiple samples from the same RNG
        let mut results = Vec::new();
        for _ in 0..20 {
            results.push(sample_from_logits(&logits, 1.0, 0, 1.0, &mut rng));
        }

        // Should see variety in sequential samples
        let unique: std::collections::HashSet<_> = results.iter().collect();
        assert!(
            unique.len() > 1,
            "sequential samples should show variety: {results:?}"
        );
    }

    #[test]
    fn test_temperature_affects_distribution() {
        let logits = vec![0.0, 1.0, 2.0];
        let seed = 42u64;

        // High temperature (more uniform)
        let mut high_temp_results = Vec::new();
        for i in 0..50 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed + i);
            high_temp_results.push(sample_from_logits(&logits, 2.0, 0, 1.0, &mut rng));
        }

        // Low temperature (more peaked)
        let mut low_temp_results = Vec::new();
        for i in 0..50 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed + i);
            low_temp_results.push(sample_from_logits(&logits, 0.5, 0, 1.0, &mut rng));
        }

        // Low temperature should favor index 2 (highest logit) more often
        let low_temp_top_count = low_temp_results.iter().filter(|&&x| x == 2).count();
        let high_temp_top_count = high_temp_results.iter().filter(|&&x| x == 2).count();

        assert!(
            low_temp_top_count >= high_temp_top_count,
            "low temp should favor top token more: low={low_temp_top_count}, high={high_temp_top_count}"
        );
    }
}
