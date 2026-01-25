//! Model quality test suite.
//!
//! Tests for model output quality, constraint adherence, diversity, and performance.
//! These tests require trained model weights to run fully.

use ascii_gen::inference::generate::GenerationConfig;
use ascii_gen::model::{AsciiGPT, ModelConfig};
use ascii_gen::tokenizer::{AsciiTokenizer, SEP_ID};
use std::collections::HashSet;
use std::path::Path;

/// Standard benchmark prompts for consistent evaluation.
const BENCHMARK_PROMPTS: &[&str] = &[
    // Animals (test detail)
    "cat", "dog", "snake", "fish", "bird", // Objects (test structure)
    "house", "car", "tree", "star", "heart", // Banners (test text rendering)
    "HELLO", "ASCII", "TEST", // Scenes (test complexity)
    "sunset", "mountain", // Abstract (test creativity)
    "pattern", "border",
];

/// Decode only the art portion of the token sequence (after SEP).
///
/// The full sequence is: `<BOS> <WIDTH> ... <SEP> {art} <EOS>`
/// We only want to decode the {art} portion.
fn decode_art_only(tokens: &[u32], tok: AsciiTokenizer) -> String {
    let art_start = tokens
        .iter()
        .rposition(|&t| t == SEP_ID)
        .map_or(0, |idx| idx + 1);
    tok.decode(&tokens[art_start..])
}

/// Load model from the test weights path if available.
fn load_test_model() -> Option<AsciiGPT> {
    let weights_path = Path::new("test_data/model.safetensors");
    if !weights_path.exists() {
        return None;
    }

    let device = candle_core::Device::Cpu;
    let config = ModelConfig::default();
    ascii_gen::model::load_model(weights_path, config, &device).ok()
}

/// Check if test model weights are available.
#[allow(dead_code)]
fn model_available() -> bool {
    Path::new("test_data/model.safetensors").exists()
}

// ==================== Constraint Adherence Tests ====================

/// Test that width constraints are never violated.
#[test]
#[ignore = "requires model weights"]
fn test_width_constraint_strict() {
    let Some(model) = load_test_model() else {
        eprintln!("Skipping: model weights not available");
        return;
    };

    let tok = AsciiTokenizer::new();
    let mut violations = 0;
    let widths = [20, 40, 60, 80];

    for width in widths {
        let cfg = GenerationConfig {
            max_width: width,
            max_lines: 20,
            max_chars: 1000,
            temperature: 0.7,
            ..Default::default()
        };

        for _ in 0..250 {
            let prompt_tokens = tok.encode_prompt(width, 20, "art", "test");
            let result = ascii_gen::inference::generate::generate_constrained(
                &model,
                &prompt_tokens,
                &cfg,
                tok,
            );

            if let Ok(tokens) = result {
                let art = decode_art_only(&tokens, tok);
                for line in art.lines() {
                    if line.len() > width {
                        violations += 1;
                    }
                }
            }
        }
    }

    assert_eq!(
        violations, 0,
        "Width constraint violated {} times",
        violations
    );
}

/// Test that height constraints are respected.
#[test]
#[ignore = "requires model weights"]
fn test_height_constraint_strict() {
    let Some(model) = load_test_model() else {
        eprintln!("Skipping: model weights not available");
        return;
    };

    let tok = AsciiTokenizer::new();
    let mut violations = 0;
    let heights = [5, 10, 20, 30];

    for max_lines in heights {
        let cfg = GenerationConfig {
            max_width: 80,
            max_lines,
            max_chars: 2000,
            temperature: 0.7,
            ..Default::default()
        };

        for _ in 0..250 {
            let prompt_tokens = tok.encode_prompt(80, max_lines, "art", "test");
            let result = ascii_gen::inference::generate::generate_constrained(
                &model,
                &prompt_tokens,
                &cfg,
                tok,
            );

            if let Ok(tokens) = result {
                let art = decode_art_only(&tokens, tok);
                let line_count = art.lines().count();
                if line_count > max_lines {
                    violations += 1;
                }
            }
        }
    }

    assert_eq!(
        violations, 0,
        "Height constraint violated {} times",
        violations
    );
}

/// Test that only valid ASCII characters are generated.
#[test]
#[ignore = "requires model weights"]
fn test_character_set_constraint() {
    let Some(model) = load_test_model() else {
        eprintln!("Skipping: model weights not available");
        return;
    };

    let tok = AsciiTokenizer::new();
    let cfg = GenerationConfig::default();
    let mut non_ascii_count = 0;

    for _ in 0..100 {
        let prompt_tokens = tok.encode_prompt(80, 20, "art", "test");
        let result =
            ascii_gen::inference::generate::generate_constrained(&model, &prompt_tokens, &cfg, tok);

        if let Ok(tokens) = result {
            let art = decode_art_only(&tokens, tok);
            for ch in art.chars() {
                if ch != '\n' && !(' '..='~').contains(&ch) {
                    non_ascii_count += 1;
                }
            }
        }
    }

    assert_eq!(
        non_ascii_count, 0,
        "Generated {} non-ASCII characters",
        non_ascii_count
    );
}

// ==================== Diversity Tests ====================

/// Test that model produces diverse outputs for the same prompt.
#[test]
#[ignore = "requires model weights"]
fn test_output_diversity() {
    let Some(model) = load_test_model() else {
        eprintln!("Skipping: model weights not available");
        return;
    };

    let tok = AsciiTokenizer::new();
    let cfg = GenerationConfig {
        temperature: 0.8,
        ..Default::default()
    };

    for prompt in BENCHMARK_PROMPTS {
        let mut outputs = HashSet::new();
        for _ in 0..10 {
            let prompt_tokens = tok.encode_prompt(40, 20, "art", prompt);
            let result = ascii_gen::inference::generate::generate_constrained(
                &model,
                &prompt_tokens,
                &cfg,
                tok,
            );

            if let Ok(tokens) = result {
                outputs.insert(decode_art_only(&tokens, tok));
            }
        }

        assert!(
            outputs.len() >= 8,
            "Low diversity for '{}': only {}/10 unique outputs",
            prompt,
            outputs.len()
        );
    }
}

// ==================== Structural Validity Tests ====================

/// Test that art outputs have valid structure.
#[test]
#[ignore = "requires model weights"]
fn test_structural_validity() {
    let Some(model) = load_test_model() else {
        eprintln!("Skipping: model weights not available");
        return;
    };

    let tok = AsciiTokenizer::new();
    let cfg = GenerationConfig::default();

    for prompt in BENCHMARK_PROMPTS.iter().take(5) {
        let prompt_tokens = tok.encode_prompt(40, 20, "art", prompt);
        let result =
            ascii_gen::inference::generate::generate_constrained(&model, &prompt_tokens, &cfg, tok);

        if let Ok(tokens) = result {
            let art = decode_art_only(&tokens, tok);
            let lines: Vec<_> = art.lines().collect();
            let non_whitespace: usize = art.chars().filter(|c| !c.is_whitespace()).count();

            // Multi-line output
            assert!(
                lines.len() >= 3,
                "Art for '{}' has only {} lines (expected >= 3)",
                prompt,
                lines.len()
            );

            // Non-trivial content
            assert!(
                non_whitespace >= 20,
                "Art for '{}' has only {} non-whitespace chars (expected >= 20)",
                prompt,
                non_whitespace
            );
        }
    }
}

// ==================== Performance Tests ====================

/// Test that generation completes within acceptable time.
#[test]
#[ignore = "requires model weights"]
fn test_generation_speed() {
    let Some(model) = load_test_model() else {
        eprintln!("Skipping: model weights not available");
        return;
    };

    let tok = AsciiTokenizer::new();
    let cfg = GenerationConfig {
        max_width: 40,
        max_lines: 20,
        max_chars: 500,
        ..Default::default()
    };

    let mut times = Vec::new();
    for _ in 0..20 {
        let prompt_tokens = tok.encode_prompt(40, 20, "art", "benchmark");
        let start = std::time::Instant::now();
        let _ =
            ascii_gen::inference::generate::generate_constrained(&model, &prompt_tokens, &cfg, tok);
        times.push(start.elapsed().as_millis());
    }

    times.sort();
    let avg_ms: u128 = times.iter().sum::<u128>() / times.len() as u128;
    let p95_idx = (times.len() * 95) / 100;
    let p95_ms = times.get(p95_idx).copied().unwrap_or(0);

    eprintln!("Generation speed: avg={}ms, p95={}ms", avg_ms, p95_ms);

    // Performance targets
    assert!(
        avg_ms < 500,
        "Average generation too slow: {}ms (target: <500ms)",
        avg_ms
    );
    assert!(
        p95_ms < 1000,
        "P95 generation too slow: {}ms (target: <1000ms)",
        p95_ms
    );
}

// ==================== Benchmark Suite ====================

/// Run full benchmark suite with all prompts.
#[test]
#[ignore = "requires model weights"]
fn test_benchmark_all_prompts() {
    let Some(model) = load_test_model() else {
        eprintln!("Skipping: model weights not available");
        return;
    };

    let tok = AsciiTokenizer::new();
    let cfg = GenerationConfig::default();

    let mut success_count = 0;
    let mut failure_count = 0;

    for prompt in BENCHMARK_PROMPTS {
        let prompt_tokens = tok.encode_prompt(40, 20, "art", prompt);
        let result =
            ascii_gen::inference::generate::generate_constrained(&model, &prompt_tokens, &cfg, tok);

        match result {
            Ok(tokens) => {
                let art = decode_art_only(&tokens, tok);
                if !art.trim().is_empty() {
                    success_count += 1;
                    eprintln!(
                        "✓ '{}': {} chars, {} lines",
                        prompt,
                        art.len(),
                        art.lines().count()
                    );
                } else {
                    failure_count += 1;
                    eprintln!("✗ '{}': empty output", prompt);
                }
            }
            Err(e) => {
                failure_count += 1;
                eprintln!("✗ '{}': {}", prompt, e);
            }
        }
    }

    eprintln!(
        "\nBenchmark results: {}/{} successful",
        success_count,
        BENCHMARK_PROMPTS.len()
    );
    assert_eq!(failure_count, 0, "Failed on {} prompts", failure_count);
}
