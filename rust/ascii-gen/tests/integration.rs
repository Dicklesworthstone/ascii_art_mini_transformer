#[test]
fn crate_boots() {
    let _tok = ascii_gen::tokenizer::ascii::AsciiTokenizer::new();
    let _cfg = ascii_gen::model::config::ModelConfig::default();
}

#[test]
#[cfg(not(feature = "embedded-weights"))]
fn embedded_weights_reports_unavailable_when_feature_disabled() {
    assert!(!ascii_gen::weights::loader::embedded_available());
    let reason = ascii_gen::weights::loader::embedded_reason();
    assert_ne!(reason, "ok");
    assert!(!reason.trim().is_empty());
    assert!(
        reason.contains("feature disabled"),
        "unexpected embedded_reason(): {reason:?}"
    );
}

#[test]
fn quantized_safetensors_without_quant_config_returns_helpful_error() {
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use safetensors::Dtype;
    use safetensors::serialize;
    use safetensors::tensor::TensorView;

    let mut tensors: BTreeMap<String, TensorView<'_>> = BTreeMap::new();

    let int_data_bytes = [0u8];
    let int_data =
        TensorView::new(Dtype::I8, vec![1, 1], &int_data_bytes).expect("int_data tensor view");
    tensors.insert("blocks.0.attn.c_attn.weight.int_data".to_string(), int_data);

    let scale_bytes = 1.0_f32.to_le_bytes();
    let scale = TensorView::new(Dtype::F32, vec![1], &scale_bytes).expect("scale tensor view");
    tensors.insert("blocks.0.attn.c_attn.weight.scale".to_string(), scale);

    let bytes = serialize(tensors.iter(), None).expect("serialize");

    let mut path: PathBuf = std::env::temp_dir();
    let uniq = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("timestamp")
        .as_nanos();
    path.push(format!("ascii_gen_quant_{uniq}.safetensors"));
    std::fs::write(&path, bytes).expect("write safetensors");

    let device = candle_core::Device::Cpu;
    let err = ascii_gen::weights::loader::load_external_model(&path, &device)
        .err()
        .expect("quantized weights without quant_config.json should error");

    assert!(
        err.chain()
            .any(|e| e.to_string().contains("quant_config.json")),
        "unexpected error chain: {err:?}"
    );
    assert!(
        err.chain()
            .any(|e| e.to_string().contains("python/train/export.py")),
        "unexpected error chain: {err:?}"
    );
}

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return f32::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
        / (a.len() as f32)
}

#[test]
fn quantized_int8_fixture_loads_and_logits_are_close() -> anyhow::Result<()> {
    use std::path::Path;

    use ascii_gen::tokenizer::AsciiTokenizer;
    use candle_core::Tensor;

    let device = candle_core::Device::Cpu;
    let base = Path::new("test_data/crossval");

    let model_f =
        ascii_gen::weights::loader::load_external_model(&base.join("model.safetensors"), &device)?;
    let model_q = ascii_gen::weights::loader::load_external_model(
        &base.join("model_int8.safetensors"),
        &device,
    )?;

    let tok = AsciiTokenizer::new();
    let mut ids = tok.encode_prompt(32, 12, "art", "cat");
    let max_len = model_f.config().block_size.min(64).min(ids.len());
    ids.truncate(max_len);
    let input = Tensor::from_vec(ids, (1, max_len), &device)?;
    let logits_f = model_f.forward_last(&input)?.squeeze(0)?.to_vec1::<f32>()?;
    let logits_q = model_q.forward_last(&input)?.squeeze(0)?.to_vec1::<f32>()?;

    let mae = mean_abs_diff(&logits_f, &logits_q);
    anyhow::ensure!(
        mae < 0.05,
        "INT8 logits drift too large: mae={mae} (len={})",
        logits_f.len()
    );
    Ok(())
}

#[test]
fn quantized_int4_fixture_loads_and_logits_are_reasonable() -> anyhow::Result<()> {
    use std::path::Path;

    use ascii_gen::tokenizer::AsciiTokenizer;
    use candle_core::Tensor;

    let device = candle_core::Device::Cpu;
    let base = Path::new("test_data/crossval");

    let model_f =
        ascii_gen::weights::loader::load_external_model(&base.join("model.safetensors"), &device)?;
    let model_q = ascii_gen::weights::loader::load_external_model(
        &base.join("model_int4.safetensors"),
        &device,
    )?;

    let tok = AsciiTokenizer::new();
    let mut ids = tok.encode_prompt(32, 12, "art", "cat");
    let max_len = model_f.config().block_size.min(64).min(ids.len());
    ids.truncate(max_len);
    let input = Tensor::from_vec(ids, (1, max_len), &device)?;
    let logits_f = model_f.forward_last(&input)?.squeeze(0)?.to_vec1::<f32>()?;
    let logits_q = model_q.forward_last(&input)?.squeeze(0)?.to_vec1::<f32>()?;

    let mae = mean_abs_diff(&logits_f, &logits_q);
    anyhow::ensure!(
        mae < 0.25,
        "INT4 logits drift too large: mae={mae} (len={})",
        logits_f.len()
    );
    Ok(())
}

// ==================== quant_config scheme selection tests ====================

#[test]
fn quant_scheme_selects_int8_by_filename() {
    let config = r#"{
        "schemes": {
            "int8": {
                "weights_file": "model_int8.safetensors",
                "precision": "int8",
                "format_version": 1,
                "quantized_layers": {}
            },
            "int4": {
                "weights_file": "model_int4.safetensors",
                "precision": "int4",
                "format_version": 1,
                "quantized_layers": {}
            }
        }
    }"#;

    let scheme =
        ascii_gen::weights::quantized::select_scheme_for_weights(config, "model_int8.safetensors")
            .expect("should select int8 scheme");
    assert_eq!(scheme.weights_file(), "model_int8.safetensors");
}

#[test]
fn quant_scheme_selects_int4_by_filename() {
    let config = r#"{
        "schemes": {
            "int8": {
                "weights_file": "model_int8.safetensors",
                "precision": "int8",
                "format_version": 1,
                "quantized_layers": {}
            },
            "int4": {
                "weights_file": "model_int4.safetensors",
                "precision": "int4",
                "format_version": 1,
                "quantized_layers": {}
            }
        }
    }"#;

    let scheme =
        ascii_gen::weights::quantized::select_scheme_for_weights(config, "model_int4.safetensors")
            .expect("should select int4 scheme");
    assert_eq!(scheme.weights_file(), "model_int4.safetensors");
}

#[test]
fn quant_scheme_errors_when_no_matching_scheme() {
    let config = r#"{
        "schemes": {
            "int8": {
                "weights_file": "model_int8.safetensors",
                "precision": "int8",
                "format_version": 1,
                "quantized_layers": {}
            }
        }
    }"#;

    let err =
        ascii_gen::weights::quantized::select_scheme_for_weights(config, "model_int4.safetensors")
            .expect_err("should fail when no matching scheme");

    let msg = err.to_string();
    assert!(
        msg.contains("model_int4.safetensors"),
        "error should mention the requested filename: {msg}"
    );
    assert!(
        msg.contains("model_int8.safetensors"),
        "error should list available schemes: {msg}"
    );
}

#[test]
fn quant_scheme_errors_on_invalid_json() {
    let config = "{ invalid json }";

    let err =
        ascii_gen::weights::quantized::select_scheme_for_weights(config, "model_int8.safetensors")
            .expect_err("should fail on invalid JSON");

    let msg = format!("{err:?}");
    assert!(
        msg.contains("parse quant_config"),
        "error should mention parsing: {msg}"
    );
}

#[test]
fn quant_scheme_errors_on_empty_schemes() {
    let config = r#"{ "schemes": {} }"#;

    let err =
        ascii_gen::weights::quantized::select_scheme_for_weights(config, "model_int8.safetensors")
            .expect_err("should fail when schemes is empty");

    let msg = err.to_string();
    assert!(
        msg.contains("model_int8.safetensors"),
        "error should mention the requested filename: {msg}"
    );
}

// ==================== weights loader config discovery tests ====================

#[test]
fn loader_config_uses_config_json_when_present() {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    let uniq = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("timestamp")
        .as_nanos();
    let tmp_dir: PathBuf = std::env::temp_dir().join(format!("ascii_cfg_present_{uniq}"));
    fs::create_dir_all(&tmp_dir).expect("create temp dir");

    // Create config.json with specific values (all required fields)
    let config_json = r#"{
        "vocab_size": 107,
        "block_size": 512,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 128,
        "max_rows": 50,
        "max_cols": 100,
        "newline_token_id": 7,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2
    }"#;
    fs::write(tmp_dir.join("config.json"), config_json).expect("write config.json");

    // Create dummy model path (file doesn't need to exist for config loading)
    let model_path = tmp_dir.join("model.safetensors");

    let cfg = ascii_gen::weights::loader::load_config_for_model(&model_path)
        .expect("load config from existing config.json");

    assert_eq!(cfg.vocab_size, 107);
    assert_eq!(cfg.block_size, 512);
    assert_eq!(cfg.n_layer, 4);
    assert_eq!(cfg.n_head, 4);
    assert_eq!(cfg.n_embd, 128);
    assert_eq!(cfg.max_rows, 50);
    assert_eq!(cfg.max_cols, 100);
}

#[test]
fn loader_config_falls_back_to_default_when_config_missing() {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    let uniq = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("timestamp")
        .as_nanos();
    let tmp_dir: PathBuf = std::env::temp_dir().join(format!("ascii_cfg_missing_{uniq}"));
    fs::create_dir_all(&tmp_dir).expect("create temp dir");

    // NO config.json in directory
    let model_path = tmp_dir.join("model.safetensors");

    let cfg = ascii_gen::weights::loader::load_config_for_model(&model_path)
        .expect("load default config when config.json missing");

    let default_cfg = ascii_gen::model::ModelConfig::default();
    assert_eq!(cfg.vocab_size, default_cfg.vocab_size);
    assert_eq!(cfg.block_size, default_cfg.block_size);
    assert_eq!(cfg.n_layer, default_cfg.n_layer);
    assert_eq!(cfg.n_head, default_cfg.n_head);
    assert_eq!(cfg.n_embd, default_cfg.n_embd);
}

#[test]
fn loader_config_errors_on_invalid_json() {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    let uniq = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("timestamp")
        .as_nanos();
    let tmp_dir: PathBuf = std::env::temp_dir().join(format!("ascii_cfg_badjson_{uniq}"));
    fs::create_dir_all(&tmp_dir).expect("create temp dir");

    // Create invalid JSON config
    fs::write(tmp_dir.join("config.json"), "{ invalid json }").expect("write invalid config");
    let model_path = tmp_dir.join("model.safetensors");

    let err = ascii_gen::weights::loader::load_config_for_model(&model_path)
        .expect_err("should fail on invalid JSON");

    let msg = format!("{err:?}");
    assert!(
        msg.contains("parse") || msg.contains("config.json"),
        "error should mention parsing: {msg}"
    );
}

#[test]
fn loader_config_errors_on_invalid_config_values() {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    let uniq = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("timestamp")
        .as_nanos();
    let tmp_dir: PathBuf = std::env::temp_dir().join(format!("ascii_cfg_badval_{uniq}"));
    fs::create_dir_all(&tmp_dir).expect("create temp dir");

    // Create config with invalid values (n_embd not divisible by n_head)
    let config_json = r#"{
        "vocab_size": 107,
        "block_size": 512,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 127,
        "max_rows": 50,
        "max_cols": 100,
        "newline_token_id": 7,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2
    }"#;
    fs::write(tmp_dir.join("config.json"), config_json).expect("write invalid config");
    let model_path = tmp_dir.join("model.safetensors");

    let err = ascii_gen::weights::loader::load_config_for_model(&model_path)
        .expect_err("should fail on invalid config values");

    let msg = format!("{err:?}");
    assert!(
        msg.contains("n_embd") || msg.contains("divisible") || msg.contains("invalid"),
        "error should mention validation failure: {msg}"
    );
}

#[test]
fn loader_config_errors_when_vocab_size_zero() {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    let uniq = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("timestamp")
        .as_nanos();
    let tmp_dir: PathBuf = std::env::temp_dir().join(format!("ascii_cfg_zero_vocab_{uniq}"));
    fs::create_dir_all(&tmp_dir).expect("create temp dir");

    // Create config with vocab_size = 0
    let config_json = r#"{
        "vocab_size": 0,
        "block_size": 512,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 128,
        "max_rows": 50,
        "max_cols": 100,
        "newline_token_id": 7,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2
    }"#;
    fs::write(tmp_dir.join("config.json"), config_json).expect("write config with zero vocab");
    let model_path = tmp_dir.join("model.safetensors");

    let err = ascii_gen::weights::loader::load_config_for_model(&model_path)
        .expect_err("should fail when vocab_size is zero");

    let msg = format!("{err:?}");
    assert!(
        msg.contains("vocab_size") || msg.contains("positive") || msg.contains("invalid"),
        "error should mention vocab_size validation: {msg}"
    );
}

// ==================== CLI output format tests ====================

/// Helper to get the path to the built CLI binary.
fn cli_binary_path() -> std::path::PathBuf {
    // Prefer cargo-provided paths when available.
    //
    // Cargo sets CARGO_BIN_EXE_<name> for integration tests, but the exact key can vary by
    // platform/tooling (notably for hyphenated bin names), so try a couple.
    for key in ["CARGO_BIN_EXE_ascii-gen", "CARGO_BIN_EXE_ascii_gen"] {
        if let Ok(path) = std::env::var(key) {
            let path = std::path::PathBuf::from(path);
            if path.exists() {
                return path;
            }
        }
    }

    // Fallback: derive from target dir layout (best-effort).
    let mut path = std::env::current_exe().expect("current_exe");
    path.pop(); // Remove test binary name
    path.pop(); // Remove deps/
    path.push("ascii-gen");
    path
}

/// Helper to get the crossval model path.
fn crossval_model_path() -> std::path::PathBuf {
    std::path::PathBuf::from("test_data/crossval/model.safetensors")
}

#[test]
fn cli_info_prints_embedded_availability() {
    use std::process::Command;

    let output = Command::new(cli_binary_path())
        .arg("--info")
        .output()
        .expect("run cli --info");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Embedded weights:"),
        "--info should print embedded weights line: {stdout}"
    );
    // Should print either "yes" or "no"
    assert!(
        stdout.contains("yes") || stdout.contains("no"),
        "--info should indicate yes/no for embedded: {stdout}"
    );
    // Should include a reason in parentheses
    assert!(
        stdout.contains('(') && stdout.contains(')'),
        "--info should include reason in parentheses: {stdout}"
    );
}

#[test]
fn cli_info_prints_model_configuration() {
    use std::process::Command;

    let output = Command::new(cli_binary_path())
        .arg("--info")
        .output()
        .expect("run cli --info");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should print model info header
    assert!(
        stdout.contains("Model Info") || stdout.contains("Configuration"),
        "--info should print model info: {stdout}"
    );
    // Should print key config values
    assert!(
        stdout.contains("Vocabulary") || stdout.contains("vocab"),
        "--info should print vocabulary size: {stdout}"
    );
    assert!(
        stdout.contains("Layers") || stdout.contains("layer"),
        "--info should print layer count: {stdout}"
    );
}

#[test]
fn cli_format_json_produces_valid_json() {
    use std::process::Command;

    let model_path = crossval_model_path();
    if !model_path.exists() {
        eprintln!("Skipping: crossval model not available");
        return;
    }

    let output = Command::new(cli_binary_path())
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--format",
            "json",
            "--width",
            "30",
            "--max-lines",
            "10",
            "--max-chars",
            "50",
            "--temperature",
            "0",
            "--seed",
            "42",
            "test",
        ])
        .output()
        .expect("run cli --format json");

    assert!(
        output.status.success(),
        "CLI should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value =
        serde_json::from_str(&stdout).expect("--format json should produce valid JSON");

    // Check required fields
    assert!(
        json.get("prompt").is_some(),
        "JSON should have 'prompt' field"
    );
    assert!(
        json.get("style").is_some(),
        "JSON should have 'style' field"
    );
    assert!(json.get("art").is_some(), "JSON should have 'art' field");
    assert!(
        json.get("width").is_some(),
        "JSON should have 'width' field"
    );
    assert!(
        json.get("height").is_some(),
        "JSON should have 'height' field"
    );
    assert!(
        json.get("total_chars").is_some(),
        "JSON should have 'total_chars' field"
    );
    assert!(
        json.get("generation_time_ms").is_some(),
        "JSON should have 'generation_time_ms' field"
    );
    assert!(
        json.get("temperature").is_some(),
        "JSON should have 'temperature' field"
    );
    assert!(
        json.get("top_k").is_some(),
        "JSON should have 'top_k' field"
    );
    assert!(
        json.get("top_p").is_some(),
        "JSON should have 'top_p' field"
    );
    assert!(
        json.get("max_width").is_some(),
        "JSON should have 'max_width' field"
    );
    assert!(
        json.get("max_lines").is_some(),
        "JSON should have 'max_lines' field"
    );
    assert!(
        json.get("max_chars").is_some(),
        "JSON should have 'max_chars' field"
    );

    // Verify prompt matches
    assert_eq!(json["prompt"].as_str(), Some("test"));
    // Verify constraints match
    assert_eq!(json["max_width"].as_u64(), Some(30));
    assert_eq!(json["max_lines"].as_u64(), Some(10));
}

#[test]
fn cli_format_markdown_wraps_in_code_fences() {
    use std::process::Command;

    let model_path = crossval_model_path();
    if !model_path.exists() {
        eprintln!("Skipping: crossval model not available");
        return;
    }

    let output = Command::new(cli_binary_path())
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--format",
            "markdown",
            "--width",
            "30",
            "--max-lines",
            "10",
            "--max-chars",
            "50",
            "--temperature",
            "0",
            "--seed",
            "42",
            "test",
        ])
        .output()
        .expect("run cli --format markdown");

    assert!(
        output.status.success(),
        "CLI should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Markdown format should start with ``` and end with ```
    assert!(
        stdout.starts_with("```"),
        "--format markdown should start with code fence: {stdout}"
    );
    assert!(
        stdout.trim_end().ends_with("```"),
        "--format markdown should end with code fence: {stdout}"
    );
}

#[test]
fn cli_format_plain_outputs_raw_art() {
    use std::process::Command;

    let model_path = crossval_model_path();
    if !model_path.exists() {
        eprintln!("Skipping: crossval model not available");
        return;
    }

    let output = Command::new(cli_binary_path())
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--format",
            "plain",
            "--width",
            "30",
            "--max-lines",
            "10",
            "--max-chars",
            "50",
            "--temperature",
            "0",
            "--seed",
            "42",
            "test",
        ])
        .output()
        .expect("run cli --format plain");

    assert!(
        output.status.success(),
        "CLI should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Plain format should NOT have markdown fences or JSON structure
    assert!(
        !stdout.starts_with("```"),
        "--format plain should not have markdown fences"
    );
    assert!(
        !stdout.starts_with('{'),
        "--format plain should not be JSON"
    );
    // Should have some content
    assert!(
        !stdout.trim().is_empty(),
        "--format plain should produce output"
    );
}
