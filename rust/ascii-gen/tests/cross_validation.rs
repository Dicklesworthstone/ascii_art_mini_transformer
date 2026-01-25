use anyhow::{Context, Result};
use ascii_gen::model::{ModelConfig, load_model};
use ascii_gen::tokenizer::AsciiTokenizer;
use candle_core::{IndexOp, Tensor};
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct GoldenCaseMeta {
    prompt: String,
    width: usize,
    height: usize,
    style: String,
    seed: u64,
}

#[derive(Debug, Deserialize)]
struct GoldenPayload {
    case: GoldenCaseMeta,
    input_ids: Vec<u32>,
    logits_sum: f32,
    logits_first_10: Vec<f32>,
    argmax_token: u32,
}

#[derive(Debug, Deserialize)]
struct TokenizerPromptCase {
    prompt: String,
    width: usize,
    height: usize,
    style: String,
    ids: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct TokenizerTextCase {
    text: String,
    ids: Vec<u32>,
    decoded: String,
}

#[derive(Debug, Deserialize)]
struct TokenizerGolden {
    encode_inference_prompt: Vec<TokenizerPromptCase>,
    encode_text: Vec<TokenizerTextCase>,
}

fn load_crossval_model() -> Result<(ModelConfig, ascii_gen::model::AsciiGPT)> {
    let device = candle_core::Device::Cpu;
    let base = Path::new("test_data/crossval");

    let config_path = base.join("config.json");
    let config_text = fs::read_to_string(&config_path)
        .with_context(|| format!("read {}", config_path.display()))?;
    let config: ModelConfig = serde_json::from_str(&config_text)
        .with_context(|| format!("parse {}", config_path.display()))?;
    config
        .validate()
        .map_err(anyhow::Error::msg)
        .context("invalid ModelConfig")?;

    let weights_path = base.join("model.safetensors");
    let model = load_model(&weights_path, config.clone(), &device)
        .with_context(|| format!("load weights {}", weights_path.display()))?;

    Ok((config, model))
}

#[test]
fn tokenizer_matches_python_golden() -> Result<()> {
    let tok = AsciiTokenizer::new();
    let path = Path::new("test_data/crossval/tokenizer_golden.json");
    let text = fs::read_to_string(path).context("read tokenizer_golden.json")?;
    let golden: TokenizerGolden =
        serde_json::from_str(&text).context("parse tokenizer_golden.json")?;

    for case in golden.encode_inference_prompt {
        let ids = tok.encode_prompt(case.width, case.height, &case.style, &case.prompt);
        anyhow::ensure!(
            ids == case.ids,
            "encode_prompt mismatch for prompt={:?} style={:?} w={} h={}",
            case.prompt,
            case.style,
            case.width,
            case.height
        );
    }

    for case in golden.encode_text {
        let ids = tok.encode(&case.text);
        anyhow::ensure!(ids == case.ids, "encode mismatch for text={:?}", case.text);
        let decoded = tok.decode(&ids);
        anyhow::ensure!(
            decoded == case.decoded,
            "decode mismatch for text={:?}: got={:?} expected={:?}",
            case.text,
            decoded,
            case.decoded
        );
    }

    Ok(())
}

#[test]
fn model_logits_match_python_golden() -> Result<()> {
    let (_config, model) = load_crossval_model()?;
    let device = candle_core::Device::Cpu;

    let golden_dir = Path::new("test_data/crossval/golden");
    let mut entries: Vec<_> = fs::read_dir(golden_dir)
        .with_context(|| format!("read {}", golden_dir.display()))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("list {}", golden_dir.display()))?;
    entries.sort_by_key(|e| e.path());

    // These tolerances are intentionally loose to account for small numeric differences between
    // PyTorch and Candle implementations (matmul/softmax/GELU/layernorm).
    let sum_tol: f32 = 1e-2;
    let first10_tol: f32 = 1e-2;

    for entry in entries {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }

        let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let golden: GoldenPayload =
            serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;

        let t = golden.input_ids.len();
        anyhow::ensure!(t > 0, "empty input_ids in {}", path.display());

        let input = Tensor::from_vec(golden.input_ids.clone(), (1, t), &device)
            .with_context(|| format!("build input tensor for {}", path.display()))?;
        let logits = model
            .forward(&input)
            .with_context(|| format!("forward pass for {}", path.display()))?;

        let last = logits
            .i((0, t - 1, ..))
            .with_context(|| format!("slice last logits for {}", path.display()))?;
        let last_vec = last
            .to_vec1::<f32>()
            .with_context(|| format!("materialize logits for {}", path.display()))?;

        let rust_sum: f32 = last_vec.iter().sum();
        let sum_diff = (rust_sum - golden.logits_sum).abs();
        anyhow::ensure!(
            sum_diff <= sum_tol,
            "logits_sum mismatch for {:?}: rust={} python={} diff={} tol={}",
            golden.case,
            rust_sum,
            golden.logits_sum,
            sum_diff,
            sum_tol
        );

        anyhow::ensure!(
            last_vec.len() >= 10 && golden.logits_first_10.len() == 10,
            "expected first_10 length 10 for {}",
            path.display()
        );
        for (i, (rust_v, py_v)) in last_vec
            .iter()
            .take(10)
            .zip(golden.logits_first_10.iter())
            .enumerate()
        {
            let diff = (rust_v - py_v).abs();
            anyhow::ensure!(
                diff <= first10_tol,
                "logits_first_10 mismatch for {:?} idx={}: rust={} python={} diff={} tol={}",
                golden.case,
                i,
                rust_v,
                py_v,
                diff,
                first10_tol
            );
        }

        let (mut argmax, mut best) = (0u32, f32::NEG_INFINITY);
        for (i, &v) in last_vec.iter().enumerate() {
            if v > best {
                best = v;
                argmax = i as u32;
            }
        }
        anyhow::ensure!(
            argmax == golden.argmax_token,
            "argmax mismatch for {:?}: rust={} python={}",
            golden.case,
            argmax,
            golden.argmax_token
        );
    }

    Ok(())
}

#[derive(Debug, Deserialize)]
struct GreedyGoldenCase {
    case: GoldenCaseMeta,
    prompt_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    #[allow(dead_code)]
    full_sequence: Vec<u32>,
}

#[test]
fn greedy_generation_matches_python_golden() -> Result<()> {
    use ascii_gen::inference::constraints::{ConstrainedDecoder, apply_constraints_to_logits};

    let (_config, model) = load_crossval_model()?;
    let device = candle_core::Device::Cpu;
    let tok = AsciiTokenizer::new();

    let path = Path::new("test_data/crossval/greedy_golden.json");
    let text = fs::read_to_string(path).context("read greedy_golden.json")?;
    let cases: Vec<GreedyGoldenCase> =
        serde_json::from_str(&text).context("parse greedy_golden.json")?;

    for golden in cases {
        let max_new_tokens = golden.generated_ids.len();
        let mut tokens = golden.prompt_ids.clone();

        // Find where art starts (after last SEP token)
        let art_start = tokens
            .iter()
            .rposition(|&t| t == tok.sep_id())
            .map_or(0, |idx| idx + 1);

        let mut decoder = ConstrainedDecoder::new(golden.case.width, golden.case.height, 32);
        for &t in tokens.iter().skip(art_start) {
            decoder.update(t, tok);
        }

        let mut generated: Vec<u32> = Vec::new();

        for _ in 0..max_new_tokens {
            let block_size = model.config().block_size;
            let ctx = if tokens.len() > block_size {
                &tokens[tokens.len() - block_size..]
            } else {
                &tokens[..]
            };

            let input = Tensor::from_vec(ctx.to_vec(), (1, ctx.len()), &device)?;
            let logits = model.forward_last(&input)?;
            let logits = logits.squeeze(0)?;
            let mut logits_vec = logits.to_vec1::<f32>()?;

            apply_constraints_to_logits(&mut logits_vec, &decoder, tok);

            // Greedy: argmax
            let (argmax, _) = logits_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            let next_token = argmax as u32;

            generated.push(next_token);
            tokens.push(next_token);

            if next_token == tok.eos_id() {
                break;
            }

            decoder.update(next_token, tok);
            if decoder.should_stop() {
                break;
            }
        }

        anyhow::ensure!(
            generated == golden.generated_ids,
            "greedy generation mismatch for {:?}:\n  rust={:?}\n  python={:?}",
            golden.case,
            generated,
            golden.generated_ids
        );
    }

    Ok(())
}
