#[test]
fn crate_boots() {
    let _tok = ascii_gen::tokenizer::ascii::AsciiTokenizer::new();
    let _cfg = ascii_gen::model::config::ModelConfig::default();
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

    let bytes = serialize(tensors.iter(), &None).expect("serialize");

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
