use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use safetensors::{Dtype as SafeDtype, SafeTensors};
use serde::Deserialize;

use crate::model::{AsciiGPT, ModelConfig};

#[derive(Debug, Deserialize)]
struct QuantConfigFile {
    schemes: HashMap<String, QuantScheme>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantScheme {
    weights_file: String,
    #[allow(dead_code)]
    precision: String,
    #[allow(dead_code)]
    format_version: u32,
    quantized_layers: HashMap<String, QuantLayer>,
}

impl QuantScheme {
    /// Returns the weights filename this scheme applies to.
    #[must_use]
    pub fn weights_file(&self) -> &str {
        &self.weights_file
    }
}

#[derive(Debug, Clone, Deserialize)]
struct QuantLayer {
    bits: u8,
    orig_shape: Vec<usize>,

    // INT4 export includes this, INT8 does not.
    #[serde(default)]
    orig_in_features: Option<usize>,

    // Helpful for validating dtype; non-essential for dequantization.
    #[serde(default)]
    int_data_dtype: Option<String>,

    // Present for INT4.
    #[serde(default)]
    packed: Option<bool>,
    #[serde(default)]
    pack_format: Option<String>,
}

pub(super) fn find_quantized_tensor_name(st: &SafeTensors<'_>) -> Option<String> {
    st.names()
        .into_iter()
        .find(|name| is_quantized_tensor_name(name))
        .cloned()
}

fn is_quantized_tensor_name(name: &str) -> bool {
    if name.ends_with(".int_data") {
        return true;
    }

    // Avoid clippy's file-extension heuristic; this is a tensor name, not a filename.
    name.rsplit_once('.')
        .is_some_and(|(_, suffix)| suffix == "scale")
}

/// Select the quantization scheme for the given weights filename.
///
/// Parses the `quant_config.json` content and returns the scheme whose
/// `weights_file` field matches the given filename.
///
/// # Errors
///
/// Returns an error if:
/// - The JSON cannot be parsed as a valid `quant_config.json`
/// - No scheme's `weights_file` matches the given filename
pub fn select_scheme_for_weights(
    quant_config_json: &str,
    weights_filename: &str,
) -> Result<QuantScheme> {
    let cfg: QuantConfigFile =
        serde_json::from_str(quant_config_json).context("parse quant_config")?;

    let mut candidates = Vec::new();
    for scheme in cfg.schemes.into_values() {
        if scheme.weights_file == weights_filename {
            return Ok(scheme);
        }
        candidates.push(scheme.weights_file);
    }

    candidates.sort();
    candidates.dedup();
    anyhow::bail!(
        "quant_config.json does not define a scheme for weights_file={weights_filename:?}. \
Available weights_file entries: {}",
        candidates.join(", ")
    );
}

pub(super) fn load_quantized_model(
    st: &SafeTensors<'_>,
    scheme: &QuantScheme,
    config: ModelConfig,
    device: &Device,
) -> Result<AsciiGPT> {
    let mut tensors: HashMap<String, Tensor> = HashMap::new();

    let quantized_keys: HashSet<&str> =
        scheme.quantized_layers.keys().map(String::as_str).collect();

    for name in st.names() {
        if is_quantized_tensor_name(name) {
            continue;
        }
        // Quantized exports shouldn't include the original `{name}.weight` tensors for quantized
        // layers, but avoid accidentally overriding a dequantized tensor if they do.
        if quantized_keys.contains(name.as_str()) {
            continue;
        }

        let view = st
            .tensor(name)
            .with_context(|| format!("read tensor {name:?}"))?;
        let dtype = match view.dtype() {
            SafeDtype::F32 => DType::F32,
            SafeDtype::F16 => DType::F16,
            SafeDtype::BF16 => DType::BF16,
            other => anyhow::bail!("unsupported dtype {other:?} for tensor {name:?}"),
        };
        let t = Tensor::from_raw_buffer(view.data(), dtype, view.shape(), device)
            .with_context(|| format!("load tensor {name:?}"))?;
        tensors.insert(name.clone(), t);
    }

    for (name, meta) in &scheme.quantized_layers {
        let Some((out_features, in_features)) = shape2(&meta.orig_shape) else {
            anyhow::bail!(
                "invalid orig_shape for {name:?}: expected 2 dims, got {:?}",
                meta.orig_shape
            );
        };

        let scales = read_scale_vector(st, name, out_features)
            .with_context(|| format!("read {name:?} scale"))?;

        let int_data_name = format!("{name}.int_data");
        let int_data = st
            .tensor(&int_data_name)
            .with_context(|| format!("read tensor {int_data_name:?}"))?;

        let w = match meta.bits {
            8 => dequantize_int8_per_row(
                &int_data,
                &scales,
                out_features,
                in_features,
                meta.int_data_dtype.as_deref(),
                name,
                device,
            )
            .with_context(|| format!("dequantize int8 for {name:?}"))?,
            4 => {
                let orig_in = meta.orig_in_features.unwrap_or(in_features);
                dequantize_int4_packed_per_row(
                    &int_data,
                    &scales,
                    out_features,
                    in_features,
                    orig_in,
                    meta.int_data_dtype.as_deref(),
                    meta.packed,
                    meta.pack_format.as_deref(),
                    name,
                    device,
                )
                .with_context(|| format!("dequantize int4 for {name:?}"))?
            }
            bits => anyhow::bail!("unsupported quant bits={bits} for {name:?}"),
        };

        if tensors.insert(name.clone(), w).is_some() {
            anyhow::bail!("duplicate tensor name {name:?} while inserting dequantized weight");
        }
    }

    let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
    let model = AsciiGPT::new(config, vb).context("build AsciiGPT from dequantized tensors")?;
    Ok(model)
}

fn shape2(shape: &[usize]) -> Option<(usize, usize)> {
    match shape {
        [a, b] => Some((*a, *b)),
        _ => None,
    }
}

fn read_scale_vector(st: &SafeTensors<'_>, base: &str, out_features: usize) -> Result<Vec<f32>> {
    let scale_name = format!("{base}.scale");
    let scale = st
        .tensor(&scale_name)
        .with_context(|| format!("read tensor {scale_name:?}"))?;
    if scale.dtype() != SafeDtype::F32 {
        anyhow::bail!("expected {scale_name:?} dtype F32, got {:?}", scale.dtype());
    }
    if scale.shape() != [out_features] {
        anyhow::bail!(
            "expected {scale_name:?} shape [{out_features}], got {:?}",
            scale.shape()
        );
    }

    let data = scale.data();
    if data.len() != out_features * 4 {
        anyhow::bail!(
            "expected {scale_name:?} byte length {}, got {}",
            out_features * 4,
            data.len()
        );
    }

    let mut out = Vec::with_capacity(out_features);
    for chunk in data.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn dequantize_int8_per_row(
    int_data: &safetensors::tensor::TensorView<'_>,
    scales: &[f32],
    out_features: usize,
    in_features: usize,
    expected_dtype: Option<&str>,
    name: &str,
    device: &Device,
) -> Result<Tensor> {
    if let Some(exp) = expected_dtype
        && exp != "int8"
    {
        anyhow::bail!("quant_config says {name:?} int_data_dtype={exp:?}, expected \"int8\"");
    }

    if int_data.dtype() != SafeDtype::I8 {
        anyhow::bail!(
            "expected {name:?}.int_data dtype I8, got {:?}",
            int_data.dtype()
        );
    }
    if int_data.shape() != [out_features, in_features] {
        anyhow::bail!(
            "expected {name:?}.int_data shape [{out_features}, {in_features}], got {:?}",
            int_data.shape()
        );
    }

    let q = int_data.data();
    if q.len() != out_features * in_features {
        anyhow::bail!(
            "expected {name:?}.int_data byte length {}, got {}",
            out_features * in_features,
            q.len()
        );
    }
    if scales.len() != out_features {
        anyhow::bail!(
            "expected {name:?}.scale length {out_features}, got {}",
            scales.len()
        );
    }

    let mut w = vec![0f32; out_features * in_features];
    for (r, &s) in scales.iter().enumerate() {
        let row = r * in_features;
        for c in 0..in_features {
            let qi = q[row + c].cast_signed();
            w[row + c] = f32::from(qi) * s;
        }
    }

    Tensor::from_vec(w, (out_features, in_features), device).context("build dequantized tensor")
}

#[allow(clippy::too_many_arguments)]
fn dequantize_int4_packed_per_row(
    int_data: &safetensors::tensor::TensorView<'_>,
    scales: &[f32],
    out_features: usize,
    in_features: usize,
    orig_in_features: usize,
    expected_dtype: Option<&str>,
    packed: Option<bool>,
    pack_format: Option<&str>,
    name: &str,
    device: &Device,
) -> Result<Tensor> {
    if let Some(exp) = expected_dtype
        && exp != "uint8"
    {
        anyhow::bail!("quant_config says {name:?} int_data_dtype={exp:?}, expected \"uint8\"");
    }
    if let Some(packed) = packed {
        anyhow::ensure!(packed, "expected quant_config packed=true for {name:?}");
    }
    if let Some(fmt) = pack_format {
        anyhow::ensure!(
            fmt == "u4u4_to_u8",
            "unsupported pack_format={fmt:?} for {name:?}"
        );
    }

    if int_data.dtype() != SafeDtype::U8 {
        anyhow::bail!(
            "expected {name:?}.int_data dtype U8, got {:?}",
            int_data.dtype()
        );
    }
    if int_data.shape().len() != 2 || int_data.shape()[0] != out_features {
        anyhow::bail!(
            "expected {name:?}.int_data shape [out, ceil(in/2)] with out={out_features}, got {:?}",
            int_data.shape()
        );
    }

    let packed_in = int_data.shape()[1];
    let expected_packed_in = orig_in_features.div_ceil(2);
    anyhow::ensure!(
        packed_in == expected_packed_in,
        "expected {name:?}.int_data packed_in={expected_packed_in} (ceil(orig_in/2)), got {packed_in}"
    );

    let q = int_data.data();
    let expected_len = out_features * packed_in;
    anyhow::ensure!(
        q.len() == expected_len,
        "expected {name:?}.int_data byte length {}, got {}",
        expected_len,
        q.len()
    );
    anyhow::ensure!(
        scales.len() == out_features,
        "expected {name:?}.scale length {out_features}, got {}",
        scales.len()
    );

    let mut w = vec![0f32; out_features * in_features];
    for (r, &s) in scales.iter().enumerate() {
        let row_packed = r * packed_in;
        let row_out = r * in_features;
        for j in 0..packed_in {
            let byte = q[row_packed + j];
            let lo = (byte & 0x0F).cast_signed() - 8;
            let hi = (byte >> 4).cast_signed() - 8;

            let idx0 = row_out + 2 * j;
            if idx0 < row_out + in_features {
                w[idx0] = f32::from(lo) * s;
            }
            let idx1 = row_out + 2 * j + 1;
            if idx1 < row_out + in_features {
                w[idx1] = f32::from(hi) * s;
            }
        }
    }

    Tensor::from_vec(w, (out_features, in_features), device).context("build dequantized tensor")
}
