use std::path::Path;

use anyhow::{Context, Result};
use candle_core::Device;
use safetensors::SafeTensors;

use crate::model::{AsciiGPT, ModelConfig, load_model_from_bytes};

use super::embedded;
use super::quantized;

/// Return whether the current build includes embedded model assets.
#[must_use]
pub fn embedded_available() -> bool {
    embedded::EMBEDDED_MODEL_PRESENT
}

/// Return the build-time reason embedded assets are unavailable (or `"ok"`).
#[must_use]
pub fn embedded_reason() -> &'static str {
    embedded::EMBEDDED_MODEL_REASON
}

/// Load a model from embedded weights/config, if they were compiled in.
///
/// # Errors
/// Returns an error if embedded assets are present but invalid (e.g., config JSON parse failure,
/// weight loading error).
pub fn try_load_embedded_model(device: &Device) -> Result<Option<AsciiGPT>> {
    if !embedded::EMBEDDED_MODEL_PRESENT {
        return Ok(None);
    }

    let config: ModelConfig =
        serde_json::from_str(embedded::MODEL_CONFIG_JSON).context("parse embedded config.json")?;
    config
        .validate()
        .map_err(anyhow::Error::msg)
        .context("invalid embedded ModelConfig")?;

    let data = embedded::MODEL_WEIGHTS;
    let quant_override = if embedded::EMBEDDED_QUANT_CONFIG_PRESENT {
        Some(embedded::QUANT_CONFIG_JSON)
    } else {
        None
    };
    let model = load_bytes_auto(
        data,
        config,
        device,
        Some(Path::new(embedded::EMBEDDED_MODEL_PATH)),
        quant_override,
    )
    .context("load embedded weights")?;
    Ok(Some(model))
}

/// Load a model from a weights file on disk, reading `config.json` from the same directory when
/// available.
///
/// # Errors
/// Returns an error if the weights or config cannot be loaded.
pub fn load_external_model(model_path: &Path, device: &Device) -> Result<AsciiGPT> {
    let config = load_config_for_model(model_path).unwrap_or_else(|_| ModelConfig::default());
    let data = std::fs::read(model_path)
        .with_context(|| format!("read weights {}", model_path.display()))?;
    load_bytes_auto(&data, config, device, Some(model_path), None).context("load external weights")
}

fn load_config_for_model(model_path: &Path) -> Result<ModelConfig> {
    let Some(parent) = model_path.parent() else {
        return Ok(ModelConfig::default());
    };

    let config_path = parent.join("config.json");
    if !config_path.exists() {
        return Ok(ModelConfig::default());
    }

    let raw = std::fs::read_to_string(&config_path)
        .with_context(|| format!("read config {}", config_path.display()))?;
    let cfg: ModelConfig =
        serde_json::from_str(&raw).with_context(|| format!("parse {}", config_path.display()))?;
    cfg.validate()
        .map_err(anyhow::Error::msg)
        .context("invalid ModelConfig")?;
    Ok(cfg)
}

fn load_bytes_auto(
    data: &[u8],
    config: ModelConfig,
    device: &Device,
    model_path_hint: Option<&Path>,
    quant_config_override: Option<&str>,
) -> Result<AsciiGPT> {
    let Ok(safetensors) = SafeTensors::deserialize(data) else {
        // If it isn't safetensors, downstream loading will fail with a better error.
        return load_model_from_bytes(data, config, device).context("load weights bytes");
    };

    let Some(quantized_name) = quantized::find_quantized_tensor_name(&safetensors) else {
        return load_model_from_bytes(data, config, device).context("load float safetensors");
    };

    let quant_config_json = if let Some(json) = quant_config_override {
        if json.trim().is_empty() {
            load_quant_config_json(model_path_hint).context("load quant_config.json")?
        } else {
            json.to_string()
        }
    } else {
        load_quant_config_json(model_path_hint).context("load quant_config.json")?
    };
    let weights_filename = model_path_hint
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    if weights_filename.is_empty() {
        anyhow::bail!(
            "Quantized safetensors detected (found tensor {quantized_name:?}), \
but could not infer a weights filename to select a scheme from quant_config.json."
        );
    }
    let scheme = quantized::select_scheme_for_weights(&quant_config_json, weights_filename)
        .context("select quantization scheme")?;
    quantized::load_quantized_model(&safetensors, &scheme, config, device)
        .with_context(|| format!("load quantized safetensors (found tensor {quantized_name:?})"))
}

fn load_quant_config_json(model_path_hint: Option<&Path>) -> Result<String> {
    let Some(model_path) = model_path_hint else {
        anyhow::bail!(
            "Quantized safetensors detected, but no model path is available to locate quant_config.json."
        );
    };
    let dir = model_path.parent().unwrap_or_else(|| Path::new("."));
    let cfg_path = dir.join("quant_config.json");
    if !cfg_path.exists() {
        anyhow::bail!(
            "Quantized safetensors detected, but quant_config.json was not found at {}. \
Export quantized weights via `python/train/export.py --quantize int8|int4|both` which writes quant_config.json alongside the weights.",
            cfg_path.display()
        );
    }
    std::fs::read_to_string(&cfg_path).with_context(|| format!("read {}", cfg_path.display()))
}
