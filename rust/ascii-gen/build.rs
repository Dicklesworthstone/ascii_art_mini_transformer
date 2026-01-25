use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=ASCII_GEN_EXPORT_DIR");
    println!("cargo:rerun-if-env-changed=ASCII_GEN_MODEL_PATH");
    println!("cargo:rerun-if-env-changed=ASCII_GEN_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=ASCII_GEN_TOKENIZER_PATH");
    println!("cargo:rerun-if-env-changed=ASCII_GEN_QUANT_CONFIG_PATH");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR must be set"));
    let dest_path = out_dir.join("embedded_assets.rs");

    let embedded_feature_enabled = env::var_os("CARGO_FEATURE_EMBEDDED_WEIGHTS").is_some();
    let generated = if embedded_feature_enabled {
        generate_embedded_assets().unwrap_or_else(|err| {
            println!("cargo:warning=embedded-weights enabled but assets not embedded: {err}");
            generate_stub("embedded-weights enabled but assets not found; see build warnings")
        })
    } else {
        generate_stub("embedded-weights feature disabled")
    };

    fs::write(&dest_path, generated).expect("write embedded_assets.rs");
}

fn generate_embedded_assets() -> anyhow::Result<String> {
    let project_root = project_root()?;
    let export_dir = resolve_path_env_or_default(
        &project_root,
        "ASCII_GEN_EXPORT_DIR",
        project_root.join("models/exported"),
    );

    let model_path = if let Some(model_path) =
        resolve_optional_env_path(&project_root, "ASCII_GEN_MODEL_PATH")
    {
        model_path
    } else {
        pick_default_model_path(&export_dir)?
    };

    let config_path = resolve_path_env_or_default(
        &project_root,
        "ASCII_GEN_CONFIG_PATH",
        model_path
            .parent()
            .unwrap_or(&export_dir)
            .join("config.json"),
    );
    let tokenizer_path = resolve_path_env_or_default(
        &project_root,
        "ASCII_GEN_TOKENIZER_PATH",
        model_path
            .parent()
            .unwrap_or(&export_dir)
            .join("tokenizer.json"),
    );
    let quant_config_path = resolve_path_env_or_default(
        &project_root,
        "ASCII_GEN_QUANT_CONFIG_PATH",
        model_path
            .parent()
            .unwrap_or(&export_dir)
            .join("quant_config.json"),
    );

    let model_path = model_path
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("model path {}: {e}", model_path.display()))?;
    let config_path = config_path
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("config path {}: {e}", config_path.display()))?;
    let tokenizer_path = tokenizer_path
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("tokenizer path {}: {e}", tokenizer_path.display()))?;

    let quant_config_path = if quant_config_path.exists() {
        Some(quant_config_path.canonicalize().map_err(|e| {
            anyhow::anyhow!("quant_config path {}: {e}", quant_config_path.display())
        })?)
    } else {
        None
    };

    println!("cargo:rerun-if-changed={}", model_path.display());
    println!("cargo:rerun-if-changed={}", config_path.display());
    println!("cargo:rerun-if-changed={}", tokenizer_path.display());
    if let Some(p) = &quant_config_path {
        println!("cargo:rerun-if-changed={}", p.display());
    }

    let (quant_present, quant_path, quant_decl) = if let Some(p) = &quant_config_path {
        (
            "true",
            format!("{p:?}"),
            format!("pub static QUANT_CONFIG_JSON: &str = include_str!({p:?});\n"),
        )
    } else {
        (
            "false",
            "\"\"".to_string(),
            "pub static QUANT_CONFIG_JSON: &str = \"\";\n".to_string(),
        )
    };

    Ok(format!(
        "\
pub const EMBEDDED_MODEL_PRESENT: bool = true;
pub const EMBEDDED_MODEL_REASON: &str = \"ok\";
pub const EMBEDDED_MODEL_PATH: &str = {model_path:?};
pub const EMBEDDED_CONFIG_PATH: &str = {config_path:?};
pub const EMBEDDED_TOKENIZER_PATH: &str = {tokenizer_path:?};
pub const EMBEDDED_QUANT_CONFIG_PRESENT: bool = {quant_present};
pub const EMBEDDED_QUANT_CONFIG_PATH: &str = {quant_path};

pub static MODEL_WEIGHTS: &[u8] = include_bytes!({model_path:?});
pub static MODEL_CONFIG_JSON: &str = include_str!({config_path:?});
pub static TOKENIZER_JSON: &str = include_str!({tokenizer_path:?});
{quant_decl}\
"
    ))
}

fn generate_stub(reason: &str) -> String {
    format!(
        "\
pub const EMBEDDED_MODEL_PRESENT: bool = false;
pub const EMBEDDED_MODEL_REASON: &str = {reason:?};
pub const EMBEDDED_MODEL_PATH: &str = \"\";
pub const EMBEDDED_CONFIG_PATH: &str = \"\";
pub const EMBEDDED_TOKENIZER_PATH: &str = \"\";
pub const EMBEDDED_QUANT_CONFIG_PRESENT: bool = false;
pub const EMBEDDED_QUANT_CONFIG_PATH: &str = \"\";

pub static MODEL_WEIGHTS: &[u8] = &[];
pub static MODEL_CONFIG_JSON: &str = \"\";
pub static TOKENIZER_JSON: &str = \"\";
pub static QUANT_CONFIG_JSON: &str = \"\";
"
    )
}

fn project_root() -> anyhow::Result<PathBuf> {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"));
    let root = manifest_dir.join("../..").canonicalize().map_err(|e| {
        anyhow::anyhow!("resolve project root from {}: {e}", manifest_dir.display())
    })?;
    Ok(root)
}

fn resolve_optional_env_path(project_root: &Path, key: &str) -> Option<PathBuf> {
    let raw = env::var_os(key)?;
    let p = PathBuf::from(raw);
    Some(if p.is_relative() {
        project_root.join(p)
    } else {
        p
    })
}

fn resolve_path_env_or_default(project_root: &Path, key: &str, default: PathBuf) -> PathBuf {
    resolve_optional_env_path(project_root, key).unwrap_or(default)
}

fn pick_default_model_path(export_dir: &Path) -> anyhow::Result<PathBuf> {
    // Prefer float weights by default.
    //
    // Users can explicitly opt into quantized weights via `ASCII_GEN_MODEL_PATH`.
    let candidates = [
        "model.safetensors",
        "model_int8.safetensors",
        "model_int4.safetensors",
    ];

    for name in candidates {
        let p = export_dir.join(name);
        if p.exists() {
            return Ok(p);
        }
    }

    Err(anyhow::anyhow!(
        "no model weights found in {} (expected one of: {})",
        export_dir.display(),
        candidates.join(", ")
    ))
}
