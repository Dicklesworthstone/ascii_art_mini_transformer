//! ASCII Art Generator CLI
//!
//! A tiny transformer-based ASCII art generator.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use serde::Serialize;

use ascii_gen::inference::generate::{GenerationConfig, generate_constrained};
use ascii_gen::model::{AsciiGPT, ModelConfig};
use ascii_gen::tokenizer::{AsciiTokenizer, SEP_ID};
use ascii_gen::weights::loader as weights_loader;

/// Style of ASCII art to generate.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum Style {
    /// Realistic ASCII art with shading
    #[default]
    Art,
    /// FIGlet-style text banner
    Banner,
    /// Simple line drawing
    Simple,
    /// Detailed art with fine shading
    Detailed,
}

impl Style {
    /// Style name used by the tokenizer prompt format.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Art => "art",
            Self::Banner => "banner",
            Self::Simple => "simple",
            Self::Detailed => "detailed",
        }
    }
}

/// Output format for generated art.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum OutputFormat {
    /// Plain text output
    #[default]
    Plain,
    /// JSON with metadata
    Json,
    /// Markdown code block
    Markdown,
}

/// ASCII Art Generator - Create ASCII art using a tiny transformer model.
#[derive(Debug, Parser)]
#[command(
    name = "ascii-gen",
    version,
    about = "Generate ASCII art using a tiny transformer model",
    long_about = "A CPU-efficient transformer (~10-50MB) specialized for generating high-quality ASCII art.\n\n\
                  Uses character-level tokenization and 2D positional encoding for spatial reasoning."
)]
struct Cli {
    /// The subject/prompt for generation (e.g., "a cat", "mountains at sunset")
    #[arg(index = 1)]
    prompt: Option<String>,

    /// Maximum width in characters
    #[arg(short, long, default_value = "80")]
    width: usize,

    /// Maximum number of lines (height)
    #[arg(long, default_value = "50")]
    max_lines: usize,

    /// Maximum total characters
    #[arg(long, default_value = "4000")]
    max_chars: usize,

    /// Art style to generate
    #[arg(short, long, value_enum, default_value = "art")]
    style: Style,

    /// Sampling temperature (higher = more creative, lower = more focused)
    #[arg(short, long, default_value = "0.7")]
    temperature: f32,

    /// Top-k sampling (0 = disabled)
    #[arg(long, default_value = "50")]
    top_k: usize,

    /// Top-p (nucleus) sampling (0 or 1 = disabled)
    #[arg(long, default_value = "0.9")]
    top_p: f32,

    /// Random seed for reproducible generation
    #[arg(long)]
    seed: Option<u64>,

    /// Path to custom model weights file
    #[arg(short, long)]
    model: Option<PathBuf>,

    /// Interactive REPL mode
    #[arg(short, long)]
    interactive: bool,

    /// Print model info and exit
    #[arg(long)]
    info: bool,

    /// Output format
    #[arg(long, value_enum, default_value = "plain")]
    format: OutputFormat,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Debug, Clone, Copy)]
struct ArtMetrics {
    width: usize,
    height: usize,
    total_chars: usize,
}

#[derive(Debug, Serialize)]
struct JsonOutput {
    prompt: String,
    style: String,
    art: String,
    width: usize,
    height: usize,
    total_chars: usize,
    generation_time_ms: u64,
    seed: Option<u64>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    max_width: usize,
    max_lines: usize,
    max_chars: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.verbose {
        eprintln!("ASCII Art Generator v{}", env!("CARGO_PKG_VERSION"));
    }

    // Handle --info flag
    if cli.info {
        return print_model_info(&cli);
    }

    if cli.interactive {
        let Some(model) = load_cli_model(&cli)? else {
            println!("No model available for interactive mode.");
            println!(
                "Embedded weights: {} ({})",
                if weights_loader::embedded_available() {
                    "yes"
                } else {
                    "no"
                },
                weights_loader::embedded_reason()
            );
            println!("Pass `--model path/to/model.safetensors` or build with embedded weights.");
            return Ok(());
        };
        return run_interactive_mode(&model, &cli);
    }

    // Check for prompt
    let prompt = cli.prompt.as_deref().context(
        "No prompt provided. Usage: ascii-gen \"a cat\"\n\
         Or run interactive mode: ascii-gen --interactive\n\
         Run ascii-gen --help for more options.",
    )?;

    let Some(model) = load_cli_model(&cli)? else {
        return print_stub(&cli, prompt);
    };
    let (art, metrics, generation_time_ms) = generate_once(&model, &cli, prompt)?;
    output_art(&cli, prompt, &art, metrics, generation_time_ms)
}

/// Print information about the model.
fn print_model_info(cli: &Cli) -> Result<()> {
    let config = ModelConfig::default();

    println!("ASCII Art Generator Model Info");
    println!("==============================");
    println!();
    println!("Default Configuration:");
    println!("  Vocabulary size: {} tokens", config.vocab_size);
    println!("  Context length:  {} chars", config.block_size);
    println!("  Layers:          {}", config.n_layer);
    println!("  Attention heads: {}", config.n_head);
    println!("  Embedding dim:   {}", config.n_embd);
    println!("  Max rows:        {}", config.max_rows);
    println!("  Max columns:     {}", config.max_cols);
    println!();

    if let Some(model_path) = &cli.model {
        println!("Model path: {}", model_path.display());
        if model_path.exists() {
            let metadata = std::fs::metadata(model_path)?;
            println!("Model size: {} MB", metadata.len() / (1024 * 1024));
        } else {
            println!("Model file not found at specified path");
        }
    }

    println!(
        "Embedded weights: {} ({})",
        if weights_loader::embedded_available() {
            "yes"
        } else {
            "no"
        },
        weights_loader::embedded_reason()
    );

    println!();
    println!("Available styles: art, banner, simple, detailed");

    Ok(())
}

fn print_stub(cli: &Cli, prompt: &str) -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                  ASCII Art Generator (Stub)                    ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║                                                                ║");
    println!("║  Model weights not available.                                  ║");
    println!("║                                                                ║");
    println!("║  To generate art, you need:                                    ║");
    println!("║  1. Exported model dir with model.safetensors + config.json    ║");
    println!("║  2. Run: ascii-gen --model models/exported/model.safetensors   ║");
    println!("║                                                                ║");
    println!(
        "║  Embedded weights: {:width$} ║",
        if weights_loader::embedded_available() {
            "yes"
        } else {
            "no"
        },
        width = 44
    );
    println!("║                                                                ║");
    println!("║  Prompt: {:width$} ║", prompt, width = 53);
    println!("║  Style:  {:width$} ║", cli.style.as_str(), width = 53);
    println!("║  Width:  {:width$} ║", cli.width, width = 53);
    println!("║                                                                ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    Ok(())
}

fn load_cli_model(cli: &Cli) -> Result<Option<AsciiGPT>> {
    let device = candle_core::Device::Cpu;

    if let Some(model_path) = cli.model.as_ref() {
        if !model_path.exists() {
            anyhow::bail!("Model file not found: {}", model_path.display());
        }
        if cli.verbose {
            eprintln!("Loading external model from: {}", model_path.display());
        }
        let model = weights_loader::load_external_model(model_path, &device)
            .context("Failed to load external model")?;
        return Ok(Some(model));
    }

    if cli.verbose {
        eprintln!(
            "No --model provided; embedded weights available: {} ({})",
            weights_loader::embedded_available(),
            weights_loader::embedded_reason()
        );
    }

    weights_loader::try_load_embedded_model(&device).context("Failed to load embedded model")
}

fn run_interactive_mode(model: &AsciiGPT, cli: &Cli) -> Result<()> {
    println!("ASCII Art Generator - Interactive Mode");
    println!("Type a prompt and press Enter. Type 'quit' or 'exit' to leave.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush()?;

        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;
        let input = input.trim();

        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            break;
        }
        if input.is_empty() {
            continue;
        }

        match generate_once(model, cli, input) {
            Ok((art, metrics, generation_time_ms)) => {
                output_art(cli, input, &art, metrics, generation_time_ms)?;
                println!();
            }
            Err(err) => {
                eprintln!("Error: {err}");
                eprintln!();
            }
        }
    }

    Ok(())
}

fn generate_once(model: &AsciiGPT, cli: &Cli, prompt: &str) -> Result<(String, ArtMetrics, u64)> {
    if cli.verbose {
        eprintln!("Generating: {prompt}");
        eprintln!("Style: {}", cli.style.as_str());
        eprintln!(
            "Constraints: {}x{} (max {} chars)",
            cli.width, cli.max_lines, cli.max_chars
        );
        eprintln!(
            "Sampling: temp={} top_k={} top_p={}",
            cli.temperature, cli.top_k, cli.top_p
        );
        if let Some(seed) = cli.seed {
            eprintln!("Seed: {seed}");
        }
    }

    let tok = AsciiTokenizer::new();
    let prompt_tokens = tok.encode_prompt(cli.width, cli.max_lines, cli.style.as_str(), prompt);

    let cfg = GenerationConfig {
        max_new_tokens: cli.max_chars,
        max_width: cli.width,
        max_lines: cli.max_lines,
        max_chars: cli.max_chars,
        temperature: cli.temperature,
        top_k: cli.top_k,
        top_p: cli.top_p,
        seed: cli.seed,
    };

    let start = Instant::now();
    let tokens = generate_constrained(model, &prompt_tokens, &cfg, tok)?;
    let generation_time_ms = start.elapsed().as_millis().try_into().unwrap_or(u64::MAX);

    let art_start = tokens
        .iter()
        .rposition(|&t| t == SEP_ID)
        .map_or(0, |idx| idx + 1);
    let art = tok.decode(&tokens[art_start..]);
    let metrics = compute_art_metrics(&art);

    Ok((art, metrics, generation_time_ms))
}

fn output_art(
    cli: &Cli,
    prompt: &str,
    art: &str,
    metrics: ArtMetrics,
    generation_time_ms: u64,
) -> Result<()> {
    match cli.format {
        OutputFormat::Plain => {
            print!("{art}");
            if !art.ends_with('\n') {
                println!();
            }
        }
        OutputFormat::Markdown => {
            println!("```");
            print!("{art}");
            if !art.ends_with('\n') {
                println!();
            }
            println!("```");
        }
        OutputFormat::Json => {
            let payload = JsonOutput {
                prompt: prompt.to_string(),
                style: cli.style.as_str().to_string(),
                art: art.to_string(),
                width: metrics.width,
                height: metrics.height,
                total_chars: metrics.total_chars,
                generation_time_ms,
                seed: cli.seed,
                temperature: cli.temperature,
                top_k: cli.top_k,
                top_p: cli.top_p,
                max_width: cli.width,
                max_lines: cli.max_lines,
                max_chars: cli.max_chars,
            };
            println!("{}", serde_json::to_string_pretty(&payload)?);
        }
    }

    Ok(())
}

fn compute_art_metrics(art: &str) -> ArtMetrics {
    let mut lines = art.split('\n').collect::<Vec<_>>();
    if art.ends_with('\n') && lines.last().is_some_and(|v| v.is_empty()) {
        lines.pop();
    }

    let height = lines.len();
    let width = lines.iter().map(|line| line.len()).max().unwrap_or(0);
    let total_chars = art.len();

    ArtMetrics {
        width,
        height,
        total_chars,
    }
}
