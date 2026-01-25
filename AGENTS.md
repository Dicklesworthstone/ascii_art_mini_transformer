# AGENTS.md — ASCII Art Mini Transformer

> Guidelines for AI coding agents working on this project.

---

## Project Overview

This project builds a tiny, CPU-efficient transformer model (~10-50MB) that generates high-quality ASCII art. Unlike large LLMs which struggle with ASCII art due to gestalt visual-symbolic reasoning gaps, this specialized model is trained specifically for this task.

**Key Innovation**: Character-level tokenization + 2D positional encoding (treating ASCII art as a 2D grid, not a 1D sequence).

**Deliverables**:
1. SQLite database with 500k-3M+ ASCII art pieces
2. Trained transformer model (10-30M parameters)
3. Single-file Rust binary CLI for inference

---

## RULE NUMBER 1: NO FILE DELETION

**YOU ARE NEVER ALLOWED TO DELETE A FILE WITHOUT EXPRESS PERMISSION.** Even a new file that you yourself created, such as a test code file. You have a horrible track record of deleting critically important files or otherwise throwing away tons of expensive work. As a result, you have permanently lost any and all rights to determine that a file or folder should be deleted.

**YOU MUST ALWAYS ASK AND RECEIVE CLEAR, WRITTEN PERMISSION BEFORE EVER DELETING A FILE OR FOLDER OF ANY KIND.**

---

## Irreversible Git & Filesystem Actions — DO NOT EVER BREAK GLASS

1. **Absolutely forbidden commands:** `git reset --hard`, `git clean -fd`, `rm -rf`, or any command that can delete or overwrite code/data must never be run unless the user explicitly provides the exact command and states, in the same message, that they understand and want the irreversible consequences.
2. **No guessing:** If there is any uncertainty about what a command might delete or overwrite, stop immediately and ask the user for specific approval.
3. **Safer alternatives first:** When cleanup or rollbacks are needed, request permission to use non-destructive options first.
4. **Mandatory explicit plan:** Even after explicit user authorization, restate the command verbatim, list exactly what will be affected, and wait for confirmation.
5. **Document the confirmation:** When running any approved destructive command, record the exact user text that authorized it.

---

## Toolchain

### Python (Data & Training)
- **Python 3.11+** for data processing and model training
- **PyTorch** for model implementation
- **HuggingFace datasets** for data loading
- **safetensors** for model weight export

### Rust (Inference)
- **Rust 2024 Edition** (nightly)
- **candle** for tensor operations (pure Rust, no native deps)
- **clap** for CLI
- **safetensors** for weight loading

### Key Dependencies

| Component | Library | Purpose |
|-----------|---------|---------|
| Training | PyTorch | Model training |
| Training | TorchAO | Quantization |
| Training | datasets | HuggingFace datasets |
| Inference | candle | Rust tensor ops |
| Inference | clap | CLI parsing |
| Database | rusqlite/sqlite3 | Data storage |

### Release Profile (Rust)

```toml
[profile.release]
opt-level = "z"     # Optimize for size
lto = true          # Link-time optimization
codegen-units = 1   # Single codegen unit
panic = "abort"     # Smaller binary
strip = true        # Remove debug symbols
```

---

## Code Editing Discipline

### No Script-Based Changes

**NEVER** run a script that processes/changes code files in this repo. Make code changes manually.

### No File Proliferation

**NEVER** create variations like `model_v2.py` or `train_improved.py`. Revise existing files in place.

---

## Project Structure

```
ascii_art_mini_transformer/
├── AGENTS.md                 # This file
├── .beads/                   # Issue tracking (br/bv)
│   └── issues.jsonl
├── data/
│   ├── ascii_art.db          # SQLite database
│   └── raw/                  # Raw scraped data
├── python/
│   ├── data/
│   │   ├── ingest_huggingface.py
│   │   ├── scrape_asciiart.py
│   │   ├── scrape_16colors.py
│   │   ├── generate_figlet.py
│   │   └── quality_pipeline.py
│   ├── model/
│   │   ├── tokenizer.py
│   │   ├── positional_encoding.py
│   │   ├── transformer.py
│   │   └── config.py
│   ├── train/
│   │   ├── dataset.py
│   │   ├── train.py
│   │   └── export.py
│   └── requirements.txt
├── rust/
│   └── ascii-gen/
│       ├── Cargo.toml
│       ├── src/
│       │   ├── main.rs
│       │   ├── lib.rs
│       │   ├── model/
│       │   ├── tokenizer/
│       │   ├── inference/
│       │   └── weights/
│       └── tests/
└── models/
    ├── checkpoints/           # Training checkpoints
    └── exported/              # Quantized models for Rust
```

---

## Compiler/Linter Checks (CRITICAL)

### Python
```bash
# Type checking
mypy python/ --strict

# Linting
ruff check python/
ruff format python/
```

### Rust
```bash
# Check for compiler errors
cargo check --all-targets

# Clippy lints
cargo clippy --all-targets -- -D warnings

# Formatting
cargo fmt --check
```

---

## Testing

### Python
```bash
pytest python/tests/
pytest python/tests/ -v --tb=short
```

### Rust
```bash
cargo test
cargo test -- --nocapture  # With output
```

---

## Issue Tracking with br (beads_rust)

All issue tracking goes through **br** (beads_rust). No other TODO systems.

**Important:** `br` is non-invasive—it NEVER executes git commands. After `br sync --flush-only`, you must manually run `git add .beads/ && git commit`.

### Basics

```bash
br ready --json                    # Show unblocked work
br list --status=open              # All open issues
br show <id>                       # Full issue details
br create "Title" -t task -p 2     # Create issue
br update <id> --status in_progress
br close <id> --reason "Done"
br sync --flush-only               # Export to JSONL
```

### Issue Types
- `epic` - Major phase/milestone
- `feature` - User-facing capability
- `task` - Implementation work
- `bug` - Defect fix
- `chore` - Maintenance

### Priorities
- `0` critical (security, data loss)
- `1` high (blocking other work)
- `2` medium (default)
- `3` low
- `4` backlog

### Agent Workflow

1. `br ready` to find unblocked work
2. Claim: `br update <id> --status in_progress`
3. Implement + test
4. If you discover new work, create a new bead
5. Close when done: `br close <id>`
6. Run `br sync --flush-only && git add .beads/` before committing

---

## Project-Specific Guidelines

### Data Collection
- **Always respect rate limits** when scraping (1-2s delay between requests)
- **Preserve exact whitespace** in ASCII art (it's structural, not decoration)
- **Compute metadata** at ingest time (width, height, charset)
- **Deduplicate** by content hash

### Model Training
- Target **10-30M parameters** (too large = slow CPU inference)
- Use **character-level tokenization** (not BPE)
- Implement **2D positional encoding** (row + column, not 1D position)
- Train with **constraint conditioning** (width/height as prefix tokens)

### Rust Inference
- Use **candle** for tensor operations (pure Rust, small binary)
- Target **<50MB total binary size** (code + weights)
- Implement **constrained decoding** (enforce width/height limits)
- Support **temperature/top-k/top-p sampling**

### Quality Criteria
- Generated art should be **recognizable** as the requested subject
- Should **NOT memorize** training data (test for similarity)
- Must **respect all constraints** (width, height, max chars)
- CPU inference should be **<100ms** for typical generation

---

## Key Research References

- **ASCII Art Generation**: arXiv:2503.14375 (March 2025)
- **2D Positional Encoding**: ViTARC, GridPE, 2D-TPE papers
- **Efficient Transformers**: TinyViT, MobileViT, nanoGPT
- **Quantization**: TorchAO (ICML 2025)
- **Byte-Level Models**: Byte Latent Transformer (Meta)

---

## Session Completion

Before ending a work session:

1. **File issues** for remaining work via `br create`
2. **Run quality gates** if code changed (tests, linters)
3. **Update issue status** via `br update` / `br close`
4. **PUSH TO REMOTE**:
   ```bash
   br sync --flush-only
   git add .beads/
   git add <changed files>
   git commit -m "description"
   git push
   ```
5. **Verify** with `git status` showing "up to date with origin"

**CRITICAL:** Work is NOT complete until `git push` succeeds.

---

## Data Sources Quick Reference

### HuggingFace Datasets
| Dataset | Size | Notes |
|---------|------|-------|
| Csplk/THE.ASCII.ART.EMPORIUM | 3.1M | Massive collection |
| mrzjy/ascii_art_generation_140k | 138k | Instruction pairs |
| apehex/ascii-art | 47k | Rich metadata |
| jdpressman/retro-ascii-art-v1 | 6k | Retro styled |

### Web Archives
- **16colo.rs** - Demoscene artpacks (1990-present)
- **asciiart.eu** - 11,000+ categorized artworks
- **asciiart.website** - Christopher Johnson's collection
- **textfiles.com/artscene** - BBS era archives

### FIGlet Fonts
- **cmatsuoka/figlet-fonts** - ALL known fonts
- **xero/figlet-fonts** - Curated collection
