<div align="center">

```text
   ___   _____ _____ _____   ___      __      __
  / _ | / ___// ___//  _/  / _ |____/ /_____/ /_
 / __ |/ /__ / /__ _/ /   / __ / __/ __/ __/ __/
/_/ |_|\___/ \___/ /___/ /_/ |_\__/\__/\__/\__/

         tiny 2D-aware transformer for ASCII art
```

[![CI](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/actions/workflows/ci.yml/badge.svg)](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/actions/workflows/ci.yml)
[![E2E](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/actions/workflows/e2e.yml/badge.svg)](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/actions/workflows/e2e.yml)

</div>

**ASCII Art Mini Transformer** trains a small, CPU-friendly decoder-only transformer and runs **constraint-respecting** ASCII generation in both **Python** and a **single Rust CLI binary**.

Core idea: ASCII art is a **2D grid** (rows/columns), not just a 1D token stream — so this project uses **character-level tokenization** + **2D positional encoding**, then decodes with **hard width/height/max-chars constraints**.

**One-command smoke test (recommended):**

```bash
tests/e2e/e2e_full.sh
```

This runs: DB sanity → tiny CPU train → export → Python↔Rust parity checks → Rust inference → embedded-weights smoke.  
E2E scripts write timestamped logs into a temp directory and intentionally do **not** auto-delete it (project policy forbids `rm -rf`).

---

## TL;DR

### The Problem

General LLMs can describe ASCII art, but they often struggle to **compose coherent 2D shapes** while also obeying hard constraints like “80 columns, 40 lines”.

### The Solution

A tiny transformer trained on ASCII/ANSI art **as a 2D grid**, with:
- **2D positional encoding** (row + column)
- **character-level vocabulary** (stable + small)
- **constrained decoding** (width/height/max-chars enforced every step)
- **Python↔Rust parity tests** to prevent silent drift

### Why use this repo?

| Feature | Why it matters | Concrete proof in repo |
|---|---|---|
| 2D-aware modeling | Better structure and alignment | `python/model/positional_encoding.py`, `rust/ascii-gen/src/model/embedding.rs` |
| Hard constraints | Never “just a bit over” width/height | `python/inference/constraints.py`, `rust/ascii-gen/src/inference/constraints.rs` |
| CPU-first Rust inference | Fast, shippable CLI | `rust/ascii-gen/src/main.rs` |
| Export format is stable | Train in PyTorch, ship in Rust | `python/train/export.py` → `.safetensors` + `config.json` |
| Deterministic parity fixtures | Catch regressions in decoding/model math | `rust/ascii-gen/test_data/crossval/` |
| No-mock tests + verbose E2E logs | Confidence without “fake” coverage | `python/tests/`, `tests/e2e/*.sh` |

---

## Quick Example (5–10 commands)

```bash
# 1) Python deps
python3 -m venv .venv
.venv/bin/pip install -r python/requirements.txt

# 2) Rust deps (nightly toolchain pinned in CI)
rustup toolchain install nightly-2026-01-20
rustup override set nightly-2026-01-20

# 3) End-to-end smoke test (keeps a temp dir with logs/artifacts)
tests/e2e/e2e_full.sh

# 4) Run Rust CLI with your exported weights (example path)
cargo run --manifest-path rust/ascii-gen/Cargo.toml -- \
  --model models/exported/prod_medium_50k/model.safetensors \
  --format markdown --width 80 --max-lines 40 --max-chars 4000 --seed 0 \
  "cat"
```

---

## Design Philosophy

1) **Grid-first modeling**: treat newlines as geometry, not punctuation.  
2) **Determinism beats vibes**: parity fixtures + seeded sampling prevent “works on my machine”.  
3) **Constraints are non-negotiable**: decoding enforces width/height/max-chars *per step*.  
4) **CPU inference is a product constraint**: Rust path stays small and dependency-light.  
5) **No destructive automation**: scripts keep logs/artifacts; no auto-cleanups.

---

## Comparison (What this is / isn’t)

| Approach | Strengths | Weaknesses | When to choose it |
|---|---|---|---|
| This repo | Structured 2D output + strict constraints + shippable Rust CLI | Needs dataset + training to get good | You want real ASCII art generation, not just “ASCII-ish text” |
| General LLM prompting | Easy, no training | Weak spatial consistency; constraint violations | Quick demos, low stakes |
| FIGlet only | Great for banners | Not general ASCII art | CLI banners / logos only |
| Classic image→ASCII converters | Accurate *given an image* | Doesn’t “imagine” from text | You already have a source image |

---

## Installation

### Option A: “Just run tests” (recommended)

```bash
python3 -m venv .venv
.venv/bin/pip install -r python/requirements.txt
tests/e2e/e2e_full.sh
```

### Option B: Python-only

```bash
python3 -m venv .venv
.venv/bin/pip install -r python/requirements.txt
export PYTHONPATH=python
# Run inference from an exported model directory (see "Quick Start" below for a tiny export):
.venv/bin/python -m inference.cli "cat" --model models/exported/smoke/model.safetensors
```

### Option C: Rust CLI (from source)

```bash
rustup toolchain install nightly-2026-01-20
cargo build --release --manifest-path rust/ascii-gen/Cargo.toml
./rust/ascii-gen/target/release/ascii-gen --help
```

### Optional system tools

- `sqlite3` (for inspecting DB files)
- `figlet` (only required if you run `python/data/generate_figlet.py`)

---

## Quick Start

1) **Run the E2E suite** to confirm your environment (recommended):
   ```bash
   tests/e2e/e2e_full.sh
   ```
2) **Create a tiny local SQLite DB** (the production DB is not committed):
   ```bash
   export PYTHONPATH=python
   .venv/bin/python - <<'PY'
from __future__ import annotations

from pathlib import Path

from data.db import connect, initialize, insert_ascii_art

db_path = Path("data/ascii_art.db")
conn = connect(db_path)
initialize(conn)

samples = [
    ("  /\\\\_/\\\\\\n ( o.o )\\n  > ^ <", "cat", "animal"),
    (" __|__\\n(  o o )\\n \\\\  ^ /\\n  |||", "robot", "object"),
    ("#####\\n#   #\\n#####", "box", "simple"),
]
for raw_text, desc, category in samples:
    insert_ascii_art(
        conn,
        raw_text=raw_text,
        source="quickstart",
        title=desc,
        description=desc,
        category=category,
    )
print("db:", db_path, "rows:", conn.execute("SELECT COUNT(*) FROM ascii_art").fetchone()[0])
conn.close()
PY
   ```
3) **Train a tiny model (CPU smoke):**
   ```bash
   .venv/bin/python -m train.train \
     --db-path data/ascii_art.db \
     --checkpoint-dir models/checkpoints/smoke \
     --device cpu --dtype float32 \
     --block-size 256 --n-layer 2 --n-head 2 --n-embd 64 --dropout 0.0 \
     --batch-size 4 --gradient-accumulation-steps 1 \
     --learning-rate 0.001 --warmup-iters 1 --max-iters 50 \
     --eval-interval 1000000 --save-interval 1000000 --num-workers 0
   ```
4) **Export to Rust-compatible safetensors:**
   ```bash
   .venv/bin/python -m train.export \
     --checkpoint models/checkpoints/smoke/final.pt \
     --output-dir models/exported/smoke \
     --dtype float32 --quantize none
   ```
5) **Generate with Rust:**
   ```bash
   cargo run --manifest-path rust/ascii-gen/Cargo.toml -- \
     --model models/exported/smoke/model.safetensors \
     --width 60 --max-lines 20 --max-chars 800 --seed 0 \
     "cat"
   ```

---

## Command Reference

### E2E scripts (with detailed logging)

- Full pipeline: `tests/e2e/e2e_full.sh`
- Data sanity + quality pipeline: `tests/e2e/e2e_data.sh`
- CPU training smoke test: `tests/e2e/e2e_train.sh`
- Export smoke test: `tests/e2e/e2e_python_export.sh <export_dir>`
- Python inference smoke: `tests/e2e/e2e_python_infer.sh` (uses `E2E_MODEL_PATH`)
- Rust parity + inference: `tests/e2e/e2e_rust.sh` (uses `E2E_MODEL_PATH`)
- Quantized weight loading: `tests/e2e/e2e_quant.sh` (uses `E2E_MODEL_DIR`)
- Embedded weights smoke: `tests/e2e/e2e_embedded.sh <export_dir>`
- Lint/typecheck gates: `tests/e2e/e2e_lint.sh`

All scripts keep a temp dir with logs and artifacts; they print the location on failure.

### Python CLIs

All Python CLIs assume:

```bash
export PYTHONPATH=python
```

- Scrape ASCIIArt.eu gallery: `python -m data.scrape_asciiart --help`
- Scrape 16colo.rs demoscene packs: `python -m data.scrape_16colors --help`
- Scrape textfiles.com artscene: `python -m data.scrape_textfiles --help`
- Ingest HuggingFace datasets: `python -m data.ingest_huggingface --help`
- Generate FIGlet banners (requires `figlet`): `python -m data.generate_figlet --help`
- Run data quality pipeline: `python -m data.quality_pipeline --help`
- Train: `python -m train.train --help`
- Export: `python -m train.export --help`
- Python inference: `python -m inference.cli --help`

### Rust CLI (`ascii-gen`)

```bash
cargo run --manifest-path rust/ascii-gen/Cargo.toml -- --help
```

Examples:

```bash
# External float weights
cargo run --manifest-path rust/ascii-gen/Cargo.toml -- \
  --model models/exported/model.safetensors \
  --format markdown \
  --width 80 --max-lines 50 --max-chars 4000 \
  --seed 0 \
  "cat"

# Quantized weight-only export (requires quant_config.json next to the weights)
cargo run --manifest-path rust/ascii-gen/Cargo.toml -- \
  --model models/exported/model_int4.safetensors \
  "cat"
```

Embedded (single-file) binary:

```bash
ASCII_GEN_EXPORT_DIR=models/exported \
  cargo build --release --manifest-path rust/ascii-gen/Cargo.toml --features embedded-weights
./rust/ascii-gen/target/release/ascii-gen "cat"
```

---

## Configuration

### Export directory layout

An export directory is Rust-loadable when it contains:

```text
model.safetensors
config.json
tokenizer.json
```

Optional (weight-only quantization):

```text
model_int8.safetensors
model_int4.safetensors
quant_config.json
```

### Example `config.json`

```json
{
  "vocab_size": 107,
  "block_size": 2048,
  "n_layer": 6,
  "n_head": 6,
  "n_embd": 384,
  "dropout": 0.1,
  "max_rows": 100,
  "max_cols": 200,
  "newline_token_id": 7,
  "pad_token_id": 0,
  "bos_token_id": 1,
  "eos_token_id": 2
}
```

Notes:
- `n_embd` must be divisible by `n_head`.
- `block_size` is the context window (maximum prompt+generated tokens the model “sees”).
- `max_rows/max_cols` are the bounds for 2D positional encoding (clamped during inference).

### Training presets

`python -m train.train --preset small|medium|large` sets default `n_layer/n_head/n_embd/block_size`, and you can override any of them via flags.

---

## Architecture (Data Flow)

```text
          (scrape / ingest / generate)
      ┌──────────────────────────────────┐
      │ python/data/* (sources + cleanup)│
      └───────────────┬──────────────────┘
                      │ upsert + dedup (sha256)
                      v
               ┌───────────────┐
               │ SQLite DB      │
               │ data/ascii_art │
               └───────┬───────┘
                       │ batches (tokenize + constraints prefix)
                       v
             ┌───────────────────┐
             │ PyTorch training   │
             │ python/train/train │
             └─────────┬─────────┘
                       │ checkpoint.pt
                       v
             ┌───────────────────┐
             │ Export (safetensors)│
             │ python/train/export │
             └─────────┬─────────┘
                       │ config.json + tokenizer.json
                       v
      ┌──────────────────────────────────┐
      │ Rust CLI inference (ascii-gen)   │
      │ candle + constrained decoding    │
      └──────────────────────────────────┘
```

The repo includes parity fixtures and integration tests to keep Python and Rust behavior aligned.

---

## Troubleshooting

**1) `ModuleNotFoundError: No module named 'torch'`**
- Install Python deps: `.venv/bin/pip install -r python/requirements.txt`

**2) `figlet: command not found`**
- Install `figlet` using your package manager (only needed for FIGlet dataset generation).

**3) Rust build fails / edition toolchain mismatch**
- Use the pinned toolchain from CI:
  ```bash
  rustup toolchain install nightly-2026-01-20
  rustup override set nightly-2026-01-20
  ```

**4) “Quantized safetensors detected, but quant_config.json was not found…”**
- When running quantized weights, keep `quant_config.json` next to `model_int8.safetensors` / `model_int4.safetensors`.

**5) “Failed to load checkpoint in safe weights_only mode…” (Python)**
- Older checkpoints may require pickle-based loading. Only if you trust the file:
  ```bash
  .venv/bin/python -m inference.cli "cat" --checkpoint path/to.ckpt.pt --unsafe-load
  ```

---

## Limitations (Honest)

- **Quality depends on data + training**: the tiny CPU-smoke model is for verification, not art quality.
- **GitHub-hosted CI is CPU-only**: the production run (`bd-19v`) requires a CUDA machine.
- **Scrapers can break**: sites change; always use rate limits.
- **No web UI**: this is a dataset/training/CLI repo, not an app.
- **No license file yet**: assume proprietary until a license is added.

---

## FAQ

**Why character-level tokens instead of BPE?**  
ASCII art is literally made of characters; keeping them atomic stabilizes training and makes exports small.

**Why 2D positional encoding?**  
Rows/columns are the “geometry” of ASCII art. 2D positions make it easier for a small model to reason about structure.

**Where is the big dataset DB?**  
The SQLite DB is not committed at production size. E2E scripts generate temporary DBs for testing.

**Can I run the full training in CI?**  
Not on GitHub-hosted runners (CPU-only). Use `.github/workflows/train_gpu.yml` on a self-hosted runner labeled `gpu`.

**How do Python and Rust stay consistent?**  
Small deterministic fixtures + parity tests under `rust/ascii-gen/test_data/crossval/` catch drift.

**Can I ship a single binary with weights?**  
Yes. Build Rust with `--features embedded-weights` and set `ASCII_GEN_EXPORT_DIR` to your export directory.

---

## About Contributions

> *About Contributions:* Please don't take this the wrong way, but I do not accept outside contributions for any of my projects. I simply don't have the mental bandwidth to review anything, and it's my name on the thing, so I'm responsible for any problems it causes; thus, the risk-reward is highly asymmetric from my perspective. I'd also have to worry about other "stakeholders," which seems unwise for tools I mostly make for myself for free. Feel free to submit issues, and even PRs if you want to illustrate a proposed fix, but know I won't merge them directly. Instead, I'll have Claude or Codex review submissions via `gh` and independently decide whether and how to address them. Bug reports in particular are welcome. Sorry if this offends, but I want to avoid wasted time and hurt feelings. I understand this isn't in sync with the prevailing open-source ethos that seeks community contributions, but it's the only way I can move at this velocity and keep my sanity.

---

## License

No license file is currently included in this repository. Treat the code as proprietary until a license is added.
