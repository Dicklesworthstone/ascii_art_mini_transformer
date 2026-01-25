# ASCII Art Mini Transformer

A tiny, CPU-efficient transformer (~10–50MB) specialized for generating high-quality ASCII art.

Core idea: treat ASCII art as a **2D grid** (row/column) with **character-level tokenization** + **2D positional encoding**, plus **constraint-conditioned generation** (width/height limits).

## Status

This repo is currently being bootstrapped from beads issues in `.beads/`.

## Quickstart (E2E)

```bash
tests/e2e/e2e_full.sh
```

This runs: data sanity → tiny train smoke test → export → Rust inference (cross-validated) → embedded-weights smoke test.

Scripts write timestamped logs into a temp directory and intentionally do not auto-delete it (project policy forbids `rm -rf`).

## Python Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r python/requirements.txt
```

## Database (Phase 1)

- Canonical SQLite schema: `data/schema.sql`
- Python utilities (schema init, hashing/dedup, FTS search): `python/data/db.py`

## Training + Export (Python)

All Python CLIs assume:

```bash
export PYTHONPATH=python
```

Train (writes checkpoints under `models/checkpoints/` by default):

```bash
.venv/bin/python -m train.train --db-path data/ascii_art.db
```

Train with a model size preset (overridable via `--n-layer/--n-head/--n-embd/--block-size`):

```bash
.venv/bin/python -m train.train --db-path data/ascii_art.db --preset small
```

Export from a training checkpoint to a Rust-compatible safetensors directory:

```bash
.venv/bin/python -m train.export \
  --checkpoint models/checkpoints/final.pt \
  --output-dir models/exported \
  --dtype float32 \
  --quantize none
```

Optional: export quantized weights (weight-only) for smaller files:

```bash
.venv/bin/python -m train.export --checkpoint models/checkpoints/final.pt --output-dir models/exported --quantize int8
# or: --quantize int4
# or: --quantize both
```

Fresh (untrained) exports are also supported for smoke tests via `--preset` and `--n-layer/--n-head/--n-embd/--block-size`.

## Python Inference CLI

Run from a training checkpoint:

```bash
.venv/bin/python -m inference.cli "cat" --checkpoint models/checkpoints/final.pt
```

Run from exported float safetensors (expects `config.json` next to the weights):

```bash
.venv/bin/python -m inference.cli "cat" --model models/exported/model.safetensors
```

## Rust CLI (Phase 3)

The `ascii-gen` CLI lives at `rust/ascii-gen/`.

External weights:

```bash
cargo run --manifest-path rust/ascii-gen/Cargo.toml -- --model models/exported/model.safetensors "cat"
```

Quantized weights (weight-only): export with `train.export --quantize int8|int4|both` which writes `model_int8.safetensors` / `model_int4.safetensors` plus `quant_config.json`; then run:

```bash
cargo run --manifest-path rust/ascii-gen/Cargo.toml -- --model models/exported/model_int4.safetensors "cat"
```

Keep `quant_config.json` in the same directory as the quantized weights.

Embedded weights (single-file): build with `--features embedded-weights`. By default this embeds from `models/exported/`, or you can point at a specific export directory:

```bash
ASCII_GEN_EXPORT_DIR=models/exported cargo build --release --manifest-path rust/ascii-gen/Cargo.toml --features embedded-weights
./rust/ascii-gen/target/release/ascii-gen --info
./rust/ascii-gen/target/release/ascii-gen "cat"   # no --model needed
```

Issue tracking is via `br` / `bv` (beads_rust). See `AGENTS.md` for workflow.

## E2E tests

- Full suite: `tests/e2e/e2e_full.sh`
- Data-only: `tests/e2e/e2e_data.sh`
- Train smoke: `tests/e2e/e2e_train.sh`
- Python export: `tests/e2e/e2e_python_export.sh`
- Rust-only (includes Python↔Rust cross-validation): `tests/e2e/e2e_rust.sh`
- Embedded weights smoke: `tests/e2e/e2e_embedded.sh`
