# ASCII Art Mini Transformer

A tiny, CPU-efficient transformer (~10–50MB) specialized for generating high-quality ASCII art.

Core idea: treat ASCII art as a **2D grid** (row/column) with **character-level tokenization** + **2D positional encoding**, plus **constraint-conditioned generation** (width/height limits).

## Status

This repo is currently being bootstrapped from beads issues in `.beads/`.

## Database (Phase 1)

- Canonical SQLite schema: `data/schema.sql`
- Python utilities (schema init, hashing/dedup, FTS search): `python/data/db.py`

## Rust CLI (Phase 3)

The `ascii-gen` CLI lives at `rust/ascii-gen/`.

- External weights: `cargo run --manifest-path rust/ascii-gen/Cargo.toml -- --model models/exported/model.safetensors "cat"`
- Quantized weights (weight-only): export with `python3 python/train/export.py --quantize int8|int4|both` which writes `model_int8.safetensors` / `model_int4.safetensors` plus `quant_config.json`; then run `cargo run --manifest-path rust/ascii-gen/Cargo.toml -- --model models/exported/model_int4.safetensors "cat"` (keep `quant_config.json` in the same directory).
- Embedded weights (single-file): export a model into `models/exported/`, then build with `--features embedded-weights` (optionally set `ASCII_GEN_MODEL_PATH`).

Issue tracking is via `br` / `bv` (beads_rust). See `AGENTS.md` for workflow.

## E2E tests

- Full suite: `tests/e2e/e2e_full.sh`
- Rust-only (includes Python↔Rust cross-validation): `tests/e2e/e2e_rust.sh`

These scripts write timestamped logs into a temp directory and intentionally do not auto-delete it (project policy forbids `rm -rf`).
