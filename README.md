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

## Production training (GPU)

This repo's GitHub-hosted CI is CPU-only. The P1 production run (bd-19v) needs a CUDA machine for ~50k iters on the 500k-row SQLite DB.

Recommended manual run on a CUDA machine (example defaults from bd-19v comments):

```bash
PYTHONPATH=python ./.venv/bin/python -m train.train \
  --db-path data/ascii_art.db \
  --preset medium \
  --device cuda --dtype bfloat16 \
  --checkpoint-dir models/checkpoints/prod_medium_50k \
  --max-iters 50000 --lr-decay-iters 50000 \
  --save-interval 5000 --eval-interval 500 --log-interval 10 \
  --batch-size 64 --gradient-accumulation-steps 4 \
  --learning-rate 6e-4 --warmup-iters 2000 --min-lr 6e-5 \
  --num-workers 4
```

Export + validate:

```bash
PYTHONPATH=python ./.venv/bin/python -m train.export \
  --checkpoint models/checkpoints/prod_medium_50k/final.pt \
  --output-dir models/exported/prod_medium_50k \
  --dtype float32 --quantize none
```

Rust validation:

```bash
cargo run --manifest-path rust/ascii-gen/Cargo.toml -- \
  --model models/exported/prod_medium_50k/model.safetensors \
  --format markdown --width 80 --max-lines 50 --max-chars 4000 --seed 0 \
  "cat"
```

### GitHub Actions (self-hosted GPU runner)

If you have a self-hosted runner labeled `gpu`, you can run the same training/export via `.github/workflows/train_gpu.yml` (manual `workflow_dispatch`).

Notes:
- The workflow expects the DB to already exist on the runner at the configured `db_path` (it is not committed).
- The workflow does **not** pip-install `torch` (to avoid overwriting CUDA wheels); point `python_bin` at your pre-provisioned venv/conda env that has a CUDA-enabled torch install.

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

## Python Tests + Coverage

Run tests:

```bash
pytest python/tests/ -v
```

Run tests with coverage report (terminal output):

```bash
pytest python/tests/ --cov=python --cov-report=term-missing
```

Generate HTML coverage report (written under a temp dir and kept; no auto-delete per project policy):

```bash
tests/coverage/run_coverage.sh --html
```
Optionally, also emit an XML report (in the same artifacts dir):

```bash
tests/coverage/run_coverage.sh --xml
```

## E2E tests

- Full suite: `tests/e2e/e2e_full.sh`
- Data-only: `tests/e2e/e2e_data.sh`
- Train smoke: `tests/e2e/e2e_train.sh`
- Python export: `tests/e2e/e2e_python_export.sh`
- Rust-only (includes Python↔Rust cross-validation): `tests/e2e/e2e_rust.sh`
- Embedded weights smoke: `tests/e2e/e2e_embedded.sh`
- Lint/typecheck gates: `tests/e2e/e2e_lint.sh`

The lint script runs:
- Python: `ruff check`, `ruff format --check`, `mypy --strict`
- Rust: `cargo fmt --check`, `cargo clippy --all-targets`

## Parity fixtures (Python ↔ Rust)

The Rust integration tests validate Python ↔ Rust parity using small, deterministic fixtures under `rust/ascii-gen/test_data/crossval/`.

When to regenerate:
- Tokenizer changes (IDs, special tokens, exported `tokenizer.json` format)
- Constraint/decoding changes (logit masking, stopping rules, width/height behavior)
- Model math changes (attention/LN behavior, output filtering) that affect logits/generation

Workflow:
1. Regenerate fixtures:
   - `PYTHONPATH=. .venv/bin/python python/tests/generate_crossval_data.py`
2. Review the diffs:
   - `git diff rust/ascii-gen/test_data/crossval/`
   - Drift should be explainable by the intentional change; otherwise treat as a bug.
3. Re-run parity checks:
   - `cargo test --manifest-path rust/ascii-gen/Cargo.toml --test integration`

Keep fixtures small/reviewable:
- Prefer tiny model configs (the generator uses a small, fixed-seed model).
- Limit cases to a handful of prompts/seeds; keep generated token counts short.
- If a change is expected to alter outputs, mention why in the commit message.
