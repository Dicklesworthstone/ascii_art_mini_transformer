#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
TMP_DIR="$(mktemp -d -t ascii_e2e_rust_${TS}_XXXXXX)"
LOG_FILE="$TMP_DIR/e2e_rust_${TS}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE" >/dev/null; }

log "Repo: $REPO_ROOT"
log "Temp: $TMP_DIR"
log "Log:  $LOG_FILE"

log "Running Rust tests (includes Python cross-validation fixtures)..."
cd "$REPO_ROOT/rust/ascii-gen"
cargo test 2>&1 | tee -a "$LOG_FILE"
cargo clippy --all-targets -- -D warnings 2>&1 | tee -a "$LOG_FILE"

log "Running Rust CLI smoke generation using crossval fixtures..."
OUT_FILE="$TMP_DIR/generated.txt"
cargo run --quiet -- \
  --model test_data/crossval/model.safetensors \
  --width 40 \
  --max-lines 20 \
  --max-chars 400 \
  --temperature 0 \
  --top-k 0 \
  --top-p 1 \
  --seed 0 \
  cat >"$OUT_FILE"

python3 - <<PY | tee -a "$LOG_FILE"
from __future__ import annotations

from pathlib import Path

out_path = Path(${OUT_FILE@Q})
text = out_path.read_text(encoding="utf-8")
lines = text.splitlines()
max_width = max((len(line) for line in lines), default=0)
print(f"lines={len(lines)} max_width={max_width}")
assert max_width <= 40, f"width constraint violated: {max_width} > 40"
assert len(lines) <= 20, f"height constraint violated: {len(lines)} > 20"
PY

log "E2E Rust: PASSED"
log "Artifacts kept under: $TMP_DIR"
log "Note: project policy forbids auto-cleanup via rm -rf; delete $TMP_DIR manually if desired."
