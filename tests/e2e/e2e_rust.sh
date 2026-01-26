#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
TMP_DIR="$(mktemp -d -t ascii_e2e_rust_${TS}_XXXXXX)"
LOG_FILE="$TMP_DIR/e2e_rust_${TS}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }

CURRENT_STEP="setup"
SECONDS=0
on_exit() {
  status=$?
  if [[ $status -ne 0 ]]; then
    echo "E2E rust: FAILED (step=$CURRENT_STEP status=$status elapsed=${SECONDS}s)" >&2
    echo "Artifacts kept under: $TMP_DIR" >&2
    echo "Log file: $LOG_FILE" >&2
  fi
}
trap on_exit EXIT

log "Repo: $REPO_ROOT"
log "Temp: $TMP_DIR"
log "Log:  $LOG_FILE"

CURRENT_STEP="env_snapshot"
bash "$REPO_ROOT/tests/e2e/env_snapshot.sh" "$REPO_ROOT" 2>&1 | tee -a "$LOG_FILE"

log "Running Rust tests (includes Python cross-validation fixtures)..."
cd "$REPO_ROOT/rust/ascii-gen"
CURRENT_STEP="cargo_test"
cargo test 2>&1 | tee -a "$LOG_FILE"
CURRENT_STEP="cargo_clippy"
cargo clippy --all-targets -- -D warnings 2>&1 | tee -a "$LOG_FILE"

MODEL_PATH="${E2E_MODEL_PATH:-test_data/crossval/model.safetensors}"
MAX_CHARS="${E2E_MAX_CHARS:-}"
if [[ -z "$MAX_CHARS" ]]; then
  if [[ -n "${E2E_MODEL_PATH:-}" ]]; then
    MAX_CHARS=80
  else
    MAX_CHARS=400
  fi
fi
log "Running Rust CLI smoke generation..."
log "Model: $MODEL_PATH"
log "Smoke limits: width=40 lines=20 max_chars=$MAX_CHARS"
OUT_FILE="$TMP_DIR/generated.txt"
CURRENT_STEP="cli_smoke"
cargo run --quiet -- \
  --model "$MODEL_PATH" \
  --width 40 \
  --max-lines 20 \
  --max-chars "$MAX_CHARS" \
  --temperature 0 \
  --top-k 0 \
  --top-p 1 \
  --seed 0 \
  cat >"$OUT_FILE"

CURRENT_STEP="validate_output"
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
