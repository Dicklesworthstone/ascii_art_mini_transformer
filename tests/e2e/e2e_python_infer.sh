#!/usr/bin/env bash
# E2E smoke test for Python inference CLI against exported safetensors weights.
#
# Environment variables:
#   E2E_MODEL_PATH - Path to float safetensors weights (expects config.json next to it)
#   E2E_MAX_TOKENS - Max tokens to generate (default: 256)
#
# Notes:
# - Keeps artifacts (no auto-delete per project policy).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
TMP_DIR="$(mktemp -d -t ascii_e2e_py_infer_${TS}_XXXXXX)"
LOG_FILE="$TMP_DIR/e2e_py_infer_${TS}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }
fail() { log "FAILED: $*"; exit 1; }

CURRENT_STEP="setup"
SECONDS=0
on_exit() {
  status=$?
  if [[ $status -ne 0 ]]; then
    echo "E2E python inference: FAILED (step=$CURRENT_STEP status=$status elapsed=${SECONDS}s)" >&2
    echo "Artifacts kept under: $TMP_DIR" >&2
    echo "Log file: $LOG_FILE" >&2
  fi
}
trap on_exit EXIT

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

MODEL_PATH="${E2E_MODEL_PATH:-$REPO_ROOT/models/exported/model.safetensors}"
MAX_TOKENS="${E2E_MAX_TOKENS:-256}"

WIDTH=40
HEIGHT=20

log "Repo: $REPO_ROOT"
log "Temp: $TMP_DIR"
log "Log:  $LOG_FILE"
log "Python: $PYTHON_BIN"
log "Model: $MODEL_PATH"
log "Limits: width=$WIDTH height=$HEIGHT max_tokens=$MAX_TOKENS"

CURRENT_STEP="env_snapshot"
bash "$REPO_ROOT/tests/e2e/env_snapshot.sh" "$REPO_ROOT" "$PYTHON_BIN" 2>&1 | tee -a "$LOG_FILE"

if [[ ! -f "$MODEL_PATH" ]]; then
  fail "Model weights not found: $MODEL_PATH"
fi
if [[ ! -f "$(dirname "$MODEL_PATH")/config.json" ]]; then
  fail "Expected config.json next to model weights: $(dirname "$MODEL_PATH")/config.json"
fi

export PYTHONPATH="$REPO_ROOT/python"

OUT_FILE="$TMP_DIR/generated.txt"
log "Running python inference CLI..."
CURRENT_STEP="inference_cli"
"$PYTHON_BIN" -m inference.cli \
  cat \
  --model "$MODEL_PATH" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --style art \
  --temperature 0 \
  --top-k 0 \
  --top-p 1 \
  --seed 0 \
  --max-tokens "$MAX_TOKENS" \
  >"$OUT_FILE" 2>>"$LOG_FILE" || fail "python inference.cli failed"

log "Python output:"
cat "$OUT_FILE" | tee -a "$LOG_FILE"

log "Validating constraints..."
CURRENT_STEP="validate_output"
"$PYTHON_BIN" - <<PY | tee -a "$LOG_FILE"
from __future__ import annotations

from pathlib import Path

out_path = Path(${OUT_FILE@Q})
text = out_path.read_text(encoding="utf-8")
lines = text.splitlines()
max_width = max((len(line) for line in lines), default=0)
print(f"lines={len(lines)} max_width={max_width}")
assert len(lines) > 0, "output is empty"
assert max_width <= ${WIDTH}, f"width constraint violated: {max_width} > ${WIDTH}"
assert len(lines) <= ${HEIGHT}, f"height constraint violated: {len(lines)} > ${HEIGHT}"
for i, ch in enumerate(text):
    if ch != "\\n":
        assert 32 <= ord(ch) <= 126, f"invalid char at pos {i}: {repr(ch)}"
print("constraints validated")
PY

log "E2E Python inference: PASSED"
log "Artifacts kept under: $TMP_DIR"
log "Note: project policy forbids auto-cleanup via rm -rf; delete $TMP_DIR manually if desired."
