#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
TMP_DIR="$(mktemp -d -t ascii_e2e_export_${TS}_XXXXXX)"
LOG_FILE="$TMP_DIR/e2e_export_${TS}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE" >/dev/null; }

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

log "Repo: $REPO_ROOT"
log "Temp: $TMP_DIR"
log "Log:  $LOG_FILE"
log "Python: $PYTHON_BIN"

export PYTHONPATH="$REPO_ROOT/python"
OUT_DIR="${1:-$TMP_DIR/exported}"
mkdir -p "$OUT_DIR"
log "Export dir: $OUT_DIR"

log "Running export smoke test (fresh tiny model + int8/int4 exports)..."
"$PYTHON_BIN" -m train.export \
  --output-dir "$OUT_DIR" \
  --quantize both \
  --preset small \
  --n-layer 2 \
  --n-head 2 \
  --n-embd 64 \
  --block-size 256 \
  2>&1 | tee -a "$LOG_FILE"

log "Verifying expected artifacts exist..."
test -f "$OUT_DIR/model.safetensors"
test -f "$OUT_DIR/config.json"
test -f "$OUT_DIR/tokenizer.json"
test -f "$OUT_DIR/model_int8.safetensors"
test -f "$OUT_DIR/model_int4.safetensors"
test -f "$OUT_DIR/quant_config.json"

log "E2E export: PASSED"
log "Artifacts kept under: $TMP_DIR"
log "Note: project policy forbids auto-cleanup via rm -rf; delete $TMP_DIR manually if desired."
