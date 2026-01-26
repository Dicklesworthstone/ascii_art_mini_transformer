#!/usr/bin/env bash
# E2E smoke tests for INT8/INT4 quantized model weights.
#
# This script validates that the Rust CLI can load and run inference
# with quantized weights (int8 and int4), respecting output constraints.
#
# Environment variables:
#   E2E_MODEL_DIR - Path to directory containing quantized models (default: models/exported/)
#   E2E_MAX_CHARS - Maximum chars for generation (default: 80)
#
# Prerequisites:
#   - Exported quantized models: model_int8.safetensors, model_int4.safetensors
#   - config.json and quant_config.json in the model directory
#
# Usage:
#   ./tests/e2e/e2e_quant.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
TMP_DIR="$(mktemp -d -t ascii_e2e_quant_${TS}_XXXXXX)"
LOG_FILE="$TMP_DIR/e2e_quant_${TS}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }
fail() { log "FAILED: $*"; exit 1; }

log "Repo: $REPO_ROOT"
log "Temp: $TMP_DIR"
log "Log:  $LOG_FILE"

bash "$REPO_ROOT/tests/e2e/env_snapshot.sh" "$REPO_ROOT" 2>&1 | tee -a "$LOG_FILE"

MODEL_DIR="${E2E_MODEL_DIR:-$REPO_ROOT/models/exported}"
MAX_CHARS="${E2E_MAX_CHARS:-80}"

log "Model directory: $MODEL_DIR"
log "Max chars: $MAX_CHARS"

# Verify quantized model files exist
INT8_MODEL="$MODEL_DIR/model_int8.safetensors"
INT4_MODEL="$MODEL_DIR/model_int4.safetensors"
QUANT_CONFIG="$MODEL_DIR/quant_config.json"
CONFIG_JSON="$MODEL_DIR/config.json"

for required in "$INT8_MODEL" "$INT4_MODEL" "$QUANT_CONFIG" "$CONFIG_JSON"; do
  if [[ ! -f "$required" ]]; then
    fail "Required file not found: $required"
  fi
done
log "All required model files present"

cd "$REPO_ROOT/rust/ascii-gen"

# Test INT8 quantized model
log "Testing INT8 quantized model..."
OUT_INT8="$TMP_DIR/generated_int8.txt"
cargo run --quiet -- \
  --model "$INT8_MODEL" \
  --width 40 \
  --max-lines 20 \
  --max-chars "$MAX_CHARS" \
  --temperature 0 \
  --top-k 0 \
  --top-p 1 \
  --seed 42 \
  cat >"$OUT_INT8" 2>>"$LOG_FILE" || fail "INT8 model generation failed"

log "INT8 output:"
cat "$OUT_INT8" | tee -a "$LOG_FILE"

# Validate INT8 output
python3 - <<PY | tee -a "$LOG_FILE"
from __future__ import annotations
from pathlib import Path

out_path = Path(${OUT_INT8@Q})
text = out_path.read_text(encoding="utf-8")
lines = text.splitlines()
max_width = max((len(line) for line in lines), default=0)
print(f"INT8: lines={len(lines)} max_width={max_width}")
assert len(lines) > 0, "INT8: output is empty"
assert max_width <= 40, f"INT8: width constraint violated: {max_width} > 40"
assert len(lines) <= 20, f"INT8: height constraint violated: {len(lines)} > 20"
for i, ch in enumerate(text):
    if ch != '\n':
        assert 32 <= ord(ch) <= 126, f"INT8: invalid char at pos {i}: {repr(ch)}"
print("INT8: constraints validated")
PY

log "INT8 quantized model: PASSED"

# Test INT4 quantized model
log "Testing INT4 quantized model..."
OUT_INT4="$TMP_DIR/generated_int4.txt"
cargo run --quiet -- \
  --model "$INT4_MODEL" \
  --width 40 \
  --max-lines 20 \
  --max-chars "$MAX_CHARS" \
  --temperature 0 \
  --top-k 0 \
  --top-p 1 \
  --seed 42 \
  cat >"$OUT_INT4" 2>>"$LOG_FILE" || fail "INT4 model generation failed"

log "INT4 output:"
cat "$OUT_INT4" | tee -a "$LOG_FILE"

# Validate INT4 output
python3 - <<PY | tee -a "$LOG_FILE"
from __future__ import annotations
from pathlib import Path

out_path = Path(${OUT_INT4@Q})
text = out_path.read_text(encoding="utf-8")
lines = text.splitlines()
max_width = max((len(line) for line in lines), default=0)
print(f"INT4: lines={len(lines)} max_width={max_width}")
assert len(lines) > 0, "INT4: output is empty"
assert max_width <= 40, f"INT4: width constraint violated: {max_width} > 40"
assert len(lines) <= 20, f"INT4: height constraint violated: {len(lines)} > 20"
for i, ch in enumerate(text):
    if ch != '\n':
        assert 32 <= ord(ch) <= 126, f"INT4: invalid char at pos {i}: {repr(ch)}"
print("INT4: constraints validated")
PY

log "INT4 quantized model: PASSED"

# Summary
log ""
log "=========================================="
log "E2E Quantized Models: ALL PASSED"
log "=========================================="
log "INT8 model: $INT8_MODEL"
log "INT4 model: $INT4_MODEL"
log "Artifacts kept under: $TMP_DIR"
log "Note: project policy forbids auto-cleanup via rm -rf; delete $TMP_DIR manually if desired."
