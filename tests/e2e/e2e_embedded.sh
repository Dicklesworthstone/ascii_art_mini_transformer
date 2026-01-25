#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
TMP_DIR="$(mktemp -d -t ascii_e2e_embedded_${TS}_XXXXXX)"
LOG_FILE="$TMP_DIR/e2e_embedded_${TS}.log"

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

EXPORT_DIR="${1:-$TMP_DIR/exported}"
mkdir -p "$EXPORT_DIR"
log "Export dir: $EXPORT_DIR"

if [[ ! -f "$EXPORT_DIR/model.safetensors" ]]; then
  log "Exporting tiny model for embedding..."
  "$PYTHON_BIN" -m train.export \
    --output-dir "$EXPORT_DIR" \
    --quantize none \
    --preset small \
    --n-layer 2 \
    --n-head 2 \
    --n-embd 64 \
    --block-size 256 \
    2>&1 | tee -a "$LOG_FILE"
fi

log "Building Rust CLI (release) with embedded weights..."
cd "$REPO_ROOT/rust/ascii-gen"
ASCII_GEN_EXPORT_DIR="$EXPORT_DIR" cargo build --release --features embedded-weights 2>&1 | tee -a "$LOG_FILE"

BIN="$REPO_ROOT/rust/ascii-gen/target/release/ascii-gen"
test -x "$BIN"

log "Verifying embedded assets are present..."
"$BIN" --info 2>&1 | tee -a "$LOG_FILE" | grep -q "Embedded weights: yes"

log "Running embedded CLI smoke generation (no --model)..."
OUT_FILE="$TMP_DIR/generated.txt"
"$BIN" \
  --width 40 \
  --max-lines 20 \
  --max-chars 80 \
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

log "E2E embedded: PASSED"
log "Artifacts kept under: $TMP_DIR"
log "Note: project policy forbids auto-cleanup via rm -rf; delete $TMP_DIR manually if desired."

