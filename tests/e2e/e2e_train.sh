#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
TMP_DIR="$(mktemp -d -t ascii_e2e_train_${TS}_XXXXXX)"
LOG_FILE="$TMP_DIR/e2e_train_${TS}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }

CURRENT_STEP="setup"
SECONDS=0
on_exit() {
  status=$?
  if [[ $status -ne 0 ]]; then
    echo "E2E train: FAILED (step=$CURRENT_STEP status=$status elapsed=${SECONDS}s)" >&2
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

log "Repo: $REPO_ROOT"
log "Temp: $TMP_DIR"
log "Log:  $LOG_FILE"
log "Python: $PYTHON_BIN"

CURRENT_STEP="env_snapshot"
bash "$REPO_ROOT/tests/e2e/env_snapshot.sh" "$REPO_ROOT" "$PYTHON_BIN" 2>&1 | tee -a "$LOG_FILE"

export PYTHONPATH="$REPO_ROOT/python"

DB_PATH="$TMP_DIR/train_ascii.db"
CKPT_DIR="$TMP_DIR/checkpoints"
OUT_DIR="$TMP_DIR/exported"

log "Creating small training DB..."
CURRENT_STEP="create_db"
"$PYTHON_BIN" - <<PY | tee -a "$LOG_FILE"
from __future__ import annotations

from pathlib import Path

from data.db import connect, initialize, insert_ascii_art

db_path = Path(${DB_PATH@Q})
conn = connect(db_path)
initialize(conn)

def make_box(label: str) -> str:
    top = "+" + "-" * (len(label) + 2) + "+"
    mid = "| " + label + " |"
    return "\n".join([top, mid, top])

inserted = 0
for i in range(120):
    label = f"smoke-{i:03d}"
    raw_text = make_box(label)
    row_id = insert_ascii_art(
        conn,
        raw_text=raw_text,
        source="e2e_train",
        title=f"Smoke {i}",
        description="ASCII box",
        category="simple",
    )
    if row_id is not None:
        inserted += 1

count = conn.execute("SELECT COUNT(*) FROM ascii_art").fetchone()[0]
print(f"Inserted={inserted} total_rows={count} db={db_path}")
assert count >= 50, "training DB too small for train/val split"
conn.close()
PY

log "Running training smoke test (CPU, tiny model, few iters)..."
CURRENT_STEP="train"
"$PYTHON_BIN" -m train.train \
  --db-path "$DB_PATH" \
  --checkpoint-dir "$CKPT_DIR" \
  --device cpu \
  --dtype float32 \
  --block-size 256 \
  --n-layer 2 \
  --n-head 2 \
  --n-embd 64 \
  --dropout 0.0 \
  --batch-size 4 \
  --gradient-accumulation-steps 1 \
  --learning-rate 0.001 \
  --warmup-iters 1 \
  --max-iters 3 \
  --eval-interval 1000000 \
  --save-interval 1000000 \
  --num-workers 0 \
  2>&1 | tee -a "$LOG_FILE"

log "Verifying expected checkpoint artifacts exist..."
CURRENT_STEP="verify_checkpoint"
test -f "$CKPT_DIR/final.pt"

log "Exporting from checkpoint (smoke test)..."
CURRENT_STEP="export"
"$PYTHON_BIN" -m train.export \
  --checkpoint "$CKPT_DIR/final.pt" \
  --output-dir "$OUT_DIR" \
  --dtype float32 \
  --quantize none \
  2>&1 | tee -a "$LOG_FILE"

log "Verifying expected export artifacts exist..."
CURRENT_STEP="verify_export"
test -f "$OUT_DIR/model.safetensors"
test -f "$OUT_DIR/config.json"
test -f "$OUT_DIR/tokenizer.json"

log "Verifying export config matches training hyperparameters..."
CURRENT_STEP="verify_export_config"
"$PYTHON_BIN" - <<PY | tee -a "$LOG_FILE"
from __future__ import annotations

import json
from pathlib import Path

cfg_path = Path(${OUT_DIR@Q}) / "config.json"
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
assert cfg["n_layer"] == 2, cfg
assert cfg["n_head"] == 2, cfg
assert cfg["n_embd"] == 64, cfg
assert cfg["block_size"] == 256, cfg
print("config.json ok:", {k: cfg[k] for k in ("n_layer", "n_head", "n_embd", "block_size")})
PY

log "E2E train: PASSED"
log "Artifacts kept under: $TMP_DIR"
log "Note: project policy forbids auto-cleanup via rm -rf; delete $TMP_DIR manually if desired."
