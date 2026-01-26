#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
TMP_DIR="$(mktemp -d -t ascii_e2e_data_${TS}_XXXXXX)"
LOG_FILE="$TMP_DIR/e2e_data_${TS}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }

CURRENT_STEP="setup"
SECONDS=0
on_exit() {
  status=$?
  if [[ $status -ne 0 ]]; then
    echo "E2E data: FAILED (step=$CURRENT_STEP status=$status elapsed=${SECONDS}s)" >&2
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
DB_PATH="$TMP_DIR/test_ascii.db"

log "Creating test DB + inserting sample rows..."
CURRENT_STEP="create_db"
"$PYTHON_BIN" - <<PY | tee -a "$LOG_FILE"
from __future__ import annotations

import sqlite3
from pathlib import Path

from data.db import connect, initialize, insert_ascii_art

db_path = Path(${DB_PATH@Q})
conn = connect(db_path)
initialize(conn)

samples = [
    ("  /\\\\_/\\\\\\n ( o.o )\\n  > ^ <", "e2e", "Cat", "A cute cat", "animal"),
    (" __|__\\n(  o o )\\n \\\\  ^ /\\n  |||", "e2e", "Robot", "A simple robot", "object"),
    ("#####\\n#   #\\n#####", "e2e", "Box", "A simple box", "simple"),
    ("TEST\\n====\\nBanner", "e2e", "Test Banner", "FIGlet-ish banner", "banner"),
]

inserted = 0
for raw_text, source, title, desc, category in samples:
    row_id = insert_ascii_art(
        conn,
        raw_text=raw_text,
        source=source,
        title=title,
        description=desc,
        category=category,
    )
    if row_id is not None:
        inserted += 1

count = conn.execute("SELECT COUNT(*) FROM ascii_art").fetchone()[0]
print(f"Inserted={inserted} total_rows={count} db={db_path}")
conn.close()
PY

log "Running quality pipeline (dry-run, no dedup, limit=100)..."
CURRENT_STEP="quality_pipeline"
"$PYTHON_BIN" -m data.quality_pipeline \
  --db-path "$DB_PATH" \
  --output "$TMP_DIR/quality_report.json" \
  --dry-run \
  --no-dedup \
  --limit 100 \
  2>&1 | tee -a "$LOG_FILE"

log "E2E data pipeline: PASSED"
log "Artifacts kept under: $TMP_DIR"
log "Note: project policy forbids auto-cleanup via rm -rf; delete $TMP_DIR manually if desired."
