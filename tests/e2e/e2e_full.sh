#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
TMP_DIR="$(mktemp -d -t ascii_e2e_full_${TS}_XXXXXX)"
LOG_FILE="$TMP_DIR/e2e_full_${TS}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE" >/dev/null; }

log "Repo: $REPO_ROOT"
log "Temp: $TMP_DIR"
log "Log:  $LOG_FILE"

bash "$REPO_ROOT/tests/e2e/env_snapshot.sh" "$REPO_ROOT" 2>&1 | tee -a "$LOG_FILE"

log "=== E2E: data ==="
"$REPO_ROOT/tests/e2e/e2e_data.sh" 2>&1 | tee -a "$LOG_FILE"

log "=== E2E: train ==="
"$REPO_ROOT/tests/e2e/e2e_train.sh" 2>&1 | tee -a "$LOG_FILE"

EXPORT_DIR="$TMP_DIR/exported"

log "=== E2E: export ==="
"$REPO_ROOT/tests/e2e/e2e_python_export.sh" "$EXPORT_DIR" 2>&1 | tee -a "$LOG_FILE"

log "=== E2E: rust ==="
E2E_MODEL_PATH="$EXPORT_DIR/model.safetensors" E2E_MAX_CHARS=80 "$REPO_ROOT/tests/e2e/e2e_rust.sh" 2>&1 | tee -a "$LOG_FILE"

log "=== E2E: quant ==="
E2E_MODEL_DIR="$EXPORT_DIR" E2E_MAX_CHARS=80 "$REPO_ROOT/tests/e2e/e2e_quant.sh" 2>&1 | tee -a "$LOG_FILE"

log "=== E2E: embedded ==="
"$REPO_ROOT/tests/e2e/e2e_embedded.sh" "$EXPORT_DIR" 2>&1 | tee -a "$LOG_FILE"

log "=== ALL E2E STEPS PASSED ==="
log "Artifacts kept under: $TMP_DIR"
log "Note: project policy forbids auto-cleanup via rm -rf; delete $TMP_DIR manually if desired."
