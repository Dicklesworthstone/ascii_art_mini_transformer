#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
TMP_DIR="$(mktemp -d -t ascii_e2e_full_${TS}_XXXXXX)"
LOG_FILE="$TMP_DIR/e2e_full_${TS}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }

CURRENT_STAGE="setup"
SECONDS=0
on_exit() {
  status=$?
  if [[ $status -ne 0 ]]; then
    echo "E2E full: FAILED (stage=$CURRENT_STAGE status=$status elapsed=${SECONDS}s)" >&2
    echo "Artifacts kept under: $TMP_DIR" >&2
    echo "Log file: $LOG_FILE" >&2
  fi
}
trap on_exit EXIT

run_stage() {
  local name="$1"
  shift
  CURRENT_STAGE="$name"
  local start=$SECONDS
  log "=== E2E: $name ==="
  if ! "$@" 2>&1 | tee -a "$LOG_FILE"; then
    return 1
  fi
  local elapsed=$((SECONDS - start))
  log "=== E2E: $name done (${elapsed}s) ==="
}

log "Repo: $REPO_ROOT"
log "Temp: $TMP_DIR"
log "Log:  $LOG_FILE"

bash "$REPO_ROOT/tests/e2e/env_snapshot.sh" "$REPO_ROOT" 2>&1 | tee -a "$LOG_FILE"

run_stage "data" "$REPO_ROOT/tests/e2e/e2e_data.sh"

run_stage "train" "$REPO_ROOT/tests/e2e/e2e_train.sh"

EXPORT_DIR="$TMP_DIR/exported"

run_stage "export" "$REPO_ROOT/tests/e2e/e2e_python_export.sh" "$EXPORT_DIR"

run_stage "python inference" env E2E_MODEL_PATH="$EXPORT_DIR/model.safetensors" \
  "$REPO_ROOT/tests/e2e/e2e_python_infer.sh"

run_stage "rust" env E2E_MODEL_PATH="$EXPORT_DIR/model.safetensors" E2E_MAX_CHARS=80 \
  "$REPO_ROOT/tests/e2e/e2e_rust.sh"

run_stage "quant" env E2E_MODEL_DIR="$EXPORT_DIR" E2E_MAX_CHARS=80 \
  "$REPO_ROOT/tests/e2e/e2e_quant.sh"

run_stage "embedded" "$REPO_ROOT/tests/e2e/e2e_embedded.sh" "$EXPORT_DIR"

CURRENT_STAGE="done"
log "=== ALL E2E STEPS PASSED (elapsed=${SECONDS}s) ==="
log "Artifacts kept under: $TMP_DIR"
log "Note: project policy forbids auto-cleanup via rm -rf; delete $TMP_DIR manually if desired."
