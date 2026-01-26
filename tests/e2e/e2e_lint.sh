#!/usr/bin/env bash
# E2E lint/typecheck gates for Python and Rust.
#
# Runs:
#   Python: ruff check, ruff format --check, mypy --strict
#   Rust:   cargo fmt --check, cargo clippy --all-targets
#
# Logs are written to a temp directory (kept; no auto-delete per project policy).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
TMP_DIR="$(mktemp -d -t ascii_e2e_lint_${TS}_XXXXXX)"
LOG_FILE="$TMP_DIR/e2e_lint_${TS}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }

CURRENT_STEP="setup"
SECONDS=0
on_exit() {
  status=$?
  if [[ $status -ne 0 ]]; then
    echo "E2E lint: FAILED (step=$CURRENT_STEP status=$status elapsed=${SECONDS}s)" >&2
    echo "Artifacts kept under: $TMP_DIR" >&2
    echo "Log file: $LOG_FILE" >&2
  fi
}
trap on_exit EXIT

log "=== E2E Lint/Typecheck Gates ==="
log "Repo: $REPO_ROOT"
log "Temp: $TMP_DIR"
log "Log:  $LOG_FILE"
log ""

CURRENT_STEP="env_snapshot"
bash "$REPO_ROOT/tests/e2e/env_snapshot.sh" "$REPO_ROOT" 2>&1 | tee -a "$LOG_FILE"

cd "$REPO_ROOT"

# Activate Python venv if available
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  log "Python venv activated"
fi

export PYTHONPATH="$REPO_ROOT/python"

# ==================== Python Gates ====================
log "=== Python Lint Gates ==="

log "[1/3] Running: ruff check python/"
CURRENT_STEP="ruff_check"
if ruff check python/ 2>&1 | tee -a "$LOG_FILE"; then
  log "ruff check: PASSED"
else
  log "ruff check: FAILED"
  exit 1
fi

log "[2/3] Running: ruff format --check python/"
CURRENT_STEP="ruff_format"
if ruff format --check python/ 2>&1 | tee -a "$LOG_FILE"; then
  log "ruff format: PASSED"
else
  log "ruff format: FAILED (run 'ruff format python/' to fix)"
  exit 1
fi

log "[3/3] Running: mypy python/ --strict"
CURRENT_STEP="mypy_strict"
if mypy python/ --strict 2>&1 | tee -a "$LOG_FILE"; then
  log "mypy: PASSED"
else
  log "mypy: FAILED (see log for details)"
  exit 1
fi

log ""

# ==================== Rust Gates ====================
log "=== Rust Lint Gates ==="

cd "$REPO_ROOT/rust/ascii-gen"

log "[1/2] Running: cargo fmt --check"
CURRENT_STEP="cargo_fmt"
if cargo fmt --check 2>&1 | tee -a "$LOG_FILE"; then
  log "cargo fmt: PASSED"
else
  log "cargo fmt: FAILED"
  exit 1
fi

log "[2/2] Running: cargo clippy --all-targets -- -D warnings"
CURRENT_STEP="cargo_clippy"
if cargo clippy --all-targets -- -D warnings 2>&1 | tee -a "$LOG_FILE"; then
  log "cargo clippy: PASSED"
else
  log "cargo clippy: FAILED"
  exit 1
fi

log ""
log "=== E2E Lint: ALL PASSED ==="
log "Log file: $LOG_FILE"
log "Artifacts kept under: $TMP_DIR"
log "Note: project policy forbids auto-cleanup via rm -rf; delete $TMP_DIR manually if desired."
