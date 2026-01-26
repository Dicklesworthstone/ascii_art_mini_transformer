#!/usr/bin/env bash
set -uo pipefail

# Emit an environment snapshot for debugging E2E failures.
#
# Usage: tests/e2e/env_snapshot.sh <repo_root> [python_bin]
#
# Notes:
# - Best-effort: never fail the caller if an info command is missing.
# - Output is intended to be piped through `tee -a "$LOG_FILE"`.

REPO_ROOT="${1:-}"
PYTHON_BIN="${2:-}"

if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT="$(pwd)"
fi

echo "=== Environment snapshot ==="
timestamp="$(date -Is 2>/dev/null || date)"
echo "timestamp: $timestamp"

if command -v git >/dev/null 2>&1 && git -C "$REPO_ROOT" rev-parse HEAD >/dev/null 2>&1; then
  echo "git_sha: $(git -C "$REPO_ROOT" rev-parse HEAD)"
else
  echo "git_sha: (unavailable)"
fi

if command -v uname >/dev/null 2>&1; then
  echo "uname: $(uname -a)"
fi

CPU_MODEL=""
if command -v lscpu >/dev/null 2>&1; then
  CPU_MODEL="$(
    lscpu | awk -F: '/Model name/ {print $2}' | sed 's/^ *//' | head -n 1 || true
  )"
fi
if [[ -z "$CPU_MODEL" && -r /proc/cpuinfo ]]; then
  CPU_MODEL="$(
    awk -F: '/model name/ {print $2; exit}' /proc/cpuinfo | sed 's/^ *//' || true
  )"
fi
if [[ -n "$CPU_MODEL" ]]; then
  echo "cpu_model: $CPU_MODEL"
fi

if command -v rustc >/dev/null 2>&1; then
  echo "rustc: $(rustc --version)"
else
  echo "rustc: (unavailable)"
fi

if command -v cargo >/dev/null 2>&1; then
  echo "cargo: $(cargo --version)"
else
  echo "cargo: (unavailable)"
fi

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  fi
fi

if [[ -n "$PYTHON_BIN" ]]; then
  echo "python_bin: $PYTHON_BIN"
  "$PYTHON_BIN" - <<'PY' || true
from __future__ import annotations

import sys

print(f"python_version: {sys.version.splitlines()[0]}")


def pkg_version(module_name: str) -> str:
    try:
        mod = __import__(module_name)
    except Exception as exc:  # pragma: no cover
        return f"(unavailable: {exc.__class__.__name__})"
    return str(getattr(mod, "__version__", "unknown"))


print(f"torch: {pkg_version('torch')}")
print(f"safetensors: {pkg_version('safetensors')}")
PY
else
  echo "python_bin: (unavailable)"
fi

echo "=== End environment snapshot ==="
