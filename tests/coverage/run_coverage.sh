#!/usr/bin/env bash
# Run Python test coverage and generate reports.
#
# Usage: tests/coverage/run_coverage.sh [--html] [--xml]
#
# Options:
#   --html    Generate HTML report under coverage_html/
#   --xml     Generate XML report (coverage.xml) for CI integration
#
# Reports are kept (no auto-delete) per project policy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

# Parse arguments
GENERATE_HTML=false
GENERATE_XML=false
for arg in "$@"; do
    case "$arg" in
        --html) GENERATE_HTML=true ;;
        --xml) GENERATE_XML=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# Artifacts directory (optional). We keep it and print the path prominently.
ARTIFACT_DIR=""
if $GENERATE_HTML || $GENERATE_XML; then
    ARTIFACT_DIR="$(mktemp -d -t ascii_cov_XXXXXX)"
    export COVERAGE_FILE="$ARTIFACT_DIR/.coverage"
fi

# Build coverage report options
declare -a COV_REPORT_ARGS
COV_REPORT_ARGS=(--cov-report=term-missing)
if $GENERATE_HTML; then
    COV_REPORT_ARGS+=(--cov-report="html:$ARTIFACT_DIR/coverage_html")
fi
if $GENERATE_XML; then
    COV_REPORT_ARGS+=(--cov-report="xml:$ARTIFACT_DIR/coverage.xml")
fi

# Activate venv if available
if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

echo "=== Running Python tests with coverage ==="
echo "Working directory: $REPO_ROOT"
if [[ -n "$ARTIFACT_DIR" ]]; then
    echo "Artifacts dir: $ARTIFACT_DIR"
fi
echo ""

# Run pytest with coverage
# Use PYTHONPATH to ensure python/ modules are importable
PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT/python:$REPO_ROOT" pytest python/tests/ \
    --cov=python \
    "${COV_REPORT_ARGS[@]}" \
    -v

echo ""
echo "=== Coverage complete ==="

if $GENERATE_HTML; then
    echo "HTML report: $ARTIFACT_DIR/coverage_html/index.html"
fi
if $GENERATE_XML; then
    echo "XML report: $ARTIFACT_DIR/coverage.xml"
fi
if [[ -n "${COVERAGE_FILE:-}" ]]; then
    echo "Coverage data: $COVERAGE_FILE"
fi
