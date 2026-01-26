# Dependency Upgrade Log

**Date:** 2026-01-26
**Project:** ascii_art_mini_transformer
**Language:** Rust + Python
**Manifests:** `rust/ascii-gen/Cargo.toml`, `python/requirements.txt`

---

## Summary

| Metric | Count |
|--------|-------|
| **Total dependencies (direct)** | 24 (Rust 9 + Python 15) |
| **Updated** | 3 |
| **Skipped** | 1 |
| **Failed (rolled back)** | 0 |
| **Requires attention** | 1 |

---

## Baseline (before upgrades)

### Rust

- Manifest: `rust/ascii-gen/Cargo.toml`
- Lockfile: `rust/ascii-gen/Cargo.lock`
- Tests: `RUST_TEST_THREADS=1 cargo test`

### Python

- Manifest: `python/requirements.txt` (currently unpinned)
- Tests: `PYTHONPATH=python .venv/bin/python -m pytest python/tests -v --tb=short`

---

## Successfully Updated

### candle-core: 0.8.4 → 0.9.2

**Changelog:** No published CHANGELOG for 0.8→0.9 found in upstream repo; relied on compile + test suite.

**Breaking changes:** Unknown (0.x minor bump); no code changes required in this repo.

**Tests:** ✓ Passed (`RUST_TEST_THREADS=1 cargo test`)

---

### candle-nn: 0.8.4 → 0.9.2

**Changelog:** Same as `candle-core` (kept versions in lockstep).

**Breaking changes:** Unknown (0.x minor bump); no code changes required in this repo.

**Tests:** ✓ Passed (`RUST_TEST_THREADS=1 cargo test`)

---

### safetensors (Rust): 0.4.5 → 0.7.0

**Changelog:** No consolidated CHANGELOG found upstream; relied on compile errors + tests.

**Breaking changes observed (fixed here):**
- `serialize(...)` now takes `Option<...>` by value (was `&Option<...>`): updated `rust/ascii-gen/tests/integration.rs`.
- `SafeTensors::names()` now yields `&str` (was `String`-like): updated `rust/ascii-gen/src/weights/quantized.rs`.

**Tests:** ✓ Passed (`RUST_TEST_THREADS=1 cargo test`)

---

## Skipped

### Python requirements pinning / lockfile

**Reason:** `python/requirements.txt` is intentionally unpinned, so installs resolve to latest stable versions by default.

**What was done instead:**
- Verified current latest stable versions via `pip index versions ...`.
- Installed the missing direct deps listed in `python/requirements.txt`: `torchao`, `rich`, `rapidfuzz`.
- Confirmed Python unit tests pass under Python 3.13.

---

## Failed Updates (Rolled Back)

_(none yet)_

---

## Requires Attention

### Reproducible Python dependency pinning

**Issue:** Because `python/requirements.txt` is unpinned, CI/local installs can drift over time.

**Recommendation:** If you want reproducibility, introduce a lock/constraints mechanism (e.g. `uv.lock` or a pinned constraints file) and update CI/README accordingly.

---

## Security Notes

### Python (`pip-audit`)

- Result: **No known vulnerabilities found** (`python/requirements.txt`).

### Rust (`cargo audit`)

- Result: **No vulnerabilities found**, but 1 advisory warning:
  - `RUSTSEC-2024-0436` (`paste`): unmaintained (transitive via `candle-core`).

---

## Commands Used

```bash
# Rust
cd rust/ascii-gen
cargo update -p candle-core --precise 0.9.2
RUST_TEST_THREADS=1 cargo test
cargo clippy --all-targets -- -D warnings
cargo audit

# Python
cd /data/projects/ascii_art_mini_transformer
.venv/bin/python -m pip install -r python/requirements.txt
PYTHONPATH=python .venv/bin/python -m pytest python/tests -v --tb=short
.venv/bin/python -m pip install pip-audit
.venv/bin/python -m pip_audit -r python/requirements.txt
```
