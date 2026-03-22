# Changelog

All notable changes to **ASCII Art Mini Transformer** are documented here.

This project has no tagged releases yet. Changes are organized chronologically by development phase, with commit links for traceability. Commits that are purely beads-tracking metadata (`beads:`, `br sync`, `br:`) are omitted for readability.

Repository: <https://github.com/Dicklesworthstone/ascii_art_mini_transformer>

---

## [Unreleased] — HEAD

Tracks everything on `main` since the initial import on 2026-01-25.

### License & Branding (2026-02-21)

- **MIT with OpenAI/Anthropic Rider** — added `LICENSE` restricting use by OpenAI, Anthropic, and affiliates without express written permission from Jeffrey Emanuel ([`41527b7`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/41527b705886e03a2eb2041e4721166f8404fcf0))
- **Social preview image** — added 1280x640 GitHub OG share image ([`6b4471b`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6b4471bedac71a060c39cace62be1868da7d0fa4))

### Dependency Maintenance (2026-02-09)

- **Batch bump** of GH Actions (`actions/checkout`, `actions/setup-python`, `actions/upload-artifact`) and Rust crates (`anyhow`, `clap`) ([`4638512`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/46385125659b79d85c9d3752a75468bf22ebe1ab))
- Dependabot branches open for: `clap` 4.5.58/4.5.60/4.6.0, `anyhow` 1.0.101/1.0.102, `rand` 0.9.2, `actions/upload-artifact` 6.0.0/7.0.0, `actions/checkout` 6.0.2, `actions/setup-python` 6.2.0, `actions/setup-go` 6.3.0, `Swatinem/rust-cache` 2.9.1

### Data Pipeline Hardening (2026-02-02 .. 2026-02-03)

- **HuggingFace ingestion early-stop fix** — prevent premature termination during large dataset pulls ([`9ff80c1`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/9ff80c12f43433e308c1ef26639cf7e2d5caf0ca))
- **`a.out` added to `.gitignore`** ([`f7cf953`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/f7cf953d7770725801abced5cfc19b7f8fe64b31))

### Inference Parity (2026-02-02)

- **Python `max_tokens=0` semantics** now mirrors Rust behavior (generate until natural stop) ([`6f57631`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6f57631dab8ea4b422a403a7137dfa7b359c298f))
- **Fallback position computation cleanup** in test suite ([`b8b20b9`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/b8b20b9b4a081f1741860b6651b6f91e2aadf860))

### GPU Training CI (2026-02-01)

- **`train_gpu_dispatch.yml`** — new dispatchable workflow for self-hosted GPU training runs ([`e8028ba`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/e8028ba9e13fed1a2aa2225b7b22ba68ce34e8d5), [`6769195`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6769195d6f72ed0e111625841e803ff0171dc2ae))
- **GPU workflow fixes** — YAML heredoc escaping, `workflow_dispatch` input schema ([`4cb6653`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/4cb6653f949d888dbb361bae228b6c937e72b1d7), [`0cb8659`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/0cb86594e9e1a0fe1d37b0b7a0a25464f6231e5a))
- **`actionlint` validation** added to CI ([`ea7ef77`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/ea7ef77fc885c7c613274d30bbd62fa116004f37))

### Scraper & Data Robustness (2026-01-26)

- **SAUCE metadata parsing hardened** — handle malformed SAUCE records and edge-case gallery paths in `scrape_16colors` and `scrape_asciiart` ([`6fd554e`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6fd554e9eef3ba8fc8ab1adeaab3728e118393e4))
- **`scrape_textfiles` categorization fix** — correct category assignment and progress-bar semantics ([`3259145`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/32591450175bf1ca0a2a2c0cfdeb0b1aed4067f4))
- **Drop cached causal masks** from checkpoints to reduce file size ([`42a0e6f`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/42a0e6f1f3dd475f956b8b74a106557c262955d3))
- **Harden data + train tests; fix Rust `max_chars=0`** — Rust decoder now treats 0 as "no limit" instead of blocking all output ([`ab6c811`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/ab6c8118859ac1a7673eb1e84e562c54318bc664))
- **Dependabot configuration** added for Cargo and GitHub Actions ([`021815c`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/021815cb62b0ce3431c31ee39442936f17f53327))

### Type Safety & Code Quality (2026-01-26)

- **`mypy --strict` gate passes** across entire Python codebase ([`b5b8ad9`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/b5b8ad99b56d3423d9a29a38642e0eed2e9b7b02))
- **Remove mocks via RNG injection** — tests use deterministic RNG seeds instead of mock objects ([`134351d`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/134351df371f0ed3ecbbafaba13b31f72a68c345))
- **`ruff format`** applied across all Python files ([`752a644`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/752a64437eba381852a81c24e6850877d334dea6))
- **Quality pipeline typing** tightened ([`28ebbb3`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/28ebbb357a799435f3a0324d4603c99c58de88df))

### Bug Fixes (2026-01-26)

- **Fail on invalid `config.json`** — Rust loader now returns a clear error instead of silently loading defaults ([`0bcddc6`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/0bcddc6234684453d9de92de5ae2d5ee1e4e4f8b))
- **Fix last-line newline handling; cache causal mask** — prevents trailing-newline artifacts in generated art ([`85dd07d`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/85dd07db345992778d0021ec9decd1d5dd75c824))
- **Align sampling edge cases with Rust** — Python sampler now handles temperature=0, top_k=0, and empty logits identically to Rust ([`aa75749`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/aa75749ee653c4cbe3061116ad9ec9fe9870a53a))

### Comprehensive Test Coverage (2026-01-25)

Major test expansion covering every Python module. All tests use real objects (no mocks) with RNG injection for determinism.

- **Sampler + export unit tests** ([`47b2060`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/47b2060221ccaa53da892e8b7fc970d3851ffdf4))
- **Missing Python test coverage landed** — scraper helpers, FIGlet generator, sampling invariants, dataset loader, quality pipeline, DB update semantics, DB metadata edge cases ([`b62ea52`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/b62ea52bed9a4438afe7dbbb8c8b619eb9f92ebc) and 8 preceding commits: [`c5abd67`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/c5abd6769ee10f7d144865f42ebfe6e6d8360761), [`400fa7f`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/400fa7f2e249dd21a9ee3c82503f00a762a3a29c), [`e042aab`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/e042aabf1fcc514ab36036a29a920e64aaacefc5), [`72897d3`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/72897d360414d957a1f3e7a2b58bddba3f225062), [`35020a0`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/35020a0c9909e81ab87d6cca6ca0b4927efbf98b), [`d304c03`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/d304c03a1e429939b95981004958e1bacbe2750c), [`0ec0777`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/0ec0777082c810757e4f55d8ab3928f4c7741518))
- **Python export/ingest/positional encoding tests** ([`ab54ee0`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/ab54ee0976145ad12ea846b09ddba43e8550289a))
- **Model forward/forward_last + position clamp tests** ([`b23da7a`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/b23da7a429a48251d121f30dd126fa347714bed3))
- **Sampling edge-case tests** ([`0e95160`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/0e95160c1b48e01405237b2953ce8e829a496abe))
- **Rust CLI output format integration tests** ([`b852eb2`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/b852eb26a61cdf34df2afe67f68036587ec28982))
- **Constrained decoder edge-case tests** expanded ([`8d871c4`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/8d871c42248a5ba77e78a05ebd45f210b51d1e38))
- **Transformer model unit tests** ([`b23da7a`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/b23da7a429a48251d121f30dd126fa347714bed3))
- **Deprecate legacy `database.py`** in favor of `db.py` ([`51fd5c9`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/51fd5c9924573319cbdd9430870b889149ada2e7))

### CI & Coverage Infrastructure (2026-01-25)

- **Python coverage gates** added to CI workflow ([`5fe6cc7`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/5fe6cc74ea59b349f4c3a668f4a5680c10d4ae3a))
- **Optional Rust coverage workflow** (`rust_coverage.yml`) using `cargo llvm-cov` ([`1b8fd51`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/1b8fd511ef00c64dfa1c56c74202b82970358678))
- **`pytest-cov` tooling and helper** script at `tests/coverage/run_coverage.sh` ([`d090e2a`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/d090e2a1cffcdcc84f2cfe7e1fdc4e0dfb0219df))
- **Self-hosted GPU training workflow** (`train_gpu.yml`) for large-scale training on `gpu`-labeled runners ([`4cac7fa`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/4cac7fac178e4491578686224aabf920da245233))
- **E2E workflow updated** — dispatch-only mode + artifact uploads ([`6266e1a`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6266e1ac161b742ee01f08015a64b5b9e87c68b7))

### E2E Test Suite (2026-01-25)

- **E2E logging improvements** — per-phase timings and failure context ([`66dd60b`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/66dd60bb020618d5658618c1e4f028b34c8e03bb))
- **Python inference CLI e2e smoke test** (`e2e_python_infer.sh`) ([`dfae0ec`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/dfae0ec62e4e9588d3fdf2bd9d5a7838f449117a))
- **E2E lint/typecheck gate** (`e2e_lint.sh`) — runs `ruff`, `mypy --strict`, and `clippy` ([`2da283b`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/2da283b0c59c7754982967486791a62c57b41029))
- **Quantized weights E2E smoke** (`e2e_quant.sh`) + env snapshot script ([`0acba79`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/0acba79620ee5b8bad3769f8fdceb226d3aba867))
- **Weights loader + quant scheme tests** ([`70105a2`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/70105a29349203ef93ff9e7cf9f251ed6b6ff14c))
- **Crossval golden fixtures expanded** with 2 new fixtures (golden_3, golden_4) and extended greedy golden ([`31df9bc`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/31df9bc3357ea3d719bc8d12568351ad0c3217c5))

### Rust Inference Engine (2026-01-25)

- **Candle bumped to 0.9.2** ([`0da744e`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/0da744ecaf12c58632ed866389bcb2969c90f796))
- **Clippy cleanups** in quantized weight loader ([`6fbfad1`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6fbfad1ddae8dd868c9260dc09049cb2db440bcb))
- **Attention mask omitted** to enable SDPA fast-path in training ([`c894338`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/c8943388efc64303ae595b572fbc3abd3c2b59b0))

### Training Pipeline (2026-01-25)

- **Training presets** — `--preset small|medium|large` for quick configuration ([`1fc0f9e`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/1fc0f9ee03b5d0afb41cd7482895dc53bb722742))
- **Document `--preset` flag** in README and `--help` ([`aae92d2`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/aae92d290cc362707be801d70e22ec21b481ec69))
- **More CLI flags exposed** — granular control over training hyperparameters ([`9dc292b`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/9dc292b958d61090edfd298d44267c549a19f6ed))
- **CPU-friendly defaults + fail-fast** on misconfiguration ([`45dbba2`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/45dbba237b556190dc80cee548cf108e44db5d1a))
- **AMP support** on `cuda:0` ([`a121b37`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/a121b370ef9cd1ee77bf990c74bd817e683fe7a7))
- **Optional auto-export** after training completes ([`45cd993`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/45cd99341ba51daf4736fd0213b5b532de0284ce))
- **Harden LR schedule** — disable `interval=0` which caused division-by-zero ([`6705d74`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6705d74254da90d9d943dc2786ebe7defedeaa8f))
- **Handle missing `charset` column** in training dataset gracefully ([`41d65fd`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/41d65fd0e54c8fbd6d76d500d9eb382c34a6595a))
- **Safe `torch.load` defaults** — use `weights_only=True` by default, `--unsafe-load` opt-in ([`e22b2d5`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/e22b2d55d6ae7307c3ebf2461eabe55b6e07a027))
- **Drop unused row/col position tensors** from model forward pass ([`dab0627`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/dab0627c50b0806cff424bc7826b81e895c30985))

### Model & Inference Improvements (2026-01-25)

- **SDPA preferred** for attention computation (PyTorch scaled-dot-product) ([`5ecc4da`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/5ecc4da504ac609e719914e7165bbfa6a1254ab1))
- **Clamp Rust 2D positional indices** — prevents out-of-bounds when generating art larger than training max ([`1fb8a85`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/1fb8a85af25ef29208c790e5c21b37a293a7b104))
- **Tighten Python inference constraints** — stricter width/height enforcement ([`9c8d3a6`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/9c8d3a6349966aa235ed882e6cf96e2926ff0c8c))
- **Python inference CLI** — load from both checkpoints and exported safetensors ([`13a080e`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/13a080e71547aa821a9506ffe963e552d80ddbb4))

### Export Pipeline (2026-01-25)

- **Configurable fresh model** for export testing ([`93fa7b4`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/93fa7b40869f09432967e713d859105f41607965))
- **Validate checkpoint state dict types** before export ([`27815ba`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/27815baa06eadd48261793b1b1371f2cd1e4107f))
- **`--unsafe-load` on older torch** — backward compatibility for pre-2.6 checkpoints ([`8507b1f`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/8507b1f61c7f9724f51d2d4bdc88bef6efe331f7))
- **Fix tokenizer export** — ensure augmentation stays ASCII-only ([`04bff5b`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/04bff5b4167412c9d32688bcdbd772f0c0a4b5fd))
- **Guard `tokenizer.json` export** with validation test ([`64cd721`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/64cd7210a5254fde799c3b799c4eae4f8e5ce806))

### Data Layer (2026-01-25)

- **Dataset scaled to 500k+ rows** with improved FIGlet + HuggingFace ingestion ([`9598022`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/95980228da13cbe263afd2ab82ff665af9b519ae))
- **Csplk ingest heuristics** — `force`/`resume` flags, improved block parsing and art-line detection ([`233be1c`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/233be1caaf6b7b38fda1e7a57d15ecb7ef2ce610), [`de277b5`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/de277b516b44020824d4cc55480dd33682e5fbc8))
- **Normalize newlines** and skip empty ingests ([`e2ce331`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/e2ce3311fec4947d07f59c11beeaf384e9660f80))
- **Skip empty ASCII art inserts** by default in `db.py` ([`f742ffe`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/f742ffe4ffc33b530010bbc3055bd6d0878d231a))
- **Cache SQLite connection safely** in dataset loader ([`dded704`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/dded7049dc6367f9b28a80d6ea29ca71a9d9fb40))
- **`model_available()` fixed** to check for `config.json` presence ([`842b3d0`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/842b3d0119515a3b6602910ab9343d2afe5aaa13))

### Documentation (2026-01-25 .. 2026-01-26)

- **README rewrite** — comprehensive docs with architecture diagram, command reference, FAQ, troubleshooting, comparison table ([`639cfb1`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/639cfb1d43bfdd7d64fb69da6d4282734615822f))
- **Parity fixture refresh workflow** documented ([`8b7d9e7`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/8b7d9e7387afa2ebe82a700f874ee1e86bb3ad60))
- **CI workflow docs** refreshed in README ([`c3f6502`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/c3f65021c3219ccd3627634b12170ee3b0cb71b2))
- **`python3` in all examples and workflows** ([`06b7c03`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/06b7c036570a5f52d5ee6ed1f46adb55d7dde112), [`f584b7c`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/f584b7c300bdc6563b6eaa4ba99fe7cd0980341e))

### Initial Import (2026-01-25)

Full-stack ASCII art generation system in a single commit (91 files, 19,340 lines).

**Python (`python/`)**:
- `model/` — decoder-only transformer with 2D positional encoding, character-level tokenizer
- `inference/` — constrained decoding (width/height/max-chars), sampling with temperature/top-k/top-p
- `train/` — PyTorch training loop with AMP, gradient accumulation, safetensors export
- `data/` — scrapers for ASCIIArt.eu, 16colo.rs, textfiles.com; HuggingFace dataset ingestion; FIGlet banner generation; SQLite storage with SHA-256 dedup; quality pipeline

**Rust (`rust/ascii-gen/`)**:
- Full inference engine using `candle` — loads safetensors, runs constrained decoding
- Quantized weight support (INT4/INT8) via `model_int4.safetensors` / `model_int8.safetensors`
- Embedded-weights mode (`--features embedded-weights`) for single-binary shipping
- Cross-validation test suite with golden fixtures for Python-Rust parity

**E2E (`tests/e2e/`)**:
- `e2e_full.sh` — orchestrates DB sanity, training, export, Python-Rust parity, Rust inference, embedded weights
- `e2e_data.sh`, `e2e_python_export.sh`, `e2e_rust.sh` — targeted pipeline tests

**CI (`.github/workflows/`)**:
- `e2e.yml` — full end-to-end pipeline on push
- Pinned to `nightly-2026-01-20` Rust toolchain

([`fe5d603`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/fe5d603c00a540035610646a64759259e69199f7))

---

## Commit Index

Quick-reference table of all substantive commits (excluding beads-tracking metadata).

| Date | Hash | Summary |
|------|------|---------|
| 2026-01-25 | [`fe5d603`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/fe5d603c00a540035610646a64759259e69199f7) | Initial import — full Python+Rust stack (91 files) |
| 2026-01-25 | [`1c0f4eb`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/1c0f4ebc61f824b37fac514365bdac4ca04f3ce8) | Tests: cover Csplk ingest block parsing |
| 2026-01-25 | [`06b7c03`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/06b7c036570a5f52d5ee6ed1f46adb55d7dde112) | Docs: use python3 in examples |
| 2026-01-25 | [`f584b7c`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/f584b7c300bdc6563b6eaa4ba99fe7cd0980341e) | CI: use python3 in workflow |
| 2026-01-25 | [`d971b0d`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/d971b0df867578b8a929ca29177193584f3dbf08) | CI: install python requirements for full-e2e |
| 2026-01-25 | [`233be1c`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/233be1caaf6b7b38fda1e7a57d15ecb7ef2ce610) | Data: Csplk ingest heuristics and force/resume flags |
| 2026-01-25 | [`de277b5`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/de277b516b44020824d4cc55480dd33682e5fbc8) | Data: improve Csplk art line detection |
| 2026-01-25 | [`9598022`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/95980228da13cbe263afd2ab82ff665af9b519ae) | Data: reach 500k rows + improve ingestion |
| 2026-01-25 | [`95d710a`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/95d710ab75ec566e8a4c334640e21cbd4576b019) | E2E: add training smoke test |
| 2026-01-25 | [`6aec215`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6aec215fa50f78540a2705feeb284bfe372896ec) | Train: quieter CPU eval |
| 2026-01-25 | [`93fa7b4`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/93fa7b40869f09432967e713d859105f41607965) | Export: configurable fresh model |
| 2026-01-25 | [`e5f7f67`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/e5f7f67498dfa031b036a96100b4836453da3098) | E2E: rust uses python-exported model |
| 2026-01-25 | [`45dbba2`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/45dbba237b556190dc80cee548cf108e44db5d1a) | Train: CPU-friendly defaults + fail-fast |
| 2026-01-25 | [`1c8fdf0`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/1c8fdf0c2f44f04969ee95b7e3ac428dd9092b2b) | E2E: export tiny model for speed |
| 2026-01-25 | [`13a080e`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/13a080e71547aa821a9506ffe963e552d80ddbb4) | Python inference CLI: load checkpoints/exports |
| 2026-01-25 | [`8360319`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/836031956090484a1c76f9c97493a460cf44d4c1) | E2E: embedded weights smoke test |
| 2026-01-25 | [`a121b37`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/a121b370ef9cd1ee77bf990c74bd817e683fe7a7) | Train: AMP works with cuda:0 |
| 2026-01-25 | [`c3f6502`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/c3f65021c3219ccd3627634b12170ee3b0cb71b2) | bd-3ce: Refresh README workflows |
| 2026-01-25 | [`9dc292b`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/9dc292b958d61090edfd298d44267c549a19f6ed) | Train: expose more CLI flags |
| 2026-01-25 | [`1fc0f9e`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/1fc0f9ee03b5d0afb41cd7482895dc53bb722742) | Train: add --preset small\|medium\|large |
| 2026-01-25 | [`1fb8a85`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/1fb8a85af25ef29208c790e5c21b37a293a7b104) | Clamp Rust 2D positional indices |
| 2026-01-25 | [`824bade`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/824bade5d475f3db2a66800ee6de74c0daf7a54d) | Train: CLI polish |
| 2026-01-25 | [`aae92d2`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/aae92d290cc362707be801d70e22ec21b481ec69) | Document train --preset |
| 2026-01-25 | [`9c8d3a6`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/9c8d3a6349966aa235ed882e6cf96e2926ff0c8c) | Tighten python inference constraints |
| 2026-01-25 | [`41d65fd`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/41d65fd0e54c8fbd6d76d500d9eb382c34a6595a) | Handle missing charset column in training dataset |
| 2026-01-25 | [`e22b2d5`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/e22b2d55d6ae7307c3ebf2461eabe55b6e07a027) | Safe torch.load defaults |
| 2026-01-25 | [`dab0627`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/dab0627c50b0806cff424bc7826b81e895c30985) | Drop unused row/col positions |
| 2026-01-25 | [`6705d74`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6705d74254da90d9d943dc2786ebe7defedeaa8f) | Train: harden LR schedule + disable interval=0 |
| 2026-01-25 | [`27815ba`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/27815baa06eadd48261793b1b1371f2cd1e4107f) | Export: validate checkpoint state dict types |
| 2026-01-25 | [`e2ce331`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/e2ce3311fec4947d07f59c11beeaf384e9660f80) | Data: normalize newlines and skip empty ingests |
| 2026-01-25 | [`1962150`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/19621502a839acdb3c9a8b30ff68b37ba1e72fa0) | Rust tests: decode only art portion |
| 2026-01-25 | [`f742ffe`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/f742ffe4ffc33b530010bbc3055bd6d0878d231a) | DB: skip empty ASCII art inserts by default |
| 2026-01-25 | [`360f211`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/360f211cee3b08a4b6b4c74bc502710ebda98f92) | Update quality tests to load config from JSON |
| 2026-01-25 | [`8507b1f`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/8507b1f61c7f9724f51d2d4bdc88bef6efe331f7) | Export: make --unsafe-load work on older torch |
| 2026-01-25 | [`6e88bcd`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6e88bcd201e3e105dd6f8696203b8442163b473f) | Rust tests: make quality suite CI-friendly |
| 2026-01-25 | [`dded704`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/dded7049dc6367f9b28a80d6ea29ca71a9d9fb40) | Dataset: cache SQLite connection safely |
| 2026-01-25 | [`5ecc4da`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/5ecc4da504ac609e719914e7165bbfa6a1254ab1) | Model: prefer SDPA for attention |
| 2026-01-25 | [`45cd993`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/45cd99341ba51daf4736fd0213b5b532de0284ce) | Train: optional auto-export after training |
| 2026-01-25 | [`842b3d0`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/842b3d0119515a3b6602910ab9343d2afe5aaa13) | Fix model_available() to check for config.json |
| 2026-01-25 | [`248a303`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/248a3038d5b2b3e5e7ed77e12215a6154f8c802b) | Training: minor robustness tweaks |
| 2026-01-25 | [`04bff5b`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/04bff5b4167412c9d32688bcdbd772f0c0a4b5fd) | Fix tokenizer export; keep augmentation ASCII-only |
| 2026-01-25 | [`64cd721`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/64cd7210a5254fde799c3b799c4eae4f8e5ce806) | Test: guard tokenizer.json export |
| 2026-01-25 | [`d090e2a`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/d090e2a1cffcdcc84f2cfe7e1fdc4e0dfb0219df) | Coverage: add pytest-cov tooling and helper |
| 2026-01-25 | [`8b7d9e7`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/8b7d9e7387afa2ebe82a700f874ee1e86bb3ad60) | Docs: parity fixture refresh workflow |
| 2026-01-25 | [`31df9bc`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/31df9bc3357ea3d719bc8d12568351ad0c3217c5) | Expand crossval golden fixtures |
| 2026-01-25 | [`70105a2`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/70105a29349203ef93ff9e7cf9f251ed6b6ff14c) | Add weights loader + quant scheme tests |
| 2026-01-25 | [`0acba79`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/0acba79620ee5b8bad3769f8fdceb226d3aba867) | Env snapshot + quantized e2e smoke |
| 2026-01-25 | [`2da283b`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/2da283b0c59c7754982967486791a62c57b41029) | Add e2e lint/typecheck gate script |
| 2026-01-25 | [`6fbfad1`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6fbfad1ddae8dd868c9260dc09049cb2db440bcb) | Rust: clippy cleanups in quantized loader |
| 2026-01-25 | [`0da744e`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/0da744ecaf12c58632ed866389bcb2969c90f796) | Rust: bump candle to 0.9.2 |
| 2026-01-25 | [`5fe6cc7`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/5fe6cc74ea59b349f4c3a668f4a5680c10d4ae3a) | Add Python coverage gates in CI |
| 2026-01-25 | [`6266e1a`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6266e1ac161b742ee01f08015a64b5b9e87c68b7) | Update E2E workflow (dispatch-only + artifacts) |
| 2026-01-25 | [`c894338`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/c8943388efc64303ae595b572fbc3abd3c2b59b0) | Train: omit attention_mask to enable SDPA fast-path |
| 2026-01-25 | [`dfae0ec`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/dfae0ec62e4e9588d3fdf2bd9d5a7838f449117a) | Add Python inference CLI e2e smoke test |
| 2026-01-25 | [`66dd60b`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/66dd60bb020618d5658618c1e4f028b34c8e03bb) | Improve e2e logging (timings + failure context) |
| 2026-01-25 | [`8d871c4`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/8d871c42248a5ba77e78a05ebd45f210b51d1e38) | Expand constrained decoder edge-case tests |
| 2026-01-25 | [`b852eb2`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/b852eb26a61cdf34df2afe67f68036587ec28982) | Add Rust CLI output format integration tests |
| 2026-01-25 | [`0e95160`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/0e95160c1b48e01405237b2953ce8e829a496abe) | Expand sampling edge-case tests |
| 2026-01-25 | [`b23da7a`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/b23da7a429a48251d121f30dd126fa347714bed3) | Add model forward/forward_last + position clamp tests |
| 2026-01-25 | [`ab54ee0`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/ab54ee0976145ad12ea846b09ddba43e8550289a) | Python export/ingest/positional tests |
| 2026-01-25 | [`0ec0777`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/0ec0777082c810757e4f55d8ab3928f4c7741518) | db.py metadata edge-case tests |
| 2026-01-25 | [`d304c03`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/d304c03a1e429939b95981004958e1bacbe2750c) | db.py update semantics tests |
| 2026-01-25 | [`35020a0`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/35020a0c9909e81ab87d6cca6ca0b4927efbf98b) | Add quality_pipeline.py validation tests |
| 2026-01-25 | [`72897d3`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/72897d360414d957a1f3e7a2b58bddba3f225062) | Add train/dataset.py coverage tests |
| 2026-01-25 | [`e042aab`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/e042aabf1fcc514ab36036a29a920e64aaacefc5) | Add generate() sampling invariant tests |
| 2026-01-25 | [`400fa7f`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/400fa7f2e249dd21a9ee3c82503f00a762a3a29c) | Add FIGlet dataset generator helper tests |
| 2026-01-25 | [`c5abd67`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/c5abd6769ee10f7d144865f42ebfe6e6d8360761) | Add scraper helper utilities tests |
| 2026-01-25 | [`b62ea52`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/b62ea52bed9a4438afe7dbbb8c8b619eb9f92ebc) | Land missing Python test coverage |
| 2026-01-25 | [`51fd5c9`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/51fd5c9924573319cbdd9430870b889149ada2e7) | Deprecate legacy database.py |
| 2026-01-25 | [`1b8fd51`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/1b8fd511ef00c64dfa1c56c74202b82970358678) | Add optional Rust coverage workflow |
| 2026-01-25 | [`47b2060`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/47b2060221ccaa53da892e8b7fc970d3851ffdf4) | Add unit tests for sampler.py and export.py |
| 2026-01-25 | [`4cac7fa`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/4cac7fac178e4491578686224aabf920da245233) | Add self-hosted GPU training workflow |
| 2026-01-25 | [`0e8cf75`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/0e8cf75e8148596b4f2687971974208c000b00bd) | Record dependency upgrade log |
| 2026-01-26 | [`0bcddc6`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/0bcddc6234684453d9de92de5ae2d5ee1e4e4f8b) | Fail on invalid config.json |
| 2026-01-26 | [`85dd07d`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/85dd07db345992778d0021ec9decd1d5dd75c824) | Fix last-line newline; cache causal mask |
| 2026-01-26 | [`aa75749`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/aa75749ee653c4cbe3061116ad9ec9fe9870a53a) | Align sampling edge cases with Rust |
| 2026-01-26 | [`752a644`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/752a64437eba381852a81c24e6850877d334dea6) | ruff format python |
| 2026-01-26 | [`134351d`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/134351df371f0ed3ecbbafaba13b31f72a68c345) | Remove mocks via RNG injection |
| 2026-01-26 | [`b5b8ad9`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/b5b8ad99b56d3423d9a29a38642e0eed2e9b7b02) | Make mypy --strict gate pass |
| 2026-01-26 | [`28ebbb3`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/28ebbb357a799435f3a0324d4603c99c58de88df) | Tighten quality_pipeline typing |
| 2026-01-26 | [`021815c`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/021815cb62b0ce3431c31ee39442936f17f53327) | Add dependabot config |
| 2026-01-26 | [`ab6c811`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/ab6c8118859ac1a7673eb1e84e562c54318bc664) | Harden data+train tests; fix Rust max_chars=0 |
| 2026-01-26 | [`3259145`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/32591450175bf1ca0a2a2c0cfdeb0b1aed4067f4) | Fix scrape_textfiles categorization + progress semantics |
| 2026-01-26 | [`42a0e6f`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/42a0e6f1f3dd475f956b8b74a106557c262955d3) | Model: drop cached causal masks from checkpoints |
| 2026-01-26 | [`6fd554e`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6fd554e9eef3ba8fc8ab1adeaab3728e118393e4) | Scrapers: harden SAUCE parsing and gallery paths |
| 2026-01-26 | [`639cfb1`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/639cfb1d43bfdd7d64fb69da6d4282734615822f) | Docs: rewrite README |
| 2026-02-01 | [`0cb8659`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/0cb86594e9e1a0fe1d37b0b7a0a25464f6231e5a) | CI: fix train_gpu workflow_dispatch inputs |
| 2026-02-01 | [`e8028ba`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/e8028ba9e13fed1a2aa2225b7b22ba68ce34e8d5) | CI: add train_gpu dispatch workflow |
| 2026-02-01 | [`6769195`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6769195d6f72ed0e111625841e803ff0171dc2ae) | CI: make train_gpu_dispatch dispatchable |
| 2026-02-01 | [`4cb6653`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/4cb6653f949d888dbb361bae228b6c937e72b1d7) | CI: fix GPU workflow YAML heredocs |
| 2026-02-01 | [`ea7ef77`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/ea7ef77fc885c7c613274d30bbd62fa116004f37) | CI: add actionlint workflow validation |
| 2026-02-02 | [`6f57631`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6f57631dab8ea4b422a403a7137dfa7b359c298f) | feat(inference): mirror Rust max_tokens=0 semantics |
| 2026-02-02 | [`b8b20b9`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/b8b20b9b4a081f1741860b6651b6f91e2aadf860) | refactor(tests): clean up fallback position computation |
| 2026-02-02 | [`9ff80c1`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/9ff80c12f43433e308c1ef26639cf7e2d5caf0ca) | fix(data): harden HF ingestion early-stop |
| 2026-02-03 | [`f7cf953`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/f7cf953d7770725801abced5cfc19b7f8fe64b31) | chore(gitignore): Add a.out to ignored files |
| 2026-02-09 | [`4638512`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/46385125659b79d85c9d3752a75468bf22ebe1ab) | chore(deps): bump GH Actions and Rust dependencies |
| 2026-02-21 | [`6b4471b`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6b4471bedac71a060c39cace62be1868da7d0fa4) | chore: add GitHub social preview image |
| 2026-02-21 | [`41527b7`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/41527b705886e03a2eb2041e4721166f8404fcf0) | chore: update license to MIT with OpenAI/Anthropic Rider |
