# Changelog

All notable changes to **ASCII Art Mini Transformer** are documented here.

This project has no formal releases or tags. Changes are organized by date
and grouped by capability area, not by diff order. Every commit link points to
the canonical GitHub repository.

---

## 2026-02-21 -- License and Repository Metadata

### Licensing

- **Add MIT license with OpenAI/Anthropic Rider** -- the repository was previously
  unlicensed (proprietary-by-default). The new license is standard MIT with an
  additional rider restricting use by OpenAI, Anthropic, and their affiliates
  without express written permission from Jeffrey Emanuel.
  ([`41527b7`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/41527b705886e03a2eb2041e4721166f8404fcf0))

### Repository

- Add GitHub social preview image (1280x640) for consistent link previews.
  ([`6b4471b`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6b4471bedac71a060c39cace62be1868da7d0fa4))

---

## 2026-02-09 -- Dependency Consolidation

### Dependencies

- **Bulk bump of GH Actions and Rust crates**: actions/checkout 4.3.1 to 6.0.2,
  actions/setup-python 5.6.0 to 6.2.0, actions/upload-artifact 4.6.2 to 6.0.0,
  clap 4.5.54 to 4.5.57, anyhow 1.0.100 to 1.0.101.
  ([`4638512`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/46385125659b79d85c9d3752a75468bf22ebe1ab))

---

## 2026-02-01 through 2026-02-03 -- GPU Training Workflows and Inference Parity

### Inference (Python)

- **Mirror Rust `max_tokens=0` semantics**: when `max_tokens` is 0 or negative,
  disable the hard character cap but still bound the decoding loop via
  `width * height` (or a 500-token fallback) to prevent infinite generation.
  This aligns Python with the Rust CLI behavior.
  ([`6f57631`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6f57631dab8ea4b422a403a7137dfa7b359c298f))

### Data Ingestion

- **Harden HuggingFace ingestion early-stop logic** to handle edge cases in
  dataset iteration that could cause premature termination.
  ([`9ff80c1`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/9ff80c12f43433e308c1ef26639cf7e2d5caf0ca))

### CI / GPU Training

- Add `train_gpu` and `train_gpu_dispatch` workflows for self-hosted GPU
  runners, enabling production-scale training outside GitHub-hosted CPU runners.
  ([`e8028ba`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/e8028ba9e13fed1a2aa2225b7b22ba68ce34e8d5),
  [`6769195`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6769195d6f72ed0e111625841e803ff0171dc2ae))
- Fix GPU workflow `workflow_dispatch` inputs and YAML heredoc quoting.
  ([`0cb8659`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/0cb86594e9e1a0fe1d37b0b7a0a25464f6231e5a),
  [`4cb6653`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/4cb6653f949d888dbb361bae228b6c937e72b1d7))
- Add actionlint workflow validation to catch YAML syntax issues before merge.
  ([`ea7ef77`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/ea7ef77fc885c7c613274d30bbd62fa116004f37))

### Tests

- Clean up fallback position computation in tests (replace `exec()` hack with
  proper function definition).
  ([`b8b20b9`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/b8b20b9b4a081f1741860b6651b6f91e2aadf860))

### Chore

- Add `a.out` to `.gitignore`.
  ([`f7cf953`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/f7cf953d7770725801abced5cfc19b7f8fe64b31))

---

## 2026-01-26 -- Bug Fixes, Type Safety, and Scraper Hardening

### Bug Fixes (Rust)

- **Fix `max_chars=0` in Rust CLI**: when `max_chars` was 0, the constraint was
  incorrectly applied as a zero-length cap instead of being disabled. Now
  mirrors the Python behavior of treating 0 as "unbounded".
  ([`ab6c811`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/ab6c8118859ac1a7673eb1e84e562c54318bc664))
- **Fix last-line newline handling** in Rust constrained decoder; also cache the
  causal mask in the Rust transformer to avoid redundant recomputation.
  ([`85dd07d`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/85dd07db345992778d0021ec9decd1d5dd75c824))
- **Fail on invalid `config.json`** during Rust weight loading instead of
  silently using defaults.
  ([`0bcddc6`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/0bcddc6234684453d9de92de5ae2d5ee1e4e4f8b))

### Bug Fixes (Python)

- **Align Python sampling edge cases with Rust** to ensure parity on
  temperature=0, top-k=1, and degenerate distribution scenarios.
  ([`aa75749`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/aa75749ee653c4cbe3061116ad9ec9fe9870a53a))
- Fix `scrape_textfiles` categorization logic and progress-reporting semantics.
  ([`3259145`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/32591450175bf1ca0a2a2c0cfdeb0b1aed4067f4))

### Model Internals

- **Drop cached causal masks from checkpoints** -- masks are now regenerated on
  load, shrinking checkpoint file size and avoiding shape mismatches when
  `block_size` changes.
  ([`42a0e6f`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/42a0e6f1f3dd475f956b8b74a106557c262955d3))

### Type Safety and Code Quality

- **`mypy --strict` gate passes** across the entire Python codebase. Added
  `mypy.ini` configuration and fixed annotations in data ingestion, inference,
  model, and training modules.
  ([`b5b8ad9`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/b5b8ad99b56d3423d9a29a38642e0eed2e9b7b02))
- Remove mocks from tests via RNG injection, improving test determinism without
  patching internals.
  ([`134351d`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/134351df371f0ed3ecbbafaba13b31f72a68c345))
- Tighten `quality_pipeline` typing and tidy inference test imports.
  ([`28ebbb3`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/28ebbb357a799435f3a0324d4603c99c58de88df),
  [`0701c44`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/0701c446017c7910c0df3463169cfa5458a27c13))
- Apply `ruff format` across Python codebase.
  ([`752a644`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/752a64437eba381852a81c24e6850877d334dea6))

### Scrapers

- Harden SAUCE metadata parsing and gallery path resolution in 16colors and
  asciiart.eu scrapers.
  ([`6fd554e`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6fd554e9eef3ba8fc8ab1adeaab3728e118393e4))

### Tests (Expanded Coverage)

- Harden data and training tests; add scraper test coverage for all three
  scrapers (asciiart.eu, 16colors, textfiles.com) plus FIGlet generator tests.
  ([`ab6c811`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/ab6c8118859ac1a7673eb1e84e562c54318bc664),
  [`3259145`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/32591450175bf1ca0a2a2c0cfdeb0b1aed4067f4))

### Deprecation

- **Deprecate legacy `database.py`** in favor of the newer `db.py` module. The
  old module is preserved but wrapped with deprecation notices.
  ([`51fd5c9`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/51fd5c9924573319cbdd9430870b889149ada2e7),
  [`cd1e474`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/cd1e474ad3566d6121aea53521afe7c606236003))

### CI

- Add Dependabot configuration for automated Rust crate and GH Actions version
  bumps.
  ([`021815c`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/021815cb62b0ce3431c31ee39442936f17f53327))

### Documentation

- **Rewrite README** with full architecture diagram, comparison table, FAQ,
  troubleshooting, and contribution policy.
  ([`639cfb1`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/639cfb1d43bfdd7d64fb69da6d4282734615822f))

---

## 2026-01-25 -- Initial Import and Rapid Build-Out

This date covers the initial import and a large burst of feature work,
hardening, test coverage, and CI plumbing that brought the project from
first commit to a fully tested, dual-language (Python + Rust) pipeline.

### Core Architecture (Initial Import)

The initial import landed the complete end-to-end pipeline:
([`fe5d603`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/fe5d603c00a540035610646a64759259e69199f7))

- **Python model**: decoder-only transformer with 2D positional encoding
  (`positional_encoding.py`), character-level tokenizer (`tokenizer.py`),
  constrained decoding (`constraints.py`, `generate.py`), and sampling
  (`sampler.py`).
- **Python training**: full training loop with gradient accumulation, mixed
  precision, checkpointing, and eval (`train.py`); data augmentation
  (`augmentation.py`); SQLite-backed dataset (`dataset.py`, `db.py`).
- **Python export**: checkpoint-to-safetensors pipeline with `config.json` and
  `tokenizer.json` emission (`export.py`).
- **Python inference CLI**: prompt-based generation with constraint flags
  (`cli.py`).
- **Data ingestion**: scrapers for ASCIIArt.eu, 16colo.rs, textfiles.com;
  HuggingFace dataset ingestor; FIGlet banner generator. All backed by a
  deduplicated SQLite store with SHA-256 content hashing.
- **Data quality pipeline**: validation for empty art, sparse content, oversized
  dimensions, control characters, encoding errors, and null bytes.
- **Rust CLI (`ascii-gen`)**: full inference binary using candle, with safetensors
  weight loading, 2D positional encoding, constrained decoding, top-k/top-p
  sampling, and markdown/raw/plain output formats.
- **Rust quantization**: INT8 and INT4 weight-only quantization support with
  `quant_config.json`.
- **Rust embedded weights**: `--features embedded-weights` compiles weights
  directly into the binary via `build.rs`.
- **Cross-validation fixtures**: deterministic golden outputs under
  `rust/ascii-gen/test_data/crossval/` for Python-Rust parity testing.
- **E2E test scripts**: `e2e_full.sh` orchestrating DB sanity, training, export,
  Python inference, Rust parity, and embedded-weights smoke tests.
- **91 files, 19,340 lines** in the initial commit.

### Training Enhancements

- **Add `--preset small|medium|large`** for quick model-size selection with
  sensible defaults for `n_layer`, `n_head`, `n_embd`, and `block_size`.
  ([`1fc0f9e`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/1fc0f9ee03b5d0afb41cd7482895dc53bb722742))
- **CPU-friendly defaults + fail-fast**: gracefully handle missing CUDA,
  default to CPU with float32, and fail early on config errors.
  ([`45dbba2`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/45dbba237b556190dc80cee548cf108e44db5d1a))
- **AMP support for `cuda:0`**: automatic mixed precision works correctly on
  GPU training.
  ([`a121b37`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/a121b370ef9cd1ee77bf990c74bd817e683fe7a7))
- Expose additional CLI flags for fine-grained training control.
  ([`9dc292b`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/9dc292b958d61090edfd298d44267c549a19f6ed))
- Quieter CPU eval output to reduce log noise.
  ([`6aec215`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6aec215fa50f78540a2705feeb284bfe372896ec))
- CLI polish and documentation of `--preset` flag.
  ([`824bade`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/824bade5d475f3db2a66800ee6de74c0daf7a54d),
  [`aae92d2`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/aae92d290cc362707be801d70e22ec21b481ec69))
- Harden learning rate schedule and reject `interval=0` configurations.
  ([`6705d74`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6705d74254da90d9d943dc2786ebe7defedeaa8f))
- Optional auto-export after training completes.
  ([`45cd993`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/45cd99341ba51daf4736fd0213b5b532de0284ce))
- Minor robustness tweaks to training loop.
  ([`248a303`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/248a3038d5b2b3e5e7ed77e12215a6154f8c802b))
- Handle missing `charset` column gracefully in training dataset.
  ([`41d65fd`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/41d65fd0e54c8fbd6d76d500d9eb382c34a6595a))

### Model Improvements

- **Prefer SDPA (Scaled Dot-Product Attention)** for the attention mechanism,
  enabling PyTorch's fused kernels when available.
  ([`5ecc4da`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/5ecc4da504ac609e719914e7165bbfa6a1254ab1))
- **Omit `attention_mask` to enable SDPA fast-path**: the causal mask is built
  internally, avoiding the overhead of an explicit mask tensor.
  ([`c894338`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/c8943388efc64303ae595b572fbc3abd3c2b59b0))
- Drop unused `row`/`col` position fields from model forward pass.
  ([`dab0627`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/dab0627c50b0806cff424bc7826b81e895c30985))

### Inference (Python)

- **Python inference CLI loads both checkpoints and exports**: supports
  `--checkpoint` (raw `.pt`) and `--model` (exported `.safetensors`).
  ([`13a080e`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/13a080e71547aa821a9506ffe963e552d80ddbb4))
- Tighten Python inference constraints to match Rust strictness.
  ([`9c8d3a6`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/9c8d3a6349966aa235ed882e6cf96e2926ff0c8c))

### Inference (Rust)

- **Clamp 2D positional indices** in Rust to prevent out-of-bounds panics when
  generated art exceeds `max_rows`/`max_cols`.
  ([`1fb8a85`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/1fb8a85af25ef29208c790e5c21b37a293a7b104))

### Export Pipeline

- Configurable fresh-model export (export a randomly initialized model for
  testing without training).
  ([`93fa7b4`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/93fa7b40869f09432967e713d859105f41607965))
- Make `--unsafe-load` work on older PyTorch versions that lack
  `weights_only` support.
  ([`8507b1f`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/8507b1f61c7f9724f51d2d4bdc88bef6efe331f7))
- Validate checkpoint state dict types before export to catch corruption early.
  ([`27815ba`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/27815baa06eadd48261793b1b1371f2cd1e4107f))
- Fix tokenizer export to emit correct `tokenizer.json`; keep augmentation
  ASCII-only to avoid non-printable characters leaking into the vocabulary.
  ([`04bff5b`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/04bff5b4167412c9d32688bcdbd772f0c0a4b5fd))
- Export tiny model in E2E for speed.
  ([`1c8fdf0`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/1c8fdf0c2f44f04969ee95b7e3ac428dd9092b2b))

### Data Ingestion

- Csplk ingest: add force/resume flags, improve art-line detection heuristics
  and block parsing.
  ([`233be1c`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/233be1caaf6b7b38fda1e7a57d15ecb7ef2ce610),
  [`de277b5`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/de277b516b44020824d4cc55480dd33682e5fbc8))
- Improve FIGlet and HuggingFace ingestion to reach 500k rows.
  ([`9598022`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/95980228da13cbe263afd2ab82ff665af9b519ae))
- Normalize newlines and skip empty ingests at the data layer.
  ([`e2ce331`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/e2ce3311fec4947d07f59c11beeaf384e9660f80))
- DB: skip empty ASCII art inserts by default.
  ([`f742ffe`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/f742ffe4ffc33b530010bbc3055bd6d0878d231a))
- Cache SQLite connection safely in dataset to avoid re-opening per batch.
  ([`dded704`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/dded7049dc6367f9b28a80d6ea29ca71a9d9fb40))

### Security

- **Safe `torch.load` defaults**: use `weights_only=True` by default to prevent
  arbitrary code execution from untrusted checkpoint files.
  ([`e22b2d5`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/e22b2d55d6ae7307c3ebf2461eabe55b6e07a027))

### Rust Dependency Updates

- Bump candle to 0.9.2.
  ([`0da744e`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/0da744ecaf12c58632ed866389bcb2969c90f796))
- Clippy cleanups in quantized weight loader.
  ([`6fbfad1`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6fbfad1ddae8dd868c9260dc09049cb2db440bcb))

### Bug Fixes

- Fix `model_available()` to require `config.json` (not just
  `model.safetensors`).
  ([`842b3d0`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/842b3d0119515a3b6602910ab9343d2afe5aaa13))

### Test Coverage (Major Expansion)

A large batch of test additions brought coverage across the entire Python and
Rust codebase. All tests use real (temp) SQLite databases -- no mocks.

**Python unit tests added:**

- `db.py` metadata edge cases and update semantics.
  ([`0ec0777`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/0ec0777082c810757e4f55d8ab3928f4c7741518),
  [`d304c03`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/d304c03a1e429939b95981004958e1bacbe2750c))
- `quality_pipeline.py` validation (31 tests covering all issue codes, dry-run,
  limits, size buckets, report serialization).
  ([`35020a0`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/35020a0c9909e81ab87d6cca6ca0b4927efbf98b))
- `train/dataset.py` (22 tests: width/height computation, charset filtering,
  constraint conditioning, collation, pickling).
  ([`72897d3`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/72897d360414d957a1f3e7a2b58bddba3f225062))
- `generate()` sampling invariant tests.
  ([`e042aab`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/e042aabf1fcc514ab36036a29a920e64aaacefc5))
- FIGlet dataset generator helper tests.
  ([`400fa7f`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/400fa7f2e249dd21a9ee3c82503f00a762a3a29c))
- Scraper helper utility tests.
  ([`c5abd67`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/c5abd6769ee10f7d144865f42ebfe6e6d8360761))
- Export and ingest/positional encoding tests.
  ([`ab54ee0`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/ab54ee0976145ad12ea846b09ddba43e8550289a))
- Model `forward`/`forward_last` and position-clamp tests.
  ([`b23da7a`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/b23da7a429a48251d121f30dd126fa347714bed3))
- Sampling edge-case tests (temperature, top-k, degenerate distributions).
  ([`0e95160`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/0e95160c1b48e01405237b2953ce8e829a496abe))
- Constrained decoder edge-case tests.
  ([`8d871c4`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/8d871c42248a5ba77e78a05ebd45f210b51d1e38))
- Sampler and export unit tests.
  ([`47b2060`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/47b2060221ccaa53da892e8b7fc970d3851ffdf4))
- Csplk ingest block-parsing tests.
  ([`1c0f4eb`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/1c0f4ebc61f824b37fac514365bdac4ca04f3ce8))
- Tokenizer export guard test.
  ([`64cd721`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/64cd7210a5254fde799c3b799c4eae4f8e5ce806))

**Rust tests added:**

- Rust CLI output-format integration tests (markdown, raw, plain).
  ([`b852eb2`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/b852eb26a61cdf34df2afe67f68036587ec28982))
- Expand cross-validation golden fixtures.
  ([`31df9bc`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/31df9bc3357ea3d719bc8d12568351ad0c3217c5))
- Weights loader and quantization scheme tests.
  ([`70105a2`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/70105a29349203ef93ff9e7cf9f251ed6b6ff14c))
- Make Rust quality suite CI-friendly (skip tests requiring large models).
  ([`6e88bcd`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6e88bcd201e3e105dd6f8696203b8442163b473f))
- Decode-only-art-portion test to isolate generated content from framing.
  ([`1962150`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/19621502a839acdb3c9a8b30ff68b37ba1e72fa0))
- Quality tests updated to load config from JSON for model-size independence.
  ([`360f211`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/360f211cee3b08a4b6b4c74bc502710ebda98f92))

### E2E Test Scripts

- Add training smoke test (`e2e_train.sh`).
  ([`95d710a`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/95d710ab75ec566e8a4c334640e21cbd4576b019))
- E2E Rust script uses Python-exported model for true pipeline coverage.
  ([`e5f7f67`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/e5f7f67498dfa031b036a96100b4836453da3098))
- Add embedded-weights smoke test (`e2e_embedded.sh`).
  ([`8360319`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/836031956090484a1c76f9c97493a460cf44d4c1))
- Add Python inference CLI E2E smoke test.
  ([`dfae0ec`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/dfae0ec62e4e9588d3fdf2bd9d5a7838f449117a))
- Improve E2E logging: add per-step timings and failure context.
  ([`66dd60b`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/66dd60bb020618d5658618c1e4f028b34c8e03bb))
- Add quantized E2E smoke test (`e2e_quant.sh`) and environment snapshot.
  ([`0acba79`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/0acba79620ee5b8bad3769f8fdceb226d3aba867))
- Add E2E lint/typecheck gate script (`e2e_lint.sh`).
  ([`2da283b`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/2da283b0c59c7754982967486791a62c57b41029))

### CI / Workflows

- Add Python coverage gates in CI workflow.
  ([`5fe6cc7`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/5fe6cc74ea59b349f4c3a668f4a5680c10d4ae3a))
- Update E2E workflow to dispatch-only with artifact uploads.
  ([`6266e1a`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/6266e1ac161b742ee01f08015a64b5b9e87c68b7))
- Add optional Rust coverage workflow.
  ([`1b8fd51`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/1b8fd511ef00c64dfa1c56c74202b82970358678))
- Add self-hosted GPU training workflow.
  ([`4cac7fa`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/4cac7fac178e4491578686224aabf920da245233))
- Use `python3` in CI workflows and documentation examples.
  ([`f584b7c`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/f584b7c300bdc6563b6eaa4ba99fe7cd0980341e),
  [`06b7c03`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/06b7c036570a5f52d5ee6ed1f46adb55d7dde112))
- Install Python requirements in full-E2E CI job.
  ([`d971b0d`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/d971b0df867578b8a929ca29177193584f3dbf08))
- Refresh README workflow references.
  ([`c3f6502`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/c3f65021c3219ccd3627634b12170ee3b0cb71b2))

### Tooling

- Add pytest-cov tooling and `run_coverage.sh` helper.
  ([`d090e2a`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/d090e2a1cffcdcc84f2cfe7e1fdc4e0dfb0219df))
- Fix `PYTHONPATH` in `run_coverage.sh`.
  ([`47db41f`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/47db41f6090431532602aafc1df13f5ed07ae5c8))
- Document parity fixture refresh workflow.
  ([`8b7d9e7`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/8b7d9e7387afa2ebe82a700f874ee1e86bb3ad60))
- Record dependency upgrade log.
  ([`0e8cf75`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/0e8cf75e8148596b4f2687971974208c000b00bd))

---

## Dependabot Branches (Unmerged)

The following dependency-bump PRs remain on their Dependabot branches and have
not been merged to `main`:

| Branch | Bump | Commit |
|--------|------|--------|
| `dependabot/cargo/rust/ascii-gen/clap-4.6.0` | clap 4.5.57 to 4.6.0 | [`7a96cf9`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/7a96cf923c61259a9a2b095dd4d1609a1f923d5d) |
| `dependabot/cargo/rust/ascii-gen/clap-4.5.60` | clap 4.5.57 to 4.5.60 | [`73092b1`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/73092b1ae8c137ec3c5d77a9a3b668303f85ce7b) |
| `dependabot/cargo/rust/ascii-gen/clap-4.5.58` | clap 4.5.57 to 4.5.58 | [`2b6d6e9`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/2b6d6e900c845140e013c83f6f4e4a7bb6674de4) |
| `dependabot/cargo/rust/ascii-gen/anyhow-1.0.102` | anyhow 1.0.101 to 1.0.102 | [`976417e`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/976417e7b22286116674d22e3e9010f3d238eb51) |
| `dependabot/cargo/rust/ascii-gen/rand-0.9.2` | rand 0.8.5 to 0.9.2 | [`86ff407`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/86ff4075ff77d2b87690cfb8a72bff772f3ee352) |
| `dependabot/github_actions/actions/upload-artifact-7.0.0` | upload-artifact 6.0.0 to 7.0.0 | [`33dfa48`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/33dfa487b4b9075e08fc620789dfcadd4e644b57) |
| `dependabot/github_actions/actions/setup-go-6.3.0` | setup-go 6.2.0 to 6.3.0 | [`da179e5`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/da179e5fb815e4470595426e3066e39434704e60) |
| `dependabot/github_actions/Swatinem/rust-cache-2.9.1` | rust-cache 2.8.2 to 2.9.1 | [`2664d33`](https://github.com/Dicklesworthstone/ascii_art_mini_transformer/commit/2664d3306da169e7a925512d2920406cc65ed3cb) |

Previously-merged Dependabot bumps (via the 2026-02-09 consolidation commit)
are listed in the [2026-02-09](#2026-02-09----dependency-consolidation) section
above.
