# Changelog

## [Unreleased] — 2026-07-25

Post-0.2.0 robustness, security, and correctness polish. Pre-0.2.0 sample
leaderboards were reset/archived (`results/archive/`); the live `results/`
directory starts empty and is regenerated on the next run.

### Security — code-execution sandbox (Windows)
- **Three-layer sandbox**: model-generated code now runs under (1) an AST static
  scan, (2) a hardened subprocess (scrubbed env allowlist, throwaway cwd, wall-clock
  timeout), and (3) on Windows a fresh Job Object (`windows_sandbox.py`) that blocks
  grandchild processes and UI/clipboard access. `windows_sandbox.py` is now wired into
  the real code-exec path via `execute_sandboxed`.
- **Fail-closed opt-in gate** enforced *at the sandbox layer*
  (`execute_sandboxed(... allow_execution=False)` runs nothing) — a direct caller can
  never execute model code by accident, even if it skips the engine's bench-level skip.
- **Hardened AST scan**: added reflection / introspection modules (`gc`, `ast`,
  `pkgutil`, `_thread`, `importlib`, …) to the import blocklist; `getattr`/`hasattr`/
  `setattr`/`delattr`/`vars` are blocked outright, closing the name-based dunder escape
  (`getattr(o, "__subclasses__")`) at the root. Self-escape by importing the sandbox /
  harness modules (`windows_sandbox`, `lite_bench`) is vetoed.
- **API-key exfiltration hole closed**: the sandboxed child inherits only an allow-listed
  env (no `*_API_KEY` / `HF_TOKEN` / `PYTHONPATH`); the `/api/run` server keeps keys in a
  server-side registry and never echoes them.

### Scoring correctness
- **MATH-500 / AIME**: numeric comparison only fires when the *gold* answer is itself a
  plain number, so symbolic golds (`\sqrt{2}`, `3\sqrt{2}`, `2^5`) are no longer matched
  to digits pulled out of the model response. `\frac`/`\dfrac`/`\tfrac` no longer leak
  incidental digits.
- **Multiple-choice extraction**: lowercase article "a"/"i" and the pronoun "I" ("I
  think …") no longer hijack the answer letter — relevant for 10-option sets (A–J) where
  "I" is a valid option.
- **GPQA**: option-shuffle + gold-letter logic now handles both the Idavidrein column
  schema and the JSON-`metadata` mirror schema (`Correct Answer` / `Incorrect Answer N`),
  seeded by `seed:question` for determinism.
- **SuperGPQA**: hard subset filtered *before* sampling so a true n≈50 is drawn.
- **IFEval**: verifiers re-aligned to official semantics — `number_paragraphs` counts
  `***`-separated paragraphs; `more than` / `at most` / `equal to` relations; regex-safe
  keyword matching (`C++`).
- **Aggregation**: `None` (failed/unscored) benchmarks are excluded — never coerced to 0.0
  — so a single provider failure no longer tanks an otherwise-good model's category score.

### Engine & providers
- Reasoning-model telemetry (thinking tokens), retry-with-exponential-backoff + jitter,
  per-run checkpoint isolation, and None-score handling throughout.

### Web dashboard
- Fixed startup blocker, CWD-relative asset paths, and robustness; fixed log-freeze,
  an `escapeAttr` XSS, and silent fetch failures in the SPA. `/api/run` validates
  Content-Type, Origin, and CSRF.

### Reports
- `README.md` footer honors `SOURCE_DATE_EPOCH` (hermetic/reproducible builds); chart
  links are anchored to the real `charts_dir`; `None`-score and zero-model results render
  gracefully (N/A / "No results yet") instead of crashing.

### Tests & hygiene
- Expanded offline tests (MC non-hijack, MATH false-positive guard, GPQA mirror schema,
  None-score exclusion, hardened AST, sandbox fail-closed gate, Windows Job-Object self
  tests). Full suite green offline; `python-dotenv` declared as a dependency; generated
  charts/results/sandbox scratch are now gitignored and untracked.

## [0.2.0] - 2026-07-23

### ⚠️ Scoring v2 (Breaking Changes)
- **Seeded Random Sampling**: Replaced deterministic uniform strided dataset indexing with true seeded random sampling (`random.Random(seed).sample(...)`). Questions sampled differ from v0.1.0; results are not directly comparable to pre-v0.2.0 runs.
- **Strict HLE Scorer**: Removed substring free-pass matching; exact boxed, single-letter, or numerical extraction is required. Multimodal questions are filtered out at load time.
- **Tau-Bench Tool Matching**: Evaluates precise function name AND argument dictionary match.

### Added
- **Unified Engine (`lite_bench/engine.py`)**: Extracted all execution loops into a single engine with callback support.
- **Results Schema v2 (`lite_bench/results_store.py`)**: Stores Python, LiteLLM, Datasets versions, git SHA, per-benchmark question hashes, and full prompt/response text. Automatically drops stale benchmark keys from legacy runs.
- **Cost Tracking**: Integrated LiteLLM cost calculation per question (`total_cost_usd`) and added "Est. Cost" column to token usage tables.
- **Per-Model `max_tokens`**: Configurable per model in `config.yaml` or Web UI (defaults to 16,384 for reasoning models).
- **Wilson Confidence Intervals**: Per-benchmark tables in README show 95% Wilson score interval half-widths (±pp).
- **Web-First Dashboard**: Web UI (`web_app.py`) is the sole execution interface; removed deprecated CLI and terminal dashboard scripts.
- **Security Hardening**: Binding defaults to `127.0.0.1`, removed POST `env_vars` injection, sanitized `/api/config` responses, and added strict path traversal protection.

### Removed
- Removed `run_benchmark.py` CLI and `run_dashboard.py` terminal dashboard.
- Cleaned up tracked binaries (`archive.tar`, `files.zip`) and leftover test data fixtures.
