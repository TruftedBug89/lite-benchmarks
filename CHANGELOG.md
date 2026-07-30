# Changelog

## [Unreleased] — 2026-07-29

Deterministic-verifier hardening and code-benchmark fairness.

### Code-execution benchmarks: no more guaranteed-fail tasks
- **Runtime confinement shim** (`lite_bench/sandbox_child.py`, Layer 2.5): prepended
  to every sandboxed child. `open`/`io`/`os.open`/`sqlite3.connect` are confined to
  the sandbox directory (reads also allow the Python install so imports work); sockets
  are confined to loopback, with non-loopback connects failing fast (`OSError`) instead
  of hanging; and `os`/`shutil` are allowed but confined — process-spawning
  (`os.system`/`popen`/`exec*`/`spawn*`/`kill`/…) is blocked outright and every
  filesystem call is restricted to the sandbox dir (mirroring `windows_sandbox.py`).
  This makes BigCodeBench's file-I/O, `os`/`shutil`, and network tasks legitimately
  runnable.
- **AST blocklist realignment**: unblocked the modules the shim confines
  (`open`, `io`, `tempfile`, `pathlib`, `glob`, archives, `os`, `shutil`, `socket`,
  `ssl`, `select`, `smtplib`, `requests`, `urllib`, `http`, `getpass`, `sqlite3`,
  `codecs`, path helpers). Genuinely unconfineable modules (`subprocess`,
  `multiprocessing`, `ctypes`, `importlib`, `pickle`, `psutil`, `sys`, …) and all
  dunder/reflection escapes stay blocked. Previously ~23% of BigCodeBench-Hard was
  unpassable because its forced preambles use `open()`/`os`/`socket`.
- **Deterministic pre-sampling filter**: HumanEval/MBPP/BigCodeBench now drop, *before
  sampling*, tasks whose reference solution the sandbox would still reject. BigCodeBench
  scans the FULL reconstructed solution (forced `code_prompt` imports + body), excluding
  only the 9/148 Hard tasks that force `subprocess`/`psutil`/`sys` or use `hasattr`
  (139 remain passable); HumanEval drops `/160` (`eval`); MBPP drops `/596` (`sys`).
  `num_samples` is now an honest count of passable tasks. Note: this changes the sampled
  pools, so prior code-benchmark results won't resume.
- **Exclusion transparency**: benchmarks record how many tasks were filtered as
  ungradeable (`excluded_count` in the results JSON) and emit a note to the dashboard
  log, so a score is auditable.

### Verifier correctness
- **Single evaluation per question**: the engine now calls `evaluate_detailed` once and
  takes the score + metadata from it, instead of calling `evaluate()` *and*
  `evaluate_detailed()`. Code benchmarks no longer run the sandbox twice per question
  (2× cost, port-binding collisions, score/judge-log divergence).
- **Scientific-notation extraction**: `_extract_number` now understands `4.4e-9` and
  LaTeX `4.4 \times 10^{-9}` / `\cdot 10^{n}`, fixing SciBench false negatives (and a
  gold-side bug where a float gold like `4.4e-09` was misread as its exponent).
- **SciBench tolerance**: the absolute floor now scales with the gold magnitude
  (`1e-9 * max(1, |g|)`) instead of a fixed `1e-8` that marked any two sub-1e-8 answers
  equal.
- **MATH-500 symbolic equivalence**: a deterministic sympy fallback (no LLM) now marks
  factually-equal answers correct across forms (`2^5`≡`32`, `\frac{1}{2}`≡`0.5`,
  `\sqrt{4}`≡`2`) while preserving false-positive guards (`\sqrt{2}`≠`2`).

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
- **Wait for a good response**: with `max_retries <= 0` (new default), a failed question
  stops advancing and retries transient provider errors (timeouts, 429, 5xx, connection
  resets, empty responses) with exponential backoff until a good response arrives or the
  user hits Force Stop. Stays visible in the dashboard via per-question retry notes and a
  throttled system log. Permanent errors (context length, content filter) and fatal
  auth/quota errors still give up immediately — they can never succeed. A positive
  `max_retries` keeps the old capped behaviour.
- **Interruptible backoff** (0.5s slices) so Force Stop unwinds during a retry wait instead
  of running a hung question; stopped mid-wait questions record as `cancelled`.

### Web dashboard
- Fixed startup blocker, CWD-relative asset paths, and robustness; fixed log-freeze,
  an `escapeAttr` XSS, and silent fetch failures in the SPA. `/api/run` validates
  Content-Type, Origin, and CSRF.
- **Error visibility**: the live cards show a capped error/retry line (char-limited so the
  grid stays readable) with the complete error available on hover; long provider
  exceptions no longer flood the web log. Live progress is now the count of completed
  questions (monotonic) instead of the out-of-order question id, and the scored/failed
  accounting matches the engine (provider errors are failures, not zeros).

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
