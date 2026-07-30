# Comprehensive Codebase Audit Report: Critical & Detrimental Bugs

> **Repository**: `TruftedBug89/lite-benchmarks`  
> **Audit Method**: Parallel 10-Agent Swarm + Manual Code Inspection  
> **Scope**: 100% of codebase (`lite_bench/`, `windows_sandbox.py`, `web_app.py`, `refresh_site.py`, `site/`, `tests/`, `config.yaml`)  
> **Focus**: Security vulnerabilities, sandbox escapes, concurrency race conditions, handle/thread leaks, scoring corruption, and unhandled crashes.

---

## Table of Contents
1. [Security & Sandbox Isolation](#1-security--sandbox-isolation)
2. [Execution Engine & Concurrency](#2-execution-engine--concurrency)
3. [Verification & Scoring Integrity](#3-verification--scoring-integrity)
4. [Data Caching, API Providers & Config](#4-data-caching-api-providers--config)
5. [Persistence, Math & Report Generation](#5-persistence-math--report-generation)
6. [Frontend Astro & Client Scripts](#6-frontend-astro--client-scripts)
7. [Audit Summary Table](#7-audit-summary-table)

---

## 1. Security & Sandbox Isolation

### 🚨 1.1 SymPy Arbitrary Code Execution in `MATH500Benchmark`
- **Location**: [`lite_bench/benchmarks.py:1072–1073`](file:///C:/Users/lavvo/Documents/lite-benchmarks/lite_bench/benchmarks.py#L1072-L1073)
- **Category**: Security / Remote Code Execution (RCE) Vector
- **Code Snippet**:
  ```python
  # lite_bench/benchmarks.py
  def _symbolic_equivalent(extracted: str, gold: str) -> bool:
      ...
      e_sym = parse_expr(extracted)
  ```
- **Root Cause**: `_symbolic_equivalent` passes model predictions extracted from `\boxed{...}` directly into `sympy.parsing.sympy_parser.parse_expr()`. SymPy's default parser uses Python's standard `eval()` without restricting `__builtins__` or dunder attributes.
- **Impact**: If a model output contains Python dunder attributes or execution calls inside `\boxed{...}` (e.g. `\boxed{__import__('os').system('...')}`), Python executes arbitrary code during benchmark evaluation, bypassing execution sandboxing even when `requires_code_execution = False`.

---

### 🚨 1.2 Silent Hook Bypass on C Extension Types (`SSLContext.wrap_socket`)
- **Location**: [`windows_sandbox.py:978`](file:///C:/Users/lavvo/Documents/lite-benchmarks/windows_sandbox.py#L978) & [`windows_sandbox.py:804–811`](file:///C:/Users/lavvo/Documents/lite-benchmarks/windows_sandbox.py#L804-L811)
- **Category**: Security / Sandbox Confinement Bypass
- **Code Snippet**:
  ```python
  # windows_sandbox.py L978
  self._swap(ctx, "wrap_socket", _raiser(PermissionError, "SSL wrap_socket blocked by Windows sandbox"))
  
  # windows_sandbox.py L804-811
  def _swap(self, target, attr, new_val):
      try:
          old_val = getattr(target, attr)
          setattr(target, attr, new_val)
      except (AttributeError, TypeError):
          return
  ```
- **Root Cause**: Line 978 attempts to hook `ssl.SSLContext.wrap_socket`. In CPython, `ssl.SSLContext` is a built-in C extension type whose class attributes cannot be modified dynamically; `setattr(SSLContext, ...)` raises a `TypeError`. `_swap()` silently catches `TypeError` and ignores the failure.
- **Impact**: Code executing inside the Windows sandbox can instantiate `ssl.SSLContext()` and invoke `wrap_socket(...)` to initiate outbound encrypted TLS/SSL connections, completely bypassing network isolation.

---

### 🚨 1.3 Job Object Kernel Handle & Token Handle Leaks
- **Location**: [`windows_sandbox.py:612–636`](file:///C:/Users/lavvo/Documents/lite-benchmarks/windows_sandbox.py#L612-L636) & [`lite_bench/sandbox.py:285–306`](file:///C:/Users/lavvo/Documents/lite-benchmarks/lite_bench/sandbox.py#L285-L306)
- **Category**: Resource Leak / Kernel Handle Leak
- **Root Cause**: 
  1. `create_child_job()` returns a `ChildJob` wrapping an open Win32 Job Object handle (`self.handle`). `ChildJob` has no `__del__` finalizer or context manager implementation (`__enter__`/`__exit__`). Callers in `sandbox.py` never call `child_job.close()`.
  2. In `_TokenGuard.engage()` ([`windows_sandbox.py:691–730`](file:///C:/Users/lavvo/Documents/lite-benchmarks/windows_sandbox.py#L691-L730)), token handles (`h_tok`, `h_imp`, `h_res`) opened during security token restriction are not cleaned up if an intermediate step raises an `OSError`.
- **Impact**: Every sandboxed subprocess invocation leaks open Win32 kernel handles in the host process, leading to OS handle exhaustion on long benchmark runs.

---

### 🚨 1.4 UNC & Device Path Prefix Check Bypass via Forward Slashes
- **Location**: [`windows_sandbox.py:463`](file:///C:/Users/lavvo/Documents/lite-benchmarks/windows_sandbox.py#L463)
- **Category**: Security / Path Traversal Bypass
- **Code Snippet**:
  ```python
  if s.startswith(("\\\\?\\", "\\\\.\\", "\\\\")):
      return True
  ```
- **Root Cause**: The prefix check for NT device namespaces and network UNC shares occurs *before* path separator normalization. Windows APIs normalize forward slashes (`/`) into backslashes (`\`). Paths formatted with forward slashes (e.g. `"//?/C:/..."` or `"//127.0.0.1/share"`) evaluate to `False` on `s.startswith("\\\\...")`.
- **Impact**: Malicious file path arguments can bypass the UNC/Device namespace gatekeeper check prior to path canonicalization.

---

### 🚨 1.5 Timeout Exception Corruption & Masking in Windows Sandbox
- **Location**: [`windows_sandbox.py:1201, 1222–1223, 1273–1276`](file:///C:/Users/lavvo/Documents/lite-benchmarks/windows_sandbox.py#L1201)
- **Category**: Error Handling / Exception Corruption
- **Root Cause**: When a thread times out in `run_in_sandbox`, line 1201 injects `SystemExit` into the thread. The worker catches `SystemExit` and stores it in `result.exception`. Line 1223 (`if result.timed_out and result.exception is None:`) fails because `result.exception` is `SystemExit`. Later in `run_source_in_sandbox`, lines 1273–1276 check `isinstance(res.exception, SystemExit)` and set `res.exception = None`.
- **Impact**: Timed-out executions report `timed_out = True`, but `exception = None`, masking the actual timeout exception from caller code.

---

## 2. Execution Engine & Concurrency

### 🐛 2.1 Resource & Thread Leak via Non-Blocking Executor Shutdown
- **Location**: [`lite_bench/engine.py:457`](file:///C:/Users/lavvo/Documents/lite-benchmarks/lite_bench/engine.py#L457)
- **Category**: Resource / Thread Leak
- **Code Snippet**:
  ```python
  finally:
      executor.shutdown(wait=False, cancel_futures=True)
  ```
- **Root Cause**: When a benchmark finishes or is cancelled, `executor.shutdown(wait=False, cancel_futures=True)` is called. In Python's `ThreadPoolExecutor`, `cancel_futures=True` only cancels pending queue items; it cannot interrupt active worker threads. `wait=False` causes `run_engine` to return immediately while running worker threads block on network calls or sandboxed execution in the background.
- **Impact**: Orphaned background threads accumulate across benchmark runs, causing memory leaks, unhandled API requests, and socket/port exhaustion.

---

### 🐛 2.3 Lost Jitter & Thundering Herd in Rate Limit (HTTP 429) Backoff
- **Location**: [`lite_bench/engine.py:173–174`](file:///C:/Users/lavvo/Documents/lite-benchmarks/lite_bench/engine.py#L173-L174)
- **Category**: Concurrency / Network Performance
- **Code Snippet**:
  ```python
  backoff = min(60.0, 5.0 * (2 ** min(attempt - 1, 4))) + random.uniform(0, 5.0)
  if "429" in err_str or "rate limit" in err_str:
      backoff = max(60.0, backoff)
  ```
- **Root Cause**: Randomized jitter (`random.uniform(0, 5.0)`) is added to `backoff` *before* the `max(60.0, backoff)` check. On HTTP 429 rate limit errors, `max(60.0, backoff)` evaluates to **exactly 60.0 seconds**, stripping out jitter completely.
- **Impact**: When parallel threads hit a rate limit, all worker threads sleep for exactly 60.0 seconds and wake up at the exact same millisecond, triggering a thundering herd that repeatedly re-hits HTTP 429 rate limits.

---

### 🐛 2.4 Repeated Error Circuit Breaker `isinstance` Exception Bug
- **Location**: [`lite_bench/engine.py:157–163`](file:///C:/Users/lavvo/Documents/lite-benchmarks/lite_bench/engine.py#L157-L163)
- **Category**: Logic / Infinite Retry Loop Risk
- **Code Snippet**:
  ```python
  if isinstance(e, RuntimeError) and err_msg_str == last_error_msg:
      consecutive_same_error += 1
  ```
- **Root Cause**: The repeated error circuit breaker strictly checks `isinstance(e, RuntimeError)`. Third-party SDK exceptions (such as `openai.APIError`, `httpx.HTTPStatusError`, `litellm.APIConnectionError`) are not instances of `RuntimeError`.
- **Impact**: If an API provider repeatedly raises a non-`RuntimeError` exception, `consecutive_same_error` resets to `1` on every attempt, creating an infinite retry loop if `max_retries` is unconstrained.

---

## 3. Verification & Scoring Integrity

### 🐛 3.1 False Positive Score (100% Pass) on Invalid Tool Arguments in `TauBenchBenchmark`
- **Location**: [`lite_bench/benchmarks.py:1404–1406`](file:///C:/Users/lavvo/Documents/lite-benchmarks/lite_bench/benchmarks.py#L1404-L1406)
- **Category**: Scoring Correctness / False Positive
- **Code Snippet**:
  ```python
  if isinstance(pred_args, dict):
      p_norm = {k: v for k, v in pred_args.items() if v is not None}
  else:
      p_norm = {}
  ```
- **Root Cause**: If a model outputs invalid tool arguments (e.g. `null`, integer, or malformed JSON), `pred_args` is not a dict, so `p_norm` falls back to `{}`. When the benchmark gold question expects no arguments (`gold_args == {}`), `not p_norm` evaluates to `True`, awarding a `1.0` (100% pass) score to malformed predictions.
- **Impact**: Models returning corrupted tool arguments receive perfect scores for zero-argument tool calls.

---

### 🐛 3.2 Scientific Notation Rejection in `MATH500Benchmark`
- **Location**: [`lite_bench/benchmarks.py:1131`](file:///C:/Users/lavvo/Documents/lite-benchmarks/lite_bench/benchmarks.py#L1131)
- **Category**: Scoring Correctness / False Negative
- **Root Cause**: Numerical answer evaluation verifies predictions using `re.fullmatch(r"-?[\d.]+", pred_num)`. However, `_extract_number` extracts numbers in scientific notation (e.g. `'1e-05'`). The regex fails on scientific notation exponent characters (`e`/`+`), marking valid scientific answers as `0.0`.

---

### 🐛 3.3 Python `str` Treated as `Sequence` in IFEval Verifiers
- **Location**: [`lite_bench/ifeval_verifiers.py:119, 163, 180`](file:///C:/Users/lavvo/Documents/lite-benchmarks/lite_bench/ifeval_verifiers.py#L119)
- **Category**: Verification Correctness
- **Root Cause**: Python strings inherit from `collections.abc.Sequence`, so `isinstance("string", Sequence)` evaluates to `True`. If `forbidden_words` or `keywords` in dataset kwargs are provided as a string instead of a list of strings, verifiers iterate over individual character elements (`'s'`, `'t'`, `'r'`, `'i'`, `'n'`, `'g'`) instead of full words.

---

## 4. Data Caching, API Providers & Config

### 🐛 4.1 Cache Collision via Generic `row_filter` Flag
- **Location**: [`lite_bench/datasets.py:30–36`](file:///C:/Users/lavvo/Documents/lite-benchmarks/lite_bench/datasets.py#L30-L36)
- **Category**: Data Integrity / Caching Bug
- **Code Snippet**:
  ```python
  cache_key = (
      f"{bench.name}/{bench.dataset}/{bench.subset}/{bench.split}/"
      f"{bench.revision or 'main'}/{bench.num_samples}/{settings.seed}/"
      f"{'filtered' if row_filter is not None else 'all'}"
  )
  ```
- **Root Cause**: The cache key represents `row_filter` as a binary string (`"filtered"` vs `"all"`). If two benchmarks target the same dataset, split, seed, and count but supply different `row_filter` functions, the second benchmark hits a false cache entry and receives questions filtered by the first benchmark's filter.

---

### 🔒 4.2 API Key Exposure in Exception Tracebacks & Logs
- **Location**: [`lite_bench/providers.py:61–92, 192–205`](file:///C:/Users/lavvo/Documents/lite-benchmarks/lite_bench/providers.py#L61-L92)
- **Category**: Security / Information Disclosure
- **Root Cause**: API keys read from environment variables are passed directly in `kwargs["api_key"]`. When LiteLLM or `httpx` raises an exception (`AuthenticationError`, `APIError`), the exception object includes `kwargs` and request headers in `str(e)`. Re-raising `e` logs unredacted API keys to console/file logs.

---

### 🐛 4.3 Unhandled `yaml.YAMLError` & Empty List Fallback Masking
- **Location**: [`lite_bench/config.py:161–163`](file:///C:/Users/lavvo/Documents/lite-benchmarks/lite_bench/config.py#L161-L163)
- **Category**: Configuration / Crash Vector
- **Root Cause**: 
  1. `yaml.safe_load(config_file)` is called without a `try...except` block, causing raw `ScannerError`/`ParserError` crashes on invalid YAML syntax.
  2. `raw = yaml.safe_load(...) or {}` converts empty top-level YAML arrays (`[]`) into `{}` because `[]` evaluates to falsy in Python, silently bypassing array schema validation.

---

## 5. Persistence, Math & Report Generation

### 🐛 5.1 Multi-Process Race Condition & History Data Loss
- **Location**: [`lite_bench/results_store.py:23, 212–285`](file:///C:/Users/lavvo/Documents/lite-benchmarks/lite_bench/results_store.py#L23)
- **Category**: Data Persistence / Race Condition
- **Root Cause**: `_history_lock = threading.Lock()` only synchronizes threads within a single process. When multiple benchmarking processes or background workers run concurrently, both read `history.json`, append their run data, and overwrite the file via `os.replace`. The last process to write overwrites the other process's history records.

---

### 🐛 5.2 Out-of-Spec JSON Serialization (`NaN` / `Inf` Literals)
- **Location**: [`lite_bench/results_store.py:181`](file:///C:/Users/lavvo/Documents/lite-benchmarks/lite_bench/results_store.py#L181) & [`refresh_site.py:167`](file:///C:/Users/lavvo/Documents/lite-benchmarks/refresh_site.py#L167)
- **Category**: Data Serialization / Standard Violation
- **Root Cause**: Standard `json.dump()` converts Python `float('nan')` and `float('inf')` to unquoted `NaN` and `Infinity` literals, which violate RFC 8259 JSON standards. If telemetry produces `NaN` TPS or average time values, writing `latest.json` or `summary.json` causes fatal `SyntaxError` crashes in web browsers (`JSON.parse`).

---

## 6. Frontend Astro & Client Scripts

### 🔒 6.1 Client-Side XSS Vulnerabilities in Raw ECharts Tooltips
- **Location**: [`site/src/lib/charts.ts:96–97, 222–225, 332–333`](file:///C:/Users/lavvo/Documents/lite-benchmarks/site/src/lib/charts.ts#L96-L97)
- **Category**: Security / Cross-Site Scripting (XSS)
- **Code Snippet**:
  ```typescript
  formatter: (p: any) => {
    const [cost, score, tps, name] = p.data;
    return `<b style="color:${t.fg}">${name}</b><br/>...`;
  }
  ```
- **Root Cause**: ECharts HTML tooltip formatters interpolate `name` (model name) and `display` (benchmark title) directly into raw HTML strings without escaping special HTML characters (`<`, `>`, `&`). If dataset JSON files contain unescaped strings, client-side XSS executes when hovering over chart elements.

---

### 🐛 6.2 Medals Awarded to Worst Models on Ascending Table Sort
- **Location**: [`site/src/lib/client.ts:140–149`](file:///C:/Users/lavvo/Documents/lite-benchmarks/site/src/lib/client.ts#L140-L149)
- **Category**: UI / Business Logic Error
- **Code Snippet**:
  ```typescript
  pairs.forEach((p, i) => {
    ...
    if (i === 0) rank.textContent = "🥇";
    else if (i === 1) rank.textContent = "🥈";
    else if (i === 2) rank.textContent = "🥉";
  });
  ```
- **Root Cause**: Clicking table headers to sort ascending (`dirs[key] = -1`) places the lowest-scoring models at array indices `0, 1, 2`. `sortRows` assigns medals purely based on array index `i`, incorrectly awarding 🥇 Gold, 🥈 Silver, and 🥉 Bronze medals to the worst-performing models.

---

### 🐛 6.3 `fmtInt` Unhandled `TypeError` Crash on Null/Undefined Values
- **Location**: [`site/src/lib/data.ts:83`](file:///C:/Users/lavvo/Documents/lite-benchmarks/site/src/lib/data.ts#L83)
- **Category**: Runtime Crash
- **Code Snippet**:
  ```typescript
  export const fmtInt = (v: number): string => v.toLocaleString("en-US");
  ```
- **Root Cause**: Invoking `fmtInt(null)` or `fmtInt(undefined)` raises an unhandled `TypeError: Cannot read properties of undefined (reading 'toLocaleString')`. Used directly in `LeaderboardTable.astro` (line 164) and `TokenTable.astro` (lines 48–51), crashing table rendering if token or question count fields are missing.

---

## 7. Test Suite Critical Errors & Flaws

### 🐛 7.2 Flaky Environment Dependency in Historical Results Loading Test
- **Location**: [`tests/test_judge_integrity.py:44–45`](file:///C:/Users/lavvo/Documents/lite-benchmarks/tests/test_judge_integrity.py#L44-L45)
- **Category**: Test Stability / Flaky Assertion
- **Code Snippet**:
  ```python
  json_files = glob.glob("results/**/*.json", recursive=True)
  self.assertTrue(len(json_files) > 0, "No result files found in results/ directory")
  ```
- **Root Cause**: Hard-asserts that `results/**/*.json` is non-empty. On a fresh repository checkout or in clean CI pipelines where benchmarks have not yet been run, `json_files` is empty, causing `pytest` to fail.

---

### 🚨 7.3 Host Home Directory File Pollution Risk in `test_windows_sandbox.py`
- **Location**: [`tests/test_windows_sandbox.py:70–78`](file:///C:/Users/lavvo/Documents/lite-benchmarks/tests/test_windows_sandbox.py#L70-L78)
- **Category**: Test Safety / Host Pollution Risk
- **Code Snippet**:
  ```python
  with open(os.path.join(os.path.expanduser("~"), "pwned.txt"), "w") as f:
  ```
- **Root Cause**: The test verifies that sandbox file containment blocks writes outside the sandbox by attempting to create `~/pwned.txt` in the user's host home directory. If a sandbox security regression occurs, the test will pollute or overwrite a file directly inside the user's host home folder.

---

## 8. Audit Summary Table

| Category | High / Critical Bug | Affected Files | Impact Summary |
| :--- | :--- | :--- | :--- |
| **Security** | SymPy Unsandboxed `eval` RCE | `lite_bench/benchmarks.py` | Arbitrary execution during math string evaluation |
| **Security** | C-Extension `SSLContext` Hook Bypass | `windows_sandbox.py` | Outbound TLS socket restriction bypass in Windows sandbox |
| **Security** | UNC / Device Path Prefix Bypass | `windows_sandbox.py` | Forward-slash prefix check bypass before canonicalization |
| **Security** | ECharts Raw Tooltip Injection | `site/src/lib/charts.ts` | Client-side XSS on chart hover |
| **Security** | API Key Exposure in Error Logs | `lite_bench/providers.py` | Provider secret keys logged to console/files in tracebacks |
| **Engine** | Non-blocking Thread Leak | `lite_bench/engine.py` | Orphaned worker threads linger after run completion |
| **Engine** | Rate Limit Jitter Loss (429) | `lite_bench/engine.py` | Thundering herd storms on rate limit backoff |
| **Scoring** | TauBench False Positive 100% Pass | `lite_bench/benchmarks.py` | Non-dict tool args awarded 1.0 score for zero-arg tools |
| **Scoring** | MATH500 Scientific Notation Rejection | `lite_bench/benchmarks.py` | Valid scientific notation predictions scored 0.0 |
| **Scoring** | IFEval `str` Sequence Type Matching | `lite_bench/ifeval_verifiers.py` | Single string kwargs matched character-by-character |
| **Data & Config** | Row Filter Cache Key Collision | `lite_bench/datasets.py` | Wrong questions served across custom filtered benchmarks |
| **Persistence** | Multi-Process History Overwrite | `lite_bench/results_store.py` | History record loss on concurrent process execution |
| **Persistence** | `NaN`/`Inf` Invalid JSON Serialization | `lite_bench/results_store.py`, `refresh_site.py` | JSON.parse syntax error crashes on frontends |
| **Frontend UI** | Medal Inversion on Ascending Sort | `site/src/lib/client.ts` | Worst models awarded Gold/Silver/Bronze medals |
| **Frontend UI** | `fmtInt(null)` Table Crash | `site/src/lib/data.ts` | Unhandled TypeError crashing UI page render |
| **Test Suite** | Host Home Folder File Creation Risk | `tests/test_windows_sandbox.py` | Polluting user's `~/pwned.txt` if sandbox regressions occur |

