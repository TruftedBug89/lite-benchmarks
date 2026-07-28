<div align="center">

# 🏆 Lite Benchmarks

### Personal LLM Leaderboard & Benchmark Studio

[![CI](https://img.shields.io/github/actions/workflow/status/TruftedBug89/lite-benchmarks/ci.yml?branch=main&style=flat-square&logo=github&label=CI)](https://github.com/TruftedBug89/lite-benchmarks/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
![Benchmarks](https://img.shields.io/badge/Benchmarks-12-green?style=flat-square)
![Categories](https://img.shields.io/badge/Categories-5-purple?style=flat-square)
![Models](https://img.shields.io/badge/Models_tested-3-orange?style=flat-square)
![Version](https://img.shields.io/badge/version-0.2.0-red?style=flat-square)

*Small, repeatable samples of established benchmarks with **100% programmatic scoring**. No LLM-as-judge. Deterministic sampling. Sandboxed code execution.*

</div>

## ✨ Why Lite Benchmarks?

<table>
<tr>
<td width="50%">

🎯 **Programmatic Scoring**

Every answer is verified by code — regex extraction, unit test execution, exact match. Zero LLM-as-judge bias.

</td>
<td width="50%">

🔒 **3-Layer Sandboxed Execution**

AST scan → hardened subprocess → Windows Job Object. Model code runs isolated with no API keys, no network, no escape.

</td>
</tr>
<tr>
<td width="50%">

🌐 **Web Dashboard**

Select models, pick benchmarks, run, and generate reports — all from a local browser UI. No CLI wrangling.

</td>
<td width="50%">

🎲 **Deterministic Sampling**

Fixed seed (42) random sampling means the exact same questions every run. Reproducible by design.

</td>
</tr>
<tr>
<td width="50%">

📊 **Statistical Rigor**

Wilson score confidence intervals on every benchmark score. Know exactly how much noise is in the numbers.

</td>
<td width="50%">

💰 **Cost & Token Tracking**

Per-model token breakdown (input/output/thinking), throughput (TPS), latency, and estimated API cost.

</td>
</tr>
</table>

## ⚡ Quick Start

```bash
pip install -e .[dev]    # install dependencies
py web_app.py            # launch dashboard → http://127.0.0.1:8000
```

Then select models, pick benchmarks, hit **Run Benchmarks**, and **Generate Reports**.

## 📑 Table of Contents

- [Benchmarks](#-benchmarks)
- [Leaderboard](#-leaderboard)
- [Charts](#-charts)
- [Token Usage & Performance](#-token-usage--performance)
- [Architecture](#-architecture)
- [Methodology](#-methodology)
- [How to Run](#-how-to-run)
- [Adding Models](#-adding-models)
- [Project Structure](#-project-structure)

## 📝 Benchmarks

12 established benchmarks, ~50 questions each, grouped into 5 categories:

| Benchmark | Category | Full Dataset | Sampled | Verification | Source |
|-----------|----------|:-----------:|:-------:|-------------|--------|
| **BigCodeBench-Hard** | Coding | 148 | 50 | Python unittest execution (explicit opt-in required) | `bigcode/bigcodebench-hard (v0.1.4)` |
| **HumanEval+** | Coding | 164 | 50 | Python test execution (explicit opt-in required) | `evalplus/humanevalplus` |
| **MBPP+** | Coding | 378 | 50 | Python test execution (explicit opt-in required) | `evalplus/mbppplus` |
| **GPQA Diamond** | Science | 198 | 50 | Multiple choice (4 options) | `nichenshun/gpqa_diamond (community mirror of Idavidrein/gpqa)` |
| **SciBench** | Science | 692 | 50 | Numerical / Formula exact match | `xw27/scibench` |
| **AIME 2024/2025** | Math | 90 | 50 | Integer exact match (000-999) | `AI-MO/aimo-validation-aime` |
| **MATH-500** | Math | 500 | 50 | Exact match / \boxed{} extraction | `HuggingFaceH4/MATH-500` |
| **MMLU-Pro** | Knowledge | 12,032 | 50 | Multiple choice (10 options) | `TIGER-Lab/MMLU-Pro` |
| **IFEval** | Instruction | 541 | 50 | 25 programmatic verifiers (strict) | `google/IFEval` |
| **SciCode** | Coding | 65 | 50 | Python code execution & unit test assertions | `SciCode1/SciCode` |
| **SuperGPQA** | Knowledge | 26,529 (7,050 hard) | 50 | Multiple choice (up to 10 options) | `m-a-p/SuperGPQA` |
| **Tau-Bench (Retail)** | Instruction | 82 | 50 | Agentic tool-call function & argument matching | `amityco/tau-bench-retail-train-next-action` |

## 🏅 Leaderboard

| Rank | Model | Overall | 💻 Coding | 🔬 Science | 📐 Math | 📚 Knowledge | 📋 Instruction |
|:----:|-------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| 🥇 | **environment** | N/A | N/A | N/A | N/A | N/A | N/A |
| 🥈 | **settings** | N/A | N/A | N/A | N/A | N/A | N/A |
| 🥉 | **models** | N/A | N/A | N/A | N/A | N/A | N/A |

*🧠 indicates reasoning models that utilize thinking tokens or have explicit thinking effort configured.*

### Per-Benchmark Scores

| Model | BigCodeBench-Hard | HumanEval+ | MBPP+ | GPQA Diamond | SciBench | AIME 2024/2025 | MATH-500 | MMLU-Pro | IFEval | SciCode | SuperGPQA | Tau-Bench (Retail) |
|-------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| environment | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| settings | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| models | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

*±pp indicates 95% Wilson score confidence interval half-width.*

## 📊 Charts

### Category Breakdown

*Grouped bar chart comparing each model across the 5 categories.*

![Category Breakdown](charts/categories.png)

### Benchmark Heatmap

*Per-benchmark scores for every model. Green = high, red = low.*

![Benchmark Heatmap](charts/heatmap.png)

### Overall Scores

*Horizontal bar chart ranked by overall score (average of all category scores).*

![Overall Scores](charts/leaderboard.png)

### Category Radar

*Spider chart showing each model's profile across categories. Larger area = stronger overall.*

![Category Radar](charts/radar.png)

### Thinking Effort vs Performance

*Scatter plot showing if models that use more thinking tokens achieve higher overall scores.*

![Thinking Effort vs Performance](charts/thinking_scatter.png)

### Token Breakdown

*Stacked bar chart of input, thinking, and output tokens per model.*

![Token Breakdown](charts/tokens.png)

## 🪙 Token Usage & Performance

| Model | Input | Output | Thinking | Total | Out % | Think % | Avg TPS | Avg Time | Est. Cost |
|-------|------:|-------:|---------:|------:|------:|--------:|--------:|---------:|----------:|
| environment | 0 | 0 | 0 | 0 | — | — | — | — | — |
| settings | 0 | 0 | 0 | 0 | — | — | — | — | — |
| models | 0 | 0 | 0 | 0 | — | — | — | — | — |

*TPS = output tokens/second (cloud APIs only, skipped for local models). Est. Cost calculated via LiteLLM cost tables.*

## 🏗️ Architecture

```mermaid
flowchart LR
    A[config.yaml] --> B[datasets.py<br/>HF sampling]
    B --> C[engine.py<br/>concurrent execution]
    C --> D[providers.py<br/>litellm calls]
    D --> E[benchmarks.py<br/>scoring & verification]
    E --> F[results_store.py<br/>schema v2 JSON]
    F --> G[charts.py<br/>matplotlib PNGs]
    F --> H[readme_gen.py<br/>this README]

    C --> I[sandbox.py<br/>3-layer isolation]
    I --> E
```

## 🔬 Methodology

<details>
<summary><strong>📐 Sampling & Statistical Significance</strong></summary>

- **~50 questions** are sampled from each benchmark's full dataset
- Sampling uses a **fixed seed (42)** via random sampling so exact questions are stable across runs
- Samples of n=50 have 95% confidence intervals of roughly ±7–14pp; treat small ranking gaps as noise
- **Scoring v2 Notice**: Sampling and scoring strictness updated in v0.2.0; results are not directly comparable with pre-v0.2.0 runs

</details>

<details>
<summary><strong>✅ Scoring & Verification</strong></summary>

- **All scoring is programmatic** — no LLM-as-judge is used anywhere
- Code benchmarks require explicit opt-in (``allow_unsafe_code_execution``) and run in a layered sandbox: an AST scan of generated code rejects destructive / escape constructs, the child runs with a scrubbed environment (no API keys, temp working dir, no OS/network/process access), and on Windows it is additionally confined by a Job Object that blocks grandchild processes and UI access. The opt-in gate is enforced at the sandbox layer, so it fails closed even for direct callers.
- Multiple-choice benchmarks extract the answer letter and compare to ground truth
- Math benchmarks extract boxed/numerical answers and evaluate via normalized string or numerical comparison
- IFEval uses its 25 strict programmatic verifiers (word count, format, keywords, etc.)
- Tau-Bench verifies tool function name AND argument dictionary match

</details>

<details>
<summary><strong>📊 Category & Overall Scores</strong></summary>

- **Category score** = average of its benchmark scores
  - 💻 **Coding** = avg(BigCodeBench-Hard, HumanEval+, MBPP+, SciCode)
  - 🔬 **Science** = avg(GPQA Diamond, SciBench)
  - 📐 **Math** = avg(AIME 2024/2025, MATH-500)
  - 📚 **Knowledge** = avg(MMLU-Pro, SuperGPQA)
  - 📋 **Instruction** = avg(IFEval, Tau-Bench (Retail))
- **Overall score** = average of completed category scores (equal weight per category)
- Provider failures are excluded and recorded separately; scorer exceptions score 0.0 without retrying provider

</details>

<details>
<summary><strong>⚙️ Inference Settings</strong></summary>

- `temperature`: 0.0
- `max_tokens`: 4096
- `timeout`: 300s per request
- `retries`: transient errors retry with exponential backoff until a good response arrives (no cap); permanent errors (context length, content filter) are never retried

</details>

## 🚀 How to Run

### Prerequisites

```bash
pip install -e .[dev]
```

### API Keys

Set environment variables for the providers you want to test.
[litellm](https://docs.litellm.ai/docs/providers) picks them up automatically.

| Provider | Environment Variable | Get a key |
|----------|---------------------|-----------|
| DeepSeek | `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com) |
| Groq | `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |
| Google Gemini | `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) |
| LM Studio (local) | *(none needed)* | [lmstudio.ai](https://lmstudio.ai) |
| HuggingFace | `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

### Running Benchmarks via Web Dashboard

```bash
# Launch the local web dashboard (serves on http://127.0.0.1:8000)
py web_app.py

# Launch without automatically opening browser
py web_app.py --no-browser
```

1. Open the dashboard in your browser.
2. Select models, benchmarks, and settings.
3. Click **Run Benchmarks**.
4. Click **Generate Reports** to update `README.md` and `charts/`.

## ➕ Adding Models

Edit `config.yaml` or add models directly in the Web UI:

```yaml
models:
  - id: anthropic/claude-sonnet-4-20250514
    name: Claude Sonnet 4
    max_tokens: 16384
  - id: openai/gpt-4o
    name: GPT-4o
  - id: lm_studio/qwen2.5-coder-7b-instruct
    name: Qwen 2.5 Coder 7B (local)
```

## 📁 Project Structure

```
├── config.yaml              # Models, benchmarks, categories, settings
├── web_app.py               # Web dashboard server
├── windows_sandbox.py       # Windows Job-object/restricted-token sandbox
├── README.md                # ← this file (auto-generated)
├── web/                     # Web dashboard frontend (HTML/CSS/JS)
├── lite_bench/
│   ├── engine.py            # Unified execution engine & thread concurrency
│   ├── results_store.py     # Results persistence, schema v2, atomic writes
│   ├── metadata.py          # Benchmark display metadata & category mapping
│   ├── config.py            # Config loading & validation
│   ├── providers.py         # litellm wrapper & telemetry
│   ├── datasets.py          # Deterministic HuggingFace sampling
│   ├── benchmarks.py        # Benchmark implementations & verifiers
│   ├── ifeval_verifiers.py  # 25 strict IFEval verifiers
│   ├── sandbox.py           # Code-exec sandbox (AST scan + subprocess + Win job)
│   ├── charts.py            # matplotlib chart generation
│   └── readme_gen.py        # README generator
├── results/                 # JSON results per run
│   └── latest.json          # Leaderboard results (schema v2)
└── charts/                  # Generated PNG charts
```

---

<div align="center">

*Auto-generated by [lite-benchmarks](.) on 2026-07-28 20:57 UTC · Licensed under [MIT](LICENSE) · Built with [litellm](https://github.com/BerriAI/litellm) + [HuggingFace Datasets](https://github.com/huggingface/datasets)*

**⭐ Star this repo if you find it useful!**

</div>
