# 🏆 Lite Benchmarks — Personal LLM Leaderboard

> **Small, repeatable samples of established benchmarks with programmatic scoring.
> No LLM-as-judge. Sampling and scoring are deterministic; model outputs may vary.**

This repo benchmarks LLMs on ~50 questions sampled from 12 established benchmarks
grouped into 5 core categories. Results, rankings, and charts below are
**auto-generated** after every run.

## 📝 Benchmarks

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
| 🥇 | **Gemma 4 31B** 🧠 | **31.0%** | N/A | 42.0% | 20.0% | N/A | N/A |

*🧠 indicates reasoning models that utilize thinking tokens or have explicit thinking effort configured.*

### Per-Benchmark Scores

| Model | BigCodeBench-Hard | HumanEval+ | MBPP+ | GPQA Diamond | SciBench | AIME 2024/2025 | MATH-500 | MMLU-Pro | IFEval | SciCode | SuperGPQA | Tau-Bench (Retail) |
|-------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Gemma 4 31B | N/A | N/A | N/A | 62% (29.0/47) ±13.4pp | 22% (10.0/45) ±11.9pp | 20% (1.0/5) ±29.4pp | N/A | N/A | N/A | N/A | N/A | N/A |

*±pp indicates 95% Wilson score confidence interval half-width.*

## 📊 Charts

### Overall Scores

*Horizontal bar chart ranked by overall score (average of all category scores).*

![Overall Scores](C:/Users/lavvo/Documents/lite-benchmarks/charts/leaderboard.png)

### Category Breakdown

*Grouped bar chart comparing each model across the 5 categories.*

![Category Breakdown](C:/Users/lavvo/Documents/lite-benchmarks/charts/categories.png)

### Category Radar

*Spider chart showing each model's profile across categories. Larger area = stronger overall.*

![Category Radar](C:/Users/lavvo/Documents/lite-benchmarks/charts/radar.png)

### Benchmark Heatmap

*Per-benchmark scores for every model. Green = high, red = low.*

![Benchmark Heatmap](C:/Users/lavvo/Documents/lite-benchmarks/charts/heatmap.png)

### Token Breakdown

*Stacked bar chart of input, thinking, and output tokens per model.*

![Token Breakdown](C:/Users/lavvo/Documents/lite-benchmarks/charts/tokens.png)

### Thinking Effort vs Performance

*Scatter plot showing if models that use more thinking tokens achieve higher overall scores.*

![Thinking Effort vs Performance](C:/Users/lavvo/Documents/lite-benchmarks/charts/thinking_scatter.png)

## 🪙 Token Usage & Performance

| Model | Input | Output | Thinking | Total | Out % | Think % | Avg TPS | Avg Time | Est. Cost |
|-------|------:|-------:|---------:|------:|------:|--------:|--------:|---------:|----------:|
| Gemma 4 31B | 20,224 | 17,272 | 0 | 364,541 | 5% | — | 8.7 | 103.9s | — |

*TPS = output tokens/second (cloud APIs only, skipped for local models). Est. Cost calculated via LiteLLM cost tables.*

## 🔬 Methodology

### Sampling & Statistical Significance
- **~50 questions** are sampled from each benchmark's full dataset
- Sampling uses a **fixed seed (42)** via random sampling so exact questions are stable across runs
- Samples of n=50 have 95% confidence intervals of roughly ±7–14pp; treat small ranking gaps as noise
- **Scoring v2 Notice**: Sampling and scoring strictness updated in v0.2.0; results are not directly comparable with pre-v0.2.0 runs

### Scoring
- **All scoring is programmatic** — no LLM-as-judge is used anywhere
- Code benchmarks require explicit opt-in (``allow_unsafe_code_execution``) and run in a layered sandbox: an AST scan of generated code rejects destructive / escape constructs, the child runs with a scrubbed environment (no API keys, temp working dir, no OS/network/process access), and on Windows it is additionally confined by a Job Object that blocks grandchild processes and UI access. The opt-in gate is enforced at the sandbox layer, so it fails closed even for direct callers.
- Multiple-choice benchmarks extract the answer letter and compare to ground truth
- Math benchmarks extract boxed/numerical answers and evaluate via normalized string or numerical comparison
- IFEval uses its 25 strict programmatic verifiers (word count, format, keywords, etc.)
- Tau-Bench verifies tool function name AND argument dictionary match

### Category & Overall Scores
- **Category score** = average of its benchmark scores
  - 💻 **Coding** = avg(BigCodeBench-Hard, HumanEval+, MBPP+, SciCode)
  - 🔬 **Science** = avg(GPQA Diamond, SciBench)
  - 📐 **Math** = avg(AIME 2024/2025, MATH-500)
  - 📚 **Knowledge** = avg(MMLU-Pro, SuperGPQA)
  - 📋 **Instruction** = avg(IFEval, Tau-Bench (Retail))
- **Overall score** = average of completed category scores (equal weight per category)
- Provider failures are excluded and recorded separately; scorer exceptions score 0.0 without retrying provider

### Inference Settings
- `temperature`: 0.0
- `max_tokens`: 4096
- `timeout`: 300s per request
- `retries`: up to 3 retries with exponential backoff and jitter

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

*Auto-generated by [lite-benchmarks](.) on 2026-07-25 03:14 UTC. Use the Web Dashboard to update.*
