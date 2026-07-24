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
| 🥇 | **deepseek/deepseek-v4-flash** 🧠 | **59.8%** | 40.0% | 39.0% | 56.0% | 80.0% | 84.0% |
| 🥈 | **gemini/gemini-3.1-flash-lite** 🧠 | **52.0%** | 29.5% | 51.0% | 78.0% | 60.0% | 41.3% |

*🧠 indicates reasoning models that utilize thinking tokens or have explicit thinking effort configured.*

### Per-Benchmark Scores

| Model | BigCodeBench-Hard | HumanEval+ | MBPP+ | GPQA Diamond | SciBench | AIME 2024/2025 | MATH-500 | MMLU-Pro | IFEval | SciCode | SuperGPQA | Tau-Bench (Retail) |
|-------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| deepseek/deepseek-v4-flash | 20% (10/50) ±10.9pp | 90% (45/50) ±8.5pp | 10% (5/50) ±8.5pp | 58% (29/50) ±13.2pp | 20% (10/50) ±10.9pp | 24% (12/50) ±11.6pp | 88% (44/50) ±9.1pp | 80% (40/50) ±10.9pp | 84% (42/50) ±10.1pp | N/A | N/A | N/A |
| gemini/gemini-3.1-flash-lite | 20% (10/50) ±10.9pp | 90% (45/50) ±8.5pp | 8% (4/50) ±7.8pp | 62% (31/50) ±13.0pp | 40% (20/50) ±13.1pp | 58% (29/50) ±13.2pp | 98% (49/50) ±5.1pp | 60% (30/50) ±13.1pp | 83% (38/46) ±10.8pp | 0% (0/50) ±3.6pp | N/A | 0% (0/3) ±28.1pp |

*±pp indicates 95% Wilson score confidence interval half-width.*

## 📊 Charts

### Overall Scores

*Horizontal bar chart ranked by overall score (average of all category scores).*

![Overall Scores](charts/leaderboard.png)

### Category Breakdown

*Grouped bar chart comparing each model across the 5 categories.*

![Category Breakdown](charts/categories.png)

### Category Radar

*Spider chart showing each model's profile across categories. Larger area = stronger overall.*

![Category Radar](charts/radar.png)

### Benchmark Heatmap

*Per-benchmark scores for every model. Green = high, red = low.*

![Benchmark Heatmap](charts/heatmap.png)

### Token Breakdown

*Stacked bar chart of input, thinking, and output tokens per model.*

![Token Breakdown](charts/tokens.png)

### Thinking Effort vs Performance

*Scatter plot showing if models that use more thinking tokens achieve higher overall scores.*

![Thinking Effort vs Performance](charts/thinking_scatter.png)

## 🪙 Token Usage & Performance

| Model | Input | Output | Thinking | Total | Out % | Think % | Avg TPS | Avg Time | Est. Cost |
|-------|------:|-------:|---------:|------:|------:|--------:|--------:|---------:|----------:|
| deepseek/deepseek-v4-flash | 66,677 | 55,762 | 680,781 | 803,220 | 7% | 85% | 94.4 | 16.8s | — |
| gemini/gemini-3.1-flash-lite | 87,458 | 186,297 | 0 | 273,755 | 68% | — | 64.2 | 5.6s | — |

*TPS = output tokens/second (cloud APIs only, skipped for local models). Est. Cost calculated via LiteLLM cost tables.*

## 🔬 Methodology

### Sampling & Statistical Significance
- **~50 questions** are sampled from each benchmark's full dataset
- Sampling uses a **fixed seed (42)** via random sampling so exact questions are stable across runs
- Samples of n=50 have 95% confidence intervals of roughly ±7–14pp; treat small ranking gaps as noise
- **Scoring v2 Notice**: Sampling and scoring strictness updated in v0.2.0; results are not directly comparable with pre-v0.2.0 runs

### Scoring
- **All scoring is programmatic** — no LLM-as-judge is used anywhere
- Code benchmarks require explicit opt-in and run in a restricted sandbox (AST scan of generated code + scrubbed subprocess environment: no API keys, temp working dir, no OS/network/process access)
- Multiple-choice benchmarks extract the answer letter and compare to ground truth
- Math benchmarks extract boxed/numerical answers and evaluate symbolically or numerically
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
│   ├── charts.py            # matplotlib chart generation
│   └── readme_gen.py        # README generator
├── results/                 # JSON results per run
│   └── latest.json          # Leaderboard results (schema v2)
└── charts/                  # Generated PNG charts
```

---

*Auto-generated by [lite-benchmarks](.) on 2026-07-23 18:31 UTC. Use the Web Dashboard to update.*
