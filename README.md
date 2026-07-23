# 🏆 Lite Benchmarks — Personal LLM Leaderboard

> **Small, repeatable samples of established benchmarks with programmatic scoring.
> No LLM-as-judge. Sampling and scoring are deterministic; model outputs may vary.**

This repo benchmarks LLMs on ~50 questions sampled from each of 8 established
benchmarks, grouped into 5 categories. Results, rankings, and charts below are
**auto-generated** by `py run_benchmark.py` after every run.

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

### Benchmark Details

<details>
<summary><b>BigCodeBench-Hard</b> — The hardest 148 practical Python programming tasks from BigCodeBench requiring d…</summary>

The hardest 148 practical Python programming tasks from BigCodeBench requiring deep integration of complex real-world libraries (pandas, numpy, scipy, etc.).

- **Paper:** Zhuo et al. 2024
- **Dataset:** `bigcode/bigcodebench-hard (v0.1.4)`
- **Verification:** Python unittest execution (explicit opt-in required)
- **Full dataset size:** 148 questions
- **Sampled:** 50 questions (seed=42)

</details>

<details>
<summary><b>HumanEval+</b> — 164 hand-written Python functions with docstrings and rigorously expanded test c…</summary>

164 hand-written Python functions with docstrings and rigorously expanded test cases to catch edge-case bugs and hallucinated solutions.

- **Paper:** Chen et al. 2021, augmented by Liu et al. 2023 (EvalPlus)
- **Dataset:** `evalplus/humanevalplus`
- **Verification:** Python test execution (explicit opt-in required)
- **Full dataset size:** 164 questions
- **Sampled:** 50 questions (seed=42)

</details>

<details>
<summary><b>MBPP+</b> — 378 crowd-sourced Python programming problems with heavily augmented test suites…</summary>

378 crowd-sourced Python programming problems with heavily augmented test suites from EvalPlus for deep coverage.

- **Paper:** Austin et al. 2021, augmented by Liu et al. 2023 (EvalPlus)
- **Dataset:** `evalplus/mbppplus`
- **Verification:** Python test execution (explicit opt-in required)
- **Full dataset size:** 378 questions
- **Sampled:** 50 questions (seed=42)

</details>

<details>
<summary><b>GPQA Diamond</b> — 198 graduate-level questions in physics, chemistry, and biology written by domai…</summary>

198 graduate-level questions in physics, chemistry, and biology written by domain experts. Google-proof questions where non-experts score only 34% with internet.

- **Paper:** Rein et al. 2023
- **Dataset:** `nichenshun/gpqa_diamond (community mirror of Idavidrein/gpqa)`
- **Verification:** Multiple choice (4 options)
- **Full dataset size:** 198 questions
- **Sampled:** 50 questions (seed=42)

</details>

<details>
<summary><b>SciBench</b> — College-level scientific textbook problem solving in physics, chemistry, and the…</summary>

College-level scientific textbook problem solving in physics, chemistry, and thermodynamics requiring multi-step quantitative calculations.

- **Paper:** Wang et al. 2023
- **Dataset:** `xw27/scibench`
- **Verification:** Numerical / Formula exact match
- **Full dataset size:** 692 questions
- **Sampled:** 50 questions (seed=42)

</details>

<details>
<summary><b>AIME 2024/2025</b> — American Invitational Mathematics Examination (AIME) high-school competition mat…</summary>

American Invitational Mathematics Examination (AIME) high-school competition math problems. Premier benchmark for evaluating advanced mathematical reasoning in SOTA AI models.

- **Paper:** MAA AIME Competition Problems
- **Dataset:** `AI-MO/aimo-validation-aime`
- **Verification:** Integer exact match (000-999)
- **Full dataset size:** 90 questions
- **Sampled:** 50 questions (seed=42)

</details>

<details>
<summary><b>MATH-500</b> — 500 challenging competition math problems (Levels 1 to 5) across algebra, geomet…</summary>

500 challenging competition math problems (Levels 1 to 5) across algebra, geometry, number theory, calculus, and probability.

- **Paper:** Hendrycks et al. 2021 / Lightman et al. 2023
- **Dataset:** `HuggingFaceH4/MATH-500`
- **Verification:** Exact match / \boxed{} extraction
- **Full dataset size:** 500 questions
- **Sampled:** 50 questions (seed=42)

</details>

<details>
<summary><b>MMLU-Pro</b> — A harder successor to MMLU with 10 answer choices instead of 4, covering 14 acad…</summary>

A harder successor to MMLU with 10 answer choices instead of 4, covering 14 academic disciplines (biology, business, chemistry, computer science, economics, engineering, health, history, law, math, philosophy, physics, psychology, other).

- **Paper:** Wang et al. 2024
- **Dataset:** `TIGER-Lab/MMLU-Pro`
- **Verification:** Multiple choice (10 options)
- **Full dataset size:** 12,032 questions
- **Sampled:** 50 questions (seed=42)

</details>

<details>
<summary><b>IFEval</b> — Tests whether models follow specific formatting and content instructions (word c…</summary>

Tests whether models follow specific formatting and content instructions (word counts, paragraph structure, keyword inclusion/exclusion, JSON output, language constraints, etc.). Each prompt has one or more verifiable constraints checked by 25 deterministic programmatic verifiers.

- **Paper:** Zhou et al. 2023
- **Dataset:** `google/IFEval`
- **Verification:** 25 programmatic verifiers (strict)
- **Full dataset size:** 541 questions
- **Sampled:** 50 questions (seed=42)

</details>

## 🏅 Leaderboard

| Rank | Model | Overall | 💻 Coding | 🔬 Science | 📐 Math | 📚 Knowledge | 📋 Instruction |
|:----:|-------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| 🥇 | **deepseek/deepseek-v4-flash** 🧠 | **59.8%** | 40.0% | 39.0% | 56.0% | 80.0% | 84.0% |

*🧠 indicates reasoning models that utilize thinking tokens or have explicit thinking effort configured.*

### Per-Benchmark Scores

| Model | BigCodeBench-Hard | HumanEval+ | MBPP+ | GPQA Diamond | SciBench | AIME 2024/2025 | MATH-500 | MMLU-Pro | IFEval |
|-------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| deepseek/deepseek-v4-flash | 20% (10/50) | 90% (45/50) | 10% (5/50) | 58% (29/50) | 20% (10/50) | 24% (12/50) | 88% (44/50) | 80% (40/50) | 84% (42/50) |

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

| Model | Input | Output | Thinking | Total | Out % | Think % | Avg TPS | Avg Time |
|-------|------:|-------:|---------:|------:|------:|--------:|--------:|---------:|
| deepseek/deepseek-v4-flash | 66,677 | 55,762 | 680,781 | 803,220 | 7% | 85% | 94.4 | 16.8s |

*TPS = output tokens/second (cloud APIs only, skipped for local models). Thinking tokens are reasoning/chain-of-thought tokens (e.g. DeepSeek R1).*

## 🔬 Methodology

### Sampling
- **~50 questions** are sampled from each benchmark's full dataset
- Sampling uses a **fixed seed (42)** so the same questions are used across runs and models
- Pin a dataset `revision` in `config.yaml` to make samples reproducible across dataset updates
- Temperature zero reduces variance, but provider-side inference is not guaranteed deterministic

### Scoring
- **All scoring is programmatic** — no LLM-as-judge is used anywhere
- Code benchmarks are skipped unless `--unsafe` is passed in an isolated sandbox
- Multiple-choice benchmarks extract the answer letter and compare to ground truth
- GSM8K extracts the final number (after `####`) and compares numerically
- IFEval uses its 25 strict programmatic verifiers (word count, format, keywords, etc.)

### Category & Overall Scores
- **Category score** = average of its benchmark scores
  - 💻 **Coding** = avg(BigCodeBench-Hard, HumanEval+, MBPP+)
  - 🔬 **Science** = avg(GPQA Diamond, SciBench)
  - 📐 **Math** = avg(AIME 2024/2025, MATH-500)
  - 📚 **Knowledge** = avg(MMLU-Pro)
  - 📋 **Instruction** = avg(IFEval)
- **Overall score** = average of completed category scores (equal weight per category)
- A failed request is excluded and recorded separately; it is never silently scored as incorrect

### Inference Settings
- `temperature`: 0.0
- `max_tokens`: 4096
- `timeout`: 300s per request
- `retries`: up to 3 provider retries

## 🚀 How to Run

### Prerequisites

```bash
pip install -r requirements.txt
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

> **Note:** All datasets are public. `HF_TOKEN` is optional but speeds up
> downloads and avoids rate limits on HuggingFace.

### Commands

```bash
# Run all benchmarks on all configured models
py run_benchmark.py

# Run code benchmarks only in an isolated sandbox; this executes model-generated Python
py run_benchmark.py --unsafe --benchmarks humaneval mbpp bigcodebench

# Run specific benchmarks only
py run_benchmark.py --benchmarks humaneval mbpp gsm8k

# Run specific models only (by id or display name)
py run_benchmark.py --models deepseek/deepseek-chat gemini/gemini-2.5-flash

# List configured models and benchmarks
py run_benchmark.py --list

# Regenerate README + charts from latest results (no API calls)
py run_benchmark.py --generate-only
```

After each run, this README is **automatically regenerated** with updated
rankings, tables, charts, and token stats. Commit the changes to update
your GitHub leaderboard.

## ➕ Adding Models

Edit `config.yaml` and add any [litellm-supported model](https://docs.litellm.ai/docs/providers):

```yaml
models:
  - id: anthropic/claude-sonnet-4-20250514
    name: Claude Sonnet 4
  - id: openai/gpt-4o
    name: GPT-4o
  - id: lm_studio/qwen2.5-coder-7b-instruct
    name: Qwen 2.5 Coder 7B (local)
```

The `id` is a litellm model identifier (`provider/model-name`).
The `name` is the display name shown in the leaderboard.

## 📁 Project Structure

```
├── config.yaml              # Models, benchmarks, categories, settings
├── run_benchmark.py         # CLI entry point
├── requirements.txt         # Python dependencies
├── README.md                # ← this file (auto-generated)
├── lite_bench/
│   ├── config.py            # Config loading & validation
│   ├── providers.py         # litellm wrapper (100+ providers)
│   ├── datasets.py          # HuggingFace dataset sampling
│   ├── benchmarks.py        # 8 benchmark implementations
│   ├── ifeval_verifiers.py  # 25 strict IFEval verifiers
│   ├── charts.py            # matplotlib chart generation
│   └── readme_gen.py        # This README generator
├── results/                 # JSON results per run
│   ├── latest.json          # Most recent run
│   └── results_YYYYMMDD_HHMMSS.json
├── charts/                  # Generated PNG charts
    ├── leaderboard.png
    ├── categories.png
    ├── radar.png
    └── heatmap.png
└── tests/                   # Regression tests
```

---

*Auto-generated by [lite-benchmarks](.) on 2026-07-23 16:27 UTC. Run `py run_benchmark.py` to update.*
