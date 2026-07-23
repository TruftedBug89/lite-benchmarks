# 🏆 Lite Benchmarks — Personal LLM Leaderboard

> **Lite subsets of professionally-made benchmarks, each scored with its
> own built-in verification system. No LLM-as-judge. Fully deterministic.**

This repo benchmarks LLMs on ~50 questions sampled from each of 7 established
benchmarks, grouped into 5 categories. Results, rankings, and charts below are
**auto-generated** by `py run_benchmark.py` after every run.

## 📝 Benchmarks

| Benchmark | Category | Full Dataset | Sampled | Verification | Source |
|-----------|----------|:-----------:|:-------:|-------------|--------|
| **HumanEval+** | Coding | 164 | 50 | Code execution (EvalPlus, 80× tests) | `evalplus/humanevalplus` |
| **MBPP+** | Coding | 399 | 50 | Code execution (EvalPlus augmented) | `evalplus/mbppplus` |
| **GPQA Diamond** | Science | 198 | 50 | Multiple choice (4 options) | `Idavidrein/gpqa (gpqa_diamond)` |
| **ARC-Challenge** | Science | 1,172 | 50 | Multiple choice | `allenai/ai2_arc (ARC-Challenge)` |
| **GSM8K** | Math | 1,319 | 50 | Numerical exact match (#### format) | `openai/gsm8k (main)` |
| **MMLU-Pro** | Knowledge | 12,032 | 50 | Multiple choice (10 options) | `TIGER-Lab/MMLU-Pro` |
| **IFEval** | Instruction | 541 | 50 | 24 programmatic verifiers | `google/IFEval` |

### Benchmark Details

<details>
<summary><b>HumanEval+</b> — 164 hand-written Python functions with docstrings…</summary>

164 hand-written Python functions with docstrings. The model must generate a working
implementation. EvalPlus augments the original ~10 unit tests per problem to ~764,
catching edge-case bugs the original HumanEval misses. Scored by executing the generated
code against the full augmented test suite.

- **Paper:** Chen et al. 2021, augmented by Liu et al. 2023 (EvalPlus)
- **Dataset:** `evalplus/humanevalplus`
- **Verification:** Code execution (EvalPlus, 80× tests)
- **Full dataset size:** 164 questions
- **Sampled:** 50 questions (seed=42)

</details>

<details>
<summary><b>MBPP+</b> — 399 crowd-sourced Python programming problems…</summary>

399 crowd-sourced Python programming problems (sanitized subset) designed for
entry-level programmers. Each problem has a natural-language description and
assert-based test cases. EvalPlus expands the original 3 tests per problem with
mutation-based fuzzing for deeper coverage.

- **Paper:** Austin et al. 2021, augmented by Liu et al. 2023 (EvalPlus)
- **Dataset:** `evalplus/mbppplus`
- **Verification:** Code execution (EvalPlus augmented)
- **Full dataset size:** 399 questions
- **Sampled:** 50 questions (seed=42)

</details>

<details>
<summary><b>GPQA Diamond</b> — 198 graduate-level questions in physics, chemistry, and biology…</summary>

198 graduate-level questions in physics, chemistry, and biology written by domain
experts. The Diamond subset contains questions where both domain experts agreed on
the answer but non-experts scored only 34% even with unrestricted internet access —
making them genuinely "Google-proof". This is the hardest standard science benchmark
for LLMs.

- **Paper:** Rein et al. 2023
- **Dataset:** `Idavidrein/gpqa (gpqa_diamond)`
- **Verification:** Multiple choice (4 options)
- **Full dataset size:** 198 questions
- **Sampled:** 50 questions (seed=42)

</details>

<details>
<summary><b>ARC-Challenge</b> — Grade-school science questions from the AI2 Reasoning Challenge…</summary>

Grade-school science questions from the AI2 Reasoning Challenge. The Challenge subset
contains questions that neither a retrieval-based algorithm nor a word-co-occurrence
algorithm could answer correctly — requiring genuine scientific reasoning rather than
pattern matching.

- **Paper:** Clark et al. 2018
- **Dataset:** `allenai/ai2_arc (ARC-Challenge)`
- **Verification:** Multiple choice
- **Full dataset size:** 1,172 questions
- **Sampled:** 50 questions (seed=42)

</details>

<details>
<summary><b>GSM8K</b> — Grade-school math word problems requiring multi-step arithmetic…</summary>

Grade-school math word problems requiring multi-step arithmetic reasoning. Each problem
has a chain-of-thought solution ending with a final numerical answer after '####'.
Scored by extracting the model's final number and comparing it to the ground truth.

- **Paper:** Cobbe et al. 2021
- **Dataset:** `openai/gsm8k (main)`
- **Verification:** Numerical exact match (#### format)
- **Full dataset size:** 1,319 questions
- **Sampled:** 50 questions (seed=42)

</details>

<details>
<summary><b>MMLU-Pro</b> — A harder successor to MMLU with 10 answer choices…</summary>

A harder successor to MMLU with 10 answer choices instead of 4, covering 14 academic
disciplines (biology, business, chemistry, computer science, economics, engineering,
health, history, law, math, philosophy, physics, psychology, other). The extra
distractors significantly reduce random-guess success and require deeper reasoning.

- **Paper:** Wang et al. 2024
- **Dataset:** `TIGER-Lab/MMLU-Pro`
- **Verification:** Multiple choice (10 options)
- **Full dataset size:** 12,032 questions
- **Sampled:** 50 questions (seed=42)

</details>

<details>
<summary><b>IFEval</b> — Tests whether models follow specific formatting and content instructions…</summary>

Tests whether models follow specific formatting and content instructions (word counts,
paragraph structure, keyword inclusion/exclusion, JSON output, language constraints,
etc.). Each prompt has one or more verifiable constraints checked by 24 deterministic
programmatic verifiers — no LLM-as-judge needed. A response passes only if ALL
constraints are satisfied.

- **Paper:** Zhou et al. 2023
- **Dataset:** `google/IFEval`
- **Verification:** 24 programmatic verifiers
- **Full dataset size:** 541 questions
- **Sampled:** 50 questions (seed=42)

</details>

## 🏅 Leaderboard

*No results yet. Run `py run_benchmark.py` to generate the leaderboard.*

## 📊 Charts

*Charts will appear here after the first benchmark run.*

## 🔬 Methodology

### Sampling
- **~50 questions** are sampled from each benchmark's full dataset
- Sampling uses a **fixed seed (42)** so the same questions are used across runs and models
- This makes results **reproducible** and **comparable** across models

### Scoring
- **All scoring is deterministic** — no LLM-as-judge is used anywhere
- Coding benchmarks execute generated code against built-in test suites
- Multiple-choice benchmarks extract the answer letter and compare to ground truth
- GSM8K extracts the final number (after `####`) and compares numerically
- IFEval uses 24 programmatic verifiers (word count, format, keywords, etc.)

### Category & Overall Scores
- **Category score** = average of its benchmark scores
  - 💻 **Coding** = avg(HumanEval+, MBPP+)
  - 🔬 **Science** = avg(GPQA Diamond, ARC-Challenge)
  - 📐 **Math** = avg(GSM8K)
  - 📚 **Knowledge** = avg(MMLU-Pro)
  - 📋 **Instruction** = avg(IFEval)
- **Overall score** = average of all category scores (equal weight per category)

### Inference Settings
- `temperature`: 0.0
- `max_tokens`: 4096
- `timeout`: 120s per request
- `retries`: 3 with exponential backoff

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
│   ├── benchmarks.py        # 7 benchmark implementations
│   ├── ifeval_verifiers.py  # 24 programmatic IFEval verifiers
│   ├── charts.py            # matplotlib chart generation
│   └── readme_gen.py        # This README generator
├── results/                 # JSON results per run
│   ├── latest.json          # Most recent run
│   └── results_YYYYMMDD_HHMMSS.json
└── charts/                  # Generated PNG charts
    ├── leaderboard.png
    ├── categories.png
    ├── radar.png
    └── heatmap.png
```

---

*Run `py run_benchmark.py` to generate the leaderboard, charts, and update this README.*
