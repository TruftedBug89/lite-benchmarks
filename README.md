# 🏆 Lite Benchmarks

Personal LLM benchmark leaderboard. Lite subsets of professionally-made
benchmarks, each scored with its own built-in verification system.

> **Run `py run_benchmark.py` to generate the full leaderboard, charts, and this README.**

## Benchmarks

| Benchmark | Category | Full Dataset | Sampled | Verification |
|-----------|----------|-------------|---------|-------------|
| HumanEval+ | Coding | 164 | 50 | Code execution (EvalPlus, 80× tests) |
| MBPP+ | Coding | 399 | 50 | Code execution (EvalPlus augmented) |
| GPQA Diamond | Science | 198 | 50 | Multiple choice (grad-level) |
| ARC-Challenge | Science | 1,172 | 50 | Multiple choice (science) |
| GSM8K | Math | 1,319 | 50 | Numerical exact match |
| MMLU-Pro | Knowledge | 12,032 | 50 | Multiple choice (10 options) |
| IFEval | Instruction | 541 | 50 | Programmatic verifiers |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys (only for providers you use)
set DEEPSEEK_API_KEY=...
set GROQ_API_KEY=...
set GEMINI_API_KEY=...
# For gated datasets like GPQA:
set HF_TOKEN=...

# Run all benchmarks on all models
py run_benchmark.py

# Run specific benchmarks or models
py run_benchmark.py --benchmarks humaneval mbpp
py run_benchmark.py --models deepseek/deepseek-chat

# List configured models and benchmarks
py run_benchmark.py --list

# Regenerate README + charts from latest results (no API calls)
py run_benchmark.py --generate-only
```

## Adding Models

Edit `config.yaml` and add any [litellm-supported model](https://docs.litellm.ai/docs/providers):

```yaml
models:
  - id: anthropic/claude-sonnet-4-20250514
    name: Claude Sonnet 4
  - id: ollama/qwen2.5-coder:7b
    name: Qwen 2.5 Coder 7B (local)
```

## Project Structure

```
├── config.yaml              # Models, benchmarks, settings
├── run_benchmark.py         # CLI entry point
├── lite_bench/
│   ├── config.py            # Config loading
│   ├── providers.py         # litellm wrapper (100+ providers)
│   ├── datasets.py          # HuggingFace dataset sampling
│   ├── benchmarks.py        # 7 benchmark implementations
│   ├── ifeval_verifiers.py  # 24 programmatic IFEval verifiers
│   ├── charts.py            # matplotlib chart generation
│   └── readme_gen.py        # Auto README generation
├── results/                 # JSON results per run
└── charts/                  # Generated PNG charts
```

---
*Run `py run_benchmark.py` to generate the leaderboard.*
