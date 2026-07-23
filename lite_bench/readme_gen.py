"""Auto-generate a detailed README.md from benchmark results.

Every section with data (leaderboard, tables, charts, token stats) is
rebuilt from the latest results JSON each time run_benchmark.py finishes,
so the README on GitHub always reflects the most recent run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .benchmarks import BENCHMARK_CLASSES
from .config import Config

CATEGORY_LABELS = {
    "coding": "Coding",
    "science": "Science",
    "math": "Math",
    "knowledge": "Knowledge",
    "instruction": "Instruction",
}

CATEGORY_ICONS = {
    "coding": "💻",
    "science": "🔬",
    "math": "📐",
    "knowledge": "📚",
    "instruction": "📋",
}

BENCHMARK_INFO: dict[str, dict] = {
    "humaneval": {
        "display": "HumanEval+",
        "category": "Coding",
        "total": "164",
        "verification": "Code execution (EvalPlus, 80× tests)",
        "source": "evalplus/humanevalplus",
        "paper": "Chen et al. 2021, augmented by Liu et al. 2023 (EvalPlus)",
        "description": (
            "164 hand-written Python functions with docstrings. The model must "
            "generate a working implementation. EvalPlus augments the original "
            "~10 unit tests per problem to ~764, catching edge-case bugs the "
            "original HumanEval misses. Scored by executing the generated code "
            "against the full augmented test suite."
        ),
    },
    "mbpp": {
        "display": "MBPP+",
        "category": "Coding",
        "total": "399",
        "verification": "Code execution (EvalPlus augmented)",
        "source": "evalplus/mbppplus",
        "paper": "Austin et al. 2021, augmented by Liu et al. 2023 (EvalPlus)",
        "description": (
            "399 crowd-sourced Python programming problems (sanitized subset) "
            "designed for entry-level programmers. Each problem has a natural-language "
            "description and assert-based test cases. EvalPlus expands the original "
            "3 tests per problem with mutation-based fuzzing for deeper coverage."
        ),
    },
    "bigcodebench": {
        "display": "BigCodeBench",
        "category": "Coding",
        "total": "1,140",
        "verification": "Code execution (unittest)",
        "source": "bigcode/bigcodebench (v0.1.4)",
        "paper": "Zhuo et al. 2024",
        "description": (
            "Practical Python programming tasks requiring use of real-world libraries "
            "(collections, itertools, json, re, os, and more). Unlike HumanEval/MBPP "
            "which test algorithmic function completion, BigCodeBench tests whether "
            "models can write code that integrates multiple library calls to solve "
            "realistic tasks. Verified with unittest-based test suites."
        ),
    },
    "gpqa": {
        "display": "GPQA Diamond",
        "category": "Science",
        "total": "198",
        "verification": "Multiple choice (4 options)",
        "source": "nichenshun/gpqa_diamond (community mirror of Idavidrein/gpqa)",
        "paper": "Rein et al. 2023",
        "description": (
            "198 graduate-level questions in physics, chemistry, and biology written "
            "by domain experts. The Diamond subset contains questions where both domain "
            "experts agreed on the answer but non-experts scored only 34% even with "
            "unrestricted internet access — making them genuinely \"Google-proof\". "
            "This is the hardest standard science benchmark for LLMs."
        ),
    },
    "arc": {
        "display": "ARC-Challenge",
        "category": "Science",
        "total": "1,172",
        "verification": "Multiple choice",
        "source": "allenai/ai2_arc (ARC-Challenge)",
        "paper": "Clark et al. 2018",
        "description": (
            "Grade-school science questions from the AI2 Reasoning Challenge. The "
            "Challenge subset contains questions that neither a retrieval-based algorithm "
            "nor a word-co-occurrence algorithm could answer correctly — requiring "
            "genuine scientific reasoning rather than pattern matching."
        ),
    },
    "gsm8k": {
        "display": "GSM8K",
        "category": "Math",
        "total": "1,319",
        "verification": "Numerical exact match (#### format)",
        "source": "openai/gsm8k (main)",
        "paper": "Cobbe et al. 2021",
        "description": (
            "Grade-school math word problems requiring multi-step arithmetic reasoning. "
            "Each problem has a chain-of-thought solution ending with a final numerical "
            "answer after '####'. Scored by extracting the model's final number and "
            "comparing it to the ground truth."
        ),
    },
    "mmlu_pro": {
        "display": "MMLU-Pro",
        "category": "Knowledge",
        "total": "12,032",
        "verification": "Multiple choice (10 options)",
        "source": "TIGER-Lab/MMLU-Pro",
        "paper": "Wang et al. 2024",
        "description": (
            "A harder successor to MMLU with 10 answer choices instead of 4, covering "
            "14 academic disciplines (biology, business, chemistry, computer science, "
            "economics, engineering, health, history, law, math, philosophy, physics, "
            "psychology, other). The extra distractors significantly reduce random-guess "
            "success and require deeper reasoning."
        ),
    },
    "ifeval": {
        "display": "IFEval",
        "category": "Instruction",
        "total": "541",
        "verification": "24 programmatic verifiers",
        "source": "google/IFEval",
        "paper": "Zhou et al. 2023",
        "description": (
            "Tests whether models follow specific formatting and content instructions "
            "(word counts, paragraph structure, keyword inclusion/exclusion, JSON output, "
            "language constraints, etc.). Each prompt has one or more verifiable "
            "constraints checked by 24 deterministic programmatic verifiers — no "
            "LLM-as-judge needed. A response passes only if ALL constraints are satisfied."
        ),
    },
}


def generate(results: dict, config: Config, chart_paths: list[str]) -> str:
    bench_names = list(config.enabled_benchmarks().keys())
    cat_names = list(config.categories.keys())
    model_names = list(results.keys())

    # ── Compute scores ──────────────────────────────────────────────
    bench_scores: dict[str, dict[str, float]] = {}
    cat_scores: dict[str, dict[str, float]] = {}
    overall: dict[str, float] = {}
    total_in_tokens: dict[str, int] = {}
    total_out_tokens: dict[str, int] = {}

    for mname in model_names:
        mdata = results[mname]
        bench_scores[mname] = {
            b: mdata.get(b, {}).get("score", 0.0) for b in bench_names
        }
        cat_scores[mname] = {}
        for cat in cat_names:
            s = config.category_score(bench_scores[mname], cat)
            cat_scores[mname][cat] = s if s is not None else 0.0
        o = config.overall_score(bench_scores[mname])
        overall[mname] = o if o is not None else 0.0
        total_in_tokens[mname] = sum(
            mdata.get(b, {}).get("input_tokens", 0) for b in bench_names
        )
        total_out_tokens[mname] = sum(
            mdata.get(b, {}).get("output_tokens", 0) for b in bench_names
        )

    ranked = sorted(model_names, key=lambda m: overall[m], reverse=True)

    L: list[str] = []
    _a = L.append

    # ── Header ──────────────────────────────────────────────────────
    _a("# 🏆 Lite Benchmarks — Personal LLM Leaderboard")
    _a("")
    _a("> **Lite subsets of professionally-made benchmarks, each scored with its")
    _a("> own built-in verification system. No LLM-as-judge. Fully deterministic.**")
    _a("")
    _a("This repo benchmarks LLMs on ~50 questions sampled from each of 7 established")
    _a("benchmarks, grouped into 5 categories. Results, rankings, and charts below are")
    _a("**auto-generated** by `py run_benchmark.py` after every run.")
    _a("")

    # ── Benchmarks ──────────────────────────────────────────────────
    _a("## 📝 Benchmarks")
    _a("")
    _a("| Benchmark | Category | Full Dataset | Sampled | Verification | Source |")
    _a("|-----------|----------|:-----------:|:-------:|-------------|--------|")
    for bname in bench_names:
        info = BENCHMARK_INFO.get(bname)
        if info:
            sampled = config.benchmarks[bname].num_samples
            _a(
                f"| **{info['display']}** | {info['category']} "
                f"| {info['total']} | {sampled} "
                f"| {info['verification']} | `{info['source']}` |"
            )
    _a("")

    # Detailed descriptions
    _a("### Benchmark Details")
    _a("")
    for bname in bench_names:
        info = BENCHMARK_INFO.get(bname)
        if not info:
            continue
        _a(f"<details>")
        _a(f"<summary><b>{info['display']}</b> — {info['description'][:80]}…</summary>")
        _a("")
        _a(info["description"])
        _a("")
        _a(f"- **Paper:** {info['paper']}")
        _a(f"- **Dataset:** `{info['source']}`")
        _a(f"- **Verification:** {info['verification']}")
        _a(f"- **Full dataset size:** {info['total']} questions")
        _a(f"- **Sampled:** {config.benchmarks[bname].num_samples} questions (seed={config.settings.seed})")
        _a("")
        _a("</details>")
        _a("")

    # ── Leaderboard ─────────────────────────────────────────────────
    _a("## 🏅 Leaderboard")
    _a("")
    if not ranked:
        _a("*No results yet. Run `py run_benchmark.py` to generate the leaderboard.*")
        _a("")
    else:
        header = "| Rank | Model | Overall |"
        sep = "|:----:|-------|:-------:|"
        for cat in cat_names:
            icon = CATEGORY_ICONS.get(cat, "")
            label = CATEGORY_LABELS.get(cat, cat)
            header += f" {icon} {label} |"
            sep += ":-------:|"
        _a(header)
        _a(sep)
        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        for i, mname in enumerate(ranked, 1):
            medal = medals.get(i, str(i))
            row = f"| {medal} | **{mname}** | **{overall[mname]*100:.1f}%** |"
            for cat in cat_names:
                row += f" {cat_scores[mname].get(cat, 0)*100:.1f}% |"
            _a(row)
        _a("")

    # ── Per-benchmark detail ────────────────────────────────────────
    _a("### Per-Benchmark Scores")
    _a("")
    if ranked:
        header = "| Model |"
        sep = "|-------|"
        for bname in bench_names:
            disp = BENCHMARK_INFO.get(bname, {}).get("display", bname)
            header += f" {disp} |"
            sep += ":-------:|"
        _a(header)
        _a(sep)
        for mname in ranked:
            row = f"| {mname} |"
            for bname in bench_names:
                s = bench_scores[mname].get(bname, 0)
                correct = results[mname].get(bname, {}).get("correct", 0)
                total = results[mname].get(bname, {}).get("total", 0)
                row += f" {s*100:.0f}% ({correct}/{total}) |"
            _a(row)
        _a("")

    # ── Charts ──────────────────────────────────────────────────────
    _a("## 📊 Charts")
    _a("")
    chart_titles = {
        "leaderboard.png": ("Overall Scores", "Horizontal bar chart ranked by overall score (average of all category scores)."),
        "categories.png": ("Category Breakdown", "Grouped bar chart comparing each model across the 5 categories."),
        "radar.png": ("Category Radar", "Spider chart showing each model's profile across categories. Larger area = stronger overall."),
        "heatmap.png": ("Benchmark Heatmap", "Per-benchmark scores for every model. Green = high, red = low."),
    }
    for cp in chart_paths:
        fname = cp.split("/")[-1]
        title, desc = chart_titles.get(fname, (fname, ""))
        _a(f"### {title}")
        _a("")
        if desc:
            _a(f"*{desc}*")
            _a("")
        _a(f"![{title}]({cp})")
        _a("")

    # ── Token usage ─────────────────────────────────────────────────
    _a("## 🪙 Token Usage")
    _a("")
    if ranked:
        _a("| Model | Input Tokens | Output Tokens | Total |")
        _a("|-------|:-----------:|:------------:|:-----:|")
        for mname in ranked:
            tin = total_in_tokens.get(mname, 0)
            tout = total_out_tokens.get(mname, 0)
            _a(f"| {mname} | {tin:,} | {tout:,} | {tin + tout:,} |")
        _a("")
        _a("*Token counts are approximate and depend on the provider's tokenizer.*")
        _a("")

    # ── Methodology ─────────────────────────────────────────────────
    _a("## 🔬 Methodology")
    _a("")
    _a("### Sampling")
    sample_sizes = {config.benchmarks[b].num_samples for b in bench_names}
    sample_str = str(sample_sizes.pop()) if len(sample_sizes) == 1 else "varies"
    _a(f"- **~{sample_str} questions** are sampled from each benchmark's full dataset")
    _a(f"- Sampling uses a **fixed seed ({config.settings.seed})** so the same questions are used across runs and models")
    _a("- This makes results **reproducible** and **comparable** across models")
    _a("")
    _a("### Scoring")
    _a("- **All scoring is deterministic** — no LLM-as-judge is used anywhere")
    _a("- Coding benchmarks execute generated code against built-in test suites")
    _a("- Multiple-choice benchmarks extract the answer letter and compare to ground truth")
    _a("- GSM8K extracts the final number (after `####`) and compares numerically")
    _a("- IFEval uses 24 programmatic verifiers (word count, format, keywords, etc.)")
    _a("")
    _a("### Category & Overall Scores")
    _a("- **Category score** = average of its benchmark scores")
    for cat in cat_names:
        benches = config.categories.get(cat, [])
        bench_labels = [BENCHMARK_INFO.get(b, {}).get("display", b) for b in benches]
        icon = CATEGORY_ICONS.get(cat, "")
        _a(f"  - {icon} **{CATEGORY_LABELS.get(cat, cat)}** = avg({', '.join(bench_labels)})")
    _a("- **Overall score** = average of all category scores (equal weight per category)")
    _a("")
    _a("### Inference Settings")
    _a(f"- `temperature`: {config.settings.temperature}")
    _a(f"- `max_tokens`: {config.settings.max_tokens}")
    _a(f"- `timeout`: {config.settings.request_timeout}s per request")
    _a(f"- `retries`: {config.settings.max_retries} with exponential backoff")
    _a("")

    # ── How to run ──────────────────────────────────────────────────
    _a("## 🚀 How to Run")
    _a("")
    _a("### Prerequisites")
    _a("")
    _a("```bash")
    _a("pip install -r requirements.txt")
    _a("```")
    _a("")
    _a("### API Keys")
    _a("")
    _a("Set environment variables for the providers you want to test.")
    _a("[litellm](https://docs.litellm.ai/docs/providers) picks them up automatically.")
    _a("")
    _a("| Provider | Environment Variable | Get a key |")
    _a("|----------|---------------------|-----------|")
    _a("| DeepSeek | `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com) |")
    _a("| Groq | `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |")
    _a("| Google Gemini | `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) |")
    _a("| LM Studio (local) | *(none needed)* | [lmstudio.ai](https://lmstudio.ai) |")
    _a("| HuggingFace | `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |")
    _a("")
    _a("> **Note:** All datasets are public. `HF_TOKEN` is optional but speeds up")
    _a("> downloads and avoids rate limits on HuggingFace.")
    _a("")
    _a("### Commands")
    _a("")
    _a("```bash")
    _a("# Run all benchmarks on all configured models")
    _a("py run_benchmark.py")
    _a("")
    _a("# Run specific benchmarks only")
    _a("py run_benchmark.py --benchmarks humaneval mbpp gsm8k")
    _a("")
    _a("# Run specific models only (by id or display name)")
    _a("py run_benchmark.py --models deepseek/deepseek-chat gemini/gemini-2.5-flash")
    _a("")
    _a("# List configured models and benchmarks")
    _a("py run_benchmark.py --list")
    _a("")
    _a("# Regenerate README + charts from latest results (no API calls)")
    _a("py run_benchmark.py --generate-only")
    _a("```")
    _a("")
    _a("After each run, this README is **automatically regenerated** with updated")
    _a("rankings, tables, charts, and token stats. Commit the changes to update")
    _a("your GitHub leaderboard.")
    _a("")

    # ── Adding models ───────────────────────────────────────────────
    _a("## ➕ Adding Models")
    _a("")
    _a("Edit `config.yaml` and add any [litellm-supported model](https://docs.litellm.ai/docs/providers):")
    _a("")
    _a("```yaml")
    _a("models:")
    _a("  - id: anthropic/claude-sonnet-4-20250514")
    _a("    name: Claude Sonnet 4")
    _a("  - id: openai/gpt-4o")
    _a("    name: GPT-4o")
    _a("  - id: lm_studio/qwen2.5-coder-7b-instruct")
    _a("    name: Qwen 2.5 Coder 7B (local)")
    _a("```")
    _a("")
    _a("The `id` is a litellm model identifier (`provider/model-name`).")
    _a("The `name` is the display name shown in the leaderboard.")
    _a("")

    # ── Project structure ───────────────────────────────────────────
    _a("## 📁 Project Structure")
    _a("")
    _a("```")
    _a("├── config.yaml              # Models, benchmarks, categories, settings")
    _a("├── run_benchmark.py         # CLI entry point")
    _a("├── requirements.txt         # Python dependencies")
    _a("├── README.md                # ← this file (auto-generated)")
    _a("├── lite_bench/")
    _a("│   ├── config.py            # Config loading & validation")
    _a("│   ├── providers.py         # litellm wrapper (100+ providers)")
    _a("│   ├── datasets.py          # HuggingFace dataset sampling")
    _a("│   ├── benchmarks.py        # 7 benchmark implementations")
    _a("│   ├── ifeval_verifiers.py  # 24 programmatic IFEval verifiers")
    _a("│   ├── charts.py            # matplotlib chart generation")
    _a("│   └── readme_gen.py        # This README generator")
    _a("├── results/                 # JSON results per run")
    _a("│   ├── latest.json          # Most recent run")
    _a("│   └── results_YYYYMMDD_HHMMSS.json")
    _a("└── charts/                  # Generated PNG charts")
    _a("    ├── leaderboard.png")
    _a("    ├── categories.png")
    _a("    ├── radar.png")
    _a("    └── heatmap.png")
    _a("```")
    _a("")

    # ── Footer ──────────────────────────────────────────────────────
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    _a("---")
    _a("")
    _a(f"*Auto-generated by [lite-benchmarks](.) on {now}. "
       f"Run `py run_benchmark.py` to update.*")
    _a("")

    return "\n".join(L)


def write_readme(results: dict, config: Config, chart_paths: list[str],
                 path: str = "README.md") -> None:
    content = generate(results, config, chart_paths)
    Path(path).write_text(content, encoding="utf-8")
