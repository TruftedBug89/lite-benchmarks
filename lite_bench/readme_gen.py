"""Auto-generate a detailed README.md from benchmark results.

Every section with data (leaderboard, tables, charts, token stats) is
rebuilt from the latest results JSON each time run_benchmark.py finishes,
so the README on GitHub always reflects the most recent run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

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
    "bigcodebench_hard": {
        "display": "BigCodeBench-Hard",
        "category": "Coding",
        "total": "148",
        "verification": "Python unittest execution (explicit opt-in required)",
        "source": "bigcode/bigcodebench-hard (v0.1.4)",
        "paper": "Zhuo et al. 2024",
        "description": (
            "The hardest 148 practical Python programming tasks from BigCodeBench requiring "
            "deep integration of complex real-world libraries (pandas, numpy, scipy, etc.)."
        ),
    },
    "humanevalplus": {
        "display": "HumanEval+",
        "category": "Coding",
        "total": "164",
        "verification": "Python test execution (explicit opt-in required)",
        "source": "evalplus/humanevalplus",
        "paper": "Chen et al. 2021, augmented by Liu et al. 2023 (EvalPlus)",
        "description": (
            "164 hand-written Python functions with docstrings and rigorously expanded "
            "test cases to catch edge-case bugs and hallucinated solutions."
        ),
    },
    "mbppplus": {
        "display": "MBPP+",
        "category": "Coding",
        "total": "378",
        "verification": "Python test execution (explicit opt-in required)",
        "source": "evalplus/mbppplus",
        "paper": "Austin et al. 2021, augmented by Liu et al. 2023 (EvalPlus)",
        "description": (
            "378 crowd-sourced Python programming problems with heavily augmented "
            "test suites from EvalPlus for deep coverage."
        ),
    },
    "humaneval": {
        "display": "HumanEval",
        "category": "Coding",
        "total": "164",
        "verification": "Python test execution (explicit opt-in required)",
        "source": "evalplus/humanevalplus",
        "paper": "Chen et al. 2021",
        "description": (
            "164 hand-written Python functions with docstrings. The model must "
            "generate a working implementation."
        ),
    },
    "mbpp": {
        "display": "MBPP",
        "category": "Coding",
        "total": "378",
        "verification": "Python test execution (explicit opt-in required)",
        "source": "evalplus/mbppplus",
        "paper": "Austin et al. 2021",
        "description": (
            "378 crowd-sourced Python programming problems designed for entry-level programmers."
        ),
    },
    "bigcodebench": {
        "display": "BigCodeBench",
        "category": "Coding",
        "total": "1,140",
        "verification": "Python unittest execution (explicit opt-in required)",
        "source": "bigcode/bigcodebench (v0.1.4)",
        "paper": "Zhuo et al. 2024",
        "description": (
            "Practical Python programming tasks requiring use of real-world libraries."
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
            "by domain experts. Google-proof questions where non-experts score only 34% with internet."
        ),
    },
    "scibench": {
        "display": "SciBench",
        "category": "Science",
        "total": "692",
        "verification": "Numerical / Formula exact match",
        "source": "xw27/scibench",
        "paper": "Wang et al. 2023",
        "description": (
            "College-level scientific textbook problem solving in physics, chemistry, "
            "and thermodynamics requiring multi-step quantitative calculations."
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
            "Grade-school science questions from the AI2 Reasoning Challenge."
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
            "Grade-school math word problems requiring multi-step arithmetic reasoning."
        ),
    },
    "aime": {
        "display": "AIME 2024/2025",
        "category": "Math",
        "total": "90",
        "verification": "Integer exact match (000-999)",
        "source": "AI-MO/aimo-validation-aime",
        "paper": "MAA AIME Competition Problems",
        "description": (
            "American Invitational Mathematics Examination (AIME) high-school competition math problems. "
            "Premier benchmark for evaluating advanced mathematical reasoning in SOTA AI models."
        ),
    },
    "math_500": {
        "display": "MATH-500",
        "category": "Math",
        "total": "500",
        "verification": "Exact match / \\boxed{} extraction",
        "source": "HuggingFaceH4/MATH-500",
        "paper": "Hendrycks et al. 2021 / Lightman et al. 2023",
        "description": (
            "500 challenging competition math problems (Levels 1 to 5) across algebra, geometry, "
            "number theory, calculus, and probability."
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
            "psychology, other)."
        ),
    },
    "ifeval": {
        "display": "IFEval",
        "category": "Instruction",
        "total": "541",
        "verification": "25 programmatic verifiers (strict)",
        "source": "google/IFEval",
        "paper": "Zhou et al. 2023",
        "description": (
            "Tests whether models follow specific formatting and content instructions "
            "(word counts, paragraph structure, keyword inclusion/exclusion, JSON output, "
            "language constraints, etc.). Each prompt has one or more verifiable "
            "constraints checked by 25 deterministic programmatic verifiers."
        ),
    },
}


def generate(results: dict, config: Config, chart_paths: list[str]) -> str:
    bench_names = list(config.enabled_benchmarks().keys())
    cat_names = list(config.categories.keys())
    model_names = list(results.keys())

    # ── Compute scores ──────────────────────────────────────────────
    bench_scores: dict[str, dict[str, float]] = {}
    cat_scores: dict[str, dict[str, float | None]] = {}
    overall: dict[str, float | None] = {}
    total_in_tokens: dict[str, int] = {}
    total_out_tokens: dict[str, int] = {}
    total_think_tokens: dict[str, int] = {}
    total_all_tokens: dict[str, int] = {}
    avg_tps: dict[str, float | None] = {}
    avg_time: dict[str, float | None] = {}

    for mname in model_names:
        mdata = results[mname]
        bench_scores[mname] = {
            b: mdata[b]["score"]
            for b in bench_names
            if isinstance(mdata.get(b), dict) and "score" in mdata[b]
        }
        cat_scores[mname] = {}
        for cat in cat_names:
            s = config.category_score(bench_scores[mname], cat)
            cat_scores[mname][cat] = s
        overall[mname] = config.overall_score(bench_scores[mname])
        completed = [mdata[b] for b in bench_names if isinstance(mdata.get(b), dict)]
        total_in_tokens[mname] = sum(result.get("input_tokens", 0) for result in completed)
        total_out_tokens[mname] = sum(result.get("output_tokens", 0) for result in completed)
        total_think_tokens[mname] = sum(result.get("thinking_tokens", 0) for result in completed)
        total_all_tokens[mname] = sum(result.get("total_tokens", 0) for result in completed)
        # Timing (only for cloud models)
        tps_vals = [
            result.get("avg_tokens_per_second")
            for result in completed
            if result.get("avg_tokens_per_second") is not None
        ]
        avg_tps[mname] = sum(tps_vals) / len(tps_vals) if tps_vals else None
        time_vals = [
            result.get("avg_time_ms")
            for result in completed
            if result.get("avg_time_ms") is not None
        ]
        avg_time[mname] = sum(time_vals) / len(time_vals) if time_vals else None

    ranked = sorted(
        model_names,
        key=lambda m: overall[m] if overall[m] is not None else float("-inf"),
        reverse=True,
    )

    L: list[str] = []
    _a = L.append

    # ── Header ──────────────────────────────────────────────────────
    _a("# 🏆 Lite Benchmarks — Personal LLM Leaderboard")
    _a("")
    _a("> **Small, repeatable samples of established benchmarks with programmatic scoring.")
    _a("> No LLM-as-judge. Sampling and scoring are deterministic; model outputs may vary.**")
    _a("")
    _a("This repo benchmarks LLMs on ~50 questions sampled from each of 8 established")
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
        _a("<details>")
        _a(f"<summary><b>{info['display']}</b> — {info['description'][:80]}…</summary>")
        _a("")
        _a(info["description"])
        _a("")
        _a(f"- **Paper:** {info['paper']}")
        _a(f"- **Dataset:** `{info['source']}`")
        _a(f"- **Verification:** {info['verification']}")
        _a(f"- **Full dataset size:** {info['total']} questions")
        _a(
            f"- **Sampled:** {config.benchmarks[bname].num_samples} questions (seed={config.settings.seed})"
        )
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
            score = overall[mname]
            overall_cell = f"**{score * 100:.1f}%**" if score is not None else "N/A"
            
            # Identify thinking models
            is_thinking = total_think_tokens.get(mname, 0) > 0 or results.get(mname, {}).get("thinking_effort") is not None
            display_name = f"**{mname}** 🧠" if is_thinking else f"**{mname}**"
            
            row = f"| {medal} | {display_name} | {overall_cell} |"
            for cat in cat_names:
                category_score = cat_scores[mname].get(cat)
                row += f" {category_score * 100:.1f}% |" if category_score is not None else " N/A |"
            _a(row)
        _a("")
        _a("*🧠 indicates reasoning models that utilize thinking tokens or have explicit thinking effort configured.*")
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
                s = bench_scores[mname].get(bname)
                correct = results[mname].get(bname, {}).get("correct", 0)
                total = results[mname].get(bname, {}).get("total", 0)
                row += f" {s * 100:.0f}% ({correct}/{total}) |" if s is not None else " N/A |"
            _a(row)
        _a("")

    # ── Charts ──────────────────────────────────────────────────────
    _a("## 📊 Charts")
    _a("")
    chart_titles = {
        "leaderboard.png": (
            "Overall Scores",
            "Horizontal bar chart ranked by overall score (average of all category scores).",
        ),
        "categories.png": (
            "Category Breakdown",
            "Grouped bar chart comparing each model across the 5 categories.",
        ),
        "radar.png": (
            "Category Radar",
            "Spider chart showing each model's profile across categories. Larger area = stronger overall.",
        ),
        "heatmap.png": (
            "Benchmark Heatmap",
            "Per-benchmark scores for every model. Green = high, red = low.",
        ),
        "tokens.png": (
            "Token Breakdown",
            "Stacked bar chart of input, thinking, and output tokens per model.",
        ),
        "thinking_scatter.png": (
            "Thinking Effort vs Performance",
            "Scatter plot showing if models that use more thinking tokens achieve higher overall scores.",
        ),
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

    # ── Token usage & performance ───────────────────────────────────
    _a("## 🪙 Token Usage & Performance")
    _a("")
    if ranked:
        _a("| Model | Input | Output | Thinking | Total | Out % | Think % | Avg TPS | Avg Time |")
        _a("|-------|------:|-------:|---------:|------:|------:|--------:|--------:|---------:|")
        for mname in ranked:
            tin = total_in_tokens.get(mname, 0)
            tout = total_out_tokens.get(mname, 0)
            tthink = total_think_tokens.get(mname, 0)
            ttot = total_all_tokens.get(mname, 0)
            out_pct = f"{tout / ttot:.0%}" if ttot else "—"
            think_pct = f"{tthink / ttot:.0%}" if ttot and tthink else "—"
            tps = avg_tps.get(mname)
            tps_str = f"{tps:.1f}" if tps is not None else "—"
            time_ms = avg_time.get(mname)
            time_str = f"{time_ms / 1000:.1f}s" if time_ms is not None else "—"
            _a(
                f"| {mname} | {tin:,} | {tout:,} | {tthink:,} | {ttot:,} "
                f"| {out_pct} | {think_pct} | {tps_str} | {time_str} |"
            )
        _a("")
        _a(
            "*TPS = output tokens/second (cloud APIs only, skipped for local models). "
            "Thinking tokens are reasoning/chain-of-thought tokens (e.g. DeepSeek R1).*"
        )
        _a("")

    # ── Methodology ─────────────────────────────────────────────────
    _a("## 🔬 Methodology")
    _a("")
    _a("### Sampling")
    sample_sizes = {config.benchmarks[b].num_samples for b in bench_names}
    sample_str = str(sample_sizes.pop()) if len(sample_sizes) == 1 else "varies"
    _a(f"- **~{sample_str} questions** are sampled from each benchmark's full dataset")
    _a(
        f"- Sampling uses a **fixed seed ({config.settings.seed})** so the same questions are used across runs and models"
    )
    _a(
        "- Pin a dataset `revision` in `config.yaml` to make samples reproducible across dataset updates"
    )
    _a(
        "- Temperature zero reduces variance, but provider-side inference is not guaranteed deterministic"
    )
    _a("")
    _a("### Scoring")
    _a("- **All scoring is programmatic** — no LLM-as-judge is used anywhere")
    _a(
        "- Code benchmarks are skipped unless `--unsafe` is passed in an isolated sandbox"
    )
    _a("- Multiple-choice benchmarks extract the answer letter and compare to ground truth")
    _a("- GSM8K extracts the final number (after `####`) and compares numerically")
    _a("- IFEval uses its 25 strict programmatic verifiers (word count, format, keywords, etc.)")
    _a("")
    _a("### Category & Overall Scores")
    _a("- **Category score** = average of its benchmark scores")
    for cat in cat_names:
        benches = config.categories.get(cat, [])
        bench_labels = [BENCHMARK_INFO.get(b, {}).get("display", b) for b in benches]
        icon = CATEGORY_ICONS.get(cat, "")
        _a(f"  - {icon} **{CATEGORY_LABELS.get(cat, cat)}** = avg({', '.join(bench_labels)})")
    _a("- **Overall score** = average of completed category scores (equal weight per category)")
    _a(
        "- A failed request is excluded and recorded separately; it is never silently scored as incorrect"
    )
    _a("")
    _a("### Inference Settings")
    _a(f"- `temperature`: {config.settings.temperature}")
    _a(f"- `max_tokens`: {config.settings.max_tokens}")
    _a(f"- `timeout`: {config.settings.request_timeout}s per request")
    _a(f"- `retries`: up to {config.settings.max_retries} provider retries")
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
    _a(
        "| HuggingFace | `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |"
    )
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
    _a("# Run code benchmarks only in an isolated sandbox; this executes model-generated Python")
    _a("py run_benchmark.py --unsafe --benchmarks humaneval mbpp bigcodebench")
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
    _a(
        "Edit `config.yaml` and add any [litellm-supported model](https://docs.litellm.ai/docs/providers):"
    )
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
    _a("│   ├── benchmarks.py        # 8 benchmark implementations")
    _a("│   ├── ifeval_verifiers.py  # 25 strict IFEval verifiers")
    _a("│   ├── charts.py            # matplotlib chart generation")
    _a("│   └── readme_gen.py        # This README generator")
    _a("├── results/                 # JSON results per run")
    _a("│   ├── latest.json          # Most recent run")
    _a("│   └── results_YYYYMMDD_HHMMSS.json")
    _a("├── charts/                  # Generated PNG charts")
    _a("    ├── leaderboard.png")
    _a("    ├── categories.png")
    _a("    ├── radar.png")
    _a("    └── heatmap.png")
    _a("└── tests/                   # Regression tests")
    _a("```")
    _a("")

    # ── Footer ──────────────────────────────────────────────────────
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    _a("---")
    _a("")
    _a(f"*Auto-generated by [lite-benchmarks](.) on {now}. Run `py run_benchmark.py` to update.*")
    _a("")

    return "\n".join(L)


def write_readme(
    results: dict, config: Config, chart_paths: list[str], path: str = "README.md"
) -> None:
    content = generate(results, config, chart_paths)
    Path(path).write_text(content, encoding="utf-8")
