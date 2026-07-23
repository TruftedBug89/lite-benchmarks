"""Auto-generate README.md from benchmark results."""

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

BENCHMARK_INFO = {
    "humaneval": ("HumanEval+", "Coding", "164", "Code execution (EvalPlus, 80× tests)"),
    "mbpp": ("MBPP+", "Coding", "399", "Code execution (EvalPlus augmented)"),
    "gpqa": ("GPQA Diamond", "Science", "198", "Multiple choice (grad-level)"),
    "arc": ("ARC-Challenge", "Science", "1,172", "Multiple choice (science)"),
    "gsm8k": ("GSM8K", "Math", "1,319", "Numerical exact match"),
    "mmlu_pro": ("MMLU-Pro", "Knowledge", "12,032", "Multiple choice (10 options)"),
    "ifeval": ("IFEval", "Instruction", "541", "Programmatic verifiers"),
}


def generate(results: dict, config: Config, chart_paths: list[str]) -> str:
    bench_names = list(config.enabled_benchmarks().keys())
    cat_names = list(config.categories.keys())
    model_names = list(results.keys())

    # Compute scores
    bench_scores: dict[str, dict[str, float]] = {}
    cat_scores: dict[str, dict[str, float]] = {}
    overall: dict[str, float] = {}
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

    ranked = sorted(model_names, key=lambda m: overall[m], reverse=True)

    lines: list[str] = []
    _a = lines.append

    _a("# 🏆 Lite Benchmarks")
    _a("")
    _a("Personal LLM benchmark leaderboard. Lite subsets of professionally-made")
    _a("benchmarks, each scored with its own built-in verification system.")
    _a("")

    # Benchmark info table
    _a("## Benchmarks")
    _a("")
    _a("| Benchmark | Category | Full Dataset | Sampled | Verification |")
    _a("|-----------|----------|-------------|---------|-------------|")
    for bname in bench_names:
        info = BENCHMARK_INFO.get(bname)
        if info:
            disp, cat, total, verif = info
            sampled = config.benchmarks[bname].num_samples
            _a(f"| {disp} | {cat} | {total} | {sampled} | {verif} |")
    _a("")

    # Leaderboard
    _a("## 🏅 Leaderboard")
    _a("")
    header = "| Rank | Model | Overall |"
    sep = "|------|-------|---------|"
    for cat in cat_names:
        header += f" {CATEGORY_LABELS.get(cat, cat)} |"
        sep += "--------|"
    _a(header)
    _a(sep)
    for i, mname in enumerate(ranked, 1):
        row = f"| {i} | {mname} | **{overall[mname]*100:.1f}%** |"
        for cat in cat_names:
            row += f" {cat_scores[mname].get(cat, 0)*100:.1f}% |"
        _a(row)
    _a("")

    # Per-benchmark detail
    _a("### Per-Benchmark Scores")
    _a("")
    header = "| Model |"
    sep = "|-------|"
    for bname in bench_names:
        disp = BENCHMARK_CLASSES[bname].display_name if bname in BENCHMARK_CLASSES else bname
        header += f" {disp} |"
        sep += "--------|"
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

    # Charts
    _a("## 📊 Charts")
    _a("")
    chart_titles = {
        "leaderboard.png": "Overall Scores",
        "categories.png": "Category Breakdown",
        "radar.png": "Category Radar",
        "heatmap.png": "Benchmark Heatmap",
    }
    for cp in chart_paths:
        fname = cp.split("/")[-1]
        title = chart_titles.get(fname, fname)
        _a(f"### {title}")
        _a(f"![{title}]({cp})")
        _a("")

    # How to run
    _a("## How to Run")
    _a("")
    _a("```bash")
    _a("# Install dependencies")
    _a("pip install -r requirements.txt")
    _a("")
    _a("# Set API keys (only for providers you use)")
    _a("set DEEPSEEK_API_KEY=...")
    _a("set GROQ_API_KEY=...")
    _a("set GEMINI_API_KEY=...")
    _a("# For gated datasets like GPQA:")
    _a("set HF_TOKEN=...")
    _a("")
    _a("# Run all benchmarks on all models")
    _a("py run_benchmark.py")
    _a("")
    _a("# Run specific benchmarks or models")
    _a("py run_benchmark.py --benchmarks humaneval mbpp")
    _a("py run_benchmark.py --models deepseek/deepseek-chat")
    _a("")
    _a("# Regenerate README + charts from latest results")
    _a("py run_benchmark.py --generate-only")
    _a("```")
    _a("")

    # Footer
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    _a("---")
    _a(f"*Last updated: {now}*")
    _a("")

    return "\n".join(lines)


def write_readme(results: dict, config: Config, chart_paths: list[str],
                 path: str = "README.md") -> None:
    content = generate(results, config, chart_paths)
    Path(path).write_text(content, encoding="utf-8")
