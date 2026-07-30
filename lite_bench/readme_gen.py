"""Auto-generate a detailed README.md from benchmark results."""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .metadata import BENCHMARK_INFO, CATEGORY_ICONS, CATEGORY_LABELS


def wilson_half_width(correct: int, total: int, z: float = 1.96) -> float:
    """Calculate 95% Wilson score interval half-width in percentage points."""
    if total <= 0:
        return 0.0
    p = correct / total
    denom = 1 + z**2 / total
    margin = (z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)) / denom
    return round(margin * 100, 1)


def _now_utc_str() -> str:
    """Footer timestamp. Honors ``SOURCE_DATE_EPOCH`` (seconds since UTC epoch)
    when set so report generation is reproducible/hermetic; otherwise the
    wall-clock UTC time."""
    sde = os.environ.get("SOURCE_DATE_EPOCH")
    if sde:
        try:
            ts = datetime.fromtimestamp(int(float(sde)), tz=timezone.utc)
            return ts.strftime("%Y-%m-%d %H:%M UTC")
        except (TypeError, ValueError):
            pass
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def generate(results: dict, config: Config, chart_paths: list[str]) -> str:
    if isinstance(results.get("models"), dict):
        results = results["models"]
    bench_names = list(config.enabled_benchmarks().keys())
    cat_names = list(config.categories.keys())
    model_names = [m for m in results if isinstance(results[m], dict)]

    bench_scores: dict[str, dict[str, float]] = {}
    cat_scores: dict[str, dict[str, float | None]] = {}
    overall: dict[str, float | None] = {}
    total_in_tokens: dict[str, int] = {}
    total_out_tokens: dict[str, int] = {}
    total_think_tokens: dict[str, int] = {}
    total_all_tokens: dict[str, int] = {}
    total_cost_usd: dict[str, float | None] = {}
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

        costs = [
            result.get("total_cost_usd") for result in completed if result.get("total_cost_usd") is not None
        ]
        total_cost_usd[mname] = sum(costs) if costs else None

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

    num_benches = len(bench_names)
    num_cats = len(cat_names)
    num_models = len(model_names)

    # ── Hero ─────────────────────────────────────────────────────────
    _a('<div align="center">')
    _a("")
    _a("# 🏆 Lite Benchmarks")
    _a("")
    _a("### Personal LLM Leaderboard & Benchmark Studio")
    _a("")
    _a(
        "[![CI](https://img.shields.io/github/actions/workflow/status/TruftedBug89/lite-benchmarks/ci.yml"
        "?branch=main&style=flat-square&logo=github&label=CI)]"
        "(https://github.com/TruftedBug89/lite-benchmarks/actions)"
    )
    _a(
        "[![Live Site](https://img.shields.io/badge/Live_Site-lite--benchmarks.netlify.app"
        "-f0b429?style=flat-square&logo=netlify&logoColor=white)]"
        "(https://lite-benchmarks.netlify.app/)"
    )
    _a(
        "[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)]"
        "(https://opensource.org/licenses/MIT)"
    )
    _a(
        "[![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-blue?style=flat-square&logo=python&logoColor=white)]"
        "(https://www.python.org)"
    )
    _a(f"![Benchmarks](https://img.shields.io/badge/Benchmarks-{num_benches}-green?style=flat-square)")
    _a(f"![Categories](https://img.shields.io/badge/Categories-{num_cats}-purple?style=flat-square)")
    _a(f"![Models](https://img.shields.io/badge/Models_tested-{num_models}-orange?style=flat-square)")
    _a("![Version](https://img.shields.io/badge/version-0.2.0-red?style=flat-square)")
    _a("")
    _a(
        "*Small, repeatable samples of established benchmarks with **100% programmatic scoring**. "
        "No LLM-as-judge. Deterministic sampling. Sandboxed code execution.*"
    )
    _a("")
    _a("</div>")
    _a("")

    # ── Live site ────────────────────────────────────────────────────
    _a("## 🌐 Live Leaderboard")
    _a("")
    _a(
        "**[lite-benchmarks.netlify.app](https://lite-benchmarks.netlify.app/)** — "
        "interactive leaderboard, charts, and per-benchmark breakdowns, "
        "auto-rebuilt from this repo on every push."
    )
    _a("")

    # ── Features ─────────────────────────────────────────────────────
    _a("## ✨ Why Lite Benchmarks?")
    _a("")
    _a("<table>")
    _a("<tr>")
    _a('<td width="50%">')
    _a("")
    _a("🎯 **Programmatic Scoring**")
    _a("")
    _a("Every answer is verified by code — regex extraction, unit test execution, exact match. Zero LLM-as-judge bias.")
    _a("")
    _a("</td>")
    _a('<td width="50%">')
    _a("")
    _a("🔒 **3-Layer Sandboxed Execution**")
    _a("")
    _a("AST scan → hardened subprocess → Windows Job Object. Model code runs isolated with no API keys, no network, no escape.")
    _a("")
    _a("</td>")
    _a("</tr>")
    _a("<tr>")
    _a('<td width="50%">')
    _a("")
    _a("🌐 **Web Dashboard**")
    _a("")
    _a("Select models, pick benchmarks, run, and generate reports — all from a local browser UI. No CLI wrangling.")
    _a("")
    _a("</td>")
    _a('<td width="50%">')
    _a("")
    _a("🎲 **Deterministic Sampling**")
    _a("")
    _a("Fixed seed (42) random sampling means the exact same questions every run. Reproducible by design.")
    _a("")
    _a("</td>")
    _a("</tr>")
    _a("<tr>")
    _a('<td width="50%">')
    _a("")
    _a("📊 **Statistical Rigor**")
    _a("")
    _a("Wilson score confidence intervals on every benchmark score. Know exactly how much noise is in the numbers.")
    _a("")
    _a("</td>")
    _a('<td width="50%">')
    _a("")
    _a("💰 **Cost & Token Tracking**")
    _a("")
    _a("Per-model token breakdown (input/output/thinking), throughput (TPS), latency, and estimated API cost.")
    _a("")
    _a("</td>")
    _a("</tr>")
    _a("</table>")
    _a("")

    # ── Quick Start ──────────────────────────────────────────────────
    _a("## ⚡ Quick Start")
    _a("")
    _a("```bash")
    _a("pip install -e .[dev]    # install dependencies")
    _a("py web_app.py            # launch dashboard → http://127.0.0.1:8000")
    _a("```")
    _a("")
    _a("Then select models, pick benchmarks, hit **Run Benchmarks**, and **Generate Reports**.")
    _a("")

    # ── Table of Contents ────────────────────────────────────────────
    _a("## 📑 Table of Contents")
    _a("")
    _a("- [Live Leaderboard](#-live-leaderboard)")
    _a("- [Benchmarks](#-benchmarks)")
    _a("- [Leaderboard](#-leaderboard)")
    _a("- [Charts](#-charts)")
    _a("- [Token Usage & Performance](#-token-usage--performance)")
    _a("- [Architecture](#-architecture)")
    _a("- [Methodology](#-methodology)")
    _a("- [How to Run](#-how-to-run)")
    _a("- [Adding Models](#-adding-models)")
    _a("- [Project Structure](#-project-structure)")
    _a("")

    # ── Benchmarks ──────────────────────────────────────────────────
    _a("## 📝 Benchmarks")
    _a("")
    _a(
        f"{num_benches} established benchmarks, ~50 questions each, "
        f"grouped into {num_cats} categories:"
    )
    _a("")
    _a("| Benchmark | Category | Full Dataset | Sampled | Verification | Source |")
    _a("|-----------|----------|:-----------:|:-------:|-------------|--------|")
    for bname in bench_names:
        info = BENCHMARK_INFO.get(
            bname, {"display": bname, "category": "Other", "total": "N/A", "verification": "Auto", "source": "HF"}
        )
        sampled = config.benchmarks[bname].num_samples
        _a(
            f"| **{info['display']}** | {info['category']} "
            f"| {info['total']} | {sampled} "
            f"| {info['verification']} | `{info['source']}` |"
        )
    _a("")

    # ── Leaderboard ─────────────────────────────────────────────────
    _a("## 🏅 Leaderboard")
    _a("")
    if not ranked:
        _a("*No results yet. Run `py web_app.py` to launch the dashboard and run benchmarks.*")
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

            is_thinking = (
                total_think_tokens.get(mname, 0) > 0
                or results.get(mname, {}).get("thinking_effort") is not None
            )
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
                if s is not None and total > 0:
                    w_margin = wilson_half_width(correct, total)
                    row += f" {s * 100:.0f}% ({correct}/{total}) ±{w_margin}pp |"
                else:
                    row += " N/A |"
            _a(row)
        _a("")
        _a("*±pp indicates 95% Wilson score confidence interval half-width.*")
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
        _a("| Model | Input | Output | Thinking | Total | Out % | Think % | Avg TPS | Avg Time | Est. Cost |")
        _a("|-------|------:|-------:|---------:|------:|------:|--------:|--------:|---------:|----------:|")
        for mname in ranked:
            tin = total_in_tokens.get(mname, 0)
            tout = total_out_tokens.get(mname, 0)
            tthink = total_think_tokens.get(mname, 0)
            ttot = total_all_tokens.get(mname, 0)
            cost_val = total_cost_usd.get(mname)
            cost_str = f"${cost_val:.4f}" if cost_val is not None else "—"
            out_pct = f"{tout / ttot:.0%}" if ttot else "—"
            think_pct = f"{tthink / ttot:.0%}" if ttot and tthink else "—"
            tps = avg_tps.get(mname)
            tps_str = f"{tps:.1f}" if tps is not None else "—"
            time_ms = avg_time.get(mname)
            time_str = f"{time_ms / 1000:.1f}s" if time_ms is not None else "—"
            _a(
                f"| {mname} | {tin:,} | {tout:,} | {tthink:,} | {ttot:,} "
                f"| {out_pct} | {think_pct} | {tps_str} | {time_str} | {cost_str} |"
            )
        _a("")
        _a(
            "*TPS = output tokens/second (cloud APIs only, skipped for local models). "
            "Est. Cost calculated via LiteLLM cost tables.*"
        )
        _a("")

    # ── Architecture ─────────────────────────────────────────────────
    _a("## 🏗️ Architecture")
    _a("")
    _a("```mermaid")
    _a("flowchart LR")
    _a("    A[config.yaml] --> B[datasets.py<br/>HF sampling]")
    _a("    B --> C[engine.py<br/>concurrent execution]")
    _a("    C --> D[providers.py<br/>litellm calls]")
    _a("    D --> E[benchmarks.py<br/>scoring & verification]")
    _a("    E --> F[results_store.py<br/>schema v2 JSON]")
    _a("    F --> G[charts.py<br/>matplotlib PNGs]")
    _a("    F --> H[readme_gen.py<br/>this README]")
    _a("")
    _a("    C --> I[sandbox.py<br/>3-layer isolation]")
    _a("    I --> E")
    _a("```")
    _a("")

    # ── Methodology ─────────────────────────────────────────────────
    _a("## 🔬 Methodology")
    _a("")
    sample_sizes = {config.benchmarks[b].num_samples for b in bench_names}
    sample_str = str(sample_sizes.pop()) if len(sample_sizes) == 1 else "varies"

    _a("<details>")
    _a("<summary><strong>📐 Sampling & Statistical Significance</strong></summary>")
    _a("")
    _a(f"- **~{sample_str} questions** are sampled from each benchmark's full dataset")
    _a(
        f"- Sampling uses a **fixed seed ({config.settings.seed})** via random sampling "
        "so exact questions are stable across runs"
    )
    _a(
        "- Samples of n=50 have 95% confidence intervals of roughly ±7–14pp; "
        "treat small ranking gaps as noise"
    )
    _a(
        "- **Scoring v2 Notice**: Sampling and scoring strictness updated in v0.2.0; "
        "results are not directly comparable with pre-v0.2.0 runs"
    )
    _a("")
    _a("</details>")
    _a("")

    _a("<details>")
    _a("<summary><strong>✅ Scoring & Verification</strong></summary>")
    _a("")
    _a("- **All scoring is programmatic** — no LLM-as-judge is used anywhere")
    _a(
        "- Code benchmarks require explicit opt-in (``allow_unsafe_code_execution``) "
        "and run in a layered sandbox: an AST scan of generated code rejects "
        "destructive / escape constructs, the child runs with a scrubbed environment "
        "(no API keys, temp working dir), a runtime confinement shim restricts file "
        "I/O to the sandbox dir and sockets to loopback, and on Windows it is "
        "additionally confined by a Job Object that blocks grandchild processes and "
        "UI access. The opt-in gate is enforced at the sandbox layer, so it fails "
        "closed even for direct callers. Tasks whose reference solution the sandbox "
        "cannot run are filtered before sampling, so every graded task is passable."
    )
    _a("- Multiple-choice benchmarks extract the answer letter and compare to ground truth")
    _a(
        "- Math benchmarks extract boxed/numerical answers (scientific notation "
        "included) and evaluate via normalized string, deterministic symbolic "
        "(sympy) equivalence, or numerical comparison"
    )
    _a("- IFEval uses its 25 strict programmatic verifiers (word count, format, keywords, etc.)")
    _a("- Tau-Bench verifies tool function name AND argument dictionary match")
    _a("")
    _a("</details>")
    _a("")

    _a("<details>")
    _a("<summary><strong>📊 Category & Overall Scores</strong></summary>")
    _a("")
    _a("- **Category score** = average of its benchmark scores")
    for cat in cat_names:
        benches = config.categories.get(cat, [])
        bench_labels = [
            BENCHMARK_INFO.get(b, {}).get("display", b) for b in benches if b in bench_names
        ]
        if bench_labels:
            icon = CATEGORY_ICONS.get(cat, "")
            _a(f"  - {icon} **{CATEGORY_LABELS.get(cat, cat)}** = avg({', '.join(bench_labels)})")
    _a("- **Overall score** = average of completed category scores (equal weight per category)")
    _a(
        "- Provider failures are excluded and recorded separately; "
        "scorer exceptions score 0.0 without retrying provider"
    )
    _a("")
    _a("</details>")
    _a("")

    _a("<details>")
    _a("<summary><strong>⚙️ Inference Settings</strong></summary>")
    _a("")
    _a(f"- `temperature`: {config.settings.temperature}")
    _a(f"- `max_tokens`: {config.settings.max_tokens}")
    _a(f"- `timeout`: {config.settings.request_timeout}s per request")
    if config.settings.max_retries > 0:
        _a(
            f"- `retries`: up to {config.settings.max_retries} attempts "
            "with exponential backoff and jitter"
        )
    else:
        _a(
            "- `retries`: transient errors retry with exponential backoff until a good "
            "response arrives (no cap); permanent errors (context length, content filter) "
            "are never retried"
        )
    _a("")
    _a("</details>")
    _a("")

    # ── How to run ──────────────────────────────────────────────────
    _a("## 🚀 How to Run")
    _a("")
    _a("### Prerequisites")
    _a("")
    _a("```bash")
    _a("pip install -e .[dev]")
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
    _a("### Running Benchmarks via Web Dashboard")
    _a("")
    _a("```bash")
    _a("# Launch the local web dashboard (serves on http://127.0.0.1:8000)")
    _a("py web_app.py")
    _a("")
    _a("# Launch without automatically opening browser")
    _a("py web_app.py --no-browser")
    _a("```")
    _a("")
    _a("1. Open the dashboard in your browser.")
    _a("2. Select models, benchmarks, and settings.")
    _a("3. Click **Run Benchmarks**.")
    _a("4. Click **Generate Reports** to update `README.md` and `charts/`.")
    _a("")

    # ── Adding models ───────────────────────────────────────────────
    _a("## ➕ Adding Models")
    _a("")
    _a("Edit `config.yaml` or add models directly in the Web UI:")
    _a("")
    _a("```yaml")
    _a("models:")
    _a("  - id: anthropic/claude-sonnet-4-20250514")
    _a("    name: Claude Sonnet 4")
    _a("    max_tokens: 16384")
    _a("  - id: openai/gpt-4o")
    _a("    name: GPT-4o")
    _a("  - id: lm_studio/qwen2.5-coder-7b-instruct")
    _a("    name: Qwen 2.5 Coder 7B (local)")
    _a("```")
    _a("")

    # ── Project structure ───────────────────────────────────────────
    _a("## 📁 Project Structure")
    _a("")
    _a("```")
    _a("├── config.yaml              # Models, benchmarks, categories, settings")
    _a("├── web_app.py               # Web dashboard server")
    _a("├── windows_sandbox.py       # Windows Job-object/restricted-token sandbox")
    _a("├── README.md                # ← this file (auto-generated)")
    _a("├── web/                     # Web dashboard frontend (HTML/CSS/JS)")
    _a("├── lite_bench/")
    _a("│   ├── engine.py            # Unified execution engine & thread concurrency")
    _a("│   ├── results_store.py     # Results persistence, schema v2, atomic writes")
    _a("│   ├── metadata.py          # Benchmark display metadata & category mapping")
    _a("│   ├── config.py            # Config loading & validation")
    _a("│   ├── providers.py         # litellm wrapper & telemetry")
    _a("│   ├── datasets.py          # Deterministic HuggingFace sampling")
    _a("│   ├── benchmarks.py        # Benchmark implementations & verifiers")
    _a("│   ├── ifeval_verifiers.py  # 25 strict IFEval verifiers")
    _a("│   ├── sandbox.py           # Code-exec sandbox (AST scan + subprocess + Win job)")
    _a("│   ├── charts.py            # matplotlib chart generation")
    _a("│   └── readme_gen.py        # README generator")
    _a("├── results/                 # JSON results per run")
    _a("│   └── latest.json          # Leaderboard results (schema v2)")
    _a("└── charts/                  # Generated PNG charts")
    _a("```")
    _a("")

    # ── Footer ──────────────────────────────────────────────────────
    now = _now_utc_str()
    _a("---")
    _a("")
    _a('<div align="center">')
    _a("")
    _a(
        f"*Auto-generated by [lite-benchmarks](.) on {now} · "
        "Licensed under [MIT](LICENSE) · "
        "Built with [litellm](https://github.com/BerriAI/litellm) + "
        "[HuggingFace Datasets](https://github.com/huggingface/datasets)*"
    )
    _a("")
    _a("**⭐ Star this repo if you find it useful!**")
    _a("")
    _a("</div>")
    _a("")

    return "\n".join(L)


def write_readme(
    results: dict, config: Config, chart_paths: list[str], path: str = "README.md"
) -> None:
    content = generate(results, config, chart_paths)
    Path(path).write_text(content, encoding="utf-8")
