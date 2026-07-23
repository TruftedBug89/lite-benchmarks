"""Chart generation for the README using matplotlib."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from .config import Config
from .metadata import BENCHMARK_INFO, CATEGORY_LABELS

COLORS = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
]


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)


def generate_all(results: dict, config: Config, charts_dir: str) -> list[str]:
    """Generate all charts. Returns list of relative paths to PNGs."""
    out = Path(charts_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    model_names = list(results.keys())
    bench_names = list(config.enabled_benchmarks().keys())
    cat_names = list(config.categories.keys())

    # Pre-compute scores
    bench_scores: dict[str, dict[str, float]] = {}
    cat_scores: dict[str, dict[str, float | None]] = {}
    overall: dict[str, float | None] = {}
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

    # Sort models by overall score descending
    ranked = sorted(
        (mname for mname in model_names if overall[mname] is not None),
        key=lambda mname: overall[mname],
        reverse=True,
    )
    if not ranked:
        return []

    paths.append(_leaderboard_bar(ranked, overall, out))
    paths.append(_category_bars(ranked, cat_scores, cat_names, out))
    paths.append(_radar(ranked, cat_scores, cat_names, out))
    paths.append(_heatmap(ranked, bench_scores, bench_names, config, out))

    # Token breakdown
    token_data: dict[str, dict[str, int]] = {}
    for mname in ranked:
        mdata = results[mname]
        token_data[mname] = {
            "input": sum(mdata.get(b, {}).get("input_tokens", 0) for b in bench_names),
            "thinking": sum(mdata.get(b, {}).get("thinking_tokens", 0) for b in bench_names),
            "output": sum(mdata.get(b, {}).get("output_tokens", 0) for b in bench_names),
        }
    paths.append(_token_breakdown(ranked, token_data, out))
    paths.append(_thinking_vs_score(ranked, token_data, overall, out))

    return paths


def _leaderboard_bar(ranked, overall, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, max(3, len(ranked) * 0.6)))
    y = range(len(ranked))
    vals = [overall[m] * 100 for m in ranked]
    bars = ax.barh(y, vals, color=[COLORS[i % len(COLORS)] for i in range(len(ranked))])
    ax.set_yticks(y)
    ax.set_yticklabels(ranked)
    ax.invert_yaxis()
    ax.set_xlabel("Overall Score (%)")
    ax.set_title("Overall Leaderboard", fontweight="bold")
    ax.set_xlim(0, 100)
    _style(ax)
    for bar, v in zip(bars, vals, strict=True):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}%",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    path = out / "leaderboard.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return f"charts/{path.name}"


def _category_bars(ranked, cat_scores, cat_names, out: Path) -> str:
    labels = [CATEGORY_LABELS.get(c, c) for c in cat_names]
    x = np.arange(len(labels))
    width = 0.8 / max(len(ranked), 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, mname in enumerate(ranked):
        vals = [
            score * 100 if (score := cat_scores[mname].get(category)) is not None else np.nan
            for category in cat_names
        ]
        offset = (i - len(ranked) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=mname, color=COLORS[i % len(COLORS)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_title("Category Breakdown", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8)
    _style(ax)
    fig.tight_layout()
    path = out / "categories.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return f"charts/{path.name}"


def _radar(ranked, cat_scores, cat_names, out: Path) -> str:
    labels = [CATEGORY_LABELS.get(c, c) for c in cat_names]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, mname in enumerate(ranked):
        vals = [
            score if (score := cat_scores[mname].get(category)) is not None else np.nan
            for category in cat_names
        ]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=mname, color=COLORS[i % len(COLORS)])
        ax.fill(angles, vals, alpha=0.08, color=COLORS[i % len(COLORS)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Category Radar", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    path = out / "radar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"charts/{path.name}"


def _heatmap(ranked, bench_scores, bench_names, config, out: Path) -> str:
    labels = [
        BENCHMARK_INFO.get(b, {}).get("display", b) for b in bench_names
    ]
    data = np.array(
        [
            [bench_scores[model].get(benchmark, np.nan) * 100 for benchmark in bench_names]
            for model in ranked
        ]
    )
    fig, ax = plt.subplots(figsize=(12, max(3, len(ranked) * 0.7)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels(ranked)
    for i in range(len(ranked)):
        for j in range(len(labels)):
            value = data[i, j]
            if np.isnan(value):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=9, color="black")
                continue
            color = "white" if value < 40 else "black"
            ax.text(j, i, f"{value:.0f}%", ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(im, ax=ax, label="Score (%)")
    ax.set_title("Benchmark Heatmap", fontweight="bold")
    fig.tight_layout()
    path = out / "heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return f"charts/{path.name}"


def _token_breakdown(ranked, token_data, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, max(3, len(ranked) * 0.6)))
    y = np.arange(len(ranked))
    inp = [token_data[m]["input"] for m in ranked]
    think = [token_data[m]["thinking"] for m in ranked]
    outp = [token_data[m]["output"] for m in ranked]

    ax.barh(y, inp, label="Input", color="#4C72B0")
    ax.barh(y, think, left=inp, label="Thinking", color="#DD8452")
    left2 = [i + t for i, t in zip(inp, think, strict=True)]
    ax.barh(y, outp, left=left2, label="Output", color="#55A868")

    ax.set_yticks(y)
    ax.set_yticklabels(ranked)
    ax.invert_yaxis()
    ax.set_xlabel("Tokens")
    ax.set_title("Token Breakdown (Input / Thinking / Output)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    _style(ax)
    fig.tight_layout()
    path = out / "tokens.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return f"charts/{path.name}"


def _thinking_vs_score(ranked, token_data, overall, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))

    x = [token_data[m]["thinking"] for m in ranked]
    y = [overall[m] * 100 for m in ranked]

    if sum(x) == 0:
        ax.text(0.5, 0.5, "No thinking tokens recorded", ha="center", va="center")
    else:
        for i, mname in enumerate(ranked):
            if x[i] > 0 or y[i] > 0:
                ax.scatter(x[i], y[i], label=mname, color=COLORS[i % len(COLORS)], s=100, alpha=0.7)
                ax.text(x[i], y[i], f" {mname}", fontsize=8, va="center")

    ax.set_xlabel("Total Thinking Tokens")
    ax.set_ylabel("Overall Score (%)")
    ax.set_title("Thinking Effort vs Performance", fontweight="bold")

    _style(ax)
    fig.tight_layout()
    path = out / "thinking_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return f"charts/{path.name}"
