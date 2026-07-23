"""Chart generation for the README using matplotlib."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from .config import Config

COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]

CATEGORY_LABELS = {
    "coding": "Coding",
    "science": "Science",
    "math": "Math",
    "knowledge": "Knowledge",
    "instruction": "Instruction",
}


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

    # Sort models by overall score descending
    ranked = sorted(model_names, key=lambda m: overall[m], reverse=True)

    paths.append(_leaderboard_bar(ranked, overall, out))
    paths.append(_category_bars(ranked, cat_scores, cat_names, out))
    paths.append(_radar(ranked, cat_scores, cat_names, out))
    paths.append(_heatmap(ranked, bench_scores, bench_names, config, out))

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
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=9)
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
        vals = [cat_scores[mname].get(c, 0) * 100 for c in cat_names]
        offset = (i - len(ranked) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=mname,
               color=COLORS[i % len(COLORS)])
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
        vals = [cat_scores[mname].get(c, 0) for c in cat_names]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=mname,
                color=COLORS[i % len(COLORS)])
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
    from .benchmarks import BENCHMARK_CLASSES
    labels = [BENCHMARK_CLASSES[b].display_name if b in BENCHMARK_CLASSES else b
              for b in bench_names]
    data = np.array([[bench_scores[m].get(b, 0) * 100 for b in bench_names]
                     for m in ranked])
    fig, ax = plt.subplots(figsize=(12, max(3, len(ranked) * 0.7)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels(ranked)
    for i in range(len(ranked)):
        for j in range(len(labels)):
            color = "white" if data[i, j] < 40 else "black"
            ax.text(j, i, f"{data[i, j]:.0f}%", ha="center", va="center",
                    fontsize=9, color=color)
    fig.colorbar(im, ax=ax, label="Score (%)")
    ax.set_title("Benchmark Heatmap", fontweight="bold")
    fig.tight_layout()
    path = out / "heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return f"charts/{path.name}"
