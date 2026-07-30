#!/usr/bin/env python
"""Refresh the public site data snapshot from benchmark results.

Reads ``results/latest.json`` and ``config.yaml``, then writes a compact
``site/src/data/summary.json`` that the Astro site bakes in at build time.

Usage:
    py refresh_site.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from lite_bench.config import Config, load_config
from lite_bench.metadata import BENCHMARK_INFO, CATEGORY_ICONS, CATEGORY_LABELS
from lite_bench.readme_gen import wilson_half_width

ROOT_DIR = Path(__file__).parent.resolve()
SITE_DATA_PATH = ROOT_DIR / "site" / "src" / "data" / "summary.json"

_PROVIDER_LABELS = {
    "deepseek": "DeepSeek",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Google",
    "google": "Google",
    "groq": "Groq",
    "mistral": "Mistral",
    "xai": "xAI",
    "moonshot": "Moonshot",
    "lm_studio": "LM Studio",
    "lmstudio": "LM Studio",
    "ollama": "Ollama",
    "huggingface": "HuggingFace",
    "cerebras": "Cerebras",
    "perplexity": "Perplexity",
    "cohere": "Cohere",
}


def provider_for(model_id: str) -> str:
    prefix = model_id.split("/", 1)[0].strip().lower() if "/" in model_id else ""
    return _PROVIDER_LABELS.get(prefix, prefix.title() or "Unknown")


def build_summary(config: Config, results: dict, run_timestamp: str | None = None) -> dict:
    """Build the compact site data snapshot from a models-results mapping."""
    if isinstance(results.get("models"), dict):
        results = results["models"]

    bench_names = list(config.enabled_benchmarks().keys())

    categories = [
        {
            "key": cat,
            "label": CATEGORY_LABELS.get(cat, cat),
            "icon": CATEGORY_ICONS.get(cat, ""),
            "benchmarks": [b for b in config.categories.get(cat, []) if b in bench_names],
        }
        for cat in config.categories
    ]

    benchmarks = []
    for bname in bench_names:
        info = BENCHMARK_INFO.get(bname, {})
        benchmarks.append(
            {
                "key": bname,
                "display": info.get("display", bname),
                "category": info.get("category", "Other").lower(),
                "full_dataset": info.get("total", "N/A"),
                "sampled": config.benchmarks[bname].num_samples,
                "verification": info.get("verification", "Programmatic"),
                "source": info.get("source", ""),
                "paper": info.get("paper", ""),
                "description": info.get("description", ""),
            }
        )

    models = []
    for mname, mdata in results.items():
        if not isinstance(mdata, dict):
            continue

        bench_entries: dict[str, dict] = {}
        bench_scores: dict[str, float] = {}
        for bname in bench_names:
            entry = mdata.get(bname)
            if not isinstance(entry, dict) or "score" not in entry:
                continue
            score = entry.get("score")
            correct = entry.get("correct", 0)
            total = entry.get("total", 0)
            if isinstance(score, (int, float)):
                bench_scores[bname] = score
                bench_entries[bname] = {
                    "score": round(score, 4),
                    "correct": correct,
                    "total": total,
                    "wilson": wilson_half_width(int(correct), int(total)) if total else 0.0,
                }
            else:
                bench_entries[bname] = {"score": None, "correct": correct, "total": total, "wilson": 0.0}

        cat_scores = {
            cat: config.category_score(bench_scores, cat) for cat in config.categories
        }
        overall = config.overall_score(bench_scores)

        attempted = [
            mdata[b] for b in bench_names if isinstance(mdata.get(b), dict) and "score" in mdata[b]
        ]
        tps_vals = [
            r.get("avg_tokens_per_second") for r in attempted if r.get("avg_tokens_per_second") is not None
        ]
        time_vals = [r.get("avg_time_ms") for r in attempted if r.get("avg_time_ms") is not None]
        costs = [r.get("total_cost_usd") for r in attempted if r.get("total_cost_usd") is not None]

        models.append(
            {
                "name": mname,
                "model_id": mdata.get("model_id", ""),
                "provider": provider_for(str(mdata.get("model_id", ""))),
                "thinking_effort": mdata.get("thinking_effort"),
                "overall": round(overall, 4) if overall is not None else None,
                "categories": {
                    cat: round(s, 4) if s is not None else None for cat, s in cat_scores.items()
                },
                "benchmarks": bench_entries,
                "completed_benchmarks": len(bench_scores),
                "tokens": {
                    "input": sum(r.get("input_tokens", 0) for r in attempted),
                    "output": sum(r.get("output_tokens", 0) for r in attempted),
                    "thinking": sum(r.get("thinking_tokens", 0) for r in attempted),
                    "total": sum(r.get("total_tokens", 0) for r in attempted),
                },
                "cost_usd": round(sum(costs), 4) if costs else None,
                "avg_tps": round(sum(tps_vals) / len(tps_vals), 1) if tps_vals else None,
                "avg_time_ms": round(sum(time_vals) / len(time_vals)) if time_vals else None,
            }
        )

    models.sort(key=lambda m: m["overall"] if m["overall"] is not None else float("-inf"), reverse=True)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_timestamp": run_timestamp,
        "seed": config.settings.seed,
        "temperature": config.settings.temperature,
        "max_tokens": config.settings.max_tokens,
        "categories": categories,
        "benchmarks": benchmarks,
        "models": models,
    }


import math

def _sanitize_json(obj: any) -> any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    return obj

def refresh_site(
    config: Config, results: dict, run_timestamp: str | None = None, path: Path = SITE_DATA_PATH
) -> Path:
    """Build the summary snapshot and write it atomically. Returns the path."""
    summary = build_summary(config, results, run_timestamp)
    summary = _sanitize_json(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def main() -> None:
    config = load_config(ROOT_DIR / "config.yaml")
    latest_path = ROOT_DIR / config.settings.results_dir / "latest.json"
    if not latest_path.is_file():
        raise SystemExit(f"No results found at {latest_path} — run benchmarks first.")

    raw = json.loads(latest_path.read_text(encoding="utf-8"))
    models = raw.get("models", {}) if isinstance(raw, dict) else {}
    if not isinstance(models, dict) or not models:
        raise SystemExit(f"No model results in {latest_path} — run benchmarks first.")

    out = refresh_site(config, models, run_timestamp=raw.get("timestamp"))
    n = len(json.loads(out.read_text(encoding="utf-8"))["models"])
    print(f"Wrote {out.relative_to(ROOT_DIR)} ({n} models)")


if __name__ == "__main__":
    main()
