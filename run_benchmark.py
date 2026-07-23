#!/usr/bin/env python
"""Lite Benchmarks CLI — run LLM benchmarks and generate a leaderboard README.

Usage:
    py run_benchmark.py                          # all models, all benchmarks
    py run_benchmark.py --benchmarks humaneval mbpp
    py run_benchmark.py --models deepseek/deepseek-chat
    py run_benchmark.py --models "lm_studio/my-model"   # ad-hoc, no config needed
    py run_benchmark.py --generate-only          # regenerate README+charts from latest results
    py run_benchmark.py --list                   # list configured models and benchmarks
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from lite_bench.benchmarks import create_benchmark
from lite_bench.charts import generate_all as generate_charts
from lite_bench.config import ModelConfig, load_config
from lite_bench.providers import generate
from lite_bench.readme_gen import write_readme

console = Console()


def _aggregate(details: list[dict]) -> dict:
    """Roll up per-question details into benchmark-level stats."""
    n = len(details)
    if n == 0:
        return {}
    inp = sum(d["input_tokens"] for d in details)
    out = sum(d["output_tokens"] for d in details)
    think = sum(d["thinking_tokens"] for d in details)
    tot = sum(d["total_tokens"] for d in details)

    timed = [d for d in details if d.get("total_time_ms") is not None]
    avg_time = sum(d["total_time_ms"] for d in timed) / len(timed) if timed else None
    tps_vals = [d["tokens_per_second"] for d in timed if d.get("tokens_per_second")]
    avg_tps = sum(tps_vals) / len(tps_vals) if tps_vals else None

    return {
        "input_tokens": inp,
        "output_tokens": out,
        "thinking_tokens": think,
        "total_tokens": tot,
        "output_ratio": round(out / tot, 4) if tot else 0.0,
        "thinking_ratio": round(think / tot, 4) if tot else 0.0,
        "avg_time_ms": round(avg_time, 1) if avg_time is not None else None,
        "avg_tokens_per_second": round(avg_tps, 2) if avg_tps is not None else None,
    }


def run_benchmarks(config, model_filter: list[str] | None, bench_filter: list[str] | None) -> dict:
    if model_filter:
        models = []
        for mid in model_filter:
            match = next((m for m in config.models if m.id == mid or m.name == mid), None)
            models.append(match or ModelConfig(id=mid, name=mid))
    else:
        models = config.models

    benchmarks = config.enabled_benchmarks()
    if bench_filter:
        benchmarks = {k: v for k, v in benchmarks.items() if k in bench_filter}
        if not benchmarks:
            console.print("[red]No benchmarks matched the filter.[/red]")
            sys.exit(1)

    results: dict = {}

    for model in models:
        console.print(f"\n[bold blue]━━━ {model.name} ({model.id}) ━━━[/bold blue]")
        results[model.name] = {"model_id": model.id}

        for bname, bconf in benchmarks.items():
            bench = create_benchmark(bname, bconf, config.settings)
            try:
                questions = bench.load()
            except Exception as e:
                console.print(f"  [red]Failed to load {bname}: {e}[/red]")
                continue

            correct = 0
            total = len(questions)
            question_details: list[dict] = []

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"  {bench.display_name}", total=total)
                for qi, q in enumerate(questions):
                    prompt = bench.format_prompt(q)
                    try:
                        result = generate(model.id, prompt, config.settings)
                        score = bench.evaluate(q, result.text)
                    except Exception as e:
                        console.print(f"\n  [yellow]Error on {model.name}/{bname} q{qi}: {e}[/yellow]")
                        score = 0.0
                        result = None

                    detail: dict = {
                        "question_index": qi,
                        "score": score,
                        "input_tokens": result.input_tokens if result else 0,
                        "output_tokens": result.output_tokens if result else 0,
                        "thinking_tokens": result.thinking_tokens if result else 0,
                        "total_tokens": result.total_tokens if result else 0,
                    }
                    if result and result.total_time_ms is not None:
                        detail["total_time_ms"] = round(result.total_time_ms, 1)
                    if result and result.tokens_per_second is not None:
                        detail["tokens_per_second"] = round(result.tokens_per_second, 2)

                    question_details.append(detail)
                    correct += int(score)
                    progress.advance(task)

            score_pct = correct / total if total else 0.0
            agg = _aggregate(question_details)

            results[model.name][bname] = {
                "score": score_pct,
                "correct": correct,
                "total": total,
                **agg,
                "questions": question_details,
            }
            console.print(
                f"  [green]{bench.display_name}:[/green] "
                f"{correct}/{total} ({score_pct:.0%})"
            )

    return results


def save_results(results: dict, config) -> Path:
    out = Path(config.settings.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out / f"results_{ts}.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "seed": config.settings.seed,
            "max_tokens": config.settings.max_tokens,
            "temperature": config.settings.temperature,
            "request_timeout": config.settings.request_timeout,
            "code_exec_timeout": config.settings.code_exec_timeout,
        },
        "benchmarks": {
            bname: {
                "dataset": bconf.dataset,
                "subset": bconf.subset,
                "num_samples": bconf.num_samples,
                "category": config.benchmark_category(bname),
            }
            for bname, bconf in config.enabled_benchmarks().items()
        },
        "models": results,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest = out / "latest.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_latest_results(config) -> dict | None:
    latest = Path(config.settings.results_dir) / "latest.json"
    if not latest.exists():
        return None
    data = json.loads(latest.read_text(encoding="utf-8"))
    return data.get("models", {})


def print_summary(results: dict, config):
    bench_names = list(config.enabled_benchmarks().keys())

    # Score table
    table = Table(title="Scores")
    table.add_column("Model", style="bold")
    for bname in bench_names:
        table.add_column(bname, justify="right")
    table.add_column("Overall", justify="right", style="bold green")

    for mname, mdata in results.items():
        bench_scores = {b: mdata.get(b, {}).get("score", 0.0) for b in bench_names}
        overall = config.overall_score(bench_scores)
        row = [mname]
        for bname in bench_names:
            row.append(f"{bench_scores.get(bname, 0):.0%}")
        row.append(f"{overall:.1%}" if overall is not None else "N/A")
        table.add_row(*row)
    console.print(table)

    # Token table
    ttable = Table(title="Token Usage")
    ttable.add_column("Model", style="bold")
    ttable.add_column("Input", justify="right")
    ttable.add_column("Output", justify="right")
    ttable.add_column("Thinking", justify="right")
    ttable.add_column("Total", justify="right")
    ttable.add_column("Out %", justify="right")
    ttable.add_column("Avg TPS", justify="right")

    for mname, mdata in results.items():
        tin = sum(mdata.get(b, {}).get("input_tokens", 0) for b in bench_names)
        tout = sum(mdata.get(b, {}).get("output_tokens", 0) for b in bench_names)
        tthink = sum(mdata.get(b, {}).get("thinking_tokens", 0) for b in bench_names)
        ttot = sum(mdata.get(b, {}).get("total_tokens", 0) for b in bench_names)
        out_pct = f"{tout/ttot:.0%}" if ttot else "—"
        tps_vals = [
            mdata.get(b, {}).get("avg_tokens_per_second")
            for b in bench_names
            if mdata.get(b, {}).get("avg_tokens_per_second") is not None
        ]
        avg_tps = f"{sum(tps_vals)/len(tps_vals):.1f}" if tps_vals else "—"
        ttable.add_row(mname, f"{tin:,}", f"{tout:,}", f"{tthink:,}", f"{ttot:,}", out_pct, avg_tps)

    console.print(ttable)


def main():
    parser = argparse.ArgumentParser(description="Lite Benchmarks — LLM evaluation suite")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--benchmarks", nargs="+", help="Run only these benchmarks")
    parser.add_argument("--models", nargs="+", help="Model IDs (litellm format) or configured names")
    parser.add_argument("--generate-only", action="store_true",
                        help="Regenerate README + charts from latest results (no API calls)")
    parser.add_argument("--list", action="store_true",
                        help="List configured models and benchmarks")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.list:
        console.print("\n[bold]Configured models:[/bold]")
        for m in config.models:
            console.print(f"  {m.id}  ({m.name})")
        console.print("\n[bold]Benchmarks:[/bold]")
        for bname, bconf in config.enabled_benchmarks().items():
            cat = config.benchmark_category(bname)
            console.print(f"  {bname}  [{cat}]  {bconf.dataset}  (n={bconf.num_samples})")
        console.print("\n[dim]Tip: --models accepts any litellm ID, e.g. "
                      "--models \"lm_studio/my-model\"[/dim]")
        return

    if args.generate_only:
        results = load_latest_results(config)
        if results is None:
            console.print("[red]No results found. Run benchmarks first.[/red]")
            sys.exit(1)
    else:
        results = run_benchmarks(config, args.models, args.benchmarks)
        path = save_results(results, config)
        console.print(f"\n[dim]Results saved to {path}[/dim]")

    console.print("\n[bold]Generating charts...[/bold]")
    chart_paths = generate_charts(results, config, config.settings.charts_dir)
    console.print(f"  {len(chart_paths)} charts saved to {config.settings.charts_dir}/")

    console.print("[bold]Generating README...[/bold]")
    write_readme(results, config, chart_paths)
    console.print("  README.md updated")

    print_summary(results, config)


if __name__ == "__main__":
    main()
