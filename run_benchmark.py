#!/usr/bin/env python
"""Lite Benchmarks CLI — run LLM benchmarks and generate a leaderboard README.

Usage:
    py run_benchmark.py                          # all models, all benchmarks
    py run_benchmark.py --benchmarks humaneval mbpp
    py run_benchmark.py --models deepseek/deepseek-chat
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
from lite_bench.config import load_config
from lite_bench.providers import generate
from lite_bench.readme_gen import write_readme

console = Console()


def run_benchmarks(config, model_filter: list[str] | None, bench_filter: list[str] | None) -> dict:
    models = config.models
    if model_filter:
        models = [m for m in models if m.id in model_filter or m.name in model_filter]
        if not models:
            console.print("[red]No models matched the filter.[/red]")
            sys.exit(1)

    benchmarks = config.enabled_benchmarks()
    if bench_filter:
        benchmarks = {k: v for k, v in benchmarks.items() if k in bench_filter}
        if not benchmarks:
            console.print("[red]No benchmarks matched the filter.[/red]")
            sys.exit(1)

    results: dict = {}

    for model in models:
        console.print(f"\n[bold blue]━━━ {model.name} ({model.id}) ━━━[/bold blue]")
        results[model.name] = {}

        for bname, bconf in benchmarks.items():
            bench = create_benchmark(bname, bconf, config.settings)
            try:
                questions = bench.load()
            except Exception as e:
                console.print(f"  [red]Failed to load {bname}: {e}[/red]")
                continue

            correct = 0
            total = len(questions)
            input_tokens = 0
            output_tokens = 0

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"  {bench.display_name}", total=total)
                for q in questions:
                    prompt = bench.format_prompt(q)
                    try:
                        result = generate(model.id, prompt, config.settings)
                        score = bench.evaluate(q, result.text)
                        input_tokens += result.input_tokens
                        output_tokens += result.output_tokens
                    except Exception as e:
                        console.print(f"\n  [yellow]Error on {model.name}/{bname}: {e}[/yellow]")
                        score = 0.0
                    correct += int(score)
                    progress.advance(task)

            score_pct = correct / total if total else 0.0
            results[model.name][bname] = {
                "score": score_pct,
                "correct": correct,
                "total": total,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
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
        "models": results,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Also save as latest.json for --generate-only
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
    table = Table(title="Results Summary")
    table.add_column("Model", style="bold")
    for bname in config.enabled_benchmarks():
        table.add_column(bname, justify="right")
    table.add_column("Overall", justify="right", style="bold green")

    for mname, mdata in results.items():
        bench_scores = {b: mdata.get(b, {}).get("score", 0.0) for b in config.enabled_benchmarks()}
        overall = config.overall_score(bench_scores)
        row = [mname]
        for bname in config.enabled_benchmarks():
            s = bench_scores.get(bname, 0)
            row.append(f"{s:.0%}")
        row.append(f"{overall:.1%}" if overall is not None else "N/A")
        table.add_row(*row)

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Lite Benchmarks — LLM evaluation suite")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--benchmarks", nargs="+", help="Run only these benchmarks")
    parser.add_argument("--models", nargs="+", help="Run only these models (id or name)")
    parser.add_argument("--generate-only", action="store_true",
                        help="Regenerate README + charts from latest results (no API calls)")
    parser.add_argument("--list", action="store_true",
                        help="List configured models and benchmarks")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.list:
        console.print("\n[bold]Models:[/bold]")
        for m in config.models:
            console.print(f"  {m.id}  ({m.name})")
        console.print("\n[bold]Benchmarks:[/bold]")
        for bname, bconf in config.enabled_benchmarks().items():
            cat = config.benchmark_category(bname)
            console.print(f"  {bname}  [{cat}]  {bconf.dataset}  (n={bconf.num_samples})")
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

    # Generate charts + README
    console.print("\n[bold]Generating charts...[/bold]")
    chart_paths = generate_charts(results, config, config.settings.charts_dir)
    console.print(f"  {len(chart_paths)} charts saved to {config.settings.charts_dir}/")

    console.print("[bold]Generating README...[/bold]")
    write_readme(results, config, chart_paths)
    console.print("  README.md updated")

    print_summary(results, config)


if __name__ == "__main__":
    main()
