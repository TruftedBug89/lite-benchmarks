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
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from lite_bench.benchmarks import create_benchmark
from lite_bench.charts import generate_all as generate_charts
from lite_bench.config import Config, ModelConfig, load_config
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


def _benchmark_scores(model_results: dict) -> dict[str, float]:
    """Return only completed benchmark scores, never synthetic zeroes."""
    return {
        name: result["score"]
        for name, result in model_results.items()
        if isinstance(result, dict) and "score" in result
    }


def run_benchmarks(
    config: Config, model_filter: list[str] | None, bench_filter: list[str] | None
) -> dict:
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

    results: dict[str, dict] = {}

    for model in models:
        console.print(f"\n[bold blue]━━━ {model.name} ({model.id}) ━━━[/bold blue]")
        results[model.name] = {"model_id": model.id}

        for bname, bconf in benchmarks.items():
            bench = create_benchmark(bname, bconf, config.settings)
            if bench.requires_code_execution and not config.settings.allow_unsafe_code_execution:
                console.print(
                    f"  [yellow]Skipped {bench.display_name}: it executes untrusted generated "
                    "code. Re-run with --allow-unsafe-code-execution only inside an isolated sandbox.[/yellow]"
                )
                continue
            try:
                questions = bench.load()
            except Exception as error:
                console.print(f"  [red]Failed to load {bname}: {type(error).__name__}[/red]")
                continue

            correct = 0
            requested = len(questions)
            scored = 0
            failed = 0
            question_details: list[dict] = []

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"  {bench.display_name}", total=requested)
                for qi, q in enumerate(questions):
                    prompt = bench.format_prompt(q)
                    try:
                        result = generate(model.id, prompt, config.settings)
                        score = bench.evaluate(q, result.text)
                    except Exception as error:
                        console.print(
                            f"\n  [yellow]Excluded {model.name}/{bname} q{qi}: "
                            f"{type(error).__name__}[/yellow]"
                        )
                        failed += 1
                        result = None
                        question_details.append(
                            {
                                "question_index": qi,
                                "status": "error",
                                "error_type": type(error).__name__,
                            }
                        )
                    else:
                        scored += 1
                        detail: dict = {
                            "question_index": qi,
                            "status": "scored",
                            "score": score,
                            "input_tokens": result.input_tokens,
                            "output_tokens": result.output_tokens,
                            "thinking_tokens": result.thinking_tokens,
                            "total_tokens": result.total_tokens,
                        }
                        if result.total_time_ms is not None:
                            detail["total_time_ms"] = round(result.total_time_ms, 1)
                        if result.tokens_per_second is not None:
                            detail["tokens_per_second"] = round(result.tokens_per_second, 2)
                        question_details.append(detail)
                        correct += int(score)
                    progress.advance(task)

            if scored == 0:
                console.print(
                    f"  [yellow]{bench.display_name}: no successful responses; omitted.[/yellow]"
                )
                continue

            score_pct = correct / scored
            agg = _aggregate(
                [detail for detail in question_details if detail["status"] == "scored"]
            )

            results[model.name][bname] = {
                "score": score_pct,
                "correct": correct,
                "total": scored,
                "requested": requested,
                "failed": failed,
                **agg,
                "questions": question_details,
            }
            console.print(
                f"  [green]{bench.display_name}:[/green] "
                f"{correct}/{scored} ({score_pct:.0%})"
                + (f" [dim]{failed} errors excluded[/dim]" if failed else "")
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
                "split": bconf.split,
                "revision": bconf.revision,
                "num_samples": bconf.num_samples,
                "category": config.benchmark_category(bname),
            }
            for bname, bconf in config.enabled_benchmarks().items()
        },
        "models": results,
    }

    def write_json_atomically(destination: Path) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(payload, temp_file, indent=2)
            temp_file.write("\n")
        temp_path.replace(destination)

    write_json_atomically(path)
    latest = out / "latest.json"
    write_json_atomically(latest)
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
        bench_scores = _benchmark_scores(mdata)
        overall = config.overall_score(bench_scores)
        row = [mname]
        for bname in bench_names:
            score = bench_scores.get(bname)
            row.append(f"{score:.0%}" if score is not None else "N/A")
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
        completed = [mdata[b] for b in bench_names if b in mdata]
        tin = sum(result.get("input_tokens", 0) for result in completed)
        tout = sum(result.get("output_tokens", 0) for result in completed)
        tthink = sum(result.get("thinking_tokens", 0) for result in completed)
        ttot = sum(result.get("total_tokens", 0) for result in completed)
        out_pct = f"{tout / ttot:.0%}" if ttot else "—"
        tps_vals = [
            result.get("avg_tokens_per_second")
            for result in completed
            if result.get("avg_tokens_per_second") is not None
        ]
        avg_tps = f"{sum(tps_vals) / len(tps_vals):.1f}" if tps_vals else "—"
        ttable.add_row(mname, f"{tin:,}", f"{tout:,}", f"{tthink:,}", f"{ttot:,}", out_pct, avg_tps)

    console.print(ttable)


def main():
    parser = argparse.ArgumentParser(description="Lite Benchmarks — LLM evaluation suite")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--benchmarks", nargs="+", help="Run only these benchmarks")
    parser.add_argument(
        "--models", nargs="+", help="Model IDs (litellm format) or configured names"
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Regenerate README + charts from latest results (no API calls)",
    )
    parser.add_argument("--list", action="store_true", help="List configured models and benchmarks")
    parser.add_argument(
        "--allow-unsafe-code-execution",
        action="store_true",
        help="Run code benchmarks locally. Use only inside an isolated sandbox.",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    if args.allow_unsafe_code_execution:
        config = replace(
            config,
            settings=replace(config.settings, allow_unsafe_code_execution=True),
        )

    if args.list:
        console.print("\n[bold]Configured models:[/bold]")
        for m in config.models:
            console.print(f"  {m.id}  ({m.name})")
        console.print("\n[bold]Benchmarks:[/bold]")
        for bname, bconf in config.enabled_benchmarks().items():
            cat = config.benchmark_category(bname)
            console.print(f"  {bname}  [{cat}]  {bconf.dataset}  (n={bconf.num_samples})")
        console.print(
            '\n[dim]Tip: --models accepts any litellm ID, e.g. --models "lm_studio/my-model"[/dim]'
        )
        return

    if args.generate_only:
        results = load_latest_results(config)
        if results is None:
            console.print("[red]No results found. Run benchmarks first.[/red]")
            sys.exit(1)
    else:
        if not config.models and not args.models:
            parser.error("No models are configured. Add one to config.yaml or pass --models.")
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
