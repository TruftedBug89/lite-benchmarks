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
import time
from dataclasses import replace
from datetime import datetime, timezone
import concurrent.futures
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


def _is_fatal_error(error: Exception) -> bool:
    err_type = type(error).__name__
    err_msg = str(error).lower()

    fatal_types = (
        "NotFoundError",
        "AuthenticationError",
        "PermissionDeniedError",
        "InvalidRequestError",
        "BadRequestError",
        "UnprocessableEntityError",
    )
    fatal_keywords = (
        "not found",
        "404",
        "invalid api key",
        "unauthorized",
        "401",
        "permission denied",
        "forbidden",
        "403",
        "is not supported",
        "unknown model",
        "does not exist",
        "invalid_model",
        "api_key_invalid",
        "no api key provided",
    )

    if any(ft in err_type for ft in fatal_types):
        return True
    if any(fk in err_msg for fk in fatal_keywords):
        return True

    return False


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
    config: Config, model_filter: list[str] | None, bench_filter: list[str] | None, existing_results: dict | None = None, quant: str | None = None, thinking: str | None = None
) -> dict:
    if existing_results is None:
        existing_results = {}
    if model_filter:
        models = []
        for mid in model_filter:
            match = next((m for m in config.models if m.id == mid or m.name == mid), None)
            base_name = match.name if match else mid
            name = f"{base_name} ({quant})" if quant else base_name
            
            if match:
                if thinking:
                    models.append(replace(match, name=name, thinking_effort=thinking))
                else:
                    models.append(replace(match, name=name))
            else:
                models.append(ModelConfig(id=mid, name=name, thinking_effort=thinking))
    else:
        models = []
        for m in config.models:
            name = f"{m.name} ({quant})" if quant and ("lm_studio" in m.id or "ollama" in m.id) else m.name
            models.append(replace(m, name=name))

    benchmarks = config.enabled_benchmarks()
    if bench_filter:
        benchmarks = {k: v for k, v in benchmarks.items() if k in bench_filter}
        if not benchmarks:
            console.print("[red]No benchmarks matched the filter.[/red]")
            sys.exit(1)

    results: dict[str, dict] = existing_results.copy()

    for model in models:
        console.print(f"\n[bold blue]━━━ {model.name} ({model.id}) ━━━[/bold blue]")
        if model.name not in results:
            results[model.name] = {"model_id": model.id}
        if model.thinking_effort:
            results[model.name]["thinking_effort"] = model.thinking_effort

        skip_model = False

        for bname, bconf in benchmarks.items():
            if skip_model:
                break
            bench = create_benchmark(bname, bconf, config.settings)
            if bench.requires_code_execution and not config.settings.allow_unsafe_code_execution:
                console.print(
                    f"  [yellow]Skipped {bench.display_name}: it executes untrusted generated "
                    "code. Re-run with --unsafe only inside an isolated sandbox.[/yellow]"
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
            fatal_error_encountered = False

            def process_question(qi: int, q: dict):
                prompt = bench.format_prompt(q)
                max_retries = max(3, config.settings.max_retries)
                last_error = None

                for attempt in range(1, max_retries + 1):
                    try:
                        result = generate(model, prompt, config.settings)
                        score = bench.evaluate(q, result.text)
                        return qi, True, result, score, None, False
                    except Exception as error:
                        last_error = error
                        err_type = type(error).__name__
                        err_msg = str(error).strip()

                        if _is_fatal_error(error):
                            console.print(
                                f"  [bold red]❌ FATAL ERROR on Q{qi+1} ({err_type}): {err_msg}[/bold red]"
                            )
                            return qi, False, None, 0.0, error, True

                        is_ratelimit = (
                            "RateLimitError" in err_type
                            or "429" in err_msg
                            or "rate limit" in err_msg.lower()
                            or "too many requests" in err_msg.lower()
                        )

                        if attempt < max_retries:
                            wait_time = 60 if is_ratelimit else min(30, 5 * attempt)
                            reason = "rate limited" if is_ratelimit else f"error ({err_type})"
                            console.print(
                                f"  [yellow]⚠️ Q{qi+1} {reason}: retrying ({attempt}/{max_retries}) in {wait_time}s...[/yellow]"
                            )
                            time.sleep(wait_time)
                        else:
                            console.print(
                                f"  [red]❌ Q{qi+1} failed after {max_retries} attempts ({err_type}: {err_msg})[/red]"
                            )

                return qi, False, None, 0.0, last_error, False

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"  {bench.display_name}", total=requested)
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=config.settings.max_concurrency) as executor:
                    futures = {executor.submit(process_question, qi, q): qi for qi, q in enumerate(questions)}
                    for future in concurrent.futures.as_completed(futures):
                        qi, success, result, score, error, is_fatal = future.result()
                        if is_fatal:
                            fatal_error_encountered = True
                            skip_model = True
                            for f in futures:
                                f.cancel()
                        if success:
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
                            correct += float(score)
                        else:
                            failed += 1
                            question_details.append(
                                {
                                    "question_index": qi,
                                    "status": "error",
                                    "error_type": type(error).__name__,
                                    "error_msg": str(error),
                                }
                            )
                        progress.advance(task)
                        if fatal_error_encountered:
                            console.print(
                                f"  [bold red]Aborting {bench.display_name} and skipping remaining benchmarks for {model.name} due to fatal model error.[/bold red]"
                            )
                            break

            question_details.sort(key=lambda x: x["question_index"])

            if scored == 0:
                first_err = next(
                    (d.get("error_msg") for d in question_details if d.get("status") == "error"),
                    "Unknown error"
                )
                console.print(
                    f"  [yellow]{bench.display_name}: no successful responses; omitted. Reason: {first_err}[/yellow]"
                )
                continue

            score_pct = correct / scored
            agg = _aggregate(
                [detail for detail in question_details if detail["status"] == "scored"]
            )

            results[model.name][bname] = {
                "score": score_pct,
                "correct": round(correct, 4),
                "total": scored,
                "requested": requested,
                "failed": failed,
                **agg,
                "questions": question_details,
            }
            console.print(
                f"  [green]{bench.display_name}:[/green] "
                f"{round(correct, 2)}/{scored} ({score_pct:.0%})"
                + (f" [dim]{failed} errors excluded[/dim]" if failed else "")
            )

        # Checkpoint partial results after each model
        try:
            save_results(results, config)
        except Exception:
            pass

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
    parser.add_argument(
        "--quant", 
        help="Quantization info for local models (e.g. Q4_K_M), appended to model name"
    )
    parser.add_argument(
        "--thinking",
        help="Set reasoning effort dynamically (e.g., max, high, low) for all models run"
    )
    parser.add_argument("--list", action="store_true", help="List configured models and benchmarks")
    parser.add_argument(
        "--unsafe",
        action="store_true",
        help="Run code benchmarks locally. Use only inside an isolated sandbox.",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    if args.unsafe:
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
        existing_results = load_latest_results(config) or {}
        if not config.models and not args.models:
            parser.error("No models are configured. Add one to config.yaml or pass --models.")
        results = run_benchmarks(config, args.models, args.benchmarks, existing_results, args.quant, args.thinking)
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
