#!/usr/bin/env python
"""Lite Benchmarks — Live Visual Multi-Model Dashboard.

Runs up to N models concurrently in real-time with an interactive, animated terminal UI.

Usage:
    py run_dashboard.py                          # all models, up to 4 concurrent
    py run_dashboard.py --concurrency 4          # run 4 models in parallel
    py run_dashboard.py --benchmarks gsm8k math_500
    py run_dashboard.py --models deepseek/deepseek-chat gemini/gemini-2.5-flash
    py run_dashboard.py --unsafe                 # enable local code execution benchmarks
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass, field
import math
import sys
import time

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from lite_bench.benchmarks import create_benchmark
from lite_bench.charts import generate_all as generate_charts
from lite_bench.config import Config, ModelConfig, Settings, load_config
from lite_bench.providers import generate
from lite_bench.readme_gen import write_readme
from run_benchmark import _aggregate, _is_fatal_error, save_results

console = Console()


@dataclass
class ModelState:
    model: ModelConfig
    status: str = "Initializing..."
    current_benchmark: str = "—"
    current_benchmark_display: str = "—"
    q_index: int = 0
    q_total: int = 0
    correct: float = 0.0
    scored: int = 0
    failed: int = 0
    latest_score: float | None = None
    latest_snippet: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0
    avg_tps: float | None = None
    total_time_ms: float = 0.0
    benchmark_scores: dict[str, float] = field(default_factory=dict)
    question_details: dict[str, list[dict]] = field(default_factory=dict)
    is_finished: bool = False
    is_failed: bool = False
    fatal_error: str | None = None
    start_time: float = field(default_factory=time.time)
    finish_time: float | None = None
    recent_events: list[str] = field(default_factory=list)

    def add_event(self, msg: str) -> None:
        self.recent_events.append(msg)
        if len(self.recent_events) > 5:
            self.recent_events.pop(0)

    @property
    def accuracy(self) -> float | None:
        return (self.correct / self.scored) if self.scored > 0 else None


class DashboardRunner:
    def __init__(self, config: Config, model_workers: int = 4):
        self.config = config
        self.model_workers = model_workers
        self.states: dict[str, ModelState] = {}
        self.start_time = time.time()
        self.results: dict[str, dict] = {}

    def run(
        self,
        model_filter: list[str] | None = None,
        bench_filter: list[str] | None = None,
        quant: str | None = None,
        thinking: str | None = None,
    ) -> dict:
        # Resolve target models
        if model_filter:
            models = []
            for mid in model_filter:
                match = next((m for m in self.config.models if m.id == mid or m.name == mid), None)
                base_name = match.name if match else mid
                name = f"{base_name} ({quant})" if quant else base_name
                if match:
                    models.append(
                        ModelConfig(
                            id=match.id,
                            name=name,
                            thinking_effort=thinking or match.thinking_effort,
                            extra_params=match.extra_params,
                        )
                    )
                else:
                    models.append(ModelConfig(id=mid, name=name, thinking_effort=thinking))
        else:
            models = list(self.config.models)

        # Resolve target benchmarks
        benchmarks = self.config.enabled_benchmarks()
        if bench_filter:
            benchmarks = {k: v for k, v in benchmarks.items() if k in bench_filter}
            if not benchmarks:
                console.print("[red]No benchmarks matched the filter.[/red]")
                sys.exit(1)

        # Initialize model states
        for m in models:
            self.states[m.name] = ModelState(model=m)
            self.results[m.name] = {"model_id": m.id}
            if m.thinking_effort:
                self.results[m.name]["thinking_effort"] = m.thinking_effort

        # Run concurrent model workers with live UI rendering
        with Live(self._render(), refresh_per_second=10, console=console) as live:

            def run_single_model(mstate: ModelState):
                mconfig = mstate.model
                mstate.status = "🚀 Starting benchmarks..."

                for bname, bconf in benchmarks.items():
                    if mstate.is_failed:
                        break

                    bench = create_benchmark(bname, bconf, self.config.settings)
                    mstate.current_benchmark = bname
                    mstate.current_benchmark_display = bench.display_name

                    if bench.requires_code_execution and not self.config.settings.allow_unsafe_code_execution:
                        mstate.add_event(f"Skipped {bench.display_name} (code execution disabled)")
                        continue

                    try:
                        questions = bench.load()
                    except Exception as error:
                        mstate.add_event(f"Failed to load dataset for {bname}: {type(error).__name__}")
                        continue

                    mstate.q_index = 0
                    mstate.q_total = len(questions)
                    b_correct = 0.0
                    b_scored = 0
                    b_failed = 0
                    b_details: list[dict] = []
                    b_fatal = False

                    def process_question(qi: int, q: dict):
                        nonlocal b_fatal
                        if b_fatal or mstate.is_failed:
                            return qi, False, None, 0.0, None, True

                        prompt = bench.format_prompt(q)
                        max_retries = max(3, self.config.settings.max_retries)

                        for attempt in range(1, max_retries + 1):
                            mstate.status = f"🧠 Processing Q{qi+1}/{mstate.q_total} (attempt {attempt})"
                            try:
                                result = generate(mconfig, prompt, self.config.settings)
                                score = bench.evaluate(q, result.text)
                                return qi, True, result, score, None, False
                            except Exception as error:
                                err_type = type(error).__name__
                                err_msg = str(error).strip()

                                if _is_fatal_error(error):
                                    mstate.add_event(f"❌ FATAL Q{qi+1} ({err_type}): {err_msg[:40]}...")
                                    return qi, False, None, 0.0, error, True

                                is_ratelimit = (
                                    "RateLimitError" in err_type
                                    or "429" in err_msg
                                    or "rate limit" in err_msg.lower()
                                )
                                if attempt < max_retries:
                                    wait_time = 60 if is_ratelimit else min(30, 5 * attempt)
                                    mstate.add_event(f"⚠️ Q{qi+1} retry in {wait_time}s ({err_type})")
                                    time.sleep(wait_time)
                                else:
                                    mstate.add_event(f"❌ Q{qi+1} failed after {max_retries} tries")
                                    return qi, False, None, 0.0, error, False

                        return qi, False, None, 0.0, Exception("Unknown error"), False

                    # Process questions per benchmark
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=self.config.settings.max_concurrency
                    ) as q_executor:
                        q_futures = {
                            q_executor.submit(process_question, qi, q): qi
                            for qi, q in enumerate(questions)
                        }
                        for q_future in concurrent.futures.as_completed(q_futures):
                            qi, success, result, score, error, is_fatal = q_future.result()
                            mstate.q_index += 1

                            if is_fatal:
                                b_fatal = True
                                mstate.is_failed = True
                                mstate.fatal_error = str(error)
                                mstate.status = "❌ FATAL ERROR (Aborted)"
                                for f in q_futures:
                                    f.cancel()

                            if success and result is not None:
                                b_scored += 1
                                mstate.scored += 1
                                b_correct += score
                                mstate.correct += score
                                mstate.input_tokens += result.input_tokens
                                mstate.output_tokens += result.output_tokens
                                mstate.thinking_tokens += result.thinking_tokens
                                mstate.total_tokens += result.total_tokens

                                snippet = result.text.strip().replace("\n", " ")
                                mstate.latest_snippet = snippet[:100] + ("..." if len(snippet) > 100 else "")
                                mstate.latest_score = score

                                if score == 1.0:
                                    mstate.add_event(f"✨ Q{qi+1} Correct! (+1.0)")
                                    mstate.status = f"✨ Q{qi+1} Correct!"
                                elif score == 0.0:
                                    mstate.add_event(f"❌ Q{qi+1} Incorrect (0.0)")
                                    mstate.status = f"❌ Q{qi+1} Incorrect"
                                else:
                                    mstate.add_event(f"⭐ Q{qi+1} Partial (+{score:.2f})")
                                    mstate.status = f"⭐ Q{qi+1} Partial"

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
                                b_details.append(detail)

                                # Update TPS
                                if result.tokens_per_second:
                                    if mstate.avg_tps is None:
                                        mstate.avg_tps = result.tokens_per_second
                                    else:
                                        mstate.avg_tps = (mstate.avg_tps * 0.8) + (result.tokens_per_second * 0.2)
                            else:
                                b_failed += 1
                                mstate.failed += 1
                                b_details.append(
                                    {
                                        "question_index": qi,
                                        "status": "error",
                                        "error_type": type(error).__name__ if error else "Error",
                                        "error_msg": str(error) if error else "",
                                    }
                                )

                            live.update(self._render())
                            if b_fatal:
                                break

                    if b_scored > 0:
                        b_score = b_correct / b_scored
                        mstate.benchmark_scores[bname] = b_score
                        agg = _aggregate([d for d in b_details if d["status"] == "scored"])
                        self.results[mconfig.name][bname] = {
                            "score": b_score,
                            "correct": round(b_correct, 4),
                            "total": b_scored,
                            "requested": len(questions),
                            "failed": b_failed,
                            **agg,
                            "questions": sorted(b_details, key=lambda x: x["question_index"]),
                        }
                        mstate.add_event(f"🏆 Completed {bench.display_name}: {b_score:.0%}")
                    elif b_fatal:
                        mstate.add_event(f"💥 Aborted {bench.display_name} (Fatal)")
                        break

                mstate.is_finished = True
                mstate.finish_time = time.time()
                if not mstate.is_failed:
                    mstate.status = "✅ Benchmarks Complete"

            # Launch concurrent model execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.model_workers) as model_executor:
                model_futures = [
                    model_executor.submit(run_single_model, mstate)
                    for mstate in self.states.values()
                ]

                # Update live display periodically until all models finish
                while any(not f.done() for f in model_futures):
                    live.update(self._render())
                    time.sleep(0.1)

            live.update(self._render())

        return self.results

    def _make_progress_bar(self, current: int, total: int, width: int = 16) -> str:
        if total <= 0:
            return "░" * width
        filled = int(round((current / total) * width))
        filled = max(0, min(width, filled))
        return "█" * filled + "░" * (width - filled)

    def _render(self) -> Panel:
        elapsed = int(time.time() - self.start_time)
        mins, secs = divmod(elapsed, 60)
        time_str = f"{mins:02d}:{secs:02d}"

        # Header banner
        total_tokens = sum(s.total_tokens for s in self.states.values())
        active_models = sum(1 for s in self.states.values() if not s.is_finished and not s.is_failed)
        finished_models = sum(1 for s in self.states.values() if s.is_finished and not s.is_failed)
        failed_models = sum(1 for s in self.states.values() if s.is_failed)

        header_text = Text()
        header_text.append(" 🏆 LITE BENCHMARKS ", style="bold gold1 on blue")
        header_text.append(" │ ", style="dim")
        header_text.append(f"⏱️ {time_str} ", style="bold white")
        header_text.append(" │ ", style="dim")
        header_text.append(f"⚡ {active_models} Active ", style="bold cyan")
        header_text.append(f"│ ✅ {finished_models} Done ", style="bold green")
        if failed_models > 0:
            header_text.append(f"│ ❌ {failed_models} Failed ", style="bold red")
        header_text.append(" │ ", style="dim")
        header_text.append(f"🪙 {total_tokens:,} Tokens", style="bold yellow")

        # Model Cards Grid (Up to 4 models rendered in visual cards)
        model_panels = []
        for name, s in self.states.items():
            if s.is_failed:
                border_style = "red"
                icon = "❌"
            elif s.is_finished:
                border_style = "green"
                icon = "✅"
            else:
                border_style = "cyan"
                icon = "🏃"

            acc_str = f"{s.accuracy:.1%}" if s.accuracy is not None else "N/A"
            pbar = self._make_progress_bar(s.q_index, s.q_total, width=14)
            pct = (s.q_index / s.q_total * 100) if s.q_total > 0 else 0.0

            card = Text()
            card.append(f"Bench: ", style="dim")
            card.append(f"{s.current_benchmark_display}\n", style="bold white")
            card.append(f"Progress: [{pbar}] {pct:.0f}%\n", style="cyan")
            card.append(f"Score: ", style="dim")
            card.append(f"{s.correct:.1f}/{s.scored} ({acc_str})\n", style="bold yellow")
            card.append(f"Speed: ", style="dim")
            card.append(f"{s.avg_tps:.1f} tok/s\n" if s.avg_tps else "—\n", style="magenta")
            card.append(f"Status: ", style="dim")
            card.append(f"{s.status}\n", style="bold italic bright_white")

            if s.latest_snippet:
                card.append(f"Output: ", style="dim")
                card.append(f'"{s.latest_snippet}"\n', style="dim italic green")

            if s.recent_events:
                card.append(f"Latest: {s.recent_events[-1]}", style="dim yellow")

            panel = Panel(
                card,
                title=f"{icon} {name}",
                subtitle=f"Model ID: {s.model.id}",
                border_style=border_style,
                padding=(0, 1),
            )
            model_panels.append(panel)

        # Arrange cards into 2 columns if 2+ models
        if len(model_panels) == 1:
            cards_renderable = model_panels[0]
        else:
            half = math.ceil(len(model_panels) / 2)
            col1 = model_panels[:half]
            col2 = model_panels[half:]

            t_grid = Table.grid(expand=True)
            t_grid.add_column(ratio=1)
            t_grid.add_column(ratio=1)

            max_rows = max(len(col1), len(col2))
            for i in range(max_rows):
                r1 = col1[i] if i < len(col1) else Text("")
                r2 = col2[i] if i < len(col2) else Text("")
                t_grid.add_row(r1, r2)
            cards_renderable = t_grid

        # Leaderboard Table
        lb_table = Table(title="🏅 LIVE LEADERBOARD", expand=True, show_header=True, header_style="bold magenta")
        lb_table.add_column("Rank", justify="center", style="bold")
        lb_table.add_column("Model", style="bold cyan")
        lb_table.add_column("Score", justify="right", style="bold green")
        lb_table.add_column("Questions", justify="right")
        lb_table.add_column("Tokens", justify="right")
        lb_table.add_column("Speed", justify="right")
        lb_table.add_column("Status", justify="center")

        sorted_states = sorted(
            self.states.values(),
            key=lambda s: (s.accuracy if s.accuracy is not None else -1.0, s.scored),
            reverse=True,
        )

        medals = {1: "🥇 1st", 2: "🥈 2nd", 3: "🥉 3rd"}
        for rank, s in enumerate(sorted_states, 1):
            r_str = medals.get(rank, f"{rank}th")
            acc_val = f"{s.accuracy:.1%}" if s.accuracy is not None else "N/A"
            q_str = f"{s.scored}/{s.q_total}" if s.q_total else "0/0"
            tok_str = f"{s.total_tokens:,}"
            tps_str = f"{s.avg_tps:.1f}/s" if s.avg_tps else "—"

            if s.is_failed:
                st_str = "[bold red]❌ Failed[/bold red]"
            elif s.is_finished:
                st_str = "[bold green]✅ Complete[/bold green]"
            else:
                st_str = "[bold cyan]🏃 Running[/bold cyan]"

            lb_table.add_row(r_str, s.model.name, acc_val, q_str, tok_str, tps_str, st_str)

        # Main Layout assembly
        main_table = Table.grid(expand=True)
        main_table.add_row(Panel(header_text, border_style="blue", padding=(0, 1)))
        main_table.add_row(cards_renderable)
        main_table.add_row(Panel(lb_table, border_style="magenta", padding=(0, 1)))

        return Panel(main_table, border_style="bold blue", title="✨ Lite Benchmarks Live Visual Runner ✨")


def main():
    parser = argparse.ArgumentParser(description="Lite Benchmarks — Live Visual Dashboard")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--benchmarks", nargs="+", help="Run only these benchmarks")
    parser.add_argument("--models", nargs="+", help="Model IDs or configured display names")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent model workers (default 4)")
    parser.add_argument("--quant", help="Quantization info for local models")
    parser.add_argument("--thinking", help="Set reasoning effort dynamically")
    parser.add_argument("--unsafe", action="store_true", help="Allow unsafe local code execution benchmarks")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    if args.unsafe:
        config = Config(
            models=config.models,
            benchmarks=config.benchmarks,
            categories=config.categories,
            settings=Settings(
                seed=config.settings.seed,
                max_tokens=config.settings.max_tokens,
                temperature=config.settings.temperature,
                request_timeout=config.settings.request_timeout,
                code_exec_timeout=config.settings.code_exec_timeout,
                max_retries=config.settings.max_retries,
                max_concurrency=config.settings.max_concurrency,
                results_dir=config.settings.results_dir,
                charts_dir=config.settings.charts_dir,
                hf_token_env=config.settings.hf_token_env,
                allow_unsafe_code_execution=True,
            ),
        )

    runner = DashboardRunner(config, model_workers=args.concurrency)
    results = runner.run(args.models, args.benchmarks, args.quant, args.thinking)

    console.print("\n[bold green]✅ All concurrent benchmarks completed![/bold green]")
    path = save_results(results, config)
    console.print(f"[dim]Results saved to {path}[/dim]")

    console.print("[bold]Generating charts...[/bold]")
    chart_paths = generate_charts(results, config, config.settings.charts_dir)
    console.print(f"  {len(chart_paths)} charts saved to {config.settings.charts_dir}/")

    console.print("[bold]Generating README...[/bold]")
    write_readme(results, config, chart_paths)
    console.print("  README.md updated")


if __name__ == "__main__":
    main()
