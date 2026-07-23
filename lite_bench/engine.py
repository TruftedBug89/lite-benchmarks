"""The unified benchmark execution engine."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol

from rich.console import Console

from .benchmarks import create_benchmark
from .config import BenchmarkConfig, Config, ModelConfig, Settings
from .providers import GenerationResult, generate
from .results_store import aggregate, compute_question_hash, is_fatal_error, save_results

console = Console()


class EngineCallbacks(Protocol):
    def on_event(self, model_name: str, message: str) -> None: ...
    def on_question_done(self, model_name: str, bench_name: str, detail: dict) -> None: ...
    def on_benchmark_done(self, model_name: str, bench_name: str, summary: dict) -> None: ...
    def on_model_done(self, model_name: str) -> None: ...


class DefaultEngineCallbacks:
    def on_event(self, model_name: str, message: str) -> None:
        console.print(f"[{model_name}] {message}")

    def on_question_done(self, model_name: str, bench_name: str, detail: dict) -> None:
        pass

    def on_benchmark_done(self, model_name: str, bench_name: str, summary: dict) -> None:
        pass

    def on_model_done(self, model_name: str) -> None:
        pass


class FatalModelError(Exception):
    """Raised when an unrecoverable account/auth error occurs for a model."""


def process_question(
    qi: int,
    q: dict,
    bench_obj: any,
    model: ModelConfig,
    settings: Settings,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Process a single question with separate provider-retry and evaluation logic."""
    if should_stop and should_stop():
        return {
            "question_id": qi,
            "status": "cancelled",
            "score": 0.0,
            "prompt": "",
            "response": "",
        }

    prompt = bench_obj.format_prompt(q)
    max_retries = max(1, settings.max_retries)
    gen_result: GenerationResult | None = None
    gen_error: Exception | None = None

    for attempt in range(max_retries):
        if should_stop and should_stop():
            break
        try:
            gen_result = generate(model, prompt, settings)
            gen_error = None
            break
        except Exception as e:
            gen_error = e
            if is_fatal_error(e):
                raise FatalModelError(f"Fatal error for {model.name}: {e}") from e

            if attempt < max_retries - 1:
                err_str = str(e).lower()
                backoff = min(60.0, 5.0 * (2**attempt)) + random.uniform(0, 5.0)
                if "429" in err_str or "rate limit" in err_str:
                    backoff = max(60.0, backoff)
                time.sleep(backoff)

    if gen_result is None:
        err_msg = str(gen_error) if gen_error else "Max retries exceeded"
        err_type = type(gen_error).__name__ if gen_error else "ProviderError"
        return {
            "question_id": qi,
            "status": "error",
            "error_type": err_type,
            "error_msg": err_msg,
            "score": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "total_tokens": 0,
            "total_time_ms": None,
            "tokens_per_second": None,
            "cost_usd": None,
            "prompt": prompt[:8000],
            "response": f"[Error: {err_msg}]",
        }

    # Separate evaluation try-block — evaluation errors score 0.0 and are never retried against provider
    try:
        score = float(bench_obj.evaluate(q, gen_result.text))
        status = "success"
        err_msg = None
    except Exception as eval_exc:
        score = 0.0
        status = "eval_error"
        err_msg = str(eval_exc)
        console.print(
            f"[bold red]Scorer error on benchmark {bench_obj.name} Q#{qi}: {eval_exc}[/bold red]"
        )

    res_dict: dict = {
        "question_id": qi,
        "status": status,
        "score": score,
        "input_tokens": gen_result.input_tokens,
        "output_tokens": gen_result.output_tokens,
        "thinking_tokens": gen_result.thinking_tokens,
        "total_tokens": gen_result.total_tokens,
        "total_time_ms": gen_result.total_time_ms,
        "tokens_per_second": gen_result.tokens_per_second,
        "cost_usd": gen_result.cost_usd,
        "prompt": prompt[:8000],
        "response": gen_result.text[:8000],
    }
    if err_msg:
        res_dict["error_msg"] = err_msg
    return res_dict


def run_engine(
    config: Config,
    models: list[ModelConfig],
    benchmarks: dict[str, BenchmarkConfig],
    callbacks: EngineCallbacks | None = None,
    should_stop: Callable[[], bool] | None = None,
    allow_unsafe: bool = False,
    existing_results: dict | None = None,
) -> dict:
    """Execute benchmarks across models using thread pool concurrency."""
    if callbacks is None:
        callbacks = DefaultEngineCallbacks()

    results: dict = dict(existing_results) if existing_results else {}
    settings = config.settings

    for model in models:
        if should_stop and should_stop():
            break

        callbacks.on_event(model.name, f"Starting model run for {model.name}")
        mdata = results.setdefault(
            model.name,
            {
                "model_id": model.id,
                "thinking_effort": model.thinking_effort,
            },
        )

        fatal_encountered = False

        for bench_name, bench_cfg in benchmarks.items():
            if should_stop and should_stop() or fatal_encountered:
                break

            try:
                bench_obj = create_benchmark(bench_name, bench_cfg, settings)
            except ValueError as e:
                callbacks.on_event(model.name, f"Skipping benchmark {bench_name}: {e}")
                continue

            # Code execution safety check
            if bench_obj.requires_code_execution and not (
                allow_unsafe or settings.allow_unsafe_code_execution
            ):
                callbacks.on_event(
                    model.name,
                    f"Skipping {bench_name} (code execution requires explicit unsafe opt-in)",
                )
                continue

            try:
                questions = bench_obj.load()
            except Exception as e:
                callbacks.on_event(model.name, f"Failed to load dataset for {bench_name}: {e}")
                continue

            if not questions:
                callbacks.on_event(model.name, f"No questions loaded for {bench_name}")
                continue

            q_hash = compute_question_hash(questions)
            callbacks.on_event(
                model.name,
                f"Running {bench_obj.display_name} ({len(questions)} questions)...",
            )

            details: list[dict] = []
            max_concurrency = max(1, settings.max_concurrency)

            with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                futures = {
                    executor.submit(
                        process_question,
                        qi,
                        q,
                        bench_obj,
                        model,
                        settings,
                        should_stop,
                    ): (qi, q)
                    for qi, q in enumerate(questions)
                }

                try:
                    for future in as_completed(futures):
                        if should_stop and should_stop():
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
                        try:
                            detail = future.result()
                            details.append(detail)
                            callbacks.on_question_done(model.name, bench_name, detail)
                        except FatalModelError as fe:
                            callbacks.on_event(model.name, f"[CRITICAL] {fe}")
                            fatal_encountered = True
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
                        except Exception as exc:
                            qi, _ = futures[future]
                            err_detail = {
                                "question_id": qi,
                                "status": "error",
                                "error_msg": str(exc),
                                "score": 0.0,
                            }
                            details.append(err_detail)
                            callbacks.on_question_done(model.name, bench_name, err_detail)
                except Exception as loop_err:
                    callbacks.on_event(model.name, f"Execution error in benchmark pool: {loop_err}")

            if fatal_encountered:
                callbacks.on_event(
                    model.name, f"Aborting remaining benchmarks for {model.name} due to fatal error"
                )

            # Summarize benchmark
            scored = [d for d in details if d["status"] in ("success", "eval_error")]
            correct_count = sum(d["score"] for d in scored)
            total_count = len(scored)
            score = (correct_count / total_count) if total_count > 0 else 0.0

            b_summary: dict = {
                "score": round(score, 4),
                "correct": correct_count,
                "total": total_count,
                "question_hash": q_hash,
                "input_tokens": sum(d.get("input_tokens", 0) for d in details),
                "output_tokens": sum(d.get("output_tokens", 0) for d in details),
                "thinking_tokens": sum(d.get("thinking_tokens", 0) for d in details),
                "total_tokens": sum(d.get("total_tokens", 0) for d in details),
                "details": details,
            }

            tps_vals = [d["tokens_per_second"] for d in details if d.get("tokens_per_second") is not None]
            if tps_vals:
                b_summary["avg_tokens_per_second"] = round(sum(tps_vals) / len(tps_vals), 2)

            time_vals = [d["total_time_ms"] for d in details if d.get("total_time_ms") is not None]
            if time_vals:
                b_summary["avg_time_ms"] = round(sum(time_vals) / len(time_vals), 1)

            cost_vals = [d["cost_usd"] for d in details if d.get("cost_usd") is not None]
            if cost_vals:
                b_summary["total_cost_usd"] = round(sum(cost_vals), 6)

            mdata[bench_name] = b_summary
            aggregate(mdata, config)

            # Checkpoint save
            save_results(results, config, settings.results_dir, "latest.json")
            callbacks.on_benchmark_done(model.name, bench_name, b_summary)

            if fatal_encountered:
                break

        callbacks.on_model_done(model.name)

    return results
