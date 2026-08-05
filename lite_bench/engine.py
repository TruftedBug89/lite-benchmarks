"""The unified benchmark execution engine."""

from __future__ import annotations

import random
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Protocol

from rich.console import Console

from .benchmarks import create_benchmark
from .config import BenchmarkConfig, Config, ModelConfig, Settings
from .logging_utils import get_logger, log_ref, run_log_rel, scrub
from .providers import GenerationResult, generate
from .results_store import aggregate, compute_question_hash, is_fatal_error, save_results

console = Console()
log = get_logger("engine")

# Permanent client/context errors that will never succeed on retry. Auth/quota
# errors are handled separately as fatal (they abort the whole model). Word
# boundaries keep e.g. "4000 tokens" from matching the "400" status code.
_NON_RETRYABLE_STATUS = re.compile(r"\b(?:400|402|403|404|405|406|409|410|413|415|422)\b")
_NON_RETRYABLE_HINTS = (
    "context_length_exceeded",
    "maximum context length",
    "context length",
    "context_window",
    "too many tokens",
    "reduce the length",
    "content_filter",
    "content_policy",
    "content policy",
    "responsibleaipolicy",
    "invalid_request_error",
    "invalid request",
)


def _is_retryable_error(e: Exception) -> bool:
    """False for errors that can never succeed (bad request, prompt too long,
    content filter, refusals) so we don't burn retries + backoff on them."""
    s = str(e).lower()
    if _NON_RETRYABLE_STATUS.search(s):
        return False
    return not any(hint in s for hint in _NON_RETRYABLE_HINTS)


class EngineCallbacks(Protocol):
    def on_event(self, model_name: str, message: str) -> None: ...
    def on_benchmark_start(self, model_name: str, bench_name: str, total_questions: int) -> None: ...
    def on_question_retry(self, model_name: str, bench_name: str, info: dict) -> None: ...
    def on_question_done(self, model_name: str, bench_name: str, detail: dict) -> None: ...
    def on_benchmark_done(self, model_name: str, bench_name: str, summary: dict) -> None: ...
    def on_model_done(self, model_name: str) -> None: ...


class DefaultEngineCallbacks:
    def on_event(self, model_name: str, message: str) -> None:
        console.print(f"[{model_name}] {message}{log_ref()}")
        log.info(f"[{model_name}] {scrub(message)}")

    def on_benchmark_start(self, model_name: str, bench_name: str, total_questions: int) -> None:
        log.info(
            f"[{model_name}] benchmark {bench_name} starting: {total_questions} questions"
        )

    def on_question_retry(self, model_name: str, bench_name: str, info: dict) -> None:
        console.print(
            f"[dim][{model_name}] Q#{info.get('question_id', 0)} attempt "
            f"{info.get('attempt', '?')} failed ({info.get('error', '')}); "
            f"retrying in {info.get('backoff', 0):.0f}s…[/dim]{log_ref()}"
        )
        log.warning(
            f"[{model_name}] [{bench_name}] Q#{info.get('question_id', 0)} retry "
            f"{info.get('attempt', '?')}/{info.get('max_attempts') or '∞'} after "
            f"{info.get('backoff', 0):.0f}s: {scrub(str(info.get('error', '')))}"
        )

    def on_question_done(self, model_name: str, bench_name: str, detail: dict) -> None:
        log.debug(
            f"[{model_name}] [{bench_name}] Q#{detail.get('question_id', 0)} "
            f"status={detail.get('status')} score={detail.get('score')} "
            f"in/out/think={detail.get('input_tokens', 0)}/{detail.get('output_tokens', 0)}/"
            f"{detail.get('thinking_tokens', 0)} time={detail.get('total_time_ms')}ms "
            f"tps={detail.get('tokens_per_second')} cost={detail.get('cost_usd')} "
            f"trunc={bool(detail.get('is_truncated'))} finish={detail.get('finish_reason')} "
            f"err={scrub(str(detail.get('error_msg', '')))}"
        )

    def on_benchmark_done(self, model_name: str, bench_name: str, summary: dict) -> None:
        console.print(
            f"[bold][{model_name}] {bench_name} done: score {summary.get('score')} "
            f"({summary.get('correct')}/{summary.get('total')}, "
            f"provider_err={summary.get('provider_error_count', 0)}, "
            f"eval_err={summary.get('eval_error_count', 0)}, "
            f"trunc={summary.get('truncated_count', 0)})[/bold]{log_ref()}"
        )
        log.info(
            f"[{model_name}] [{bench_name}] done: score={summary.get('score')} "
            f"correct={summary.get('correct')}/{summary.get('total')} "
            f"attempted={summary.get('attempted')}/{summary.get('total_questions')} "
            f"excluded={summary.get('excluded_count')} "
            f"provider_err={summary.get('provider_error_count', 0)} "
            f"eval_err={summary.get('eval_error_count', 0)} "
            f"trunc={summary.get('truncated_count', 0)} "
            f"tokens={summary.get('total_tokens')} "
            f"cost={summary.get('total_cost_usd')} "
            f"avg_tps={summary.get('avg_tokens_per_second')} "
            f"avg_ms={summary.get('avg_time_ms')}"
        )

    def on_model_done(self, model_name: str) -> None:
        log.info(f"[{model_name}] model run finished")


class FatalModelError(Exception):
    """Raised when an unrecoverable account/auth error occurs for a model."""


def _interruptible_sleep(seconds: float, should_stop: Callable[[], bool] | None) -> bool:
    """Sleep in 0.5s slices so Force Stop is honored mid-backoff.

    Returns True if the sleep was cut short by a stop request."""
    deadline = time.monotonic() + seconds
    while True:
        if should_stop and should_stop():
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.5, remaining))


def process_question(
    qi: int,
    q: dict,
    bench_obj: any,
    model: ModelConfig,
    settings: Settings,
    should_stop: Callable[[], bool] | None = None,
    on_retry: Callable[[str, str, dict], None] | None = None,
) -> dict:
    """Process a single question with separate provider-retry and evaluation logic.

    Retry policy: transient provider errors (timeouts, 429s, 5xx, connection
    drops, empty responses) are retried with exponential backoff until a good
    response arrives — with ``max_retries <= 0`` (the default) the question
    waits indefinitely and never advances on an error; a positive value caps
    the number of attempts. Permanent errors (prompt too long, content filter)
    can never succeed, so they give up immediately; fatal auth/quota errors
    abort the whole model. Force Stop cancels the wait at any point."""
    if should_stop and should_stop():
        return {
            "question_id": qi,
            "status": "cancelled",
            "score": 0.0,
            "prompt": "",
            "response": "",
        }

    bench_name = getattr(bench_obj, "name", "?")
    prompt = bench_obj.format_prompt(q)
    log.debug(
        f"[{model.name}] [{bench_name}] Q#{qi} start: prompt_len={len(prompt)} chars"
    )
    max_attempts: int | None = settings.max_retries if settings.max_retries > 0 else None
    gen_result: GenerationResult | None = None
    gen_error: Exception | None = None
    stopped = False
    attempt = 0
    consecutive_same_error = 0
    last_error_msg = ""

    while True:
        if should_stop and should_stop():
            stopped = True
            break
        try:
            gen_result = generate(model, prompt, settings)
            gen_error = None
            break
        except Exception as e:
            gen_error = e
            if is_fatal_error(e):
                log.error(
                    f"[{model.name}] [{bench_name}] Q#{qi} FATAL provider error: "
                    f"{scrub(str(e)[:500])}"
                )
                raise FatalModelError(f"Fatal error for {model.name}: {str(e)[:500]}") from e

            # Permanent errors (bad request, context too long, content filter)
            # will never succeed no matter how long we wait — record and move on.
            if not _is_retryable_error(e):
                log.warning(
                    f"[{model.name}] [{bench_name}] Q#{qi} permanent error "
                    f"(no retry): {scrub(str(e)[:500])}"
                )
                break

            # Only the engine's own semantic RuntimeErrors (empty completion,
            # no text content) get the "same error 3x in a row" bailout — those
            # are deterministic and would repeat forever. Transient provider
            # failures (connection errors, timeouts, 429s) must keep retrying
            # per the retry policy (max_retries=0 = wait indefinitely), not
            # silently score 0 after three identical messages.
            err_msg_str = str(e)
            if isinstance(e, RuntimeError) and err_msg_str == last_error_msg:
                consecutive_same_error += 1
            else:
                consecutive_same_error = 1
                last_error_msg = err_msg_str
            if consecutive_same_error >= 3:
                log.warning(
                    f"[{model.name}] [{bench_name}] Q#{qi} same provider error 3x "
                    f"in a row, giving up: {scrub(str(e)[:500])}"
                )
                break

            attempt += 1
            if max_attempts is not None and attempt >= max_attempts:
                break

            err_str = str(e).lower()
            backoff = min(60.0, 5.0 * (2 ** min(attempt - 1, 4)))
            if "429" in err_str or "rate limit" in err_str:
                backoff = max(60.0, backoff)
            backoff += random.uniform(0, 5.0)
            log.debug(
                f"[{model.name}] [{bench_name}] Q#{qi} attempt {attempt} failed "
                f"({scrub(str(e)[:500])}); backing off {backoff:.0f}s"
            )
            if on_retry:
                on_retry(
                    model.name,
                    bench_obj.name,
                    {
                        "question_id": qi,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "backoff": backoff,
                        "error": str(e)[:200],
                    },
                )
            if _interruptible_sleep(backoff, should_stop):
                stopped = True
                break

    if gen_result is None:
        if stopped:
            log.debug(f"[{model.name}] [{bench_name}] Q#{qi} cancelled")
            return {
                "question_id": qi,
                "status": "cancelled",
                "score": 0.0,
                "prompt": prompt[:8000],
                "response": "",
            }
        err_msg = str(gen_error) if gen_error else "Max retries exceeded"
        err_type = type(gen_error).__name__ if gen_error else "ProviderError"
        log.warning(
            f"[{model.name}] [{bench_name}] Q#{qi} gave up after "
            f"{attempt} attempt(s): {scrub(err_msg[:500])}"
        )
        return {
            "question_id": qi,
            "status": "error",
            "error_type": err_type,
            "error_msg": err_msg[:2000],
            "score": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "total_tokens": 0,
            "total_time_ms": None,
            "tokens_per_second": None,
            "cost_usd": None,
            "prompt": prompt[:8000],
            "response": f"[Error: {err_msg[:500]}]",
        }

    # Evaluate exactly ONCE. evaluate_detailed is the single source of truth
    # for both the score and the judge metadata; routing everything through it
    # means code-execution benchmarks run the sandbox a single time per question
    # (the old path called evaluate() AND evaluate_detailed(), executing model
    # code twice — 2x cost plus flaky port-binding collisions on network tasks
    # and possible score/judge-log divergence). Benchmarks that only override
    # evaluate() are covered by the base evaluate_detailed, which delegates to
    # evaluate(). Non-dict returns (e.g. bare Mocks in tests) fall back to a
    # direct evaluate() call.
    try:
        eval_info = None
        if hasattr(bench_obj, "evaluate_detailed"):
            candidate = bench_obj.evaluate_detailed(q, gen_result.text)
            if isinstance(candidate, dict):
                eval_info = candidate

        if eval_info is not None:
            score = float(eval_info.get("score", 0.0))
            expected_answer = str(
                eval_info.get("expected_answer")
                or q.get("answer") or q.get("target") or "N/A"
            )
            extracted_answer = str(eval_info.get("extracted_answer", "N/A"))
            judge_response = str(
                eval_info.get("judge_response", f"Evaluated score: {score}")
            )
        else:
            score = float(bench_obj.evaluate(q, gen_result.text))
            expected_answer = str(q.get("answer") or q.get("target") or "N/A")
            extracted_answer = "N/A"
            judge_response = f"Evaluated score: {score}"

        status = "success"
        err_msg = None
    except Exception as eval_exc:
        score = 0.0
        expected_answer = str(q.get("answer") or q.get("target") or "")
        extracted_answer = "N/A"
        judge_response = f"Evaluator Error: {eval_exc}"
        status = "eval_error"
        err_msg = str(eval_exc)
        log.exception(
            f"[{model.name}] [{bench_name}] Q#{qi} scorer error: {scrub(str(eval_exc)[:500])}"
        )
        console.print(
            f"[bold red]Scorer error on benchmark {bench_obj.name} Q#{qi}: "
            f"{eval_exc}[/bold red]{log_ref()}"
        )

    res_dict: dict = {
        "question_id": qi,
        "status": status,
        "score": score,
        "expected_answer": expected_answer,
        "extracted_answer": extracted_answer,
        "judge_response": judge_response,
        "input_tokens": gen_result.input_tokens,
        "output_tokens": gen_result.output_tokens,
        "thinking_tokens": gen_result.thinking_tokens,
        "total_tokens": gen_result.total_tokens,
        "total_time_ms": gen_result.total_time_ms,
        "tokens_per_second": gen_result.tokens_per_second,
        "cost_usd": gen_result.cost_usd,
        "finish_reason": gen_result.finish_reason,
        "is_truncated": gen_result.is_truncated,
        "prompt": prompt[:8000],
        "response": gen_result.text[:8000],
    }
    if err_msg:
        res_dict["error_msg"] = err_msg[:2000]
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
    """Execute benchmarks with parallel models and sequential benchmarks.

    Up to ``settings.max_concurrent_models`` models run at once; extra models
    queue. Each model works through its benchmarks one at a time, using
    ``settings.max_concurrency`` workers for a benchmark's questions.
    ``should_stop`` is polled every ~0.5s: queued work is cancelled
    immediately and running models unwind without waiting for in-flight
    requests.
    """
    if callbacks is None:
        callbacks = DefaultEngineCallbacks()

    names = [m.name for m in models]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate model names detected; each model must have a unique name")

    results: dict = dict(existing_results) if existing_results else {}
    settings = config.settings

    log.info(
        f"Engine run starting: {len(models)} models, {len(benchmarks)} benchmarks, "
        f"max_concurrency={settings.max_concurrency}, "
        f"max_concurrent_models={settings.max_concurrent_models} "
        f"(details → {run_log_rel()})"
    )
    for m in models:
        log.debug(
            f"model: id={m.id!r} name={m.name!r} thinking_effort={m.thinking_effort} "
            f"max_tokens={m.max_tokens} api_base={m.api_base} "
            f"api_key_env={m.api_key_env} extra_params={list(m.extra_params)}"
        )
    for bk, bc in benchmarks.items():
        log.debug(
            f"benchmark: {bk} enabled={bc.enabled} dataset={bc.dataset} "
            f"split={bc.split} subset={bc.subset} revision={bc.revision} "
            f"num_samples={bc.num_samples}"
        )
    log.debug(f"settings: seed={settings.seed} temperature={settings.temperature} "
              f"max_tokens={settings.max_tokens} request_timeout={settings.request_timeout} "
              f"code_exec_timeout={settings.code_exec_timeout} max_retries={settings.max_retries} "
              f"results_dir={settings.results_dir} hf_token_env={settings.hf_token_env} "
              f"allow_unsafe_code_execution={settings.allow_unsafe_code_execution}")

    results: dict = dict(existing_results) if existing_results else {}

    def _stopped() -> bool:
        return bool(should_stop and should_stop())

    # Pre-create per-model result containers single-threaded (no races later).
    for model in models:
        results.setdefault(
            model.name,
            {
                "model_id": model.id,
                "thinking_effort": model.thinking_effort,
            },
        )
        callbacks.on_event(model.name, f"Starting model run for {model.name}")

    results_lock = threading.Lock()  # guards results mutation + checkpoint saves
    done_lock = threading.Lock()  # guards done_models
    done_models: set[str] = set()

    def _run_model(model: ModelConfig) -> None:
        """Run all benchmarks for one model, sequentially. Owns its exceptions."""
        try:
            fatal_encountered = False

            for bench_name, bench_cfg in benchmarks.items():
                if _stopped() or fatal_encountered:
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
                        f"Skipping {bench_name} (sandboxed code execution requires explicit opt-in)",
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

                # Surface pre-sampling exclusions (e.g. code tasks the sandbox
                # cannot run) so a score is auditable in the log and results.
                exclusion_note = getattr(bench_obj, "exclusion_note", lambda: None)()
                if exclusion_note:
                    callbacks.on_event(model.name, exclusion_note)

                q_hash = compute_question_hash(questions)

                existing_b = results.get(model.name, {}).get(bench_name)
                ex_details = existing_b.get("details", []) if isinstance(existing_b, dict) else []
                if (isinstance(existing_b, dict) and "details" in existing_b
                    and existing_b.get("question_hash") == q_hash
                    and len(ex_details) >= len(questions)
                    and not any(d.get("status") == "cancelled" for d in ex_details if isinstance(d, dict))):
                    callbacks.on_event(
                        model.name,
                        f"Skipping {bench_name} (already completed in loaded results)",
                    )
                    callbacks.on_benchmark_done(model.name, bench_name, existing_b)
                    continue
                callbacks.on_benchmark_start(model.name, bench_name, len(questions))
                callbacks.on_event(
                    model.name,
                    f"Running {bench_obj.display_name} ({len(questions)} questions)...",
                )

                details: list[dict] = []
                max_concurrency = max(1, settings.max_concurrency)

                # Manual executor (no `with`) so stop never waits on in-flight calls.
                executor = ThreadPoolExecutor(max_workers=max_concurrency)
                # Not every callbacks object implements on_question_retry (the
                # Protocol is structural); tolerate its absence so a minimal
                # custom callbacks class doesn't AttributeError at submit time.
                _on_retry = getattr(callbacks, "on_question_retry", None)
                futures = {
                    executor.submit(
                        process_question,
                        qi,
                        q,
                        bench_obj,
                        model,
                        settings,
                        should_stop,
                        _on_retry,
                    ): (qi, q)
                    for qi, q in enumerate(questions)
                }
                pending = set(futures)
                try:
                    while pending:
                        if _stopped() or fatal_encountered:
                            for f in list(pending):
                                if f.done():
                                    try:
                                        detail = f.result()
                                        details.append(detail)
                                    except Exception:
                                        pass
                                    pending.discard(f)
                            for f in pending:
                                f.cancel()
                            break
                        done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                        for future in done:
                            try:
                                detail = future.result()
                            except FatalModelError as fe:
                                callbacks.on_event(model.name, f"[CRITICAL] {fe}")
                                fatal_encountered = True
                                for f in pending:
                                    f.cancel()
                                pending = {f for f in pending if not f.done()}
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
                                try:
                                    callbacks.on_question_done(model.name, bench_name, err_detail)
                                except Exception:
                                    pass
                                continue
                            details.append(detail)
                            try:
                                callbacks.on_question_done(model.name, bench_name, detail)
                            except Exception:
                                pass
                except Exception as loop_err:
                    callbacks.on_event(model.name, f"Execution error in benchmark pool: {loop_err}")
                finally:
                    executor.shutdown(wait=True, cancel_futures=True)

                if fatal_encountered:
                    callbacks.on_event(
                        model.name,
                        f"Aborting remaining benchmarks for {model.name} due to fatal error",
                    )

                # Summarize benchmark metrics cleanly.
                scored = [d for d in details if d["status"] in ("success", "eval_error")]
                eval_errors = [d for d in details if d["status"] == "eval_error"]
                provider_errors = [d for d in details if d["status"] == "error"]
                truncated_items = [d for d in details if d.get("is_truncated")]

                correct_count = sum(d["score"] for d in scored)
                total_count = len(scored)
                score = (correct_count / total_count) if total_count > 0 else None

                b_summary: dict = {
                    "score": round(score, 4) if score is not None else None,
                    "correct": correct_count,
                    "total": total_count,
                    "attempted": len(details),
                    "total_questions": len(questions),
                    "excluded_count": getattr(bench_obj, "excluded_count", 0),
                    "eval_error_count": len(eval_errors),
                    "provider_error_count": len(provider_errors),
                    "truncated_count": len(truncated_items),
                    "question_hash": q_hash,
                    "input_tokens": sum(d.get("input_tokens", 0) for d in details),
                    "output_tokens": sum(d.get("output_tokens", 0) for d in details),
                    "thinking_tokens": sum(d.get("thinking_tokens", 0) for d in details),
                    "total_tokens": sum(d.get("total_tokens", 0) for d in details),
                    "details": details,
                }
                if fatal_encountered or _stopped() or len(details) < len(questions):
                    b_summary["partial"] = True

                tps_vals = [d["tokens_per_second"] for d in details if d.get("tokens_per_second") is not None]
                if tps_vals:
                    b_summary["avg_tokens_per_second"] = round(sum(tps_vals) / len(tps_vals), 2)

                time_vals = [d["total_time_ms"] for d in details if d.get("total_time_ms") is not None]
                if time_vals:
                    b_summary["avg_time_ms"] = round(sum(time_vals) / len(time_vals), 1)

                cost_vals = [d["cost_usd"] for d in details if d.get("cost_usd") is not None]
                if cost_vals:
                    b_summary["total_cost_usd"] = round(sum(cost_vals), 6)

                # Critical section: save_results iterates the whole results tree
                # while other model threads mutate their own mdata.
                with results_lock:
                    mdata = results[model.name]
                    mdata[bench_name] = b_summary
                    # A checkpoint/persistence failure (read-only or full results dir,
                    # a transient Windows file lock) must NOT abort the model's
                    # remaining benchmarks — record the score, log, and continue.
                    try:
                        aggregate(mdata, config)
                        save_results(results, config, settings.results_dir, "latest.json")
                        log.info(
                            f"[{model.name}] checkpoint saved for {bench_name} "
                            f"({len(details)} details)"
                        )
                    except Exception as persist_err:
                        callbacks.on_event(
                            model.name,
                            f"Warning: checkpoint save failed after {bench_name}: {persist_err}",
                        )
                callbacks.on_benchmark_done(model.name, bench_name, b_summary)

        except Exception as model_err:
            callbacks.on_event(model.name, f"Model run error: {model_err}")
        finally:
            with done_lock:
                done_models.add(model.name)
            callbacks.on_model_done(model.name)

    if not models:
        return results

    workers = max(1, min(settings.max_concurrent_models, len(models)))
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        model_futures = [pool.submit(_run_model, m) for m in models]
        pending_models = set(model_futures)
        while pending_models:
            done_m, pending_models = wait(
                pending_models, timeout=0.5, return_when=FIRST_COMPLETED
            )
            for f in done_m:
                try:
                    f.result()
                except Exception:
                    pass  # _run_model owns and logs its exceptions
            if _stopped():
                for f in pending_models:
                    f.cancel()
                pending_models = {f for f in pending_models if not f.done()}
    finally:
        pool.shutdown(wait=True)

    # Queued models cancelled on stop never reached on_model_done — close them
    # out so the UI doesn't leave them "Running".
    for model in models:
        with done_lock:
            fire = model.name not in done_models
            if fire:
                done_models.add(model.name)
        if fire:
            callbacks.on_model_done(model.name)

    return results
