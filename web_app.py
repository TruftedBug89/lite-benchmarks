#!/usr/bin/env python
"""Lite Benchmarks — Modern Single-Page Web Dashboard & API Server.

Serves a visual web UI on http://127.0.0.1:8000 allowing users to configure,
run, monitor, and inspect LLM benchmarks directly from their web browser.

Usage:
    py web_app.py                          # starts server on 127.0.0.1:8000
    py web_app.py --port 8080              # custom port
    py web_app.py --no-browser             # start without launching browser
    py web_app.py --host 0.0.0.0          # custom host (warning on non-local)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
import webbrowser
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from lite_bench import logging_utils
from lite_bench.charts import generate_all as generate_charts
from lite_bench.config import Config, ModelConfig, Settings, load_config
from lite_bench.engine import DefaultEngineCallbacks, run_engine
from lite_bench.logging_utils import get_logger, log_ref, open_run_log, scrub
from lite_bench.metadata import BENCHMARK_INFO, CATEGORY_ICONS, CATEGORY_LABELS
from lite_bench.readme_gen import write_readme
from lite_bench.results_store import append_run_history, load_latest_results, load_run_history
from refresh_site import refresh_site

log = get_logger("web")

ROOT_DIR = Path(__file__).parent.resolve()
WEB_DIR = ROOT_DIR / "web"
CHARTS_DIR = ROOT_DIR / "charts"
CUSTOM_MODELS_FILE = ROOT_DIR / "custom_models.json"
CONFIG_FILE = ROOT_DIR / "config.yaml"


def _configure_stdio() -> None:
    """Force UTF-8 stdout/stderr so emoji and rich log output never crash on a
    non-UTF-8 console (e.g. Windows cp1252) or when output is redirected to a
    file/service. Without this, the very first ``print("✨ …")`` raises
    ``UnicodeEncodeError`` and the server never starts on a default console."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


_configure_stdio()


def _abs(path: str | Path) -> Path:
    """Resolve a configured path against the repo root.

    All file access is anchored to ``ROOT_DIR`` (not the process CWD) so the
    app behaves identically no matter which directory it is launched from."""
    p = Path(path)
    return p if p.is_absolute() else (ROOT_DIR / p)


def _latest_results_path(config: Config) -> Path:
    return _abs(config.settings.results_dir) / "latest.json"


# Load .env from the repo root if present (does NOT override real env vars).
# This is where users typically put LITE_BENCH_ALLOW_UNSAFE=1 and HF_TOKEN.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass


def _env_truthy(name: str) -> bool:
    """Robust env flag check: tolerates trailing spaces from cmd's `set X=1 `
    and common truthy spellings. Values are never logged."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


_SECRET_PATTERNS = re.compile(
    r"Bearer\s+\S+|sk-\S+|api[_-]?key[=:]\s*\S+|token[=:]\s*\S+",
    re.IGNORECASE,
)


def _scrub_secrets(text: str) -> str:
    return _SECRET_PATTERNS.sub("[REDACTED]", text)


def load_custom_models() -> list[dict]:
    if CUSTOM_MODELS_FILE.is_file():
        try:
            return json.loads(CUSTOM_MODELS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_custom_models(models: list[dict]) -> None:
    CUSTOM_MODELS_FILE.write_text(json.dumps(models, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class QuestionStatus:
    index: int
    prompt: str
    status: str = "pending"  # pending, running, success, eval_error, error
    score: float = 0.0
    expected_answer: str = ""
    extracted_answer: str = ""
    judge_response: str = ""
    response_text: str = ""
    error_msg: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0
    time_ms: float | None = None
    tps: float | None = None
    cost_usd: float | None = None


@dataclass
class LiveModelState:
    id: str
    name: str
    thinking_effort: str | None = None
    status: str = "Idle"
    current_benchmark: str = ""
    current_benchmark_display: str = ""
    q_index: int = 0
    q_total: int = 0
    correct: float = 0.0
    scored: int = 0
    failed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0
    rate_limit_count: int = 0
    truncated_count: int = 0
    provider_error_count: int = 0
    eval_error_count: int = 0
    avg_tps: float | None = None
    latest_snippet: str = ""
    last_error: str | None = None
    retry_note: str | None = None
    is_finished: bool = False
    is_failed: bool = False
    fatal_error: str | None = None
    benchmark_scores: dict[str, float] = field(default_factory=dict)
    recent_events: list[str] = field(default_factory=list)
    questions: list[dict] = field(default_factory=list)

    def add_event(self, msg: str) -> None:
        self.recent_events.append(msg)
        if len(self.recent_events) > 8:
            self.recent_events.pop(0)

    @property
    def accuracy(self) -> float | None:
        return (self.correct / self.scored) if self.scored > 0 else None


class EngineCallbacksBridge(DefaultEngineCallbacks):
    def __init__(self, engine: DashboardEngine):
        self.engine = engine
        self._run_id = engine._run_id

    def on_event(self, model_name: str, message: str) -> None:
        if self.engine._run_id != self._run_id:
            return
        with self.engine.state_lock:
            self.engine._log_locked(f"[{model_name}] {_scrub_secrets(message)}")
            state = self.engine.states.get(model_name)
            if state:
                state.add_event(_scrub_secrets(message))

    def on_benchmark_start(self, model_name: str, bench_name: str, total_questions: int) -> None:
        if self.engine._run_id != self._run_id:
            return
        with self.engine.state_lock:
            state = self.engine.states.get(model_name)
            if state:
                state.q_total += total_questions
                state.current_benchmark = bench_name
                info = BENCHMARK_INFO.get(bench_name, {})
                state.current_benchmark_display = info.get("display", bench_name)
                state.status = "Running"

    def on_question_retry(self, model_name: str, bench_name: str, info: dict) -> None:
        if self.engine._run_id != self._run_id:
            return
        with self.engine.state_lock:
            state = self.engine.states.get(model_name)
            if state:
                err_str = str(info.get("error", "")).lower()
                if "429" in err_str or "rate limit" in err_str:
                    state.rate_limit_count += 1
                state.retry_note = (
                    f"⏳ Q#{info.get('question_id', 0)} attempt {info.get('attempt', '?')} failed "
                    f"({_scrub_secrets(str(info.get('error', '')))}); retrying in {info.get('backoff', 0):.0f}s"
                )
            attempt = info.get("attempt", 0)
            if attempt <= 3 or attempt % 5 == 0:
                max_str = info.get("max_attempts") or "∞"
                self.engine._log_locked(
                    f"[{model_name}] Q#{info.get('question_id', 0)} retry {attempt}/{max_str}: "
                    f"{_scrub_secrets(str(info.get('error', '')))} — waiting {info.get('backoff', 0):.0f}s for a good response"
                )

    def on_question_done(self, model_name: str, bench_name: str, detail: dict) -> None:
        if self.engine._run_id != self._run_id:
            return
        with self.engine.state_lock:
            state = self.engine.states.get(model_name)
            if not state:
                return

            state.current_benchmark = bench_name
            info = BENCHMARK_INFO.get(bench_name, {})
            state.current_benchmark_display = info.get("display", bench_name)

            qi = detail.get("question_id", 0)
            status = detail.get("status", "error")
            score = float(detail.get("score", 0.0))

            state.q_index += 1
            state.retry_note = None
            if detail.get("is_truncated"):
                state.truncated_count += 1

            if status in ("success", "eval_error"):
                state.scored += 1
                state.correct += score
            if status == "eval_error":
                state.eval_error_count += 1
            if status == "error":
                state.failed += 1
                state.provider_error_count += 1

            state.input_tokens += detail.get("input_tokens", 0)
            state.output_tokens += detail.get("output_tokens", 0)
            state.thinking_tokens += detail.get("thinking_tokens", 0)
            state.total_tokens += detail.get("total_tokens", 0)

            tps = detail.get("tokens_per_second")
            if tps is not None:
                if state.avg_tps is None:
                    state.avg_tps = tps
                else:
                    state.avg_tps = (state.avg_tps * 0.8) + (tps * 0.2)

            resp = detail.get("response", "")
            if resp and status != "error":
                state.latest_snippet = resp[:150].replace("\n", " ")

            error_msg = _scrub_secrets(detail.get("error_msg", ""))
            if status in ("error", "eval_error") and error_msg:
                state.last_error = f"Q#{qi}: {error_msg[:4000]}"
            elif status == "success":
                state.last_error = None

            q_entry = {
                "index": qi,
                "benchmark": bench_name,
                "prompt": detail.get("prompt", "")[:8000],
                "status": status,
                "score": score,
                "expected_answer": detail.get("expected_answer", ""),
                "extracted_answer": detail.get("extracted_answer", ""),
                "judge_response": detail.get("judge_response", ""),
                "response_text": resp[:8000],
                "error_msg": error_msg[:2000],
                "input_tokens": detail.get("input_tokens", 0),
                "output_tokens": detail.get("output_tokens", 0),
                "thinking_tokens": detail.get("thinking_tokens", 0),
                "total_tokens": detail.get("total_tokens", 0),
                "finish_reason": detail.get("finish_reason"),
                "is_truncated": bool(detail.get("is_truncated")),
                "time_ms": detail.get("total_time_ms"),
                "tps": tps,
                "cost_usd": detail.get("cost_usd"),
            }
            state.questions.append(q_entry)

    def on_benchmark_done(self, model_name: str, bench_name: str, summary: dict) -> None:
        if self.engine._run_id != self._run_id:
            return
        with self.engine.state_lock:
            state = self.engine.states.get(model_name)
            if state:
                state.benchmark_scores[bench_name] = summary.get("score", 0.0)

    def on_model_done(self, model_name: str, reason: str = "completed") -> None:
        if self.engine._run_id != self._run_id:
            return
        with self.engine.state_lock:
            state = self.engine.states.get(model_name)
            if state:
                if state.status in ("Stopped", "Failed"):
                    return
                state.is_finished = True
                if reason == "stopped":
                    state.status = "Stopped"
                elif reason == "fatal":
                    state.is_failed = True
                    state.fatal_error = "Provider fatal error"
                    state.status = "Failed"
                else:
                    state.status = "Completed"


class DashboardEngine:
    def __init__(self):
        self.is_running: bool = False
        self.stop_requested: bool = False
        self.start_time: float = 0.0
        self.finish_time: float | None = None
        self.states: dict[str, LiveModelState] = {}
        self.results: dict[str, dict] = {}
        self.logs: list[str] = []
        self.current_config: Config | None = None
        self.worker_thread: threading.Thread | None = None
        self.state_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._run_id: int = 0
        self._log_counter: int = 0  # monotonic; lets the UI detect new log lines

    def _log_locked(self, msg: str) -> None:
        """Append a log entry. Caller must hold state_lock.

        Entries are single-line and capped — provider exceptions can be
        multi-KB dumps that would flood the web log and SSE frames. The full
        error text is still available in results/latest.json, the card
        tooltips, and the per-run detail log in ``logs/``."""
        ts = time.strftime("%H:%M:%S")
        msg = str(msg).replace("\n", " ")
        log.info(f"ui: {scrub(msg)}")
        if len(msg) > 800:
            msg = msg[:800] + "…"
        entry = f"[{ts}] {msg}"
        self.logs.append(entry)
        self._log_counter += 1
        if len(self.logs) > 100:
            self.logs.pop(0)

    def log(self, msg: str) -> None:
        with self.state_lock:
            self._log_locked(msg)

    def stop(self) -> None:
        """Force stop: cancel everything queued and reset state immediately.

        The worker thread unwinds in the background without blocking the UI;
        in-flight provider requests are discarded as they return. A fresh
        per-run stop event and run-id guard ensure a subsequent run is never
        affected by the previous one winding down.
        """
        with self.state_lock:
            if not self.is_running:
                return
            self._stop_event.set()
            self.stop_requested = True
            self._log_locked("⛔ Force stop — cancelling all queued work immediately.")
            log.warning("Force stop requested — cancelling queued work")
            self.is_running = False
            self.finish_time = time.time()
            for st in self.states.values():
                if not st.is_finished:
                    st.status = "Stopped"
                    st.is_finished = True

    def run_async(self, run_payload: dict, base_config: Config) -> bool:
        """Start a benchmark run in a background worker.

        Returns False if a run is already in progress (so the caller can report
        the conflict instead of silently no-oping)."""
        with self.state_lock:
            if self.is_running:
                return False
            self.is_running = True
            self.stop_requested = False
            self._stop_event = threading.Event()
            self._run_id += 1
            run_id = self._run_id
            stop_event = self._stop_event
            self.start_time = time.time()
            self.finish_time = None
            self.states = {}
        self.results = {}
        self.logs = []
        self.log("🚀 Initializing benchmark run...")

        # Per-run detail log: everything lands here (DEBUG+); the UI log and
        # console stay short and point back at this file.
        run_log_path = open_run_log(f"r{run_id}")
        log.info(f"Run #{run_id} started — detail log: {run_log_path}")
        self.log(f"📝 Detail log: {run_log_path}{log_ref()}")

        # Server-side model registry. The client may only *select* models that
        # already exist in config.yaml or custom_models.json; ``api_base`` and
        # ``api_key_env`` are taken from those server-side records and NEVER
        # from the request payload. This closes a key-exfiltration hole where a
        # (possibly cross-site) caller could point a resolved API key at an
        # attacker-controlled ``api_base``.
        registry: dict[str, dict] = {}
        for cm in base_config.models:
            registry[cm.id] = {"api_base": cm.api_base, "api_key_env": cm.api_key_env}
        for cm in load_custom_models():
            registry[str(cm.get("id", ""))] = {
                "api_base": cm.get("api_base"),
                "api_key_env": cm.get("api_key_env"),
            }

        models_data = run_payload.get("models", [])
        models: list[ModelConfig] = []
        for m in models_data:
            mid = str(m.get("id", "")).strip()
            name = str(m.get("name", mid)).strip() or mid
            thinking_effort = m.get("thinking_effort")
            max_tokens = m.get("max_tokens")
            source = registry.get(mid, {})
            api_base = source.get("api_base")
            api_key_env = source.get("api_key_env")
            try:
                max_tokens_val = int(max_tokens) if max_tokens else None
            except (TypeError, ValueError):
                max_tokens_val = None

            models.append(
                ModelConfig(
                    id=mid,
                    name=name,
                    thinking_effort=thinking_effort if thinking_effort else None,
                    max_tokens=max_tokens_val,
                    api_base=api_base if api_base else None,
                    api_key_env=api_key_env if api_key_env else None,
                    extra_params={},
                )
            )

        selected_bnames = run_payload.get("benchmarks", [])
        benchmarks = base_config.enabled_benchmarks()
        if selected_bnames:
            benchmarks = {k: v for k, v in benchmarks.items() if k in selected_bnames}

        # Override settings
        user_settings = run_payload.get("settings", {})
        num_samples_override = user_settings.get("num_samples")
        if num_samples_override and isinstance(num_samples_override, int):
            new_bconfs = {}
            for bk, bv in benchmarks.items():
                new_bconfs[bk] = type(bv)(
                    name=bv.name,
                    enabled=bv.enabled,
                    dataset=bv.dataset,
                    num_samples=num_samples_override,
                    split=bv.split,
                    subset=bv.subset,
                    revision=bv.revision,
                )
            benchmarks = new_bconfs

        # Check unsafe code execution policy
        allow_unsafe_requested = bool(user_settings.get("allow_unsafe_code_execution", False))
        allow_unsafe_env = _env_truthy("LITE_BENCH_ALLOW_UNSAFE")
        allow_unsafe = allow_unsafe_requested and allow_unsafe_env

        if allow_unsafe_requested and not allow_unsafe_env:
            self.log(
                "⚠️ Code-execution benchmarks requested but LITE_BENCH_ALLOW_UNSAFE is not "
                "set in the SERVER's environment. Set it before launching web_app.py "
                "(or add LITE_BENCH_ALLOW_UNSAFE=1 to a .env file next to it) and restart. "
                "Code-execution benchmarks will be skipped."
            )

        settings = Settings(
            seed=base_config.settings.seed,
            max_tokens=user_settings.get("max_tokens", base_config.settings.max_tokens),
            temperature=user_settings.get("temperature", base_config.settings.temperature),
            request_timeout=user_settings.get("request_timeout", base_config.settings.request_timeout),
            code_exec_timeout=user_settings.get("code_exec_timeout", base_config.settings.code_exec_timeout),
            max_retries=user_settings.get("max_retries", base_config.settings.max_retries),
            max_concurrency=user_settings.get("max_concurrency", base_config.settings.max_concurrency),
            max_concurrent_models=user_settings.get(
                "max_concurrent_models", base_config.settings.max_concurrent_models
            ),
            results_dir=str(_abs(base_config.settings.results_dir)),
            charts_dir=str(_abs(base_config.settings.charts_dir)),
            hf_token_env=base_config.settings.hf_token_env,
            allow_unsafe_code_execution=allow_unsafe,
        )

        self.current_config = Config(
            models=models,
            benchmarks=benchmarks,
            categories=base_config.categories,
            settings=settings,
        )

        log.info(
            f"Run #{run_id} config: {len(models)} models × {len(benchmarks)} benchmarks "
            f"(max_concurrency={settings.max_concurrency}, "
            f"max_concurrent_models={settings.max_concurrent_models})"
        )
        for m in models:
            log.debug(
                f"Run #{run_id} model: id={m.id!r} name={m.name!r} "
                f"thinking_effort={m.thinking_effort} max_tokens={m.max_tokens} "
                f"api_base={m.api_base} api_key_env={m.api_key_env}"
            )
        for bk, bv in benchmarks.items():
            log.debug(
                f"Run #{run_id} benchmark: {bk} dataset={bv.dataset} "
                f"split={bv.split} subset={bv.subset} revision={bv.revision} "
                f"num_samples={bv.num_samples}"
            )
        log.debug(
            f"Run #{run_id} settings: seed={settings.seed} "
            f"temperature={settings.temperature} max_tokens={settings.max_tokens} "
            f"request_timeout={settings.request_timeout} "
            f"code_exec_timeout={settings.code_exec_timeout} "
            f"max_retries={settings.max_retries} allow_unsafe={allow_unsafe}"
        )

        with self.state_lock:
            for m in models:
                self.states[m.name] = LiveModelState(
                    id=m.id,
                    name=m.name,
                    thinking_effort=m.thinking_effort,
                    status="Queued",
                )

        def _worker():
            try:
                callbacks = EngineCallbacksBridge(self)
                existing = load_latest_results(self.current_config, _latest_results_path(self.current_config))
                self.results = run_engine(
                    config=self.current_config,
                    models=models,
                    benchmarks=benchmarks,
                    callbacks=callbacks,
                    should_stop=stop_event.is_set,
                    allow_unsafe=allow_unsafe,
                    existing_results=existing,
                )
                with self.state_lock:
                    if self._run_id == run_id:
                        if stop_event.is_set():
                            self._log_locked("⛔ Benchmark run force-stopped.")
                            log.info(f"Run #{run_id} force-stopped")
                        else:
                            self._log_locked("✅ Benchmark run finished.")
                            for mname, mdata in self.results.items():
                                if not isinstance(mdata, dict):
                                    continue
                                sm = mdata.get("summary") if isinstance(mdata.get("summary"), dict) else {}
                                log.info(
                                    f"Run #{run_id} result: model={mname!r} "
                                    f"overall={sm.get('overall_score')} "
                                    f"completed={sm.get('completed_benchmarks')} "
                                    f"tokens={sm.get('total_tokens')}"
                                )
                            log.info(f"Run #{run_id} finished (results in results/latest.json)")
                            try:
                                append_run_history(
                                    self.results,
                                    self.current_config,
                                    self.current_config.settings.results_dir,
                                )
                                self._log_locked("📜 Run snapshot saved to history.")
                            except Exception as hist_err:
                                self._log_locked(f"⚠️ Failed to save run history: {hist_err}")
            except Exception as e:
                err_tb = _scrub_secrets(traceback.format_exc())
                log.error(f"Run #{run_id} failed: {_scrub_secrets(str(e))}")
                log.debug(f"Run #{run_id} traceback:\n{err_tb}")
                self.log(f"❌ Error during benchmark execution: {e}")
                with self.state_lock:
                    if self._run_id == run_id:
                        for st in self.states.values():
                            if not st.is_finished:
                                st.status = "Failed"
                                st.is_finished = True
                                st.is_failed = True
                                st.fatal_error = str(e)
            finally:
                # Only touch shared state if no newer run has started since.
                with self.state_lock:
                    if self._run_id == run_id:
                        self.is_running = False
                        self.finish_time = time.time()

        self.worker_thread = threading.Thread(target=_worker, daemon=True)
        self.worker_thread.start()
        return True

    def get_status_dict(self) -> dict:
        elapsed = (
            (self.finish_time or time.time()) - self.start_time if self.start_time > 0 else 0
        )

        with self.state_lock:
            model_states_json = {}
            for mname, s in self.states.items():
                d = asdict(s)
                # The full per-question detail list is unbounded and unused by
                # the dashboard; dropping it keeps SSE frames small even on long
                # runs (hundreds of questions × several benchmarks × models).
                d.pop("questions", None)
                d["accuracy"] = s.accuracy
                model_states_json[mname] = d

            return {
                "is_running": self.is_running,
                "stop_requested": self.stop_requested,
                "elapsed_seconds": round(elapsed, 1),
                "logs": self.logs[-30:],
                "logs_total": self._log_counter,
                "models": model_states_json,
            }


ENGINE = DashboardEngine()


class DashboardRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass  # Quiet HTTP request logging

    def _send_json(self, data: Any, status: int = 200) -> None:
        content = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self.send_error(404, "File not found")
            return
        content = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        path = url.path

        if path == "/" or path == "/index.html":
            self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
            return

        if path == "/visualizer" or path == "/visualizer.html":
            self._send_file(WEB_DIR / "visualizer.html", "text/html; charset=utf-8")
            return

        if path == "/api/config":
            try:
                config = load_config(CONFIG_FILE)
            except Exception as e:
                self._send_json({"error": f"Failed to load config: {e}"}, status=500)
                return

            # Sanitize config — strip any key/token parameters
            models_list = []
            for m in config.models:
                extra = {}
                for k, v in m.extra_params.items():
                    if any(term in k.lower() for term in ("key", "token", "secret", "password", "auth")):
                        continue
                    extra[k] = v
                models_list.append(
                    {
                        "id": m.id,
                        "name": m.name,
                        "thinking_effort": m.thinking_effort,
                        "max_tokens": m.max_tokens,
                        "extra_params": extra,
                    }
                )

            benchmarks_dict = {}
            for k, v in config.benchmarks.items():
                info = BENCHMARK_INFO.get(k, {})
                benchmarks_dict[k] = {
                    "name": v.name,
                    "enabled": v.enabled,
                    "dataset": v.dataset,
                    "num_samples": v.num_samples,
                    "split": v.split,
                    "subset": v.subset,
                    "display": info.get("display", v.name),
                    "category": info.get("category", "Other"),
                    "verification": info.get("verification", "Auto"),
                    "description": info.get("description", ""),
                }

            self._send_json(
                {
                    "models": models_list,
                    "benchmarks": benchmarks_dict,
                    "categories": config.categories,
                    "settings": asdict(config.settings),
                    "metadata": {
                        "category_labels": CATEGORY_LABELS,
                        "category_icons": CATEGORY_ICONS,
                        "benchmark_info": BENCHMARK_INFO,
                    },
                }
            )
            return

        if path == "/api/status":
            self._send_json(ENGINE.get_status_dict())
            return

        if path == "/api/model-questions":
            params = parse_qs(url.query)
            model_name = params.get("model", [None])[0]
            bench_name = params.get("benchmark", [None])[0]
            if not model_name:
                self._send_json({"error": "Missing model parameter"}, status=400)
                return

            questions_list = []
            with ENGINE.state_lock:
                state = ENGINE.states.get(model_name)
                if state:
                    questions_list = list(state.questions)

            if not questions_list:
                try:
                    config = load_config(CONFIG_FILE)
                    latest = load_latest_results(config, _latest_results_path(config))
                    mdata = latest.get(model_name, {})
                    if bench_name and bench_name in mdata:
                        candidates = {bench_name: mdata[bench_name]}
                    elif not bench_name:
                        candidates = {
                            k: v for k, v in mdata.items()
                            if isinstance(v, dict) and "details" in v
                        }
                    else:
                        candidates = {}
                    for _bk, bv in candidates.items():
                        details = bv.get("details", []) if isinstance(bv, dict) else []
                        for d in details:
                            if not isinstance(d, dict):
                                continue
                            entry = dict(d)
                            # Persisted details lack the live-state keys the
                            # visualizer reads (index/benchmark/response_text/
                            # time_ms/tps); normalize so deep links and the
                            # bench dropdown work on saved results too.
                            entry.setdefault("benchmark", _bk)
                            if "index" not in entry and "question_id" in entry:
                                entry["index"] = entry["question_id"]
                            if "response_text" not in entry and "response" in entry:
                                entry["response_text"] = entry["response"]
                            if "time_ms" not in entry and "total_time_ms" in entry:
                                entry["time_ms"] = entry["total_time_ms"]
                            if "tps" not in entry and "tokens_per_second" in entry:
                                entry["tps"] = entry["tokens_per_second"]
                            questions_list.append(entry)
                except Exception:
                    pass

            if bench_name:
                questions_list = [
                    q for q in questions_list if q.get("benchmark") == bench_name
                ]

            self._send_json({"model": model_name, "benchmark": bench_name, "questions": questions_list})
            return

        if path == "/api/results":
            try:
                config = load_config(CONFIG_FILE)
                latest = load_latest_results(config, _latest_results_path(config))
                self._send_json({"models": latest})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if path == "/api/history":
            try:
                config = load_config(CONFIG_FILE)
                runs = load_run_history(str(_abs(config.settings.results_dir)))
                self._send_json({"runs": runs})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if path == "/api/custom-models":
            self._send_json({"models": load_custom_models()})
            return

        if path == "/api/env-keys":
            known_keys = []
            for key, val in os.environ.items():
                upper = key.upper()
                if ("API_KEY" in upper or "APIKEY" in upper or upper.endswith("_KEY")) and val:
                    known_keys.append(key)
            known_keys.sort()
            self._send_json({"keys": known_keys})
            return

        if path == "/api/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            try:
                while True:
                    status_json = json.dumps(ENGINE.get_status_dict())
                    self.wfile.write(f"data: {status_json}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.3)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            return

        # Serve generated chart files with strict path traversal protection
        if path.startswith("/charts/"):
            rel_name = path[len("/charts/") :]
            target_path = (CHARTS_DIR / rel_name).resolve()
            if not target_path.is_file() or not target_path.is_relative_to(CHARTS_DIR.resolve()):
                self.send_error(404, "File not found")
                return
            self._send_file(target_path, "image/png")
            return

        # Serve assets from web directory
        rel_path = path.lstrip("/")
        target = (WEB_DIR / rel_path).resolve()
        if target.is_file() and target.is_relative_to(WEB_DIR.resolve()):
            ext = target.suffix.lower()
            ct = "text/plain"
            if ext == ".html":
                ct = "text/html; charset=utf-8"
            elif ext == ".css":
                ct = "text/css; charset=utf-8"
            elif ext == ".js":
                ct = "application/javascript; charset=utf-8"
            elif ext in (".png", ".jpg", ".jpeg"):
                ct = f"image/{ext[1:]}"
            elif ext == ".json":
                ct = "application/json"
            self._send_file(target, ct)
            return

        self.send_error(404, "Not Found")

    def _is_same_origin(self, origin: str) -> bool:
        """True if an ``Origin`` header refers to the local dashboard itself."""
        try:
            host = (urlparse(origin).hostname or "").lower()
        except ValueError:
            return False
        if host in ("127.0.0.1", "localhost", "::1"):
            return True
        our_host = (self.headers.get("Host") or "").split(":")[0].lower()
        return bool(our_host) and host == our_host

    def do_POST(self) -> None:
        url = urlparse(self.path)
        path = url.path

        # --- Guard every mutating API endpoint against CSRF ---------------
        # Requiring an explicit ``application/json`` Content-Type means a plain
        # HTML form or ``text/plain`` cross-site POST cannot reach these routes
        # (a cross-origin JSON POST triggers a CORS preflight we never grant),
        # and the Origin check blocks the rest. Essential now that the dashboard
        # can drive real API calls.
        if path.startswith("/api/"):
            ctype = self.headers.get("Content-Type", "")
            if "application/json" not in ctype.lower():
                self._send_json({"error": "Content-Type must be application/json."}, status=415)
                return
            origin = self.headers.get("Origin", "")
            if origin and not self._is_same_origin(origin):
                self._send_json({"error": "Cross-origin request rejected."}, status=403)
                return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            self._send_json({"error": "Invalid Content-Length."}, status=400)
            return
        if content_length > 5_000_000:
            self._send_json({"error": "Payload too large."}, status=413)
            return
        body = self.rfile.read(content_length) if content_length > 0 else b""
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except (ValueError, UnicodeDecodeError):
            self._send_json({"error": "Invalid JSON body."}, status=400)
            return

        if not isinstance(payload, dict):
            self._send_json({"error": "Request body must be a JSON object"}, status=400)
            return

        try:
            if path == "/api/run":
                try:
                    base_config = load_config(CONFIG_FILE)
                except Exception as e:
                    self._send_json({"error": f"Failed to load config: {e}"}, status=500)
                    return
                if not ENGINE.run_async(payload, base_config):
                    self._send_json({"error": "A benchmark run is already in progress."}, status=400)
                    return
                self._send_json({"status": "started"})
                return

            if path == "/api/stop":
                ENGINE.stop()
                self._send_json({"status": "stopping"})
                return

            if path == "/api/custom-models":
                models = load_custom_models()
                action = payload.get("action", "add")
                if action == "add":
                    model_entry = {
                        "id": payload.get("id", "").strip(),
                        "name": payload.get("name", "").strip(),
                        "thinking_effort": payload.get("thinking_effort") or None,
                        "api_base": payload.get("api_base") or None,
                        "api_key_env": payload.get("api_key_env") or None,
                    }
                    if not model_entry["id"]:
                        self._send_json({"error": "Model ID is required."}, status=400)
                        return
                    if not model_entry["name"]:
                        model_entry["name"] = model_entry["id"]
                    models = [m for m in models if m["id"] != model_entry["id"]]
                    models.append(model_entry)
                    save_custom_models(models)
                    self._send_json({"status": "added", "models": models})
                elif action == "delete":
                    mid = payload.get("id", "").strip()
                    models = [m for m in models if m["id"] != mid]
                    save_custom_models(models)
                    self._send_json({"status": "deleted", "models": models})
                else:
                    self._send_json({"error": "Unknown action."}, status=400)
                return

            if path == "/api/reports":
                try:
                    config = load_config(CONFIG_FILE)
                    latest = load_latest_results(config, _latest_results_path(config))
                    if not latest:
                        self._send_json({"error": "No results found to generate reports."}, status=400)
                        return
                    charts_dir = _abs(config.settings.charts_dir)
                    chart_paths = generate_charts(latest, config, str(charts_dir))
                    write_readme(latest, config, chart_paths, path=str(ROOT_DIR / "README.md"))
                    try:
                        refresh_site(config, latest)
                    except Exception as site_err:
                        print(f"[warn] site snapshot refresh failed: {site_err}")
                    self._send_json({"status": "success", "chart_paths": chart_paths})
                except Exception as e:
                    self._send_json({"error": f"Failed to generate reports: {e}"}, status=500)
                return

            self.send_error(404, "Not Found")
        except Exception as e:
            log.error(f"HTTP POST failed ({path}): {_scrub_secrets(str(e))}")
            log.debug(f"HTTP POST traceback ({path}):\n{_scrub_secrets(traceback.format_exc())}")
            self._send_json({"error": "Internal server error"}, status=500)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lite Benchmarks Web Dashboard")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser")
    args = parser.parse_args()

    if args.host != "127.0.0.1":
        print(f"\033[91mWARNING: Binding to non-local address {args.host}. Ensure network access is secure!\033[0m")

    server_address = (args.host, args.port)
    try:
        httpd = ThreadingHTTPServer(server_address, DashboardRequestHandler)
    except OSError as e:
        sys.exit(f"❌ Cannot bind {args.host}:{args.port} — is the port already in use? ({e})")

    # Per-request handler threads (notably the long-lived /api/events SSE loop)
    # must be daemons so they never hold the interpreter open when the console
    # is closed. ThreadingHTTPServer already defaults to this, but set it
    # explicitly so a future base-class change cannot reintroduce the hang.
    httpd.daemon_threads = True

    # 0.0.0.0/:: are not navigable addresses in a browser; open loopback instead.
    browser_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{browser_host}:{args.port}/"
    server_log = open_run_log("server")
    log.info(f"Web server starting on {url}")
    print(f"✨ Lite Benchmarks Web Dashboard running at: {url}")
    print(f"📝 Server log: {server_log}{log_ref()}")
    print(f"   (each benchmark run gets its own file in {logging_utils.LOG_DIR})")

    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(0.5), webbrowser.open(url)), daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except SystemExit:
        # Raised by Python's default handler on Windows CTRL_CLOSE_EVENT (e.g.
        # closing the terminal tab) and on os._exit-free console teardown.
        # Treat it as a normal shutdown instead of letting it escape uncaught.
        print("\nConsole closed — shutting down.")
    finally:
        # Always release the listening socket and stop the accept loop so the
        # process can exit cleanly instead of lingering as an unclosable tab.
        try:
            httpd.shutdown()
        finally:
            httpd.server_close()
    sys.exit(0)


if __name__ == "__main__":
    main()
