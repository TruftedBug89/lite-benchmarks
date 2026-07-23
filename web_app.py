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
import sys
import threading
import time
import webbrowser
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from lite_bench.charts import generate_all as generate_charts
from lite_bench.config import Config, ModelConfig, Settings, load_config
from lite_bench.engine import DefaultEngineCallbacks, run_engine
from lite_bench.metadata import BENCHMARK_INFO, CATEGORY_ICONS, CATEGORY_LABELS
from lite_bench.readme_gen import write_readme
from lite_bench.results_store import load_latest_results

ROOT_DIR = Path(__file__).parent.resolve()
WEB_DIR = ROOT_DIR / "web"
CHARTS_DIR = ROOT_DIR / "charts"


@dataclass
class QuestionStatus:
    index: int
    prompt: str
    status: str = "pending"  # pending, running, success, eval_error, error
    score: float = 0.0
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
    avg_tps: float | None = None
    latest_snippet: str = ""
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

    def on_event(self, model_name: str, message: str) -> None:
        self.engine.log(f"[{model_name}] {message}")
        state = self.engine.states.get(model_name)
        if state:
            state.add_event(message)

    def on_question_done(self, model_name: str, bench_name: str, detail: dict) -> None:
        state = self.engine.states.get(model_name)
        if not state:
            return

        state.current_benchmark = bench_name
        info = BENCHMARK_INFO.get(bench_name, {})
        state.current_benchmark_display = info.get("display", bench_name)

        qi = detail.get("question_id", 0)
        status = detail.get("status", "error")
        score = float(detail.get("score", 0.0))

        state.q_index = qi + 1
        state.scored += 1
        state.correct += score
        if status in ("error", "eval_error"):
            state.failed += 1

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
        if resp:
            state.latest_snippet = resp[:150].replace("\n", " ")

        q_entry = {
            "index": qi,
            "prompt": detail.get("prompt", "")[:300],
            "status": status,
            "score": score,
            "response_text": resp[:1000],
            "error_msg": detail.get("error_msg", ""),
            "input_tokens": detail.get("input_tokens", 0),
            "output_tokens": detail.get("output_tokens", 0),
            "thinking_tokens": detail.get("thinking_tokens", 0),
            "total_tokens": detail.get("total_tokens", 0),
            "time_ms": detail.get("total_time_ms"),
            "tps": tps,
            "cost_usd": detail.get("cost_usd"),
        }
        state.questions.append(q_entry)

    def on_benchmark_done(self, model_name: str, bench_name: str, summary: dict) -> None:
        state = self.engine.states.get(model_name)
        if state:
            state.benchmark_scores[bench_name] = summary.get("score", 0.0)

    def on_model_done(self, model_name: str) -> None:
        state = self.engine.states.get(model_name)
        if state:
            state.is_finished = True
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

    def log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.logs.append(entry)
        if len(self.logs) > 100:
            self.logs.pop(0)

    def stop(self) -> None:
        if self.is_running:
            self.stop_requested = True
            self.log("⏹️ Stop request initiated by user.")

    def run_async(self, run_payload: dict, base_config: Config) -> None:
        if self.is_running:
            return
        self.is_running = True
        self.stop_requested = False
        self.start_time = time.time()
        self.finish_time = None
        self.states = {}
        self.results = {}
        self.logs = []
        self.log("🚀 Initializing benchmark run...")

        # Parse model parameters from payload (no env_vars mutation allowed)
        models_data = run_payload.get("models", [])
        models: list[ModelConfig] = []
        for m in models_data:
            mid = str(m.get("id", "")).strip()
            name = str(m.get("name", mid)).strip() or mid
            thinking_effort = m.get("thinking_effort")
            max_tokens = m.get("max_tokens")

            models.append(
                ModelConfig(
                    id=mid,
                    name=name,
                    thinking_effort=thinking_effort if thinking_effort else None,
                    max_tokens=int(max_tokens) if max_tokens else None,
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
        allow_unsafe_env = os.environ.get("LITE_BENCH_ALLOW_UNSAFE") == "1"
        allow_unsafe = allow_unsafe_requested and allow_unsafe_env

        if allow_unsafe_requested and not allow_unsafe_env:
            self.log(
                "⚠️ Unsafe code execution requested in UI, but LITE_BENCH_ALLOW_UNSAFE=1 "
                "environment variable is not set. Code execution benchmarks will be skipped."
            )

        settings = Settings(
            seed=base_config.settings.seed,
            max_tokens=user_settings.get("max_tokens", base_config.settings.max_tokens),
            temperature=user_settings.get("temperature", base_config.settings.temperature),
            request_timeout=user_settings.get("request_timeout", base_config.settings.request_timeout),
            code_exec_timeout=user_settings.get("code_exec_timeout", base_config.settings.code_exec_timeout),
            max_retries=user_settings.get("max_retries", base_config.settings.max_retries),
            max_concurrency=user_settings.get("max_concurrency", base_config.settings.max_concurrency),
            results_dir=base_config.settings.results_dir,
            charts_dir=base_config.settings.charts_dir,
            hf_token_env=base_config.settings.hf_token_env,
            allow_unsafe_code_execution=allow_unsafe,
        )

        self.current_config = Config(
            models=models,
            benchmarks=benchmarks,
            categories=base_config.categories,
            settings=settings,
        )

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
                existing = load_latest_results(self.current_config)
                self.results = run_engine(
                    config=self.current_config,
                    models=models,
                    benchmarks=benchmarks,
                    callbacks=callbacks,
                    should_stop=lambda: self.stop_requested,
                    allow_unsafe=allow_unsafe,
                    existing_results=existing,
                )
                self.log("✅ Benchmark run finished.")
            except Exception as e:
                self.log(f"❌ Error during benchmark execution: {e}")
            finally:
                self.is_running = False
                self.finish_time = time.time()

        self.worker_thread = threading.Thread(target=_worker, daemon=True)
        self.worker_thread.start()

    def get_status_dict(self) -> dict:
        elapsed = (
            (self.finish_time or time.time()) - self.start_time if self.start_time > 0 else 0
        )

        model_states_json = {}
        for mname, s in self.states.items():
            d = asdict(s)
            d["accuracy"] = s.accuracy
            model_states_json[mname] = d

        return {
            "is_running": self.is_running,
            "stop_requested": self.stop_requested,
            "elapsed_seconds": round(elapsed, 1),
            "logs": self.logs[-30:],
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

        if path == "/api/config":
            try:
                config = load_config("config.yaml")
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

        if path == "/api/results":
            try:
                config = load_config("config.yaml")
                latest = load_latest_results(config)
                self._send_json({"models": latest})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if path == "/api/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            try:
                for _ in range(200):
                    status_json = json.dumps(ENGINE.get_status_dict())
                    self.wfile.write(f"data: {status_json}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.3)
            except (BrokenPipeError, ConnectionResetError):
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

    def do_POST(self) -> None:
        url = urlparse(self.path)
        path = url.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        payload = json.loads(body.decode("utf-8")) if body else {}

        if path == "/api/run":
            if ENGINE.is_running:
                self._send_json({"error": "A benchmark run is already in progress."}, status=400)
                return
            base_config = load_config("config.yaml")
            ENGINE.run_async(payload, base_config)
            self._send_json({"status": "started"})
            return

        if path == "/api/stop":
            ENGINE.stop()
            self._send_json({"status": "stopping"})
            return

        if path == "/api/reports":
            try:
                config = load_config("config.yaml")
                latest = load_latest_results(config)
                if not latest:
                    self._send_json({"error": "No results found to generate reports."}, status=400)
                    return
                chart_paths = generate_charts(latest, config, config.settings.charts_dir)
                write_readme(latest, config, chart_paths)
                self._send_json({"status": "success", "chart_paths": chart_paths})
            except Exception as e:
                self._send_json({"error": f"Failed to generate reports: {e}"}, status=500)
            return

        self.send_error(404, "Not Found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lite Benchmarks Web Dashboard")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser")
    args = parser.parse_args()

    if args.host != "127.0.0.1":
        print(f"\033[91mWARNING: Binding to non-local address {args.host}. Ensure network access is secure!\033[0m")

    server_address = (args.host, args.port)
    httpd = ThreadingHTTPServer(server_address, DashboardRequestHandler)

    url = f"http://{args.host}:{args.port}/"
    print(f"✨ Lite Benchmarks Web Dashboard running at: {url}")

    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(0.5), webbrowser.open(url)), daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
