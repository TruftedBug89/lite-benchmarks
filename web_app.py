#!/usr/bin/env python
"""Lite Benchmarks — Modern Single-Page Web Dashboard & API Server.

Serves a visual web UI on http://localhost:8000 allowing users to configure,
run, monitor, and inspect LLM benchmarks directly from their web browser.

Usage:
    py web_app.py                          # starts server on port 8000
    py web_app.py --port 8080              # custom port
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass, field, asdict
import json
import os
from pathlib import Path
import re
import sys
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from lite_bench.benchmarks import create_benchmark
from lite_bench.charts import generate_all as generate_charts
from lite_bench.config import Config, ModelConfig, Settings, load_config
from lite_bench.providers import generate
from lite_bench.readme_gen import write_readme
from run_benchmark import _aggregate, _is_fatal_error, save_results

ROOT_DIR = Path(__file__).parent.resolve()
WEB_DIR = ROOT_DIR / "web"


@dataclass
class QuestionStatus:
    index: int
    prompt: str
    status: str = "pending"  # pending, running, scored, error
    score: float = 0.0
    response_text: str = ""
    error_msg: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0
    time_ms: float | None = None
    tps: float | None = None


@dataclass
class LiveModelState:
    id: str
    name: str
    thinking_effort: str | None = None
    api_base: str | None = None
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


class Engine:
    def __init__(self):
        self.is_running: bool = False
        self.stop_requested: bool = False
        self.start_time: float = 0.0
        self.finish_time: float | None = None
        self.states: dict[str, LiveModelState] = {}
        self.results: dict[str, dict] = {}
        self.logs: list[str] = []
        self.current_config: Config | None = None

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

        # Parse user payload
        raw_env = run_payload.get("env_vars", {})
        for k, v in raw_env.items():
            if isinstance(k, str) and isinstance(v, str) and v.strip():
                os.environ[k.strip()] = v.strip()

        models_data = run_payload.get("models", [])
        models: list[ModelConfig] = []
        for m in models_data:
            mid = str(m.get("id", "")).strip()
            name = str(m.get("name", mid)).strip() or mid
            thinking_effort = m.get("thinking_effort")
            api_key = m.get("api_key")
            api_base = m.get("api_base")

            extra: dict = {}
            if api_key:
                extra["api_key"] = api_key.strip()
            if api_base:
                extra["api_base"] = api_base.strip()

            models.append(
                ModelConfig(
                    id=mid,
                    name=name,
                    thinking_effort=thinking_effort if thinking_effort else None,
                    extra_params=extra,
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
            allow_unsafe_code_execution=user_settings.get("allow_unsafe_code_execution", False),
        )

        self.current_config = Config(
            models=models,
            benchmarks=benchmarks,
            categories=base_config.categories,
            settings=settings,
        )

        # Initialize states
        for m in models:
            self.states[m.name] = LiveModelState(
                id=m.id,
                name=m.name,
                thinking_effort=m.thinking_effort,
                api_base=m.extra_params.get("api_base"),
            )
            self.results[m.name] = {"model_id": m.id}
            if m.thinking_effort:
                self.results[m.name]["thinking_effort"] = m.thinking_effort

        def worker():
            model_workers = user_settings.get("model_concurrency", 4)
            self.log(f"⚡ Starting {len(models)} models across {len(benchmarks)} benchmarks (workers={model_workers}).")

            def run_model_task(mstate: LiveModelState):
                mconfig = mstate.model if hasattr(mstate, "model") else ModelConfig(id=mstate.id, name=mstate.name, thinking_effort=mstate.thinking_effort)
                # Find matching ModelConfig
                mcfg = next((x for x in models if x.name == mstate.name), mconfig)

                for bname, bconf in benchmarks.items():
                    if self.stop_requested or mstate.is_failed:
                        break

                    bench = create_benchmark(bname, bconf, settings)
                    mstate.current_benchmark = bname
                    mstate.current_benchmark_display = bench.display_name

                    if bench.requires_code_execution and not settings.allow_unsafe_code_execution:
                        mstate.add_event(f"Skipped {bench.display_name} (code execution disabled)")
                        self.log(f"[{mstate.name}] Skipped {bench.display_name} (unsafe code exec disabled).")
                        continue

                    try:
                        questions = bench.load()
                    except Exception as error:
                        mstate.add_event(f"Failed to load dataset: {type(error).__name__}")
                        self.log(f"[{mstate.name}] Error loading dataset for {bname}: {error}")
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
                        if b_fatal or mstate.is_failed or self.stop_requested:
                            return qi, False, None, 0.0, None, True

                        prompt = bench.format_prompt(q)
                        max_retries = max(3, settings.max_retries)

                        for attempt in range(1, max_retries + 1):
                            mstate.status = f"🧠 Q{qi+1}/{mstate.q_total} (attempt {attempt})"
                            try:
                                result = generate(mcfg, prompt, settings)
                                score = bench.evaluate(q, result.text)
                                return qi, True, result, score, None, False
                            except Exception as error:
                                err_type = type(error).__name__
                                err_msg = str(error).strip()

                                if _is_fatal_error(error):
                                    mstate.add_event(f"❌ FATAL Q{qi+1} ({err_type}): {err_msg[:30]}...")
                                    self.log(f"[{mstate.name}] Fatal error on Q{qi+1}: {err_msg}")
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
                                    mstate.add_event(f"❌ Q{qi+1} failed after {max_retries} attempts")
                                    return qi, False, None, 0.0, error, False

                        return qi, False, None, 0.0, Exception("Unknown error"), False

                    with concurrent.futures.ThreadPoolExecutor(max_workers=settings.max_concurrency) as q_executor:
                        q_futures = {
                            q_executor.submit(process_question, qi, q): qi
                            for qi, q in enumerate(questions)
                        }
                        for q_future in concurrent.futures.as_completed(q_futures):
                            if self.stop_requested:
                                break
                            qi, success, result, score, error, is_fatal = q_future.result()
                            mstate.q_index += 1

                            if is_fatal:
                                b_fatal = True
                                mstate.is_failed = True
                                mstate.fatal_error = str(error)
                                mstate.status = "❌ FATAL ERROR"
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
                                mstate.latest_snippet = snippet[:120] + ("..." if len(snippet) > 120 else "")

                                if score == 1.0:
                                    mstate.add_event(f"✨ Q{qi+1} Correct! (+1.0)")
                                    mstate.status = f"✨ Q{qi+1} Correct"
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
                                    "prompt_snippet": bench.format_prompt(q)[:150],
                                    "response_snippet": snippet[:250],
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

                            if b_fatal:
                                break

                    if b_scored > 0:
                        b_score = b_correct / b_scored
                        mstate.benchmark_scores[bname] = b_score
                        agg = _aggregate([d for d in b_details if d["status"] == "scored"])
                        self.results[mstate.name][bname] = {
                            "score": b_score,
                            "correct": round(b_correct, 4),
                            "total": b_scored,
                            "requested": len(questions),
                            "failed": b_failed,
                            **agg,
                            "questions": sorted(b_details, key=lambda x: x["question_index"]),
                        }
                        mstate.add_event(f"🏆 Finished {bench.display_name}: {b_score:.0%}")
                        self.log(f"[{mstate.name}] {bench.display_name}: {b_score:.1%} ({round(b_correct, 1)}/{b_scored})")
                    elif b_fatal:
                        mstate.add_event(f"💥 Aborted {bench.display_name} (Fatal)")
                        break

                mstate.is_finished = True
                if not mstate.is_failed:
                    mstate.status = "✅ Complete"

            with concurrent.futures.ThreadPoolExecutor(max_workers=model_workers) as executor:
                futures = [executor.submit(run_model_task, s) for s in self.states.values()]
                for f in concurrent.futures.as_completed(futures):
                    pass

            self.finish_time = time.time()
            self.is_running = False
            self.log("✅ All benchmarks completed. Saving results & generating charts/README...")

            try:
                if self.current_config:
                    path = save_results(self.results, self.current_config)
                    self.log(f"Results saved to {path.name}")
                    chart_paths = generate_charts(self.results, self.current_config, self.current_config.settings.charts_dir)
                    write_readme(self.results, self.current_config, chart_paths)
                    self.log("README.md & charts updated successfully.")
            except Exception as ex:
                self.log(f"Error persisting results: {ex}")

        t = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        t.submit(worker)

    def get_status_dict(self) -> dict:
        elapsed = int((time.time() - self.start_time) if self.start_time > 0 else 0)
        states_dict = {}
        for k, s in self.states.items():
            states_dict[k] = {
                "id": s.id,
                "name": s.name,
                "thinking_effort": s.thinking_effort,
                "api_base": s.api_base,
                "status": s.status,
                "current_benchmark": s.current_benchmark,
                "current_benchmark_display": s.current_benchmark_display,
                "q_index": s.q_index,
                "q_total": s.q_total,
                "correct": round(s.correct, 2),
                "scored": s.scored,
                "failed": s.failed,
                "accuracy": round(s.accuracy, 4) if s.accuracy is not None else None,
                "input_tokens": s.input_tokens,
                "output_tokens": s.output_tokens,
                "thinking_tokens": s.thinking_tokens,
                "total_tokens": s.total_tokens,
                "avg_tps": round(s.avg_tps, 1) if s.avg_tps is not None else None,
                "latest_snippet": s.latest_snippet,
                "is_finished": s.is_finished,
                "is_failed": s.is_failed,
                "fatal_error": s.fatal_error,
                "benchmark_scores": s.benchmark_scores,
                "recent_events": s.recent_events,
            }

        return {
            "is_running": self.is_running,
            "elapsed_seconds": elapsed,
            "states": states_dict,
            "logs": self.logs[-20:],
        }


ENGINE = Engine()


class DashboardRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path, content_type: str) -> None:
        if not file_path.is_file():
            self.send_error(404, "File Not Found")
            return
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            index_file = WEB_DIR / "index.html"
            self._send_file(index_file, "text/html; charset=utf-8")
            return

        if path.startswith("/charts/"):
            chart_file = ROOT_DIR / path.lstrip("/")
            self._send_file(chart_file, "image/png")
            return

        if path == "/api/config":
            try:
                base_cfg = load_config("config.yaml")
                models = [
                    {
                        "id": m.id,
                        "name": m.name,
                        "thinking_effort": m.thinking_effort,
                        "extra_params": m.extra_params,
                    }
                    for m in base_cfg.models
                ]
                benchmarks = [
                    {
                        "name": name,
                        "enabled": b.enabled,
                        "dataset": b.dataset,
                        "num_samples": b.num_samples,
                        "category": base_cfg.benchmark_category(name),
                    }
                    for name, b in base_cfg.benchmarks.items()
                ]
                self._send_json(
                    {
                        "models": models,
                        "benchmarks": benchmarks,
                        "categories": base_cfg.categories,
                        "settings": {
                            "seed": base_cfg.settings.seed,
                            "max_tokens": base_cfg.settings.max_tokens,
                            "temperature": base_cfg.settings.temperature,
                            "request_timeout": base_cfg.settings.request_timeout,
                            "code_exec_timeout": base_cfg.settings.code_exec_timeout,
                            "max_retries": base_cfg.settings.max_retries,
                            "max_concurrency": base_cfg.settings.max_concurrency,
                            "allow_unsafe_code_execution": base_cfg.settings.allow_unsafe_code_execution,
                        },
                    }
                )
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if path == "/api/status":
            self._send_json(ENGINE.get_status_dict())
            return

        if path == "/api/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                for _ in range(600):  # Stream for up to 60s per connection
                    status_json = json.dumps(ENGINE.get_status_dict())
                    self.wfile.write(f"data: {status_json}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(0.3)
            except (ConnectionError, BrokenPipeError):
                pass
            return

        if path == "/api/results":
            res_dir = ROOT_DIR / "results"
            results_files = []
            if res_dir.exists():
                for p in sorted(res_dir.glob("*.json"), reverse=True):
                    results_files.append({"filename": p.name, "mtime": p.stat().st_mtime})
            self._send_json({"results": results_files})
            return

        self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b""

        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            payload = {}

        if path == "/api/run":
            if ENGINE.is_running:
                self._send_json({"error": "A benchmark run is already in progress."}, status=400)
                return

            try:
                base_cfg = load_config("config.yaml")
                ENGINE.run_async(payload, base_cfg)
                self._send_json({"status": "started"})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if path == "/api/stop":
            ENGINE.stop()
            self._send_json({"status": "stopping"})
            return

        if path == "/api/reports":
            try:
                base_cfg = load_config("config.yaml")
                latest = ROOT_DIR / "results" / "latest.json"
                if not latest.exists():
                    self._send_json({"error": "No results found."}, status=404)
                    return
                data = json.loads(latest.read_text(encoding="utf-8")).get("models", {})
                chart_paths = generate_charts(data, base_cfg, base_cfg.settings.charts_dir)
                write_readme(data, base_cfg, chart_paths)
                self._send_json({"status": "generated", "charts": chart_paths})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        self.send_error(404, "Not Found")


def main():
    parser = argparse.ArgumentParser(description="Lite Benchmarks — Web App Dashboard Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve web app on (default 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    WEB_DIR.mkdir(parents=True, exist_ok=True)
    server_address = ("", args.port)
    httpd = ThreadingHTTPServer(server_address, DashboardRequestHandler)

    url = f"http://localhost:{args.port}"
    print(f"\n🚀 Lite Benchmarks Web Dashboard running at: {url}\n")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Web Server...")
        httpd.server_close()


if __name__ == "__main__":
    main()
