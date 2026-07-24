"""Tests for run_engine's parallel-model execution and fast force-stop."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

from lite_bench.config import BenchmarkConfig, Config, ModelConfig, Settings
from lite_bench.engine import run_engine


class RecordingCallbacks:
    def __init__(self):
        self.events: list[tuple] = []
        self.lock = threading.Lock()
        self.active: set[str] = set()
        self.max_active = 0

    def on_event(self, model_name, message):
        pass

    def on_benchmark_start(self, model_name, bench_name, total):
        with self.lock:
            self.events.append(("start", model_name, bench_name))
            self.active.add(model_name)
            self.max_active = max(self.max_active, len(self.active))

    def on_question_done(self, model_name, bench_name, detail):
        pass

    def on_benchmark_done(self, model_name, bench_name, summary):
        with self.lock:
            self.events.append(("done", model_name, bench_name))

    def on_model_done(self, model_name):
        with self.lock:
            self.events.append(("model_done", model_name, None))
            self.active.discard(model_name)


def _make_config(n_models: int, n_benches: int, settings: Settings) -> Config:
    models = [ModelConfig(id=f"fake/m{i}", name=f"Model {i}") for i in range(n_models)]
    benches = {
        f"b{i}": BenchmarkConfig(name=f"b{i}", enabled=True, dataset="fake", num_samples=3)
        for i in range(n_benches)
    }
    return Config(models=models, benchmarks=benches, categories={}, settings=settings)


def _fake_bench(questions: list[dict]):
    bench = MagicMock()
    bench.requires_code_execution = False
    bench.display_name = "FakeBench"
    bench.load.return_value = questions
    return bench


def _detail(qi: int) -> dict:
    return {"question_id": qi, "status": "success", "score": 1.0}


def test_models_run_in_parallel_but_benchmarks_sequential():
    """Up to max_concurrent_models models run at once; each model does its
    benchmarks strictly one after another."""
    settings = Settings(max_concurrency=1, max_concurrent_models=4)
    config = _make_config(n_models=6, n_benches=2, settings=settings)
    cb = RecordingCallbacks()

    def slow_question(qi, q, bench, model, settings_, should_stop=None):
        time.sleep(0.2)
        return _detail(qi)

    with (
        patch("lite_bench.engine.create_benchmark", side_effect=lambda *a, **k: _fake_bench([{"i": 0}, {"i": 1}, {"i": 2}])),
        patch("lite_bench.engine.process_question", side_effect=slow_question),
        patch("lite_bench.engine.aggregate", lambda *a, **k: None),
        patch("lite_bench.engine.save_results", lambda *a, **k: None),
    ):
        results = run_engine(config, config.models, config.benchmarks, callbacks=cb)

    # 4-way model parallelism actually reached
    assert cb.max_active == 4

    # on_model_done fired exactly once per model
    model_dones = [e for e in cb.events if e[0] == "model_done"]
    assert sorted(e[1] for e in model_dones) == sorted(m.name for m in config.models)

    # Per model, benchmarks ran sequentially: start/done events alternate
    for m in config.models:
        seq = [e[0] for e in cb.events if e[1] == m.name and e[0] in ("start", "done")]
        assert seq == ["start", "done", "start", "done"], f"{m.name}: {seq}"

    # Results contain every model with both benchmark summaries
    for m in config.models:
        assert m.name in results
        assert set(results[m.name]) >= {"b0", "b1"}


def test_force_stop_cancels_queued_models_fast():
    """Stop must cancel queued models/questions and return quickly without
    waiting for hung in-flight requests."""
    settings = Settings(max_concurrency=2, max_concurrent_models=2)
    config = _make_config(n_models=3, n_benches=2, settings=settings)
    cb = RecordingCallbacks()
    stop_event = threading.Event()
    calls = {"n": 0}
    calls_lock = threading.Lock()

    def hung_question(qi, q, bench, model, settings_, should_stop=None):
        if should_stop and should_stop():
            return {"question_id": qi, "status": "cancelled", "score": 0.0}
        with calls_lock:
            calls["n"] += 1
        time.sleep(2)  # simulates a hung provider request
        return _detail(qi)

    started = time.monotonic()
    with (
        patch("lite_bench.engine.create_benchmark", side_effect=lambda *a, **k: _fake_bench([{"i": i} for i in range(4)])),
        patch("lite_bench.engine.process_question", side_effect=hung_question),
        patch("lite_bench.engine.aggregate", lambda *a, **k: None),
        patch("lite_bench.engine.save_results", lambda *a, **k: None),
    ):
        timer = threading.Timer(0.7, stop_event.set)
        timer.start()
        results = run_engine(
            config,
            config.models,
            config.benchmarks,
            callbacks=cb,
            should_stop=stop_event.is_set,
        )
        timer.cancel()
    elapsed = time.monotonic() - started

    # Serial execution would take 3 models x 2 benches x 4 questions x 2s = 48s.
    # Force stop must unwind in a few seconds (poll interval + one hung wave).
    assert elapsed < 8, f"stop took too long: {elapsed:.1f}s"

    # Queued questions were cancelled — far fewer calls than the 24 total
    assert calls["n"] < 24

    # Every model still got on_model_done so the UI closes out cleanly
    model_dones = [e[1] for e in cb.events if e[0] == "model_done"]
    assert sorted(model_dones) == sorted(m.name for m in config.models)
    assert set(results) == {m.name for m in config.models}
