"""Tests for the deterministic pre-sampling filters that drop code-execution
benchmark tasks whose reference solution the sandbox would reject (so every
sampled task is passable and num_samples stays an honest count)."""

from __future__ import annotations

from lite_bench.benchmarks import (
    BenchmarkBase,
    BigCodeBenchBenchmark,
    HumanEvalBenchmark,
    MBPPBenchmark,
    _bigcodebench_gradeable,
    _canonical_solution_gradeable,
)
from lite_bench.config import BenchmarkConfig, Settings


def _bench(cls):
    cfg = BenchmarkConfig(name="t", enabled=True, dataset="t", num_samples=10)
    return cls(cfg, Settings(code_exec_timeout=5))


def test_gradeable_helper():
    assert _canonical_solution_gradeable("def f():\n    return 1")
    assert _canonical_solution_gradeable("import math\nmath.sqrt(4)")
    # Confined modules are gradeable (the runtime shim makes them safe).
    assert _canonical_solution_gradeable("import socket\nopen('x.txt', 'w')")
    assert _canonical_solution_gradeable("import os\nos.makedirs('d')")
    assert _canonical_solution_gradeable("import shutil\nshutil.copy('a', 'b')")
    # Genuinely blocked constructs are not gradeable.
    assert not _canonical_solution_gradeable("import os\nos.system('ls')")
    assert not _canonical_solution_gradeable("eval('1')")
    assert not _canonical_solution_gradeable("import subprocess")
    assert not _canonical_solution_gradeable("import psutil")
    # Missing/empty reference code must not drop the row.
    assert _canonical_solution_gradeable(None)
    assert _canonical_solution_gradeable("")
    # Indented function-body fragments (HumanEval/MBPP) are dedented.
    assert _canonical_solution_gradeable("    return sorted(xs)\n")


def test_bigcodebench_gradeable_scans_forced_preamble():
    """A blocked import forced by code_prompt makes a task unpassable even if
    the body is clean — the model is handed code_prompt verbatim."""
    # Clean preamble + clean body -> passable.
    assert _bigcodebench_gradeable({
        "code_prompt": "import os\ndef task_func(p):\n",
        "canonical_solution": "    return os.path.basename(p)\n",
    })
    # Forced subprocess import in the preamble -> unpassable.
    assert not _bigcodebench_gradeable({
        "code_prompt": "import subprocess\ndef task_func(c):\n",
        "canonical_solution": "    return subprocess.run(c)\n",
    })
    # Forced psutil import -> unpassable.
    assert not _bigcodebench_gradeable({
        "code_prompt": "import psutil\ndef task_func():\n",
        "canonical_solution": "    return psutil.cpu_percent()\n",
    })


def test_humaneval_filter_excludes_eval_task():
    bench = _bench(HumanEvalBenchmark)
    assert bench.row_filter({"canonical_solution": "    return x + 1\n"})
    assert not bench.row_filter({"canonical_solution": "    return eval(expr)\n"})


def test_mbpp_filter_excludes_sys_task():
    bench = _bench(MBPPBenchmark)
    assert bench.row_filter({"code": "def f():\n    return 1"})
    assert not bench.row_filter({"code": "import sys\ndef f():\n    return sys.argv"})


def test_bigcodebench_filter_keeps_confined_drops_process_control():
    bench = _bench(BigCodeBenchBenchmark)
    # File-I/O, os/shutil and loopback-network tasks scan clean and stay.
    assert bench.row_filter({
        "code_prompt": "import os\ndef task_func(p):\n",
        "canonical_solution": "    with open(p) as f:\n        return f.read()",
    })
    assert bench.row_filter({
        "code_prompt": "import socket\ndef task_func():\n",
        "canonical_solution": "    s = socket.socket()\n    return s",
    })
    # Process-control tasks the sandbox cannot run are excluded.
    assert not bench.row_filter({
        "code_prompt": "import subprocess\ndef task_func(c):\n",
        "canonical_solution": "    return subprocess.run(c)",
    })
    assert not bench.row_filter({
        "code_prompt": "def task_func(o):\n",
        "canonical_solution": "    if hasattr(o, 'x'):\n        pass",
    })


def test_load_records_filter_stats(monkeypatch):
    """load() must capture how many rows the pre-sampling filter excluded."""
    rows = [
        {"canonical_solution": "    return 1\n"},
        {"canonical_solution": "    import subprocess\n"},  # excluded
        {"canonical_solution": "    return 2\n"},
    ]
    monkeypatch.setattr(
        "lite_bench.benchmarks.load_questions",
        lambda cfg, st, row_filter=None, filter_stats=None: _fake_load(rows, row_filter, filter_stats),
    )
    bench = _bench(HumanEvalBenchmark)
    qs = bench.load()
    assert len(qs) == 2
    assert bench.excluded_count == 1
    note = bench.exclusion_note()
    assert note and "excluded 1" in note


def test_no_filter_means_no_exclusions(monkeypatch):
    """A benchmark without a row_filter reports zero exclusions and no note."""
    rows = [{"answer": "A"}]
    monkeypatch.setattr(
        "lite_bench.benchmarks.load_questions",
        lambda cfg, st, row_filter=None, filter_stats=None: _fake_load(rows, row_filter, filter_stats),
    )

    class NoFilter(BenchmarkBase):
        name = "nofilter"

        def format_prompt(self, q):
            return ""

        def evaluate(self, q, response):
            return 0.0

    bench = NoFilter(BenchmarkConfig(name="t", enabled=True, dataset="t", num_samples=10), Settings())
    bench.load()
    assert bench.excluded_count == 0
    assert bench.exclusion_note() is None


def _fake_load(rows, row_filter, filter_stats):
    """Mimic datasets.load_questions: filter before sampling, record stats."""
    if row_filter is not None:
        pool = [r for r in rows if row_filter(r)]
        if filter_stats is not None:
            filter_stats.update(
                {"available": len(rows), "pool": len(pool), "excluded": len(rows) - len(pool)}
            )
        return [dict(r) for r in pool]
    return [dict(r) for r in rows]
