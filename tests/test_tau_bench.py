from __future__ import annotations

from lite_bench.benchmarks import TauBenchBenchmark
from lite_bench.config import BenchmarkConfig, Settings


def test_tau_bench_function_and_argument_matching():
    cfg = BenchmarkConfig(name="tau_bench", enabled=True, dataset="amityco/tau-bench", num_samples=5)
    settings = Settings()
    bench = TauBenchBenchmark(cfg, settings)

    question = {
        "answer": [
            {
                "tool_calls": [
                    {
                        "function": {
                            "name": "cancel_order",
                            "arguments": {"order_id": "12345", "reason": "user_request"},
                        }
                    }
                ]
            }
        ]
    }

    # Match name AND arguments -> 1.0
    correct_resp = '{"name": "cancel_order", "arguments": {"order_id": "12345", "reason": "user_request"}}'
    assert bench.evaluate(question, correct_resp) == 1.0

    # Match name but WRONG argument -> 0.0
    wrong_arg_resp = '{"name": "cancel_order", "arguments": {"order_id": "99999", "reason": "user_request"}}'
    assert bench.evaluate(question, wrong_arg_resp) == 0.0

    # Match name but missing argument -> 0.0
    missing_arg_resp = '{"name": "cancel_order", "arguments": {"order_id": "12345"}}'
    assert bench.evaluate(question, missing_arg_resp) == 0.0
