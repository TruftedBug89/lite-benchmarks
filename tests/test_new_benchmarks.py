"""Tests for HLE, SciCode, and TauBench benchmarks."""

from __future__ import annotations

import unittest

from lite_bench.benchmarks import (
    HLEBenchmark,
    SciCodeBenchmark,
    TauBenchBenchmark,
)
from lite_bench.config import BenchmarkConfig, Settings


class NewBenchmarksTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(code_exec_timeout=5)
        self.dummy_config = BenchmarkConfig(
            name="test",
            enabled=True,
            dataset="test/dataset",
            num_samples=10,
        )

    def test_hle_benchmark(self) -> None:
        bench = HLEBenchmark(self.dummy_config, self.settings)
        q = {"question": "What is the capital of France?", "answer": "Paris"}
        prompt = bench.format_prompt(q)
        self.assertIn("Humanity's Last Exam", prompt)
        self.assertIn("What is the capital of France?", prompt)

        # Test boxed evaluation
        self.assertEqual(bench.evaluate(q, "The answer is \\boxed{Paris}."), 1.0)
        self.assertEqual(bench.evaluate(q, "The answer is \\boxed{London}."), 0.0)

        # Test choice answer
        q_choice = {"question": "Select option A or B", "answer": "A"}
        self.assertEqual(bench.evaluate(q_choice, "Answer: A"), 1.0)

    def test_scicode_benchmark(self) -> None:
        bench = SciCodeBenchmark(self.dummy_config, self.settings)
        q = {
            "problem_description_main": "Compute square of x.",
            "required_dependencies": "import math",
            "general_tests": "assert square(3) == 9",
        }
        prompt = bench.format_prompt(q)
        self.assertIn("Compute square of x.", prompt)

        # Correct response
        code_resp = "def square(x):\n    return x * x"
        self.assertEqual(bench.evaluate(q, code_resp), 1.0)

        # Incorrect response
        wrong_resp = "def square(x):\n    return x + 1"
        self.assertEqual(bench.evaluate(q, wrong_resp), 0.0)

    def test_tau_bench_benchmark(self) -> None:
        bench = TauBenchBenchmark(self.dummy_config, self.settings)
        q = {
            "conversations": [
                {"role": "user", "content": "I want to return item #123."},
            ],
            "answer": [
                {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "return_item",
                                "arguments": '{"item_id": "123"}',
                            }
                        }
                    ]
                }
            ],
        }
        prompt = bench.format_prompt(q)
        self.assertIn("I want to return item #123.", prompt)

        # Correct tool call in response
        response = 'Action: {"name": "return_item", "arguments": {"item_id": "123"}}'
        self.assertEqual(bench.evaluate(q, response), 1.0)

        # Incorrect response
        wrong_response = 'I cannot help with returns.'
        self.assertEqual(bench.evaluate(q, wrong_response), 0.0)


if __name__ == "__main__":
    unittest.main()
