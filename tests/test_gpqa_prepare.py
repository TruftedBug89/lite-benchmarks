from __future__ import annotations

from lite_bench.benchmarks import GPQABenchmark
from lite_bench.config import BenchmarkConfig, Settings


def test_gpqa_option_shuffling_and_gold_alignment():
    cfg = BenchmarkConfig(name="gpqa", enabled=True, dataset="nichenshun/gpqa_diamond", num_samples=5)
    settings = Settings(seed=42)
    bench = GPQABenchmark(cfg, settings)

    raw_question = {
        "question": "What is the capital of France?",
        "Correct Answer": "Paris",
        "Incorrect Answer 1": "London",
        "Incorrect Answer 2": "Berlin",
        "Incorrect Answer 3": "Madrid",
    }

    prepared = bench.prepare(raw_question)

    assert "options" in prepared
    assert "gold_letter" in prepared
    assert len(prepared["options"]) == 4
    assert prepared["gold_letter"] in {"A", "B", "C", "D"}

    gold_idx = ord(prepared["gold_letter"]) - 65
    assert prepared["options"][gold_idx] == "Paris"

    # Evaluate correct choice
    assert bench.evaluate(prepared, prepared["gold_letter"]) == 1.0
    # Evaluate incorrect choice
    wrong_letter = "B" if prepared["gold_letter"] != "B" else "A"
    assert bench.evaluate(prepared, wrong_letter) == 0.0
