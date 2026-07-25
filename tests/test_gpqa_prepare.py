from __future__ import annotations

import json

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


def test_gpqa_metadata_mirror_schema_roundtrip():
    """The nichenshun/gpqa_diamond mirror packs the four answer options into a
    JSON-encoded ``metadata`` string (Correct Answer + Incorrect Answer 1/2/3)
    instead of the Idavidreinrein column schema. ``prepare`` must decode it,
    shuffle deterministically, derive a gold letter, and ``evaluate`` against
    that derived letter must round-trip correctly."""
    cfg = BenchmarkConfig(name="gpqa", enabled=True, dataset="nichenshun/gpqa_diamond", num_samples=5)
    settings = Settings(seed=42)
    bench = GPQABenchmark(cfg, settings)

    metadata = json.dumps(
        {
            "Correct Answer": "Paris",
            "Incorrect Answer 1": "London",
            "Incorrect Answer 2": "Berlin",
            "Incorrect Answer 3": "Madrid",
        }
    )
    raw_question = {"question": "What is the capital of France?", "metadata": metadata}

    prepared = bench.prepare(raw_question)

    assert prepared["options"] is not None
    assert len(prepared["options"]) == 4
    assert prepared["gold_letter"] in {"A", "B", "C", "D"}
    # The shuffled options must still contain exactly the four original choices.
    assert set(prepared["options"]) == {"Paris", "London", "Berlin", "Madrid"}
    # The gold-lettered option must be the correct answer.
    gold_idx = ord(prepared["gold_letter"]) - 65
    assert prepared["options"][gold_idx] == "Paris"

    # Determinism: re-preparing the SAME raw question from a fresh row yields the
    # same gold letter/options (the shuffle is seeded by seed:question_text, not
    # by call order). prepare() mutates its input, so hand it a fresh dict.
    again = bench.prepare({"question": "What is the capital of France?", "metadata": metadata})
    assert again["gold_letter"] == prepared["gold_letter"]
    assert again["options"] == prepared["options"]

    # Evaluation round-trip against the derived gold letter.
    assert bench.evaluate(prepared, prepared["gold_letter"]) == 1.0
    wrong_letter = "B" if prepared["gold_letter"] != "B" else "A"
    assert bench.evaluate(prepared, wrong_letter) == 0.0
