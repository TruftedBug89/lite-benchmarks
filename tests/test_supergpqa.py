from __future__ import annotations

from lite_bench.benchmarks import SuperGPQABenchmark
from lite_bench.config import BenchmarkConfig, Settings


def test_supergpqa_hard_filter_and_letter_extraction():
    cfg = BenchmarkConfig(name="supergpqa", enabled=True, dataset="m-a-p/SuperGPQA", num_samples=5)
    settings = Settings()
    bench = SuperGPQABenchmark(cfg, settings)

    # Easy item should be skipped
    easy_q = {
        "question": "What is 2+2?",
        "difficulty": "easy",
        "options": ["3", "4", "5"],
        "answer_letter": "B",
    }
    prep_easy = bench.prepare(easy_q)
    assert prep_easy.get("_skip") is True

    # Hard item should be kept and normalized
    hard_q = {
        "question": "Which quantum effect explains Josephson junctions?",
        "difficulty": "hard",
        "options": ["Macroscopic quantum coherence", "SQUIDs", "BCS Cooper pairs", "Tunneling"],
        "answer_letter": "A",
    }
    prep_hard = bench.prepare(hard_q)
    assert prep_hard.get("_skip") is not True
    assert prep_hard["gold_letter"] == "A"
    assert len(prep_hard["options"]) == 4

    prompt = bench.format_prompt(prep_hard)
    assert "A." in prompt
    assert "D." in prompt
    assert "(A-D)" in prompt

    # Evaluate correct letter
    assert bench.evaluate(prep_hard, "The answer is (A).") == 1.0
    # Evaluate wrong letter
    assert bench.evaluate(prep_hard, "Choice C") == 0.0
