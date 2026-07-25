"""Configuration and aggregation regression tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lite_bench.config import Config, load_config


class ConfigTests(unittest.TestCase):
    def test_partial_scores_do_not_receive_synthetic_zeroes(self) -> None:
        config = Config(categories={"one": ["a", "b"], "two": ["c"]})
        self.assertEqual(config.category_score({"a": 0.8}, "one"), 0.8)
        self.assertEqual(config.overall_score({"a": 0.8}), 0.8)

    def test_none_scores_excluded_from_aggregation(self) -> None:
        """A benchmark that failed to score (None) must be dropped, not treated
        as 0.0. The realistic failure mode: a provider error or scorer exception
        yields a None bench score, and a 0.0 must NOT be silently averaged in
        its place — that would tank an otherwise-good model's category score."""
        config = Config(categories={"math": ["aime", "math_500"], "code": ["humaneval"]})
        # aime present but None (failed); math_500 = 0.9 -> category "math" = 0.9.
        self.assertEqual(config.category_score({"aime": None, "math_500": 0.9}, "math"), 0.9)
        # A *real* 0.0 must still count (no synthetic-zero masking of failures).
        self.assertEqual(config.category_score({"aime": 0.0, "math_500": 0.8}, "math"), 0.4)
        # Every benchmark None -> whole category None (excluded from overall).
        self.assertIsNone(config.category_score({"aime": None, "math_500": None}, "math"))

    def test_overall_score_excludes_none_category_scores(self) -> None:
        """When an entire category has no numeric scores (all None), it must be
        excluded from the overall average rather than dragging it toward 0."""
        config = Config(categories={"math": ["aime", "math_500"], "code": ["humaneval", "mbpp"], "mcq": ["gpqa"]})
        scores = {"aime": None, "math_500": None, "humaneval": 0.6, "mbpp": 0.8, "gpqa": None}
        # Only "code" has a numeric category score (0.7); math and mcq are None.
        self.assertEqual(config.overall_score(scores), 0.7)
        # If everything is None, overall is None (no leaderboard entry).
        self.assertIsNone(config.overall_score({"aime": None, "humaneval": None, "gpqa": None}))

    def test_duplicate_model_names_are_rejected(self) -> None:
        content = """
models:
  - id: provider/one
    name: Same
  - id: provider/two
    name: Same
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "display names must be unique"):
                load_config(path)

    def test_dataset_revision_is_loaded(self) -> None:
        content = """
benchmarks:
  sample:
    dataset: example/dataset
    revision: abcdef
categories:
  category:
    benchmarks: [sample]
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(content, encoding="utf-8")
            self.assertEqual(load_config(path).benchmarks["sample"].revision, "abcdef")

    def test_thinking_effort_and_extra_params_are_loaded(self) -> None:
        content = """
models:
  - id: provider/model-one
    name: Model One (Max Thinking)
    thinking_effort: max
    extra_params:
      custom_param: 123
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(content, encoding="utf-8")
            cfg = load_config(path)
            self.assertEqual(cfg.models[0].thinking_effort, "max")
            self.assertEqual(cfg.models[0].extra_params, {"custom_param": 123})



if __name__ == "__main__":
    unittest.main()
