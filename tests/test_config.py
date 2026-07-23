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


if __name__ == "__main__":
    unittest.main()
