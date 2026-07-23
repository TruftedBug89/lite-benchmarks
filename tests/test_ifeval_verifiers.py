"""Regression tests for strict IFEval dataset argument handling."""

from __future__ import annotations

import unittest

from lite_bench.ifeval_verifiers import verify_all, verify_instruction


class IFEvalVerifierTests(unittest.TestCase):
    def test_official_argument_names_are_used(self) -> None:
        response = "*alpha* *beta* this has exactly seven simple words"
        self.assertTrue(
            verify_all(
                [
                    "detectable_format:number_highlighted_sections",
                    "length_constraints:number_words",
                    "punctuation:no_comma",
                ],
                response,
                [
                    {"num_highlights": 2},
                    {"relation": "at least", "num_words": 7},
                    {},
                ],
            )
        )

    def test_relation_based_constraints_are_strict(self) -> None:
        self.assertTrue(
            verify_instruction(
                "length_constraints:number_sentences",
                "One. Two.",
                {"relation": "at least", "num_sentences": 2},
            )
        )
        self.assertFalse(
            verify_instruction(
                "length_constraints:number_sentences",
                "One. Two.",
                {"relation": "less than", "num_sentences": 2},
            )
        )

    def test_dataset_specific_argument_names_are_checked(self) -> None:
        self.assertTrue(
            verify_instruction(
                "change_case:capital_word_frequency",
                "ONE TWO three",
                {"capital_relation": "at least", "capital_frequency": 2},
            )
        )
        self.assertTrue(
            verify_instruction(
                "combination:repeat_prompt",
                "Repeat this, then explain it.",
                {"prompt_to_repeat": "Repeat this"},
            )
        )

    def test_instruction_and_argument_lengths_must_match(self) -> None:
        self.assertFalse(
            verify_all(
                ["punctuation:no_comma"],
                "No commas here",
                [],
            )
        )


if __name__ == "__main__":
    unittest.main()
