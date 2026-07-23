"""Comprehensive tests for benchmark solution evaluation, precision, and verifier edge cases."""

from __future__ import annotations

import unittest

from lite_bench.benchmarks import (
    AIMEBenchmark,
    GPQABenchmark,
    HLEBenchmark,
    MATH500Benchmark,
    MMLUProBenchmark,
    SciBenchBenchmark,
    TauBenchBenchmark,
    _extract_boxed,
    _extract_letter,
    _extract_number,
    _strip_code_blocks,
)
from lite_bench.config import BenchmarkConfig, Settings
from lite_bench.ifeval_verifiers import (
    verify_forbidden_words,
    verify_json_format,
    verify_keyword_frequency,
    verify_keywords_existence,
    verify_number_bullet_lists,
    verify_number_highlighted_sections,
    verify_number_paragraphs,
    verify_number_sentences,
    verify_number_words,
)


class EvaluationPrecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(code_exec_timeout=5)
        self.config = BenchmarkConfig(name="test", enabled=True, dataset="test", num_samples=10)

    def test_strip_code_blocks(self) -> None:
        raw = "Here is the solution:\n```python\ndef solution():\n    return 42\n```\nHope this helps!"
        stripped = _strip_code_blocks(raw)
        self.assertEqual(stripped, "def solution():\n    return 42")

    def test_extract_boxed_nested_braces(self) -> None:
        raw = "First try \\boxed{1}. Final answer: \\boxed{\\frac{x^2}{y + 1}}."
        boxed = _extract_boxed(raw)
        self.assertEqual(boxed, "\\frac{x^2}{y + 1}")

    def test_extract_letter_precision(self) -> None:
        raw = "A detailed analysis shows that option A is tempting, but the correct choice is (C)."
        letter = _extract_letter(raw, {"A", "B", "C", "D"})
        self.assertEqual(letter, "C")

    def test_gpqa_benchmark(self) -> None:
        bench = GPQABenchmark(self.config, self.settings)
        q = {
            "Question": "What is the capital of France?",
            "Correct Answer": "Paris",
            "Incorrect Answer 1": "London",
            "Incorrect Answer 2": "Berlin",
            "Incorrect Answer 3": "Madrid",
        }
        prompt = bench.format_prompt(q)
        self.assertIn("A.", prompt)
        self.assertIn("Paris", prompt)

        # Evaluating response giving the right letter option
        pred_letter = None
        for letter in ["A", "B", "C", "D"]:
            if bench.evaluate(q, f"The answer is {letter}.") == 1.0:
                pred_letter = letter
                break
        self.assertIsNotNone(pred_letter)

    def test_scibench_zero_answer(self) -> None:
        bench = SciBenchBenchmark(self.config, self.settings)
        q = {"problem_text": "Compute net flux.", "answer_number": 0.0, "solution": "Flux is zero."}
        self.assertEqual(bench.evaluate(q, "The result is \\boxed{0}."), 1.0)
        self.assertEqual(bench.evaluate(q, "The result is \\boxed{0.0}."), 1.0)

    def test_aime_boxed_extraction(self) -> None:
        bench = AIMEBenchmark(self.config, self.settings)
        q = {"problem": "Find n.", "solution": "Steps... Thus \\boxed{42}. (Ref 2024)"}
        self.assertEqual(bench.evaluate(q, "Answer: \\boxed{42}"), 1.0)

    def test_math500_fraction_and_latex_precision(self) -> None:
        bench = MATH500Benchmark(self.config, self.settings)
        q = {"problem": "Simplify.", "answer": "\\dfrac{3}{5}"}
        
        # Exact/normalized match
        self.assertEqual(bench.evaluate(q, "\\boxed{\\frac{3}{5}}"), 1.0)
        
        # Wrong fraction with same denominator must NOT evaluate to 1.0
        self.assertEqual(bench.evaluate(q, "\\boxed{\\frac{4}{5}}"), 0.0)

    def test_hle_short_string_precision(self) -> None:
        bench = HLEBenchmark(self.config, self.settings)
        q = {"question": "Is it true?", "answer": "no"}
        
        # Response with boxed answer
        self.assertEqual(bench.evaluate(q, "\\boxed{no}"), 1.0)
        
        # Unrelated text containing the word "no" should NOT blindly pass unless boxed/exact
        self.assertEqual(bench.evaluate(q, "There is no doubt that the answer is yes."), 0.0)

    def test_ifeval_relation_verifiers(self) -> None:
        # "at most" relation
        self.assertTrue(verify_number_words("one two three", relation="at most", num_words=5))
        self.assertFalse(verify_number_words("one two three four five six", relation="at most", num_words=5))

        # "more than" relation
        self.assertTrue(verify_number_words("one two three four five six", relation="more than", num_words=5))

        # "equal to" relation
        self.assertTrue(verify_number_words("one two three", relation="equal to", num_words=3))

    def test_ifeval_paragraph_splitting(self) -> None:
        text = "Paragraph one is here.\n\nParagraph two is here.\n\nParagraph three is here."
        self.assertTrue(verify_number_paragraphs(text, num_paragraphs=3))

    def test_ifeval_keyword_regex_escaping(self) -> None:
        text = "I love programming in C++!"
        self.assertTrue(verify_keywords_existence(text, keywords=["C++"]))
        self.assertTrue(verify_keyword_frequency(text, keyword="C++", frequency=1))
        self.assertFalse(verify_forbidden_words(text, forbidden_words=["C++"]))

    def test_ifeval_forbidden_words_word_boundaries(self) -> None:
        text = "This item belongs in the category."
        # Substring "cat" inside "category" should NOT trigger forbidden word "cat"
        self.assertTrue(verify_forbidden_words(text, forbidden_words=["cat"]))

    def test_ifeval_json_code_blocks(self) -> None:
        response = "```json\n{\"status\": \"ok\"}\n```"
        self.assertTrue(verify_json_format(response))


if __name__ == "__main__":
    unittest.main()
