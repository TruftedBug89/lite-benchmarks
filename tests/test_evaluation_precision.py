"""Comprehensive tests for benchmark solution evaluation, precision, and verifier edge cases."""

from __future__ import annotations

import string
import unittest

from lite_bench.benchmarks import (
    AIMEBenchmark,
    GPQABenchmark,
    MATH500Benchmark,
    SciBenchBenchmark,
    SuperGPQABenchmark,
    _extract_boxed,
    _extract_letter,
    _strip_code_blocks,
)
from lite_bench.config import BenchmarkConfig, Settings
from lite_bench.ifeval_verifiers import (
    verify_forbidden_words,
    verify_json_format,
    verify_keyword_frequency,
    verify_keywords_existence,
    verify_number_paragraphs,
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

    def test_supergpqa_precision(self) -> None:
        bench = SuperGPQABenchmark(self.config, self.settings)
        q = {"difficulty": "hard", "options": ["A_val", "B_val", "C_val"], "answer_letter": "B"}
        self.assertEqual(bench.evaluate(q, "The correct choice is B."), 1.0)
        self.assertEqual(bench.evaluate(q, "Choice A is right."), 0.0)

    def test_ifeval_relation_verifiers(self) -> None:
        # "at most" relation
        self.assertTrue(verify_number_words("one two three", relation="at most", num_words=5))
        self.assertFalse(verify_number_words("one two three four five six", relation="at most", num_words=5))

        # "more than" relation
        self.assertTrue(verify_number_words("one two three four five six", relation="more than", num_words=5))

        # "equal to" relation
        self.assertTrue(verify_number_words("one two three", relation="equal to", num_words=3))

    def test_ifeval_paragraph_splitting(self) -> None:
        # Official IFEval: number_paragraphs counts paragraphs separated by the
        # "***" markdown divider (NOT blank lines).
        text = "Paragraph one is here. *** Paragraph two is here. *** Paragraph three is here."
        self.assertTrue(verify_number_paragraphs(text, num_paragraphs=3))
        # Blank-line separation alone is a single paragraph for this instruction.
        blank = "Para one.\n\nPara two.\n\nPara three."
        self.assertTrue(verify_number_paragraphs(blank, num_paragraphs=1))
        self.assertFalse(verify_number_paragraphs(blank, num_paragraphs=3))

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

    # ── Multiple-choice extraction: prose false positives must not hijack ──

    def test_extract_letter_ignores_pronoun_i_in_extended_options(self) -> None:
        """With a 10-option (A–J) answer set the letter ``I`` is a *valid*
        option, so the English pronoun "I" (as in "I am sure") must NOT be
        extracted as the answer. Before the ``_PRONOUN_I`` filter, the reverse
        scan would pick "I" over the real answer "J" here."""
        valid_ten = set(string.ascii_uppercase[:10])  # {A..J}
        # "I" trails the real answer J and is the last bare letter token: a
        # naive reverse scan would return "I" (valid for A-J) -> wrong.
        self.assertEqual(_extract_letter("J is my answer. I am sure", valid_ten), "J")
        # The pronoun never wins even when no other letter is present yet an
        # explicit "answer is X" clause gives the real one.
        self.assertEqual(_extract_letter("I think the answer is F", valid_ten), "F")

    def test_extract_letter_lowercase_article_not_hijacked(self) -> None:
        """The lowercase articles ``a``/``i`` must never be promoted to an
        option letter, even when they are the only letter-shaped tokens in the
        last line — a real answer stated elsewhere must still win."""
        valid = {"A", "B", "C", "D"}
        # Prose "a nice day" must not become option A; the real answer is C.
        self.assertEqual(_extract_letter("It is a nice day. The answer is (C).", valid), "C")
        # A trailing lowercase article alone must not produce a false letter.
        self.assertIsNone(_extract_letter("i am unsure here", valid))

    # ── MATH-500: symbolic gold must not be matched to a bare number ─────

    def test_math500_symbolic_gold_not_matched_to_bare_number(self) -> None:
        """Numeric comparison only fires when the GOLD answer is itself a plain
        number. A symbolic gold (\\sqrt{2}, 3\\sqrt{2}, 2^5) must NOT be matched
        by digits pulled out of the model's response — that used to mark wrong
        answers correct (\\sqrt{2} -> "2", 2^5 -> "5"/"32")."""
        bench = MATH500Benchmark(self.config, self.settings)

        # Symbolic gold vs. a bare numeric boxed answer -> wrong.
        self.assertEqual(bench.evaluate({"answer": "\\sqrt{2}"}, "\\boxed{2}"), 0.0)
        self.assertEqual(bench.evaluate({"answer": "3\\sqrt{2}"}, "\\boxed{2}"), 0.0)
        # 2^5 is symbolic; a model that computes 32 must still mismatch on latex
        # (gold is not a plain number, so numeric comparison is skipped).
        self.assertEqual(bench.evaluate({"answer": "2^5"}, "\\boxed{32}"), 0.0)
        # A LaTeX fraction gold vs. a bare number -> wrong.
        self.assertEqual(bench.evaluate({"answer": "\\frac{3}{5}"}, "\\boxed{2}"), 0.0)
        self.assertEqual(bench.evaluate({"answer": "\\frac{3}{5}"}, "42"), 0.0)

    def test_math500_plain_numeric_gold_still_matches(self) -> None:
        """Positive control: a plain numeric gold must still match a numeric
        boxed prediction (the no-false-positive guard did not break real
        numeric scoring)."""
        bench = MATH500Benchmark(self.config, self.settings)
        self.assertEqual(bench.evaluate({"answer": "42"}, "\\boxed{42}"), 1.0)
        self.assertEqual(bench.evaluate({"answer": "3.14"}, "The result is \\boxed{3.14}."), 1.0)
        # Comma-grouped numeric boxed still normalizes to the plain gold number.
        self.assertEqual(bench.evaluate({"answer": "1000000"}, "\\boxed{1,000,000}"), 1.0)
        # Wrong plain number -> 0.0
        self.assertEqual(bench.evaluate({"answer": "42"}, "\\boxed{43}"), 0.0)


if __name__ == "__main__":
    unittest.main()
