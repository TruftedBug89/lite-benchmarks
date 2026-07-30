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

    # ── MATH-500: false positives stay blocked, true equivalences now pass ─

    def test_math500_symbolic_gold_not_matched_to_bare_number(self) -> None:
        """A symbolic gold must NOT be matched by an UNequal bare number. The
        sympy fallback proves equivalence, so it keeps the false-positive guards
        (\\sqrt{2} != 2, 3\\sqrt{2} != 2, \\frac{3}{5} != 2) while now accepting
        answers that are genuinely equal (see the equivalence test below)."""
        bench = MATH500Benchmark(self.config, self.settings)

        # Symbolic gold vs. an unequal bare numeric boxed answer -> wrong.
        self.assertEqual(bench.evaluate({"answer": "\\sqrt{2}"}, "\\boxed{2}"), 0.0)
        self.assertEqual(bench.evaluate({"answer": "3\\sqrt{2}"}, "\\boxed{2}"), 0.0)
        # A LaTeX fraction gold vs. an unequal bare number -> wrong.
        self.assertEqual(bench.evaluate({"answer": "\\frac{3}{5}"}, "\\boxed{2}"), 0.0)
        self.assertEqual(bench.evaluate({"answer": "\\frac{3}{5}"}, "42"), 0.0)

    def test_math500_symbolic_equivalence_now_matches(self) -> None:
        """Factually-equal answers in different forms must score correct via the
        deterministic sympy fallback (no LLM). 2^5 == 32, \\frac{1}{2} == 0.5,
        \\sqrt{4} == 2 — these used to score 0.0 under pure string matching."""
        bench = MATH500Benchmark(self.config, self.settings)
        self.assertEqual(bench.evaluate({"answer": "2^5"}, "\\boxed{32}"), 1.0)
        self.assertEqual(bench.evaluate({"answer": "\\frac{1}{2}"}, "\\boxed{0.5}"), 1.0)
        self.assertEqual(bench.evaluate({"answer": "\\sqrt{4}"}, "\\boxed{2}"), 1.0)
        self.assertEqual(bench.evaluate({"answer": "10^6"}, "\\boxed{1000000}"), 1.0)
        # Still-equal symbolic forms match too.
        self.assertEqual(bench.evaluate({"answer": "\\dfrac{1}{2}"}, "\\boxed{\\frac{1}{2}}"), 1.0)

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

    # ── Scientific-notation extraction (SciBench physics answers) ──────────

    def test_sci_notation_extraction(self) -> None:
        """_extract_number must understand e-notation and LaTeX \\times 10^n so
        SciBench answers like 4.4e-9 are not misread as their exponent."""
        from lite_bench.benchmarks import _extract_number

        self.assertEqual(_extract_number("\\boxed{4.4 \\times 10^{-9}}"), "4.4e-09")
        self.assertEqual(_extract_number("\\boxed{2.5 \\cdot 10^{3}}"), "2500.0")
        self.assertEqual(_extract_number("the flux is 4.4e-9 W"), "4.4e-09")
        self.assertEqual(_extract_number("\\boxed{5.07E1}"), "50.7")
        self.assertEqual(_extract_number("-1.5e-3 meters"), "-0.0015")
        # Plain numbers and markers are unaffected.
        self.assertEqual(_extract_number("#### 17"), "17")
        self.assertEqual(_extract_number("\\boxed{42}"), "42")

    def test_scibench_sci_notation_answer_matches(self) -> None:
        """A model answering in scientific notation must match a SciBench gold."""
        bench = SciBenchBenchmark(self.config, self.settings)
        q = {"problem_text": "Compute the flux.", "answer_number": 4.4e-9}
        self.assertEqual(bench.evaluate(q, "\\boxed{4.4 \\times 10^{-9}}"), 1.0)
        self.assertEqual(bench.evaluate(q, "\\boxed{4.4e-9}"), 1.0)
        # A genuinely wrong value still scores 0.
        self.assertEqual(bench.evaluate(q, "\\boxed{9.9 \\times 10^{-9}}"), 0.0)


if __name__ == "__main__":
    unittest.main()
