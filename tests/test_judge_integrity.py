"""Comprehensive test suite for Judge System Integrity and Backward Compatibility."""

from __future__ import annotations

import glob
import os
import unittest

from lite_bench.benchmarks import (
    AIMEBenchmark,
    MATH500Benchmark,
    SciBenchBenchmark,
    _extract_boxed,
    _extract_letter,
    _extract_number,
    _normalize_latex,
)
from lite_bench.config import BenchmarkConfig, Settings, load_config
from lite_bench.ifeval_verifiers import (
    verify_forbidden_words,
    verify_json_format,
    verify_keywords_existence,
    verify_number_paragraphs,
    verify_number_words,
)
from lite_bench.results_store import compute_question_hash, load_latest_results


class JudgeIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(code_exec_timeout=5)
        self.config_yaml_path = "config.yaml" if os.path.exists("config.yaml") else None

    # ---------------------------------------------------------------------------
    # 1. Historical Results & Schema Viability Tests
    # ---------------------------------------------------------------------------

    def test_historical_results_loading_viability(self) -> None:
        """Verify that loading any legacy or active result JSON file retains models and valid scores."""
        if not self.config_yaml_path:
            self.skipTest("config.yaml not found in CWD")

        config = load_config(self.config_yaml_path)
        json_files = glob.glob("results/**/*.json", recursive=True)
        if not json_files:
            self.skipTest("No result files found in results/ directory")

        for filepath in json_files:
            results = load_latest_results(config, filepath)
            self.assertIsInstance(results, dict, f"Failed to parse {filepath} as dict")
            for mname, mdata in results.items():
                self.assertIn("summary", mdata, f"Model {mname} in {filepath} missing summary rollup")
                summary = mdata["summary"]
                self.assertIn("completed_benchmarks", summary)
                self.assertIn("overall_score", summary)

    def test_question_hash_determinism(self) -> None:
        """Verify compute_question_hash is deterministic across various question structures."""
        questions_a = [{"question": "What is 2+2?"}, {"question": "What is 3+3?"}]
        questions_b = [{"prompt": "What is 2+2?"}, {"prompt": "What is 3+3?"}]
        questions_c = [{"problem": "What is 2+2?"}, {"problem": "What is 3+3?"}]

        hash_a = compute_question_hash(questions_a)
        hash_b = compute_question_hash(questions_b)
        hash_c = compute_question_hash(questions_c)

        self.assertEqual(len(hash_a), 12)
        self.assertEqual(hash_a, hash_b)
        self.assertEqual(hash_b, hash_c)
        # Repeated call must produce identical hash
        self.assertEqual(compute_question_hash(questions_a), hash_a)

    # ---------------------------------------------------------------------------
    # 2. Answer Extraction Integrity Tests
    # ---------------------------------------------------------------------------

    def test_extract_boxed_nested_and_whitespace(self) -> None:
        """Verify _extract_boxed handles nested TeX braces and optional whitespace."""
        self.assertEqual(_extract_boxed("The answer is \\boxed{42}."), "42")
        self.assertEqual(_extract_boxed("Answer: \\boxed {100}"), "100")
        self.assertEqual(_extract_boxed("\\boxed{\\frac{x^2}{y + 1}}"), "\\frac{x^2}{y + 1}")
        self.assertIsNone(_extract_boxed("No boxed content here"))

    def test_extract_number_fraction_exclusion(self) -> None:
        """Verify _extract_number extracts correct digits without pulling TeX fraction denominators."""
        self.assertEqual(_extract_number("Result is #### 42"), "42")
        self.assertEqual(_extract_number("Answer: \\boxed{-3.14}"), "-3.14")
        self.assertEqual(_extract_number("The answer is 1,000,000."), "1000000")
        # Should not extract isolated denominator from \frac{3}{5}
        self.assertIsNone(_extract_number("The fraction is \\frac{3}{5}"))

    def test_extract_letter_prose_isolation(self) -> None:
        """Verify multi-choice option letter extraction is isolated from prose false positives."""
        valid_options = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}

        # Pronoun "I think" must not be extracted as option I when real option J is present
        self.assertEqual(_extract_letter("I think option J is correct.", valid_options), "J")

        # Lowercase article "a" should not hijack option A when explicit choice C is stated
        self.assertEqual(_extract_letter("This is a tough question. The answer is (C).", valid_options), "C")

        # Explicit boxed option wins over surrounding prose
        self.assertEqual(_extract_letter("I believe A is possible, but \\boxed{B} is right.", valid_options), "B")

    def test_normalize_latex_consistency(self) -> None:
        """Verify _normalize_latex normalizes TeX spacing, text commands, and fraction variants."""
        self.assertEqual(_normalize_latex(" \\dfrac{3}{5} "), "\\frac{3}{5}")
        self.assertEqual(_normalize_latex("3\\,x"), "3x")
        self.assertEqual(_normalize_latex("90^\\circ"), "90^")

    # ---------------------------------------------------------------------------
    # 3. Benchmark Evaluator Precision Tests
    # ---------------------------------------------------------------------------

    def test_math500_evaluator_precision(self) -> None:
        """Verify MATH-500 evaluator strictly distinguishes symbolic gold from numeric gold."""
        bench = MATH500Benchmark(
            BenchmarkConfig(name="math_500", enabled=True, dataset="HuggingFaceH4/MATH-500", num_samples=50),
            self.settings,
        )

        # Plain numeric gold vs numeric boxed
        self.assertEqual(bench.evaluate({"answer": "42"}, "Therefore \\boxed{42}."), 1.0)
        self.assertEqual(bench.evaluate({"answer": "42"}, "\\boxed{43}"), 0.0)

        # Symbolic gold vs bare number must NOT give false positive
        self.assertEqual(bench.evaluate({"answer": "\\sqrt{2}"}, "\\boxed{2}"), 0.0)

        # LaTeX fraction matching
        self.assertEqual(bench.evaluate({"answer": "\\dfrac{1}{2}"}, "\\boxed{\\frac{1}{2}}"), 1.0)

    def test_aime_evaluator_precision(self) -> None:
        """Verify AIME evaluator evaluates integer solutions correctly."""
        bench = AIMEBenchmark(
            BenchmarkConfig(name="aime", enabled=True, dataset="AI-MO/aimo-validation-aime", num_samples=50),
            self.settings,
        )

        q = {"problem": "Find integer n.", "answer": "314"}
        self.assertEqual(bench.evaluate(q, "The final answer is \\boxed{314}."), 1.0)
        self.assertEqual(bench.evaluate(q, "Answer: 314"), 1.0)
        self.assertEqual(bench.evaluate(q, "Answer: 315"), 0.0)

    def test_scibench_evaluator_tolerance(self) -> None:
        """Verify SciBench evaluator handles zero values and floating-point tolerances."""
        bench = SciBenchBenchmark(
            BenchmarkConfig(name="scibench", enabled=True, dataset="xw27/scibench", num_samples=50),
            self.settings,
        )

        q_zero = {"problem_text": "Calculate flux.", "answer_number": 0.0}
        self.assertEqual(bench.evaluate(q_zero, "\\boxed{0}"), 1.0)
        self.assertEqual(bench.evaluate(q_zero, "\\boxed{0.0}"), 1.0)

        q_float = {"problem_text": "Calculate force.", "answer_number": 9.81}
        self.assertEqual(bench.evaluate(q_float, "\\boxed{9.81}"), 1.0)
        self.assertEqual(bench.evaluate(q_float, "\\boxed{9.82}"), 1.0)  # within 1% rel_tol

    # ---------------------------------------------------------------------------
    # 4. IFEval Verifiers Constraint Tests
    # ---------------------------------------------------------------------------

    def test_ifeval_verifiers_integrity(self) -> None:
        """Verify IFEval verifier functions behave correctly under various constraints."""
        # Word counts with relations
        self.assertTrue(verify_number_words("alpha beta gamma", relation="equal to", num_words=3))
        self.assertTrue(verify_number_words("alpha beta", relation="at most", num_words=3))
        self.assertTrue(verify_number_words("alpha beta gamma delta", relation="more than", num_words=3))

        # Keyword existence & regex escaping
        self.assertTrue(verify_keywords_existence("Code in C++ works.", keywords=["C++"]))
        self.assertFalse(verify_forbidden_words("Code in Python.", forbidden_words=["Python"]))

        # Paragraph splitting
        three_paras = "Para 1 *** Para 2 *** Para 3"
        self.assertTrue(verify_number_paragraphs(three_paras, num_paragraphs=3))

        # JSON code block format
        json_resp = "```json\n{\"valid\": true}\n```"
        self.assertTrue(verify_json_format(json_resp))


if __name__ == "__main__":
    unittest.main()
