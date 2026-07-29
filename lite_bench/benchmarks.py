"""Benchmark implementations using real datasets with built-in verification."""

from __future__ import annotations

import json
import math
import random
import re
from abc import ABC, abstractmethod

from rich.console import Console

from .config import BenchmarkConfig, Settings
from .datasets import load_questions
from .ifeval_verifiers import verify_all
from .sandbox import execute_sandboxed

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_latex(text: str) -> str:
    if not text:
        return text
    # Strip common LaTeX formatting wrappers: \mathbf{}, \mathrm{}, \text{}, \textbf{}, \mathit{}, etc.
    cleaned = re.sub(
        r"\\(?:mathbf|mathrm|text|textbf|mathit|mathsf|mathtt)\{([^{}]+)\}", r"\1", text
    )
    cleaned = re.sub(r"\\(?:left|right|displaystyle)", "", cleaned)
    cleaned = cleaned.replace("$", "").strip()
    return cleaned


def _strip_code_blocks(text: str) -> str:
    text = text.strip()
    matches = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.DOTALL)
    if matches:
        with_def = [m for m in matches if "def " in m or "class " in m or "assert" in m]
        if with_def:
            return with_def[0].strip()
        return max(matches, key=len).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:python|py)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _extract_number(text: str) -> str | None:
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").strip()

    boxed = _extract_boxed(text)
    if boxed:
        cleaned_boxed = _clean_latex(boxed)
        m_boxed = re.search(r"^-?[\d,]+(?:\.\d+)?$", cleaned_boxed.replace(",", "").strip())
        if m_boxed:
            return m_boxed.group(0)

    # Avoid extracting isolated digits from LaTeX fractions like \frac{3}{5}, \dfrac{3}{5}, \tfrac{3}{5}
    cleaned = re.sub(r"\\(?:d|t)?frac\{[^{}]+\}\{[^{}]+\}", "", text)
    cleaned = _clean_latex(cleaned)
    # The lookahead deliberately allows a trailing sentence period ("...is 42.")
    # while the lookbehind still stops us splitting decimals like 3.14.
    nums = re.findall(r"(?<![\d\w.-])-?[\d,]+(?:\.\d+)?(?![\d\w-])", cleaned)
    if nums:
        return nums[-1].replace(",", "").strip()
    return None


# Matches the English pronoun "I" when followed by a lowercase word
# ("I think", "I believe", ...), which is nearly never an option letter.
_PRONOUN_I = re.compile(r"\bI\b(?=\s+[a-z])")


def _candidate_letters(text: str) -> list[str]:
    """Single-letter option candidates in ``text``, minus prose false positives:
    the article "a"/"i" (always lowercase) and the pronoun "I" ("I think")."""
    out: list[str] = []
    cleaned_text = _clean_latex(text)
    for m in re.finditer(r"\b([A-Ja-j])\b", cleaned_text):
        tok = m.group(1)
        if tok in ("a", "i"):
            continue
        if tok == "I" and _PRONOUN_I.match(cleaned_text, m.start()):
            continue
        out.append(tok)
    return out


def _extract_letter(response: str, valid: set[str] | None = None) -> str | None:
    if valid is None:
        valid = {"A", "B", "C", "D"}
    valid_upper = {v.upper() for v in valid}

    boxed = _extract_boxed(response)
    if boxed:
        cleaned_boxed = _clean_latex(boxed).strip().upper()
        if cleaned_boxed in valid_upper:
            return cleaned_boxed
        cands_boxed = _candidate_letters(cleaned_boxed)
        if cands_boxed and cands_boxed[-1].upper() in valid_upper:
            return cands_boxed[-1].upper()

    strong = list(
        re.finditer(r"(?:answer|choice)\s*(?:is|:)\s*\(?([A-Ja-j])\)?", response, flags=re.IGNORECASE)
    )
    if strong:
        last_match = strong[-1].group(1).upper()
        if last_match in valid_upper:
            return last_match

    lines = [ln for ln in response.strip().splitlines() if ln.strip()]

    standalone = None
    for ln in lines:
        cleaned_ln = _clean_latex(ln)
        m = re.fullmatch(r"\s*\(?([A-Ja-j])\)?[.\):]?\s*", cleaned_ln)
        if m and m.group(1).upper() in valid_upper:
            standalone = m.group(1).upper()
    if standalone:
        return standalone

    weak = list(
        re.finditer(r"(?:answer|choice|option)\s*(?:is|:)?\s*\(?([A-Ja-j])\)?", response, flags=re.IGNORECASE)
    )
    if weak:
        last_match = weak[-1].group(1).upper()
        if last_match in valid_upper:
            return last_match

    if lines:
        cands = _candidate_letters(lines[-1])
        if cands and cands[-1].upper() in valid_upper:
            return cands[-1].upper()

    for letter in reversed(_candidate_letters(response)):
        if letter.upper() in valid_upper:
            return letter.upper()

    return None


def _extract_boxed(text: str) -> str | None:
    idx = text.rfind(r"\boxed")
    if idx != -1:
        # Tolerate optional whitespace between \boxed and the opening brace
        # (\boxed{42} and \boxed {42}).
        brace = text.find("{", idx + len(r"\boxed"))
        if brace != -1:
            substr = text[brace + 1 :]
            open_braces = 1
            chars = []
            for c in substr:
                if c == "{":
                    open_braces += 1
                elif c == "}":
                    open_braces -= 1
                    if open_braces == 0:
                        break
                chars.append(c)
            # Support completed or truncated boxed sequences
            if chars:
                return _clean_latex("".join(chars).strip())

    m = re.findall(r"\\boxed\s*\{([^{}]+)\}", text)
    if m:
        return _clean_latex(m[-1].strip())
    return None


def _execute_code(
    untrusted_code: str, trusted_code: str, timeout: int, *, allow_execution: bool = False
) -> bool:
    """Run model-generated code + trusted test harness in the sandbox.

    ``allow_execution`` is the opt-in gate, threaded from
    ``settings.allow_unsafe_code_execution``. It is enforced INSIDE the sandbox
    (``execute_sandboxed`` fails closed when False), so even a direct
    ``evaluate()`` call cannot run model code by accident. The untrusted
    portion is AST-scanned first; on Windows the child is additionally confined
    by a Job Object, and it always runs with a scrubbed env in a throwaway dir.
    """
    ok, violations = execute_sandboxed(
        untrusted_code, trusted_code, timeout, allow_execution=allow_execution
    )
    if violations:
        console.print(
            f"[yellow]Sandbox rejected generated code: {violations[0]}[/yellow]"
        )
    return ok


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class BenchmarkBase(ABC):
    name: str = ""
    display_name: str = ""
    requires_code_execution = False

    def __init__(self, config: BenchmarkConfig, settings: Settings):
        self.config = config
        self.settings = settings
        self._questions: list[dict] | None = None

    def prepare(self, raw: dict) -> dict:
        """Hook to normalize/prepare a raw question dict once upon loading."""
        return dict(raw)

    def row_filter(self, raw: dict) -> bool:
        """Optional predicate applied to the FULL dataset before sampling.

        Override to restrict sampling to a subset (e.g. SuperGPQA's "hard"
        rows) so ``num_samples`` is the true sample size. Default keeps all rows.
        """
        return True

    def load(self) -> list[dict]:
        if self._questions is None:
            # Only pass a filter when a subclass actually overrides row_filter,
            # so unrestricted benchmarks skip the full-dataset scan.
            rf = None if type(self).row_filter is BenchmarkBase.row_filter else self.row_filter
            raw_qs = load_questions(self.config, self.settings, row_filter=rf)
            prepared = [self.prepare(q) for q in raw_qs]
            self._questions = [q for q in prepared if not q.get("_skip", False)]
        return self._questions

    @abstractmethod
    def format_prompt(self, question: dict) -> str: ...

    @abstractmethod
    def evaluate(self, question: dict, response: str) -> float: ...


# ---------------------------------------------------------------------------
# HumanEval — code execution against built-in unit tests
# ---------------------------------------------------------------------------


class HumanEvalBenchmark(BenchmarkBase):
    name = "humaneval"
    display_name = "HumanEval"
    requires_code_execution = True

    def format_prompt(self, q: dict) -> str:
        return (
            "Complete the following Python function. "
            "Return ONLY the complete function code (including the def line). "
            "No explanations, no markdown code blocks.\n\n"
            f"{q['prompt']}"
        )

    def evaluate(self, q: dict, response: str) -> float:
        code = _strip_code_blocks(response)
        header = q["prompt"].split("\ndef ", 1)[0] if "\ndef " in q["prompt"] else ""
        if code.lstrip().startswith("def "):
            if header.strip():
                code = header + "\n" + code
        else:
            code = q["prompt"] + "\n" + code
        return 1.0 if _execute_code(
            code, q["test"], self.settings.code_exec_timeout,
            allow_execution=self.settings.allow_unsafe_code_execution,
        ) else 0.0


# ---------------------------------------------------------------------------
# MBPP — code execution against assert-based tests
# ---------------------------------------------------------------------------


class MBPPBenchmark(BenchmarkBase):
    name = "mbpp"
    display_name = "MBPP"
    requires_code_execution = True

    def format_prompt(self, q: dict) -> str:
        prompt = q.get("prompt") or q.get("text", "")
        code = q.get("code", "")
        m = re.search(r"def\s+(\w+)\s*\(([^)]*)\)", code)
        if m:
            sig = f"def {m.group(1)}({m.group(2)}):"
            return (
                "Write a Python function to solve the following problem. "
                "Return ONLY the code, no explanations, no markdown code blocks.\n"
                f"Your function MUST be named exactly `{m.group(1)}`.\n\n"
                f"{prompt}\n\n"
                f"Function signature:\n{sig}"
            )
        return (
            "Write a Python function to solve the following problem. "
            "Return ONLY the code, no explanations, no markdown code blocks.\n\n"
            f"{prompt}"
        )

    def evaluate(self, q: dict, response: str) -> float:
        code = _strip_code_blocks(response)
        imports = "\n".join(q.get("test_imports", []))
        tests = "\n".join(q.get("test_list", []))
        trusted = "\n\n".join(p for p in [imports, tests] if p.strip())
        return 1.0 if _execute_code(
            code, trusted, self.settings.code_exec_timeout,
            allow_execution=self.settings.allow_unsafe_code_execution,
        ) else 0.0


# ---------------------------------------------------------------------------
# BigCodeBench — practical Python with real libraries, unittest verification
# ---------------------------------------------------------------------------


class BigCodeBenchBenchmark(BenchmarkBase):
    name = "bigcodebench"
    display_name = "BigCodeBench"
    requires_code_execution = True

    def format_prompt(self, q: dict) -> str:
        entry = q.get("entry_point", "task_func")
        prompt = q.get("instruct_prompt") or q.get("complete_prompt", "")
        return (
            f"Write a Python function named `{entry}` to solve the following task. "
            "Return ONLY the code (including any needed imports). "
            "No explanations, no markdown code blocks.\n\n"
            f"{prompt}"
        )

    def evaluate(self, q: dict, response: str) -> float:
        code = _strip_code_blocks(response)
        test = q.get("test", "")
        runner = (
            "\n\nimport sys, unittest\n"
            "suite = unittest.TestLoader().loadTestsFromTestCase(TestCases)\n"
            "result = unittest.TextTestRunner(verbosity=0).run(suite)\n"
            "sys.exit(0 if result.wasSuccessful() else 1)\n"
        )
        return 1.0 if _execute_code(
            code, test + runner, self.settings.code_exec_timeout,
            allow_execution=self.settings.allow_unsafe_code_execution,
        ) else 0.0


# ---------------------------------------------------------------------------
# GPQA Diamond — graduate-level science multiple choice (community mirror)
# ---------------------------------------------------------------------------


class GPQABenchmark(BenchmarkBase):
    name = "gpqa"
    display_name = "GPQA Diamond"

    def prepare(self, raw: dict) -> dict:
        q = raw
        question_text = q.get("question") or q.get("Question", "")
        options = q.get("options") or q.get("choices")
        gold_letter = None

        # The original Idavidrein/gpqa exposes "Correct Answer" / "Incorrect
        # Answer N" columns; the nichenshun/gpqa_diamond mirror instead packs the
        # answer choices into a JSON-encoded `metadata` string. Support both so
        # the option-shuffle + gold-letter logic works on either schema.
        meta = q.get("metadata")
        if not options and isinstance(meta, str) and meta.strip():
            try:
                meta_d = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta_d = None
            if isinstance(meta_d, dict) and meta_d.get("Correct Answer"):
                correct = meta_d.get("Correct Answer", "")
                raw_choices = [
                    correct,
                    meta_d.get("Incorrect Answer 1", ""),
                    meta_d.get("Incorrect Answer 2", ""),
                    meta_d.get("Incorrect Answer 3", ""),
                ]
                if correct and all(raw_choices):
                    rng = random.Random(f"{self.settings.seed}:{question_text}")
                    indices = list(range(4))
                    rng.shuffle(indices)
                    options = [raw_choices[i] for i in indices]
                    gold_letter = chr(65 + options.index(correct))

        if not options and "Correct Answer" in q:
            correct = q["Correct Answer"]
            inc1 = q.get("Incorrect Answer 1", "")
            inc2 = q.get("Incorrect Answer 2", "")
            inc3 = q.get("Incorrect Answer 3", "")
            raw_choices = [correct, inc1, inc2, inc3]
            rng = random.Random(f"{self.settings.seed}:{question_text}")
            indices = list(range(4))
            rng.shuffle(indices)
            options = [raw_choices[i] for i in indices]
            try:
                gold_idx = options.index(correct)
                gold_letter = chr(65 + gold_idx)
            except ValueError:
                pass

        if not gold_letter:
            solution = str(q.get("solution") or q.get("answer") or "")
            m = re.search(r"[Aa]nswer:\s*([A-Da-d])", solution)
            if m:
                gold_letter = m.group(1).upper()
            elif solution.strip().upper() in {"A", "B", "C", "D"}:
                gold_letter = solution.strip().upper()

        q["options"] = options
        q["gold_letter"] = gold_letter
        if not gold_letter:
            # No recoverable gold answer -> ungradeable; drop rather than score 0.
            q["_skip"] = True
        return q

    def format_prompt(self, q: dict) -> str:
        if "options" not in q or q.get("options") is None:
            q = self.prepare(q)
        question = q.get("question") or q.get("Question", "")
        options = q.get("options")

        if options and isinstance(options, list):
            opts_str = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options))
            return (
                "Answer the following graduate-level science question.\n\n"
                f"{question}\n\n{opts_str}\n\n"
                "Reply with ONLY the letter (A, B, C, or D) of the correct answer. "
                "State your final answer letter clearly at the end."
            )

        return (
            "Answer the following graduate-level science question.\n\n"
            f"{question}\n\n"
            "Reply with ONLY the letter (A, B, C, or D) of the correct answer. "
            "State your final answer letter clearly at the end."
        )

    def evaluate(self, q: dict, response: str) -> float:
        gold = q.get("gold_letter")
        if not gold:
            q = self.prepare(q)
            gold = q.get("gold_letter")
        if not gold:
            return 0.0
        pred = _extract_letter(response, {"A", "B", "C", "D"})
        return 1.0 if pred == gold else 0.0


# ---------------------------------------------------------------------------
# ARC-Challenge — grade-school science multiple choice
# ---------------------------------------------------------------------------


class ARCBenchmark(BenchmarkBase):
    name = "arc"
    display_name = "ARC-Challenge"

    def format_prompt(self, q: dict) -> str:
        texts = q["choices"]["text"]
        options = "\n".join(f"{chr(65 + i)}. {t}" for i, t in enumerate(texts))
        return (
            f"Question: {q['question']}\n{options}\n\n"
            "Reply with ONLY the letter of the correct answer."
        )

    def evaluate(self, q: dict, response: str) -> float:
        labels = q["choices"]["label"]
        answer_key = q["answerKey"]
        try:
            gold_idx = labels.index(answer_key)
        except ValueError:
            return 0.0
        gold = chr(65 + gold_idx)
        valid = {chr(65 + i) for i in range(len(labels))}
        pred = _extract_letter(response, valid)
        return 1.0 if pred == gold else 0.0


# ---------------------------------------------------------------------------
# GSM8K — numerical exact match
# ---------------------------------------------------------------------------


class GSM8KBenchmark(BenchmarkBase):
    name = "gsm8k"
    display_name = "GSM8K"

    def format_prompt(self, q: dict) -> str:
        return (
            "Solve the following math problem step by step. "
            'End your response with "#### <number>" where <number> is your '
            "final numerical answer.\n\n"
            f"Question: {q['question']}"
        )

    def evaluate(self, q: dict, response: str) -> float:
        gold = _extract_number(q["answer"])
        pred = _extract_number(response)
        if gold is None or pred is None:
            return 0.0
        try:
            return 1.0 if abs(float(gold) - float(pred)) < 1e-6 else 0.0
        except ValueError:
            return 0.0


# ---------------------------------------------------------------------------
# MMLU-Pro — 10-choice multiple choice
# ---------------------------------------------------------------------------


class MMLUProBenchmark(BenchmarkBase):
    name = "mmlu_pro"
    display_name = "MMLU-Pro"

    def prepare(self, raw: dict) -> dict:
        q = raw
        options = q.get("options") or []
        n = len(options)
        ans_raw = q.get("answer")
        if ans_raw is None:
            ans_raw = q.get("answer_index")
        idx = None
        if isinstance(ans_raw, int):
            idx = ans_raw
        else:
            s = str(ans_raw).strip()
            if s.isdigit():
                idx = int(s)
            elif len(s) == 1 and s.upper() in "ABCDEFGHIJ":
                idx = ord(s.upper()) - 65
        # Drop rows whose gold answer is missing or out of range for the options;
        # they can never be graded and would otherwise silently score 0.
        if idx is None or not (0 <= idx < n):
            q["_skip"] = True
        return q

    def format_prompt(self, q: dict) -> str:
        options = q["options"]
        letters = "ABCDEFGHIJ"
        opts = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options))
        category = q.get("category", "")
        header = f"Subject: {category}\n\n" if category else ""
        return (
            f"{header}Question: {q['question']}\n{opts}\n\n"
            f"Reply with ONLY the letter ({letters[0]}-{letters[len(options) - 1]}) "
            "of the correct answer. "
            "State your final answer letter clearly at the end."
        )

    def evaluate(self, q: dict, response: str) -> float:
        ans_raw = q.get("answer")
        if ans_raw is None:
            ans_raw = q.get("answer_index")

        if isinstance(ans_raw, int):
            gold = chr(65 + ans_raw)
        else:
            gold_str = str(ans_raw).strip()
            if gold_str.isdigit():
                gold = chr(65 + int(gold_str))
            else:
                gold = gold_str.upper()

        n = len(q.get("options", [])) or 10
        valid = {chr(65 + i) for i in range(n)}
        pred = _extract_letter(response, valid)
        return 1.0 if pred == gold else 0.0


# ---------------------------------------------------------------------------
# IFEval — programmatic instruction verifiers
# ---------------------------------------------------------------------------


class IFEvalBenchmark(BenchmarkBase):
    name = "ifeval"
    display_name = "IFEval"

    def format_prompt(self, q: dict) -> str:
        return q["prompt"]

    def evaluate(self, q: dict, response: str) -> float:
        ids = q.get("instruction_id_list", [])
        kwargs = q.get("kwargs", [])
        if not ids:
            return 0.0
        return 1.0 if verify_all(ids, response, kwargs) else 0.0


# ---------------------------------------------------------------------------
# HumanEval+ — EvalPlus augmented HumanEval
# ---------------------------------------------------------------------------


class HumanEvalPlusBenchmark(HumanEvalBenchmark):
    name = "humanevalplus"
    display_name = "HumanEval+"


# ---------------------------------------------------------------------------
# MBPP+ — EvalPlus augmented MBPP
# ---------------------------------------------------------------------------


class MBPPPlusBenchmark(MBPPBenchmark):
    name = "mbppplus"
    display_name = "MBPP+"


# ---------------------------------------------------------------------------
# BigCodeBench-Hard — Practical Python with complex libraries (Hard subset)
# ---------------------------------------------------------------------------


class BigCodeBenchHardBenchmark(BigCodeBenchBenchmark):
    name = "bigcodebench_hard"
    display_name = "BigCodeBench-Hard"


# ---------------------------------------------------------------------------
# SciBench — College-level scientific textbook problem solving
# ---------------------------------------------------------------------------


class SciBenchBenchmark(BenchmarkBase):
    name = "scibench"
    display_name = "SciBench"

    def format_prompt(self, q: dict) -> str:
        text = q.get("problem_text") or q.get("question", "")
        unit = q.get("unit", "")
        unit_str = f" Unit: {unit}." if unit else ""
        return (
            "Solve the following college-level science problem step-by-step.\n\n"
            f"{text}{unit_str}\n\n"
            "End your response with '\\boxed{<answer>}' containing your final numerical or symbol answer. "
            "Be concise. Give your final answer clearly."
        )

    def evaluate(self, q: dict, response: str) -> float:
        gold_val = q.get("answer_number")
        if gold_val is None:
            gold_val = q.get("answer_latex")
        if gold_val is None:
            gold_val = q.get("solution", "")

        if gold_val is None or not str(gold_val).strip():
            return 0.0

        gold_str = str(gold_val).strip()
        # Re-extract a number from inside \boxed{...} so "\boxed{x = 5}" still
        # yields 5 (a bare _extract_boxed would give "x = 5" and fail float()).
        boxed = _extract_boxed(response)
        pred = _extract_number(boxed) if boxed else _extract_number(response)
        if pred is None:
            return 0.0

        if pred.strip().lower() == gold_str.lower():
            return 1.0

        try:
            p = float(pred)
            g = float(gold_str)
        except ValueError:
            return 0.0
        # Relative tolerance (matches SciBench's evaluator). The previous
        # absolute <1e-4 was wrong both ways: it rejected a 5-sig-fig answer to
        # a large gold (e.g. 299792458 vs 300000000) yet accepted 50% error on a
        # tiny gold (0.00015 vs 0.0001).
        return 1.0 if math.isclose(p, g, rel_tol=1e-2, abs_tol=1e-8) else 0.0


# ---------------------------------------------------------------------------
# AIME — American Invitational Mathematics Examination competition math
# ---------------------------------------------------------------------------


class AIMEBenchmark(BenchmarkBase):
    name = "aime"
    display_name = "AIME 2024/2025"

    def format_prompt(self, q: dict) -> str:
        problem = q.get("problem") or q.get("question", "")
        return (
            "Solve the following competition math problem step-by-step.\n"
            "End your response with '\\boxed{<integer>}' containing your final integer answer.\n"
            "Be concise. Give your final answer clearly.\n\n"
            f"Problem: {problem}"
        )

    def evaluate(self, q: dict, response: str) -> float:
        gold_raw = q.get("answer")
        if gold_raw is None:
            gold_raw = q.get("solution", "")

        gold_str = str(gold_raw).strip()
        gold_num = _extract_boxed(gold_str) or _extract_number(gold_str)
        if gold_num is None:
            return 0.0

        pred_boxed = _extract_boxed(response)
        pred_num = _extract_number(pred_boxed) if pred_boxed else _extract_number(response)
        if pred_num is None:
            return 0.0

        try:
            return 1.0 if float(gold_num) == float(pred_num) else 0.0
        except ValueError:
            return 0.0


# ---------------------------------------------------------------------------
# MATH-500 — Competition math problems (Hendrycks MATH level 1-5)
# ---------------------------------------------------------------------------


def _normalize_latex(s: str) -> str:
    s = s.strip()
    s = s.strip("$")
    s = re.sub(r"\s+", "", s)
    s = s.replace(r"\tfrac", r"\frac").replace(r"\dfrac", r"\frac")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\mathrm", "").replace(r"\mathbf", "").replace(r"\text", "")
    # Drop TeX spacing commands and \circ so e.g. "5\," == "5" and "90^\circ"
    # normalizes consistently.
    for sp in (r"\,", r"\!", r"\;", r"\:", r"\ "):
        s = s.replace(sp, "")
    s = s.replace(r"\circ", "")
    s = s.rstrip(".")
    return s


class MATH500Benchmark(BenchmarkBase):
    name = "math_500"
    display_name = "MATH-500"

    def format_prompt(self, q: dict) -> str:
        problem = q.get("problem") or q.get("question", "")
        return (
            "Solve the following math problem step-by-step.\n"
            "End your response with '\\boxed{<answer>}' containing your final answer.\n"
            "Be concise. Give your final answer clearly.\n\n"
            f"Problem: {problem}"
        )

    def evaluate(self, q: dict, response: str) -> float:
        gold_raw = q.get("answer")
        if gold_raw is None:
            gold_raw = q.get("solution", "")
        gold_str = str(gold_raw).strip()
        gold_boxed = _extract_boxed(gold_str) or gold_str

        pred_boxed = _extract_boxed(response)
        if pred_boxed and _normalize_latex(pred_boxed) == _normalize_latex(gold_boxed):
            return 1.0

        # Numeric comparison is ONLY valid when the gold answer is itself a plain
        # number. Comparing against _extract_number(gold) would pull incidental
        # digits out of symbolic gold (\sqrt{2}->"2", 3\sqrt{2}->"2", 2^5->"5")
        # and mark wrong answers correct. m_gold is non-None iff gold_boxed is a
        # plain number.
        m_gold = re.match(r"^-?[\d,]+(?:\.\d+)?$", gold_boxed.replace(",", "").strip())
        if m_gold:
            gold_num = float(m_gold.group(0))
            # Prefer the boxed prediction; otherwise a number from the prose.
            pred_box = _extract_number(pred_boxed) if pred_boxed else None
            pred_num = pred_box or _extract_number(response)
            if pred_num is not None and re.fullmatch(r"-?[\d.]+", pred_num):
                try:
                    return 1.0 if abs(gold_num - float(pred_num)) < 1e-6 else 0.0
                except ValueError:
                    pass
        return 0.0


# ---------------------------------------------------------------------------
# SuperGPQA — scaled graduate-level multiple choice (hard subset)
# ---------------------------------------------------------------------------


class SuperGPQABenchmark(BenchmarkBase):
    name = "supergpqa"
    display_name = "SuperGPQA"

    def row_filter(self, raw: dict) -> bool:
        # Restrict to the "hard" subset BEFORE sampling so num_samples=50 really
        # yields 50 hard questions (the old code sampled first and filtered
        # after, shrinking the effective sample to ~27% of num_samples).
        return str(raw.get("difficulty", "")).strip().lower() == "hard"

    def prepare(self, raw: dict) -> dict:
        q = raw
        # Defensive: rows are pre-filtered to "hard" by row_filter, but skip any
        # non-hard row that slips through rather than scoring it.
        diff = str(q.get("difficulty", "")).strip().lower()
        if diff != "hard":
            q["_skip"] = True
            return q

        options = q.get("options") or q.get("choices")
        if not options:
            opts = []
            for code in range(65, 75):
                letter = chr(code)
                val = q.get(f"option_{letter.lower()}") or q.get(f"option_{letter}") or q.get(letter)
                if val:
                    opts.append(val)
            options = opts

        gold_letter = str(q.get("answer_letter") or q.get("answer") or q.get("gold") or "").strip().upper()
        q["options"] = options or []
        q["gold_letter"] = gold_letter
        return q

    def format_prompt(self, q: dict) -> str:
        if "options" not in q or q.get("options") is None:
            q = self.prepare(q)
        question = q.get("question") or q.get("Question", "")
        options = q.get("options", [])
        letters = "ABCDEFGHIJ"

        if options and isinstance(options, list):
            opts_str = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options) if i < len(letters))
            max_letter = letters[min(len(options), len(letters)) - 1]
            return (
                "Answer the following graduate-level multiple-choice question.\n\n"
                f"{question}\n\n{opts_str}\n\n"
                f"Reply with ONLY the letter (A-{max_letter}) of the correct answer. "
                "State your final answer letter clearly at the end."
            )

        return (
            "Answer the following graduate-level multiple-choice question.\n\n"
            f"{question}\n\n"
            "Reply with ONLY the letter of the correct answer. "
            "State your final answer letter clearly at the end."
        )

    def evaluate(self, q: dict, response: str) -> float:
        gold = q.get("gold_letter")
        if not gold:
            q = self.prepare(q)
            gold = q.get("gold_letter")
        if not gold:
            return 0.0

        n_opts = len(q.get("options", [])) or 10
        valid = {chr(65 + i) for i in range(min(n_opts, 10))}
        pred = _extract_letter(response, valid)
        return 1.0 if pred == gold else 0.0


# ---------------------------------------------------------------------------
# SciCode — scientific Python programming problem solving
# ---------------------------------------------------------------------------


class SciCodeBenchmark(BenchmarkBase):
    name = "scicode"
    display_name = "SciCode"
    requires_code_execution = True

    def format_prompt(self, q: dict) -> str:
        desc = q.get("problem_description_main") or q.get("prompt", "")
        background = q.get("problem_background_main") or ""
        deps = q.get("required_dependencies", "")
        deps_str = "\n".join(deps) if isinstance(deps, list) else str(deps)

        # The graders call specific function names/signatures. Expose the
        # skeleton (or, failing that, each sub-step's function_header) so the
        # model implements the exact symbols the tests expect — without this the
        # model invents its own names and every test NameErrors -> 0.
        skeleton = q.get("skeleton") or ""
        if not skeleton:
            headers = []
            for step in q.get("sub_steps", []) or []:
                if isinstance(step, dict):
                    hdr = step.get("function_header") or ""
                    if hdr:
                        headers.append(hdr)
            if headers:
                skeleton = "\n".join(headers)

        parts = [
            "Write a Python script to solve the following scientific problem.",
            "Return ONLY the complete Python code (including imports). "
            "Implement the functions with the exact names and signatures shown. "
            "No explanations, no markdown code blocks.",
            "",
        ]
        if deps_str.strip():
            parts.append(f"Required dependencies:\n{deps_str}\n")
        if background.strip():
            parts.append(f"Background:\n{background}\n")
        parts.append(str(desc))
        if skeleton.strip():
            parts.append(f"\nYou must implement exactly these function signatures:\n{skeleton}")
        return "\n".join(parts)

    def evaluate(self, q: dict, response: str) -> float:
        code = _strip_code_blocks(response)
        deps = q.get("required_dependencies", "")
        deps_str = "\n".join(deps) if isinstance(deps, list) else str(deps)
        tests = q.get("general_tests", "")
        tests_str = "\n\n".join(tests) if isinstance(tests, list) else str(tests)

        trusted = "\n\n".join(p for p in [deps_str, tests_str] if p.strip())
        return 1.0 if _execute_code(
            code, trusted, self.settings.code_exec_timeout,
            allow_execution=self.settings.allow_unsafe_code_execution,
        ) else 0.0


# ---------------------------------------------------------------------------
# Tau-Bench — agentic tool use and multi-turn workflow
# ---------------------------------------------------------------------------


def _extract_tool_json(response: str) -> dict | None:
    """Extract the first balanced ``{...}`` that parses as a JSON object.

    A greedy ``\\{.*\\}`` match breaks when the model writes braced prose before
    the tool call (e.g. "Per {policy 3}, call {...}"). We instead collect every
    balanced brace span, parse each, and prefer one that looks like a tool call.
    """
    spans: list[str] = []
    depth = 0
    start = None
    for i, ch in enumerate(response):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    spans.append(response[start : i + 1])
                    start = None

    parsed: list[dict] = []
    for span in spans:
        try:
            obj = json.loads(span)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            parsed.append(obj)
    for obj in parsed:
        if "name" in obj or "function" in obj or "arguments" in obj:
            return obj
    return parsed[0] if parsed else None


def _norm_arg_value(v: object) -> str:
    """Normalize an argument value for comparison; numeric values compare equal
    across representation (100 == 100.0), everything else is a lowered string."""
    s = str(v).strip().lower()
    try:
        return repr(float(s))
    except ValueError:
        return s


class TauBenchBenchmark(BenchmarkBase):
    name = "tau_bench"
    display_name = "Tau-Bench (Retail)"

    def format_prompt(self, q: dict) -> str:
        convs = q.get("conversations", [])
        formatted = []
        for turn in convs:
            role = str(turn.get("role", ""))
            content = turn.get("content")
            tool_calls = turn.get("tool_calls")
            if content:
                formatted.append(f"{role.capitalize()}: {content}")
            elif tool_calls:
                formatted.append(f"Assistant (Tool Call): {tool_calls}")

        prompt_text = "\n\n".join(formatted)
        return (
            "You are an AI assistant in a multi-turn customer service environment.\n"
            "Based on the conversation history below, output the next required tool call or response.\n"
            'If calling a tool, reply with JSON format: {"name": "<tool_name>", "arguments": {<args>}}.\n\n'
            f"Conversation History:\n{prompt_text}\n\n"
            "Next Action:"
        )

    def evaluate(self, q: dict, response: str) -> float:
        gold_answers = q.get("answer", [])
        if not gold_answers or not isinstance(gold_answers, list):
            return 0.0

        first_ans = gold_answers[0]
        if not isinstance(first_ans, dict):
            return 0.0

        gold_tool_calls = first_ans.get("tool_calls", [])
        if not gold_tool_calls:
            # Gold action is a plain text message, not a tool call.
            gold_text = str(first_ans.get("content") or "")
            if not gold_text.strip():
                return 0.0  # empty/None gold must not auto-pass every response
            return 1.0 if gold_text.strip().lower() in response.strip().lower() else 0.0

        # This is a "next-action" dataset: the gold is the single next tool call,
        # so we grade the model's emitted call against gold_tool_calls[0].
        gold_func = gold_tool_calls[0].get("function", {})
        gold_name = str(gold_func.get("name", "")).strip()
        gold_args_raw = gold_func.get("arguments", {})
        if isinstance(gold_args_raw, str):
            try:
                gold_args = json.loads(gold_args_raw)
            except Exception:
                gold_args = {}
        elif isinstance(gold_args_raw, dict):
            gold_args = gold_args_raw
        else:
            gold_args = {}

        pred_json = _extract_tool_json(response)
        if not isinstance(pred_json, dict):
            return 0.0

        pred_name = str(pred_json.get("name") or pred_json.get("function", "")).strip()
        if pred_name.lower() != gold_name.lower():
            return 0.0

        pred_args = pred_json.get("arguments", {})
        if isinstance(pred_args, str):
            try:
                pred_args = json.loads(pred_args)
            except Exception:
                pred_args = {}
        p_norm = (
            {str(k).lower(): _norm_arg_value(v) for k, v in pred_args.items()}
            if isinstance(pred_args, dict)
            else {}
        )
        if not gold_args:
            # Gold expects no arguments: accept only if the model sent none.
            return 1.0 if not p_norm else 0.0
        g_norm = {str(k).lower(): _norm_arg_value(v) for k, v in gold_args.items()}
        return 1.0 if g_norm == p_norm else 0.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BENCHMARK_CLASSES: dict[str, type[BenchmarkBase]] = {
    "humaneval": HumanEvalBenchmark,
    "humanevalplus": HumanEvalPlusBenchmark,
    "mbpp": MBPPBenchmark,
    "mbppplus": MBPPPlusBenchmark,
    "bigcodebench": BigCodeBenchBenchmark,
    "bigcodebench_hard": BigCodeBenchHardBenchmark,
    "gpqa": GPQABenchmark,
    "scibench": SciBenchBenchmark,
    "arc": ARCBenchmark,
    "gsm8k": GSM8KBenchmark,
    "aime": AIMEBenchmark,
    "math_500": MATH500Benchmark,
    "mmlu_pro": MMLUProBenchmark,
    "ifeval": IFEvalBenchmark,
    "supergpqa": SuperGPQABenchmark,
    "scicode": SciCodeBenchmark,
    "tau_bench": TauBenchBenchmark,
}


def create_benchmark(name: str, config: BenchmarkConfig, settings: Settings) -> BenchmarkBase:
    cls = BENCHMARK_CLASSES.get(name)
    if cls is None:
        raise ValueError(f"Unknown benchmark: {name}")
    return cls(config, settings)
