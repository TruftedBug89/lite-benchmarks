"""Benchmark implementations using real datasets with built-in verification."""

from __future__ import annotations

import json
import math
import random
import re
import textwrap
from abc import ABC, abstractmethod

from rich.console import Console

from .config import BenchmarkConfig, Settings
from .datasets import load_questions
from .ifeval_verifiers import verify_all
from .sandbox import execute_sandboxed, scan_code

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
    text = textwrap.dedent(text).strip()
    matches = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.DOTALL)
    if matches:
        with_def = [m for m in matches if "def " in m or "class " in m or "assert" in m]
        if with_def:
            return textwrap.dedent(with_def[0]).strip()
        return textwrap.dedent(max(matches, key=len)).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:python|py)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return textwrap.dedent(text).strip()


def _extract_sci_number(text: str) -> str | None:
    """Parse a scientific-notation number the plain regex would mangle.

    Handles both programmer notation (``4.4e-9``, ``5.07E+1``) and the LaTeX
    form models favour for physics answers (``4.4 \\times 10^{-9}``,
    ``2.5 \\cdot 10^{3}``). Returns the value as a plain decimal string so
    callers can ``float()`` it directly, or ``None`` if nothing matches. This
    closes a factual hole: the plain number regex would pull just the exponent
    out of ``\\boxed{4.4 \\times 10^{-9}}`` and score the answer wrong.
    """
    m = re.search(
        r"(-?(?:\d+(?:\.\d+)?|\.\d+))\s*[eE]\s*([+-]?\d+)",
        text,
    )
    if m:
        try:
            return repr(float(f"{m.group(1)}e{m.group(2)}"))
        except ValueError:
            return None
    m = re.search(
        r"(-?(?:\d+(?:\.\d+)?|\.\d+))\s*(?:\\times|\\cdot|\*|×)\s*10\s*\^\s*\{?\s*([+-]?\d+)\s*\}?",
        text,
    )
    if m:
        try:
            # Build the same e-notation string as the programmer form so both
            # paths share one float round-trip (no mantissa*power artefacts).
            return repr(float(f"{m.group(1)}e{m.group(2)}"))
        except (ValueError, OverflowError):
            return None
    return None


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

    # Scientific notation must take priority over the plain-number scan: the
    # plain regex would otherwise grab the exponent digits of "4.4e-09" (-> "9")
    # or the base of a "\times 10^n" form. This matters for both SciBench golds
    # (floats rendered as 4.4e-09) and model answers. Only fires when an actual
    # mantissa+exponent pair is present, so plain-number cases are unaffected.
    sci = _extract_sci_number(boxed) if boxed else None
    if sci is None:
        sci = _extract_sci_number(cleaned)
    if sci is not None:
        return sci

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


def _canonical_solution_gradeable(code: str | None) -> bool:
    """True iff a dataset's reference solution passes the sandbox AST scan.

    Used as a pre-sampling ``row_filter`` for code-execution benchmarks so we
    never ask a model to solve a task whose own canonical solution the sandbox
    would reject (e.g. a forced ``import subprocess``). Such rows are
    ungradeable by construction — every correct answer scores 0 — so excluding
    them BEFORE sampling keeps ``num_samples`` an honest count of passable tasks
    instead of silently deflating the score. File-I/O, ``os``/``shutil``, and
    loopback-network tasks are NOT excluded: the runtime confinement shim
    (sandbox_child.py) makes those safe, so their solutions scan clean.

    ``code`` may be an indented function-body fragment (HumanEval/MBPP store the
    body only), so it is dedented before scanning.
    """
    if not code or not str(code).strip():
        return True  # nothing to scan -> don't drop on missing reference code
    return not scan_code(textwrap.dedent(str(code)))


def _bigcodebench_gradeable(raw: dict) -> bool:
    """BigCodeBench ships the imports + signature in ``code_prompt`` and the
    reference *body* in ``canonical_solution``; the model is given code_prompt
    and its completion is executed with code_prompt's header prepended. A task
    is therefore passable only if that FULL reconstructed solution scans clean —
    scanning the body alone would miss a blocked import forced by the preamble
    (e.g. ``import subprocess``/``import psutil``), which the model can never
    avoid emitting."""
    prompt = str(raw.get("code_prompt") or "")
    header = prompt.split("\ndef ", 1)[0] if "\ndef " in prompt else prompt
    body = textwrap.dedent(str(raw.get("canonical_solution") or ""))
    full = (header + "\n" + body).strip()
    if not full:
        return True
    return not scan_code(full)


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


def _execute_code_with_details(
    untrusted_code: str, trusted_code: str, timeout: int, *, allow_execution: bool = False
) -> tuple[bool, str]:
    """Run sandboxed code execution and return (passed, detailed_log_message)."""
    ok, violations = execute_sandboxed(
        untrusted_code, trusted_code, timeout, allow_execution=allow_execution
    )
    if violations:
        reason = f"Sandbox test execution rejected: {'; '.join(violations)}"
        console.print(f"[yellow]{reason}[/yellow]")
        return False, reason
    return True, "Sandboxed execution succeeded. All assertions and unit tests passed cleanly."


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
        self._filter_stats: dict | None = None

    @property
    def excluded_count(self) -> int:
        """Number of dataset rows dropped by ``row_filter`` before sampling."""
        return int((self._filter_stats or {}).get("excluded", 0))

    def exclusion_note(self) -> str | None:
        """Human-readable note about pre-sampling exclusions, or None if none."""
        n = self.excluded_count
        if not n:
            return None
        return (
            f"{self.display_name or self.name}: excluded {n} ungradeable task(s) "
            "before sampling (reference solution requires sandbox-blocked "
            "constructs, e.g. subprocess/psutil/sys); the graded pool is passable by construction."
        )

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
            stats: dict | None = {} if rf is not None else None
            raw_qs = load_questions(self.config, self.settings, row_filter=rf, filter_stats=stats)
            self._filter_stats = stats
            prepared = [self.prepare(q) for q in raw_qs]
            self._questions = [q for q in prepared if not q.get("_skip", False)]
        return self._questions

    @abstractmethod
    def format_prompt(self, question: dict) -> str: ...

    @abstractmethod
    def evaluate(self, question: dict, response: str) -> float: ...

    def evaluate_detailed(self, question: dict, response: str) -> dict:
        """Detailed evaluation returning score, expected answer, extracted answer, and judge logs."""
        score = float(self.evaluate(question, response))
        expected = str(
            question.get("gold_letter")
            or question.get("answer")
            or question.get("target")
            or question.get("canonical_solution")
            or question.get("solution")
            or "N/A"
        )
        return {
            "score": score,
            "expected_answer": expected,
            "extracted_answer": "N/A",
            "judge_response": f"Evaluated using {self.display_name or self.name} verifier. Earned score: {score}",
        }


# ---------------------------------------------------------------------------
# HumanEval — code execution against built-in unit tests
# ---------------------------------------------------------------------------


class HumanEvalBenchmark(BenchmarkBase):
    name = "humaneval"
    display_name = "HumanEval"
    requires_code_execution = True

    def row_filter(self, raw: dict) -> bool:
        # Drop tasks whose reference solution the sandbox would reject (e.g.
        # HumanEval/160 uses eval()), so every sampled task is passable.
        return _canonical_solution_gradeable(raw.get("canonical_solution"))

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

    def evaluate_detailed(self, q: dict, response: str) -> dict:
        code = _strip_code_blocks(response)
        header = q["prompt"].split("\ndef ", 1)[0] if "\ndef " in q["prompt"] else ""
        if code.lstrip().startswith("def "):
            if header.strip():
                full_code = header + "\n" + code
            else:
                full_code = code
        else:
            full_code = q["prompt"] + "\n" + code
        ok, judge_log = _execute_code_with_details(
            full_code, q["test"], self.settings.code_exec_timeout,
            allow_execution=self.settings.allow_unsafe_code_execution,
        )
        score = 1.0 if ok else 0.0
        return {
            "score": score,
            "expected_answer": q.get("test", "Unit test assertions"),
            "extracted_answer": code,
            "judge_response": judge_log,
        }


# ---------------------------------------------------------------------------
# MBPP — code execution against assert-based tests
# ---------------------------------------------------------------------------


class MBPPBenchmark(BenchmarkBase):
    name = "mbpp"
    display_name = "MBPP"
    requires_code_execution = True

    def row_filter(self, raw: dict) -> bool:
        # Drop tasks whose reference solution the sandbox would reject (e.g.
        # MBPP/596 imports sys), so every sampled task is passable.
        return _canonical_solution_gradeable(raw.get("code"))

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

    def evaluate_detailed(self, q: dict, response: str) -> dict:
        code = _strip_code_blocks(response)
        imports = "\n".join(q.get("test_imports", []))
        tests = "\n".join(q.get("test_list", []))
        trusted = "\n\n".join(p for p in [imports, tests] if p.strip())
        ok, judge_log = _execute_code_with_details(
            code, trusted, self.settings.code_exec_timeout,
            allow_execution=self.settings.allow_unsafe_code_execution,
        )
        score = 1.0 if ok else 0.0
        return {
            "score": score,
            "expected_answer": tests or "Assert tests",
            "extracted_answer": code,
            "judge_response": judge_log,
        }


# ---------------------------------------------------------------------------
# BigCodeBench — practical Python with real libraries, unittest verification
# ---------------------------------------------------------------------------


class BigCodeBenchBenchmark(BenchmarkBase):
    name = "bigcodebench"
    display_name = "BigCodeBench"
    requires_code_execution = True

    def row_filter(self, raw: dict) -> bool:
        # Drop tasks whose FULL reference solution (forced code_prompt header +
        # body) the sandbox would reject even with the runtime confinement shim
        # — i.e. tasks that force an unconfineable import like subprocess/psutil/
        # sys. File-I/O, os/shutil, and loopback-network tasks scan clean and
        # stay in the pool.
        return _bigcodebench_gradeable(raw)

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

    def evaluate_detailed(self, q: dict, response: str) -> dict:
        code = _strip_code_blocks(response)
        test = q.get("test", "")
        runner = (
            "\n\nimport sys, unittest\n"
            "suite = unittest.TestLoader().loadTestsFromTestCase(TestCases)\n"
            "result = unittest.TextTestRunner(verbosity=0).run(suite)\n"
            "sys.exit(0 if result.wasSuccessful() else 1)\n"
        )
        ok, judge_log = _execute_code_with_details(
            code, test + runner, self.settings.code_exec_timeout,
            allow_execution=self.settings.allow_unsafe_code_execution,
        )
        score = 1.0 if ok else 0.0
        return {
            "score": score,
            "expected_answer": test or "Unittest cases",
            "extracted_answer": code,
            "judge_response": judge_log,
        }


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

    def evaluate_detailed(self, q: dict, response: str) -> dict:
        gold = q.get("gold_letter")
        if not gold:
            q = self.prepare(q)
            gold = q.get("gold_letter")
        if not gold:
            return {"score": 0.0, "expected_answer": "N/A", "extracted_answer": "None", "judge_response": "Gold answer letter missing."}
        pred = _extract_letter(response, {"A", "B", "C", "D"})
        score = 1.0 if pred == gold else 0.0
        verdict = "MATCHED (Correct answer)" if score == 1.0 else "MISMATCHED (Incorrect answer)"
        return {
            "score": score,
            "expected_answer": str(gold),
            "extracted_answer": str(pred or "None"),
            "judge_response": f"Multiple Choice Verifier: Extracted '{pred or 'None'}', Expected target is '{gold}'. {verdict}",
        }


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

    def evaluate_detailed(self, q: dict, response: str) -> dict:
        labels = q["choices"]["label"]
        answer_key = q["answerKey"]
        try:
            gold_idx = labels.index(answer_key)
            gold = chr(65 + gold_idx)
        except ValueError:
            gold = str(answer_key)
        valid = {chr(65 + i) for i in range(len(labels))}
        pred = _extract_letter(response, valid)
        score = 1.0 if pred == gold else 0.0
        verdict = "MATCHED (Correct answer)" if score == 1.0 else "MISMATCHED (Incorrect answer)"
        return {
            "score": score,
            "expected_answer": str(gold),
            "extracted_answer": str(pred or "None"),
            "judge_response": f"ARC Verifier: Extracted '{pred or 'None'}', Expected target is '{gold}'. {verdict}",
        }


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

    def evaluate_detailed(self, q: dict, response: str) -> dict:
        gold = _extract_number(q.get("answer", ""))
        pred = _extract_number(response)
        score = self.evaluate(q, response)
        verdict = "MATCHED (Correct answer)" if score == 1.0 else "MISMATCHED (Incorrect answer)"
        return {
            "score": score,
            "expected_answer": str(gold or q.get("answer", "")),
            "extracted_answer": str(pred or "None"),
            "judge_response": f"GSM8K Numerical Verifier: Extracted number '{pred or 'None'}', Target is '{gold or 'N/A'}'. {verdict}",
        }


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

    def evaluate_detailed(self, q: dict, response: str) -> dict:
        ans_raw = q.get("answer")
        if ans_raw is None:
            ans_raw = q.get("answer_index")
        if isinstance(ans_raw, int):
            gold = chr(65 + ans_raw)
        else:
            gold_str = str(ans_raw).strip()
            gold = chr(65 + int(gold_str)) if gold_str.isdigit() else gold_str.upper()
        n = len(q.get("options", [])) or 10
        valid = {chr(65 + i) for i in range(n)}
        pred = _extract_letter(response, valid)
        score = 1.0 if pred == gold else 0.0
        verdict = "MATCHED (Correct answer)" if score == 1.0 else "MISMATCHED (Incorrect answer)"
        return {
            "score": score,
            "expected_answer": str(gold),
            "extracted_answer": str(pred or "None"),
            "judge_response": f"MMLU-Pro Choice Verifier: Extracted '{pred or 'None'}', Expected target is '{gold}'. {verdict}",
        }


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

    def evaluate_detailed(self, q: dict, response: str) -> dict:
        ids = q.get("instruction_id_list", [])
        kwargs = q.get("kwargs", [])
        if not ids:
            return {
                "score": 0.0,
                "expected_answer": "No instruction rules",
                "extracted_answer": "N/A",
                "judge_response": "Instruction list empty.",
            }
        passed = verify_all(ids, response, kwargs)
        score = 1.0 if passed else 0.0
        rules_str = f"IFEval Rules ({len(ids)}): {', '.join(ids)}"
        verdict = "PASS (All instruction rules satisfied)" if passed else "FAIL (One or more instruction constraints violated)"
        return {
            "score": score,
            "expected_answer": rules_str,
            "extracted_answer": response[:150] + ("..." if len(response) > 150 else ""),
            "judge_response": f"IFEval Rule Verifier: Checked {len(ids)} rule constraints. Result: {verdict}",
        }


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
        # Relative tolerance (matches SciBench's evaluator). The absolute floor
        # must SCALE with the gold's magnitude: a fixed abs_tol (the old 1e-8)
        # is larger than many physics golds (e.g. 4.4e-9) and would mark ANY two
        # sub-1e-8 answers equal. Tying the floor to max(1, |g|) keeps a tight
        # 1e-9 absolute band for O(1) answers (and an exact-ish zero check) while
        # letting large golds absorb representation noise without accepting the
        # 50%-error case the relative term already rejects.
        abs_floor = 1e-9 * max(1.0, abs(g))
        return 1.0 if math.isclose(p, g, rel_tol=1e-2, abs_tol=abs_floor) else 0.0


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


def _latex_to_sympy(s: str) -> str:
    """Best-effort conversion of a normalized LaTeX answer to a sympy-parseable
    string. Handles the forms MATH-500 answers actually use (fractions, roots,
    powers, pi, implicit multiplication). Anything exotic falls through and the
    caller's try/except treats it as non-equivalent — this only ever ADDS true
    matches, never forces one."""
    t = s
    for _ in range(6):  # nested \frac{...}{...} -> ((...)/(...))
        t2 = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", t)
        if t2 == t:
            break
        t = t2
    for _ in range(4):  # nested \sqrt{...} -> sqrt(...)
        t2 = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", t)
        if t2 == t:
            break
        t = t2
    t = t.replace(r"\pi", "pi")
    t = t.replace(r"\ln", "ln")
    t = t.replace(r"\log", "log")
    t = t.replace(r"\cdot", "*").replace(r"\times", "*")
    t = t.replace("^", "**")
    # Implicit multiplication: "3sqrt(2)" -> "3*sqrt(2)", "2pi" -> "2*pi",
    # "(a)(b)" -> "(a)*(b)", "sqrt(2)sqrt(3)" -> "sqrt(2)*sqrt(3)".
    t = re.sub(r"(\d)([A-Za-z(])", r"\1*\2", t)
    t = re.sub(r"\)(\d)", r")*\1", t)
    t = re.sub(r"\)([A-Za-z(])", r")*\1", t)
    return t


def _symbolic_equivalent(pred: str, gold: str) -> bool:
    """Deterministic symbolic equivalence via sympy (no LLM, no network).

    Returns True only when sympy can prove ``pred - gold`` simplifies to zero
    or both sides numerically evaluate to the same constant. Any parse failure
    or exception returns False, so this is a safe fallback that strictly
    extends the normalized-string match (e.g. ``2^5`` == ``32``,
    ``\\frac{1}{2}`` == ``0.5``) while preserving false-positive guards
    (``\\sqrt{2}`` != ``2``).
    """
    if not pred or not gold or len(pred) > 120 or len(gold) > 120:
        return False
    try:
        import sympy
        from sympy.parsing.sympy_parser import parse_expr

        global_dict = {"__builtins__": {}}
        global_dict.update(sympy.__dict__)
        gp = parse_expr(_latex_to_sympy(gold), evaluate=False, global_dict=global_dict)
        pp = parse_expr(_latex_to_sympy(pred), evaluate=False, global_dict=global_dict)
        free = gp.free_symbols | pp.free_symbols
        if free:
            # Non-closed-form answers (contain a variable) are out of scope for
            # numeric fallback; only accept a proven symbolic identity.
            return bool(sympy.simplify(gp - pp) == 0)
        if sympy.simplify(gp - pp) == 0:
            return True
        diff = complex(sympy.N(gp - pp, 20))
        scale = max(1.0, abs(complex(sympy.N(gp, 20))))
        return abs(diff) <= 1e-9 * scale
    except Exception:
        return False


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

        # Deterministic symbolic equivalence for factually-equal answers that
        # differ in form (2^5 vs 32, \frac{1}{2} vs 0.5, \sqrt{4} vs 2). Runs
        # only on the boxed prediction so prose digits can't force a match, and
        # fails closed to the strict checks below on any parse error.
        if pred_boxed and _symbolic_equivalent(
            _normalize_latex(pred_boxed), _normalize_latex(gold_boxed)
        ):
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
            if pred_num is not None and re.fullmatch(r"-?[\d.]+(?:[eE][+-]?\d+)?", pred_num):
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
        if not isinstance(pred_args, dict):
            return 0.0
        p_norm = {str(k).lower(): _norm_arg_value(v) for k, v in pred_args.items()}
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
