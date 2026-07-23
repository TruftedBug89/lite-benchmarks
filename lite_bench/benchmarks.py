"""Benchmark implementations using real datasets with built-in verification."""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod

from rich.console import Console

from .config import BenchmarkConfig, Settings
from .datasets import load_questions
from .ifeval_verifiers import verify_all

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_code_blocks(text: str) -> str:
    text = text.strip()
    # First search for explicit markdown python/py code blocks
    matches = re.findall(r"```(?:python|py)?\s*\n(.*?)\n```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    # Fallback for code blocks missing closing tags or simple triple backticks
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
        m_boxed = re.search(r"^-?[\d,]+(?:\.\d+)?$", boxed.replace(",", "").strip())
        if m_boxed:
            return m_boxed.group(0)

    # Avoid extracting isolated digits from LaTeX fractions like \frac{3}{5}, \dfrac{3}{5}, \tfrac{3}{5}
    cleaned = re.sub(r"\\(?:d|t)?frac\{[^{}]+\}\{[^{}]+\}", "", text)
    nums = re.findall(r"(?<![\d\w.-])-?[\d,]+(?:\.\d+)?(?![\d\w.-])", cleaned)
    if nums:
        return nums[-1].replace(",", "").strip()
    return None


def _extract_letter(response: str, valid: set[str] | None = None) -> str | None:
    if valid is None:
        valid = {"A", "B", "C", "D"}
    valid_upper = {v.upper() for v in valid}

    # Check boxed content first
    boxed = _extract_boxed(response)
    if boxed and boxed.strip().upper() in valid_upper:
        return boxed.strip().upper()

    # Search for explicit "Answer is X" from the end of the text
    matches = list(re.finditer(r"(?:answer|choice|option)\s*(?:is|:)?\s*\(?([A-Ja-j])\)?", response, flags=re.IGNORECASE))
    if matches:
        last_match = matches[-1].group(1).upper()
        if last_match in valid_upper:
            return last_match

    # Search for standalone letter in the last paragraph or line
    paragraphs = [p for p in response.strip().split("\n") if p.strip()]
    if paragraphs:
        last_para = paragraphs[-1]
        letter_matches = re.findall(r"\b([A-Ja-j])\b", last_para)
        if letter_matches and letter_matches[-1].upper() in valid_upper:
            return letter_matches[-1].upper()

    # Fallback to general scan in reverse
    letter_matches = re.findall(r"\b([A-Ja-j])\b", response)
    for letter in reversed(letter_matches):
        if letter.upper() in valid_upper:
            return letter.upper()

    stripped = response.strip()
    if stripped and stripped[0].upper() in valid_upper:
        return stripped[0].upper()
    return None


def _extract_boxed(text: str) -> str | None:
    idx = text.rfind(r"\boxed{")
    if idx != -1:
        substr = text[idx + 7 :]
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
        if chars and open_braces == 0:
            return "".join(chars).strip()

    m = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if m:
        return m[-1].strip()
    return None


def _execute_code(code: str, timeout: int) -> bool:
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


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

    def load(self) -> list[dict]:
        if self._questions is None:
            raw_qs = load_questions(self.config, self.settings)
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
        if not code.lstrip().startswith("def "):
            code = q["prompt"] + "\n" + code
        full = code + "\n\n" + q["test"]
        return 1.0 if _execute_code(full, self.settings.code_exec_timeout) else 0.0


# ---------------------------------------------------------------------------
# MBPP — code execution against assert-based tests
# ---------------------------------------------------------------------------


class MBPPBenchmark(BenchmarkBase):
    name = "mbpp"
    display_name = "MBPP"
    requires_code_execution = True

    def format_prompt(self, q: dict) -> str:
        prompt = q.get("prompt") or q.get("text", "")
        return (
            "Write a Python function to solve the following problem. "
            "Return ONLY the code, no explanations, no markdown code blocks.\n\n"
            f"{prompt}"
        )

    def evaluate(self, q: dict, response: str) -> float:
        code = _strip_code_blocks(response)
        imports = "\n".join(q.get("test_imports", []))
        tests = "\n".join(q.get("test_list", []))
        parts = [p for p in [imports, code, tests] if p.strip()]
        full = "\n\n".join(parts)
        return 1.0 if _execute_code(full, self.settings.code_exec_timeout) else 0.0


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
        full = code + "\n\n" + test + runner
        return 1.0 if _execute_code(full, self.settings.code_exec_timeout) else 0.0


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
                "Reply with ONLY the letter (A, B, C, or D) of the correct answer."
            )

        return (
            "Answer the following graduate-level science question.\n\n"
            f"{question}\n\n"
            "Reply with ONLY the letter (A, B, C, or D) of the correct answer."
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

    def format_prompt(self, q: dict) -> str:
        options = q["options"]
        letters = "ABCDEFGHIJ"
        opts = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options))
        category = q.get("category", "")
        header = f"Subject: {category}\n\n" if category else ""
        return (
            f"{header}Question: {q['question']}\n{opts}\n\n"
            f"Reply with ONLY the letter ({letters[0]}-{letters[len(options) - 1]}) "
            "of the correct answer."
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
            "End your response with '\\boxed{<answer>}' containing your final numerical or symbol answer."
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
        pred = _extract_boxed(response) or _extract_number(response)
        if pred is None:
            return 0.0

        if pred.strip().lower() == gold_str.lower():
            return 1.0

        try:
            return 1.0 if abs(float(pred) - float(gold_str)) < 1e-4 else 0.0
        except ValueError:
            return 0.0


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
            "End your response with '\\boxed{<integer>}' containing your final integer answer.\n\n"
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
    s = re.sub(r"\s+", "", s)
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\mathrm", "").replace(r"\mathbf", "").replace(r"\text", "")
    return s


class MATH500Benchmark(BenchmarkBase):
    name = "math_500"
    display_name = "MATH-500"

    def format_prompt(self, q: dict) -> str:
        problem = q.get("problem") or q.get("question", "")
        return (
            "Solve the following math problem step-by-step.\n"
            "End your response with '\\boxed{<answer>}' containing your final answer.\n\n"
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

        # Only do numerical float comparison if both are simple standalone numbers
        m_gold = re.match(r"^-?[\d,]+(?:\.\d+)?$", gold_boxed.replace(",", "").strip())
        if pred_boxed:
            m_pred = re.match(r"^-?[\d,]+(?:\.\d+)?$", pred_boxed.replace(",", "").strip())
            if m_gold and m_pred:
                try:
                    return 1.0 if abs(float(m_gold.group(0)) - float(m_pred.group(0))) < 1e-6 else 0.0
                except ValueError:
                    pass

        gold_num = _extract_number(gold_str)
        pred_num = _extract_number(response)
        if gold_num is not None and pred_num is not None:
            m_g = re.match(r"^-?[\d.]+$", gold_num)
            m_p = re.match(r"^-?[\d.]+$", pred_num)
            if m_g and m_p and r"\frac" not in gold_str:
                try:
                    return 1.0 if abs(float(gold_num) - float(pred_num)) < 1e-6 else 0.0
                except ValueError:
                    pass
        return 0.0


# ---------------------------------------------------------------------------
# SuperGPQA — scaled graduate-level multiple choice (hard subset)
# ---------------------------------------------------------------------------


class SuperGPQABenchmark(BenchmarkBase):
    name = "supergpqa"
    display_name = "SuperGPQA"

    def prepare(self, raw: dict) -> dict:
        q = raw
        # Filter for hard subset only
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
                f"Reply with ONLY the letter (A-{max_letter}) of the correct answer."
            )

        return (
            "Answer the following graduate-level multiple-choice question.\n\n"
            f"{question}\n\n"
            "Reply with ONLY the letter of the correct answer."
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
        deps = q.get("required_dependencies", "")
        deps_str = "\n".join(deps) if isinstance(deps, list) else str(deps)
        return (
            "Write a Python script to solve the following scientific problem.\n"
            "Return ONLY the complete Python code (including imports). "
            "No explanations, no markdown code blocks.\n\n"
            f"{deps_str}\n\n"
            f"{desc}"
        )

    def evaluate(self, q: dict, response: str) -> float:
        code = _strip_code_blocks(response)
        deps = q.get("required_dependencies", "")
        deps_str = "\n".join(deps) if isinstance(deps, list) else str(deps)
        tests = q.get("general_tests", "")
        tests_str = "\n\n".join(tests) if isinstance(tests, list) else str(tests)

        parts = [p for p in [deps_str, code, tests_str] if p.strip()]
        full = "\n\n".join(parts)
        return 1.0 if _execute_code(full, self.settings.code_exec_timeout) else 0.0


# ---------------------------------------------------------------------------
# Tau-Bench — agentic tool use and multi-turn workflow
# ---------------------------------------------------------------------------


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
            "If calling a tool, reply with JSON format: {\"name\": \"<tool_name>\", \"arguments\": {<args>}}.\n\n"
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
            gold_text = str(first_ans.get("content", ""))
            return 1.0 if gold_text.strip().lower() in response.strip().lower() else 0.0

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

        pred_json = None
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                pred_json = json.loads(json_match.group(0))
            except Exception:
                pred_json = None

        if isinstance(pred_json, dict):
            pred_name = str(pred_json.get("name") or pred_json.get("function", "")).strip()
            pred_args = pred_json.get("arguments", {})
            if isinstance(pred_args, str):
                try:
                    pred_args = json.loads(pred_args)
                except Exception:
                    pred_args = {}

            if pred_name.lower() == gold_name.lower():
                if not gold_args:
                    return 1.0
                g_norm = {str(k).lower(): str(v).strip().lower() for k, v in gold_args.items()}
                p_norm = (
                    {str(k).lower(): str(v).strip().lower() for k, v in pred_args.items()}
                    if isinstance(pred_args, dict)
                    else {}
                )
                if g_norm == p_norm:
                    return 1.0
        return 0.0


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
