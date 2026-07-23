"""Benchmark implementations using real datasets with built-in verification.

Eight benchmarks across five categories:
  Coding:      HumanEval+ (EvalPlus), MBPP+ (EvalPlus), BigCodeBench (unittest)
  Science:     GPQA Diamond (grad-level MC), ARC-Challenge (science MC)
  Math:        GSM8K (numerical exact match)
  Knowledge:   MMLU-Pro (10-choice MC)
  Instruction: IFEval (programmatic verifiers)
"""

from __future__ import annotations

import os
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
    if text.startswith("```"):
        text = re.sub(r"^```(?:python|py)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _extract_number(text: str) -> str | None:
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "").strip() if nums else None


def _extract_letter(response: str, valid: set[str] | None = None) -> str | None:
    if valid is None:
        valid = {"A", "B", "C", "D"}
    m = re.search(r"(?:answer|Answer)\s*(?:is|:)?\s*\(?([A-Ja-j])\)?", response)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    m = re.search(r"\b([A-Ja-j])\b", response.strip())
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    stripped = response.strip()
    if stripped and stripped[0].upper() in valid:
        return stripped[0].upper()
    return None


def _extract_boxed(text: str) -> str | None:
    m = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if m:
        return m[-1].strip()
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
        if chars:
            return "".join(chars).strip()
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

    def load(self) -> list[dict]:
        if self._questions is None:
            self._questions = load_questions(self.config, self.settings)
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

    def format_prompt(self, q: dict) -> str:
        question = q.get("question", "")
        return (
            "Answer the following graduate-level science question.\n\n"
            f"{question}\n\n"
            "Reply with ONLY the letter (A, B, C, or D) of the correct answer."
        )

    def evaluate(self, q: dict, response: str) -> float:
        solution = q.get("solution", "")
        m = re.search(r"[Aa]nswer:\s*([A-Da-d])", solution)
        if not m:
            return 0.0
        gold = m.group(1).upper()
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
            return 1.0 if float(gold) == float(pred) else 0.0
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
        gold = q["answer"].strip().upper()
        n = len(q["options"])
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
        gold = q.get("answer_number") or q.get("answer_latex") or q.get("solution", "")
        pred = _extract_boxed(response) or _extract_number(response)
        if pred is None or not str(gold).strip():
            return 0.0
        gold_str = str(gold).strip()
        if pred.strip() == gold_str:
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
        gold = q.get("answer") or q.get("solution", "")
        gold_num = _extract_number(str(gold))
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
        gold = q.get("answer") or q.get("solution", "")
        gold_boxed = _extract_boxed(str(gold)) or str(gold).strip()
        pred_boxed = _extract_boxed(response)
        if pred_boxed and pred_boxed.strip() == gold_boxed:
            return 1.0
        gold_num = _extract_number(str(gold))
        pred_num = _extract_number(response)
        if gold_num is not None and pred_num is not None:
            try:
                if float(gold_num) == float(pred_num):
                    return 1.0
            except ValueError:
                pass
        return 0.0


# ---------------------------------------------------------------------------
# Humanity's Last Exam (HLE) — extreme multi-disciplinary benchmark
# ---------------------------------------------------------------------------


class HLEBenchmark(BenchmarkBase):
    name = "hle"
    display_name = "Humanity's Last Exam"

    def format_prompt(self, q: dict) -> str:
        question = q.get("question", "")
        return (
            "Solve the following question from Humanity's Last Exam step-by-step.\n"
            "End your response with '\\boxed{<answer>}' containing your final concise answer.\n\n"
            f"Question: {question}"
        )

    def evaluate(self, q: dict, response: str) -> float:
        gold = str(q.get("answer", "")).strip()
        if not gold:
            return 0.0

        pred_boxed = _extract_boxed(response)
        if pred_boxed and pred_boxed.strip().lower() == gold.lower():
            return 1.0

        if len(gold) == 1 and gold.upper() in {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}:
            pred_letter = _extract_letter(response, set("ABCDEFGHIJ"))
            if pred_letter == gold.upper():
                return 1.0

        gold_num = _extract_number(gold)
        pred_num = _extract_number(pred_boxed) if pred_boxed else _extract_number(response)
        if gold_num is not None and pred_num is not None:
            try:
                if float(gold_num) == float(pred_num):
                    return 1.0
            except ValueError:
                pass

        if gold.lower() in response.lower():
            return 1.0

        return 0.0


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
        return (
            "Write a Python script to solve the following scientific problem.\n"
            "Return ONLY the complete Python code (including imports). "
            "No explanations, no markdown code blocks.\n\n"
            f"{deps}\n\n"
            f"{desc}"
        )

    def evaluate(self, q: dict, response: str) -> float:
        code = _strip_code_blocks(response)
        deps = q.get("required_dependencies", "")
        tests = q.get("general_tests", "")
        if isinstance(tests, list):
            tests_str = "\n\n".join(tests)
        else:
            tests_str = str(tests)

        parts = [p for p in [deps, code, tests_str] if p.strip()]
        full = "\n\n".join(parts)
        return 1.0 if _execute_code(full, self.settings.code_exec_timeout) else 0.0


# ---------------------------------------------------------------------------
# Tau-Bench — agentic tool use and multi-turn workflow
# ---------------------------------------------------------------------------


class TauBenchBenchmark(BenchmarkBase):
    name = "tau_bench"
    display_name = "Tau-Bench (Banking)"

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
        gold_name = str(gold_func.get("name", ""))

        if gold_name and gold_name.lower() in response.lower():
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
    "hle": HLEBenchmark,
    "scicode": SciCodeBenchmark,
    "tau_bench": TauBenchBenchmark,
}


def create_benchmark(name: str, config: BenchmarkConfig, settings: Settings) -> BenchmarkBase:
    cls = BENCHMARK_CLASSES.get(name)
    if cls is None:
        raise ValueError(f"Unknown benchmark: {name}")
    return cls(config, settings)

