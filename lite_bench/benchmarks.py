"""Benchmark implementations using real datasets with built-in verification.

7 benchmarks across 5 categories:
  Coding:      HumanEval (unit-test execution), MBPP (assert execution)
  Science:     GPQA Diamond (grad-level MC), ARC-Challenge (science MC)
  Math:        GSM8K (numerical exact match)
  Knowledge:   MMLU-Pro (10-choice MC)
  Instruction: IFEval (programmatic verifiers)
"""

from __future__ import annotations

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
# GPQA Diamond — graduate-level science multiple choice
# ---------------------------------------------------------------------------

class GPQABenchmark(BenchmarkBase):
    name = "gpqa"
    display_name = "GPQA Diamond"

    def _choices(self, q: dict) -> tuple[list[str], str]:
        question = q.get("Question") or q.get("question", "")
        correct = q.get("Correct Answer") or q.get("correct_answer", "")
        incorrect = [
            q.get("Incorrect Answer 1") or q.get("incorrect_answer_1", ""),
            q.get("Incorrect Answer 2") or q.get("incorrect_answer_2", ""),
            q.get("Incorrect Answer 3") or q.get("incorrect_answer_3", ""),
        ]
        choices = [correct] + [c for c in incorrect if c]
        rng = random.Random(hash(question) % 2**32)
        rng.shuffle(choices)
        correct_letter = "ABCD"[choices.index(correct)]
        return choices, correct_letter

    def format_prompt(self, q: dict) -> str:
        question = q.get("Question") or q.get("question", "")
        choices, _ = self._choices(q)
        options = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        return (
            f"Answer the following graduate-level science question.\n\n"
            f"Question: {question}\n{options}\n\n"
            "Reply with ONLY the letter (A, B, C, or D) of the correct answer."
        )

    def evaluate(self, q: dict, response: str) -> float:
        _, gold = self._choices(q)
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
        options = "\n".join(f"{chr(65+i)}. {t}" for i, t in enumerate(texts))
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
            f"Reply with ONLY the letter ({letters[0]}-{letters[len(options)-1]}) "
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
# Registry
# ---------------------------------------------------------------------------

BENCHMARK_CLASSES: dict[str, type[BenchmarkBase]] = {
    "humaneval": HumanEvalBenchmark,
    "mbpp": MBPPBenchmark,
    "gpqa": GPQABenchmark,
    "arc": ARCBenchmark,
    "gsm8k": GSM8KBenchmark,
    "mmlu_pro": MMLUProBenchmark,
    "ifeval": IFEvalBenchmark,
}


def create_benchmark(name: str, config: BenchmarkConfig, settings: Settings) -> BenchmarkBase:
    cls = BENCHMARK_CLASSES.get(name)
    if cls is None:
        raise ValueError(f"Unknown benchmark: {name}")
    return cls(config, settings)
