"""Tests for finish_reason tracking, LaTeX cleaning, truncated boxed extraction, and resumption logic."""

from __future__ import annotations

import pytest
from lite_bench.benchmarks import _clean_latex, _extract_boxed, _extract_letter, _extract_number
from lite_bench.providers import GenerationResult


def test_clean_latex():
    raw = r"\mathbf{A}"
    assert _clean_latex(raw) == "A"

    raw_text = r"\text{42}"
    assert _clean_latex(raw_text) == "42"


def test_extract_boxed_truncated():
    # Unclosed boxed tag due to length truncation
    truncated_text = r"The solution is \boxed{123"
    assert _extract_boxed(truncated_text) == "123"


def test_extract_letter_with_latex():
    response = r"The answer is \boxed{\mathbf{C}}"
    assert _extract_letter(response, {"A", "B", "C", "D"}) == "C"


def test_generation_result_truncation():
    res = GenerationResult(text="Partial...", finish_reason="length")
    assert res.is_truncated is True
    d = res.to_dict()
    assert d.get("finish_reason") == "length"


def test_generation_result_normal():
    res = GenerationResult(text="Complete answer.", finish_reason="stop")
    assert res.is_truncated is False
    d = res.to_dict()
    assert d.get("finish_reason") == "stop"
