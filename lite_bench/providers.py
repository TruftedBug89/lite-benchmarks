"""Unified model provider via litellm — one API for 100+ providers.

Tracks per-request telemetry:
  - Token counts: input, output, thinking (reasoning), total
  - Timing: total wall-clock time, tokens-per-second (cloud APIs only)
  - Local models (lm_studio/, ollama/) skip speed metrics since
    they reflect hardware, not model quality.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import litellm
from rich.console import Console

from .config import Settings

console = Console()

litellm.suppress_debug_info = True

LOCAL_PREFIXES = ("lm_studio/", "ollama/")


@dataclass
class GenerationResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0
    total_time_ms: float | None = None      # None for local models
    tokens_per_second: float | None = None   # None for local models

    @property
    def output_ratio(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.output_tokens / self.total_tokens

    @property
    def thinking_ratio(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.thinking_tokens / self.total_tokens

    def to_dict(self) -> dict:
        d: dict = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "total_tokens": self.total_tokens,
            "output_ratio": round(self.output_ratio, 4),
        }
        if self.total_time_ms is not None:
            d["total_time_ms"] = round(self.total_time_ms, 1)
        if self.tokens_per_second is not None:
            d["tokens_per_second"] = round(self.tokens_per_second, 2)
        return d


def _is_local(model_id: str) -> bool:
    return model_id.startswith(LOCAL_PREFIXES)


def generate(model_id: str, prompt: str, settings: Settings) -> GenerationResult:
    """Call any litellm-supported model with full telemetry."""
    local = _is_local(model_id)

    start = time.perf_counter()
    response = litellm.completion(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        timeout=settings.request_timeout,
        num_retries=settings.max_retries,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    usage = response.usage
    inp = usage.prompt_tokens if usage else 0
    out = usage.completion_tokens if usage else 0
    total = usage.total_tokens if usage else (inp + out)

    # Thinking / reasoning tokens (DeepSeek R1, Gemini thinking, etc.)
    thinking = 0
    try:
        details = getattr(usage, "completion_tokens_details", None)
        if details:
            thinking = getattr(details, "reasoning_tokens", 0) or 0
    except Exception:
        pass

    # Speed metrics — skip for local models (hardware-dependent)
    time_ms = None if local else elapsed_ms
    tps = None
    if not local and elapsed_ms > 0 and out > 0:
        tps = out / (elapsed_ms / 1000)

    return GenerationResult(
        text=response.choices[0].message.content or "",
        input_tokens=inp,
        output_tokens=out,
        thinking_tokens=thinking,
        total_tokens=total,
        total_time_ms=time_ms,
        tokens_per_second=tps,
    )
