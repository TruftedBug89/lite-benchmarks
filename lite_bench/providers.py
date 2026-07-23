"""Unified model provider via litellm — one API for 100+ providers."""

from __future__ import annotations

from dataclasses import dataclass

import litellm
from rich.console import Console

from .config import Settings

console = Console()

# Suppress litellm's verbose logging by default
litellm.suppress_debug_info = True


@dataclass
class GenerationResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


def generate(model_id: str, prompt: str, settings: Settings) -> GenerationResult:
    """Call any litellm-supported model. Retries and timeout handled by litellm."""
    response = litellm.completion(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        timeout=settings.request_timeout,
        num_retries=settings.max_retries,
    )
    usage = response.usage
    return GenerationResult(
        text=response.choices[0].message.content or "",
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )
