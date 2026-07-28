"""Unified model provider via litellm — one API for 100+ providers.

Tracks per-request telemetry:
  - Token counts: input, output, thinking (reasoning), total
  - Timing: total wall-clock time, tokens-per-second (cloud APIs only)
  - Cost in USD (where supported by LiteLLM)
  - Local models (lm_studio/, ollama/) skip speed metrics since
    they reflect hardware, not model quality.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

import litellm
from rich.console import Console

from .config import ModelConfig, Settings

console = Console()

litellm.suppress_debug_info = True
# Many models reject parameters they don't support (e.g. reasoning models reject
# `temperature`/`max_tokens`). Rather than raising, let litellm drop unsupported
# params so a single incompatible field doesn't zero out an entire model's run.
litellm.drop_params = True

# LiteLLM response objects trigger noisy Pydantic v2 serializer warnings
# (optional fields not always populated). Harmless — silence them.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

LOCAL_PREFIXES = ("lm_studio/", "ollama/")

# Remember which (model, env-var) pairs we've already warned about so the
# "api_key_env not set" notice prints once, not once per question.
_warned_missing_keys: set[tuple[str, str]] = set()

_PARAM_ERROR_HINTS = (
    "unsupported",
    "unexpected keyword",
    "not supported",
    "invalid parameter",
    "unrecognized",
    "unknown parameter",
    "invalid_request_error",
)


def _completion_with_fallback(kwargs: dict[str, Any]) -> Any:
    """Call litellm.completion, retrying once after dropping/translating the
    specific parameter a model rejected.

    ``litellm.drop_params`` handles most incompatibilities up front; this is a
    targeted backstop that only fires when the error actually names a parameter,
    and says *which* parameter was dropped (the old code blindly stripped
    ``reasoning_effort`` on any "not supported" error, silently disabling
    thinking for reasoning models)."""
    try:
        return litellm.completion(**kwargs)
    except Exception as e:
        msg = str(e).lower()
        if not any(hint in msg for hint in _PARAM_ERROR_HINTS):
            raise

        dropped: list[str] = []
        if "reasoning_effort" in kwargs and ("reasoning" in msg or "reasoning_effort" in msg):
            kwargs.pop("reasoning_effort", None)
            dropped.append("reasoning_effort")
        if "temperature" in kwargs and "temperature" in msg:
            kwargs.pop("temperature", None)
            dropped.append("temperature")
        if "max_tokens" in kwargs and ("max_tokens" in msg or "max_completion_tokens" in msg):
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            dropped.append("max_tokens->max_completion_tokens")
        if not dropped:
            # The error looks parameter-related but names nothing we recognize;
            # shed the most-likely-optional field rather than fail the question.
            if "reasoning_effort" in kwargs:
                kwargs.pop("reasoning_effort", None)
                dropped.append("reasoning_effort")
            elif "temperature" in kwargs:
                kwargs.pop("temperature", None)
                dropped.append("temperature")
        if not dropped:
            raise
        console.print(
            f"[dim]Model {kwargs.get('model')} rejected a parameter; retrying without {', '.join(dropped)}.[/dim]"
        )
        return litellm.completion(**kwargs)


@dataclass
class GenerationResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0  # Non-reasoning completion tokens.
    thinking_tokens: int = 0
    total_tokens: int = 0
    total_time_ms: float | None = None  # None for local models
    tokens_per_second: float | None = None  # None for local models
    cost_usd: float | None = None
    finish_reason: str | None = None

    @property
    def is_truncated(self) -> bool:
        return self.finish_reason == "length"

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
        if self.finish_reason:
            d["finish_reason"] = self.finish_reason
        if self.total_time_ms is not None:
            d["total_time_ms"] = round(self.total_time_ms, 1)
        if self.tokens_per_second is not None:
            d["tokens_per_second"] = round(self.tokens_per_second, 2)
        if self.cost_usd is not None:
            d["cost_usd"] = round(self.cost_usd, 6)
        return d


def _is_local(model_id: str) -> bool:
    return model_id.startswith(LOCAL_PREFIXES)


def _map_reasoning_effort(effort: str) -> str:
    effort_lower = effort.lower()
    if effort_lower in ("max", "xhigh", "ultracode", "high"):
        return "high"
    elif effort_lower in ("medium", "mid"):
        return "medium"
    elif effort_lower in ("low", "min"):
        return "low"
    return effort_lower


def generate(model: ModelConfig | str, prompt: str, settings: Settings) -> GenerationResult:
    """Call any LiteLLM-supported model with full telemetry."""
    if isinstance(model, ModelConfig):
        model_id = model.id
        thinking_effort = model.thinking_effort
        max_tokens = model.max_tokens or settings.max_tokens
        extra_params = model.extra_params
        api_base = model.api_base
        api_key_env = model.api_key_env
    else:
        model_id = model
        thinking_effort = None
        max_tokens = settings.max_tokens
        extra_params = {}
        api_base = None
        api_key_env = None

    local = _is_local(model_id)

    kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": settings.temperature,
        "timeout": settings.request_timeout,
        "num_retries": 0,  # Engine handles retries with jitter
    }

    if thinking_effort:
        kwargs["reasoning_effort"] = _map_reasoning_effort(thinking_effort)
    if api_base:
        kwargs["api_base"] = api_base
    if api_key_env:
        import os as _os

        resolved_key = _os.environ.get(api_key_env)
        if resolved_key:
            kwargs["api_key"] = resolved_key
        elif (model_id, api_key_env) not in _warned_missing_keys:
            _warned_missing_keys.add((model_id, api_key_env))
            console.print(
                f"[yellow]Warning: env var {api_key_env!r} for {model_id} is not set; "
                f"falling back to litellm's default key lookup.[/yellow]"
            )
    if extra_params:
        kwargs.update(extra_params)

    start = time.perf_counter()
    response = _completion_with_fallback(kwargs)

    elapsed_ms = (time.perf_counter() - start) * 1000

    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    details = getattr(usage, "completion_tokens_details", None)
    thinking_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
    # OpenAI-style usage folds reasoning into completion_tokens (subtract it);
    # providers that report reasoning *separately* would otherwise be clamped to
    # 0, so when thinking >= completion we treat completion as the visible output.
    if thinking_tokens and thinking_tokens <= completion_tokens:
        output_tokens = completion_tokens - thinking_tokens
    else:
        output_tokens = completion_tokens
    total_tokens = int(getattr(usage, "total_tokens", 0) or (input_tokens + completion_tokens))

    # Speed metrics — skip for local models (hardware-dependent)
    time_ms = None if local else elapsed_ms
    tps = None
    if not local and elapsed_ms > 0 and completion_tokens > 0:
        tps = completion_tokens / (elapsed_ms / 1000)

    # Cost calculation via LiteLLM
    cost_usd = None
    if not local:
        try:
            cost_val = litellm.completion_cost(completion_response=response)
            if isinstance(cost_val, (int, float)) and cost_val >= 0:
                cost_usd = float(cost_val)
        except Exception:
            cost_usd = None

    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("The provider returned no completion choices.")
    choice = choices[0]
    content = getattr(getattr(choice, "message", None), "content", None)
    if not isinstance(content, str):
        raise RuntimeError("The provider returned a completion without text content.")

    finish_reason = getattr(choice, "finish_reason", None)
    if isinstance(finish_reason, str):
        finish_reason = finish_reason.lower()

    return GenerationResult(
        text=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        total_tokens=total_tokens,
        total_time_ms=time_ms,
        tokens_per_second=tps,
        cost_usd=cost_usd,
        finish_reason=finish_reason,
    )
