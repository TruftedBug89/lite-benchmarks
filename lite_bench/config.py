"""Configuration loading, validation, and score aggregation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """A LiteLLM model identifier and its stable leaderboard label."""

    id: str
    name: str
    thinking_effort: str | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)



@dataclass(frozen=True)
class BenchmarkConfig:
    """Dataset source and sampling configuration for one benchmark."""

    name: str
    enabled: bool
    dataset: str
    num_samples: int
    split: str = "test"
    subset: str = ""
    revision: str | None = None


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    max_tokens: int = 4096
    temperature: float = 0.0
    request_timeout: int = 120
    code_exec_timeout: int = 15
    max_retries: int = 3
    max_concurrency: int = 5
    results_dir: str = "results"
    charts_dir: str = "charts"
    hf_token_env: str = "HF_TOKEN"
    allow_unsafe_code_execution: bool = False

    @property
    def hf_token(self) -> str | None:
        """Return the optional Hugging Face token without persisting it."""
        return os.environ.get(self.hf_token_env) or None


@dataclass(frozen=True)
class Config:
    models: list[ModelConfig] = field(default_factory=list)
    benchmarks: dict[str, BenchmarkConfig] = field(default_factory=dict)
    categories: dict[str, list[str]] = field(default_factory=dict)
    settings: Settings = field(default_factory=Settings)

    def enabled_benchmarks(self) -> dict[str, BenchmarkConfig]:
        return {name: benchmark for name, benchmark in self.benchmarks.items() if benchmark.enabled}

    def benchmark_category(self, bench_name: str) -> str:
        for category, benchmarks in self.categories.items():
            if bench_name in benchmarks:
                return category
        return "other"

    def category_score(self, bench_scores: Mapping[str, float], category: str) -> float | None:
        scores = [
            bench_scores[name] for name in self.categories.get(category, []) if name in bench_scores
        ]
        return sum(scores) / len(scores) if scores else None

    def overall_score(self, bench_scores: Mapping[str, float]) -> float | None:
        category_scores = [
            score
            for category in self.categories
            if (score := self.category_score(bench_scores, category)) is not None
        ]
        return sum(category_scores) / len(category_scores) if category_scores else None


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value.strip()


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number.")
    return float(value)


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load a YAML configuration and reject invalid or ambiguous inputs early."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file) or {}
    raw = _mapping(raw, "Configuration")

    raw_models = raw.get("models", [])
    if not isinstance(raw_models, list):
        raise ValueError("models must be a list.")
    models: list[ModelConfig] = []
    for index, value in enumerate(raw_models):
        item = _mapping(value, f"models[{index}]")
        model_id = _string(item.get("id"), f"models[{index}].id")
        name = _string(item.get("name", model_id), f"models[{index}].name")
        thinking_effort = item.get("thinking_effort")
        if thinking_effort is not None:
            thinking_effort = _string(thinking_effort, f"models[{index}].thinking_effort")
        extra_params = item.get("extra_params", {})
        if not isinstance(extra_params, Mapping):
            raise ValueError(f"models[{index}].extra_params must be a mapping.")
        models.append(
            ModelConfig(
                id=model_id,
                name=name,
                thinking_effort=thinking_effort,
                extra_params=dict(extra_params),
            )
        )

    if len({model.id for model in models}) != len(models):
        raise ValueError("Model IDs must be unique.")
    if len({model.name for model in models}) != len(models):
        raise ValueError("Model display names must be unique.")

    raw_benchmarks = _mapping(raw.get("benchmarks", {}), "benchmarks")
    benchmarks: dict[str, BenchmarkConfig] = {}
    for name, value in raw_benchmarks.items():
        benchmark_name = _string(name, "benchmark name")
        item = _mapping(value, f"benchmarks.{benchmark_name}")
        enabled = item.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"benchmarks.{benchmark_name}.enabled must be a boolean.")
        revision = item.get("revision")
        if revision is not None:
            revision = _string(revision, f"benchmarks.{benchmark_name}.revision")
        benchmarks[benchmark_name] = BenchmarkConfig(
            name=benchmark_name,
            enabled=enabled,
            dataset=_string(item.get("dataset"), f"benchmarks.{benchmark_name}.dataset"),
            num_samples=_positive_int(
                item.get("num_samples", 50), f"benchmarks.{benchmark_name}.num_samples"
            ),
            split=_string(item.get("split", "test"), f"benchmarks.{benchmark_name}.split"),
            subset=str(item.get("subset", "") or "").strip(),
            revision=revision,
        )

    raw_categories = _mapping(raw.get("categories", {}), "categories")
    categories: dict[str, list[str]] = {}
    assigned: set[str] = set()
    for name, value in raw_categories.items():
        category_name = _string(name, "category name")
        item = _mapping(value, f"categories.{category_name}")
        members = item.get("benchmarks", [])
        if not isinstance(members, list) or not all(isinstance(member, str) for member in members):
            raise ValueError(f"categories.{category_name}.benchmarks must be a list of names.")
        if not members:
            raise ValueError(f"categories.{category_name}.benchmarks must not be empty.")
        if len(set(members)) != len(members):
            raise ValueError(f"categories.{category_name}.benchmarks contains duplicates.")
        unknown = set(members).difference(benchmarks)
        if unknown:
            raise ValueError(
                f"categories.{category_name} references unknown benchmarks: {sorted(unknown)}"
            )
        overlap = assigned.intersection(members)
        if overlap:
            raise ValueError(f"Benchmarks may only belong to one category: {sorted(overlap)}")
        assigned.update(members)
        categories[category_name] = members

    raw_settings = _mapping(raw.get("settings", {}), "settings")
    temperature = _number(raw_settings.get("temperature", 0.0), "settings.temperature")
    if temperature < 0:
        raise ValueError("settings.temperature must be non-negative.")
    settings = Settings(
        seed=_nonnegative_int(raw_settings.get("seed", 42), "settings.seed"),
        max_tokens=_positive_int(raw_settings.get("max_tokens", 4096), "settings.max_tokens"),
        temperature=temperature,
        request_timeout=_positive_int(
            raw_settings.get("request_timeout", 120), "settings.request_timeout"
        ),
        code_exec_timeout=_positive_int(
            raw_settings.get("code_exec_timeout", 15), "settings.code_exec_timeout"
        ),
        max_retries=_nonnegative_int(raw_settings.get("max_retries", 3), "settings.max_retries"),
        max_concurrency=_positive_int(raw_settings.get("max_concurrency", 5), "settings.max_concurrency"),
        results_dir=_string(raw_settings.get("results_dir", "results"), "settings.results_dir"),
        charts_dir=_string(raw_settings.get("charts_dir", "charts"), "settings.charts_dir"),
        hf_token_env=_string(raw_settings.get("hf_token_env", "HF_TOKEN"), "settings.hf_token_env"),
    )

    return Config(models=models, benchmarks=benchmarks, categories=categories, settings=settings)
