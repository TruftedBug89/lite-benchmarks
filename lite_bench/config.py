"""Configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    id: str   # litellm model ID, e.g. "deepseek/deepseek-chat"
    name: str  # display name for leaderboards


@dataclass
class BenchmarkConfig:
    name: str
    enabled: bool
    dataset: str
    num_samples: int
    split: str = "test"
    subset: str = ""


@dataclass
class Settings:
    seed: int = 42
    max_tokens: int = 4096
    temperature: float = 0.0
    request_timeout: int = 120
    code_exec_timeout: int = 15
    max_retries: int = 3
    retry_delay: float = 2.0
    results_dir: str = "results"
    charts_dir: str = "charts"
    hf_token_env: str = "HF_TOKEN"

    @property
    def hf_token(self) -> str | None:
        return os.environ.get(self.hf_token_env) or None


@dataclass
class Config:
    models: list[ModelConfig] = field(default_factory=list)
    benchmarks: dict[str, BenchmarkConfig] = field(default_factory=dict)
    categories: dict[str, list[str]] = field(default_factory=dict)
    settings: Settings = field(default_factory=Settings)

    def enabled_benchmarks(self) -> dict[str, BenchmarkConfig]:
        return {k: v for k, v in self.benchmarks.items() if v.enabled}

    def benchmark_category(self, bench_name: str) -> str:
        for cat, benches in self.categories.items():
            if bench_name in benches:
                return cat
        return "other"

    def category_score(self, bench_scores: dict[str, float], category: str) -> float | None:
        benches = self.categories.get(category, [])
        scores = [bench_scores[b] for b in benches if b in bench_scores]
        if not scores:
            return None
        return sum(scores) / len(scores)

    def overall_score(self, bench_scores: dict[str, float]) -> float | None:
        cat_scores = []
        for cat in self.categories:
            s = self.category_score(bench_scores, cat)
            if s is not None:
                cat_scores.append(s)
        if not cat_scores:
            return None
        return sum(cat_scores) / len(cat_scores)


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    models = [
        ModelConfig(id=m["id"], name=m.get("name", m["id"]))
        for m in raw.get("models", [])
    ]

    benchmarks = {}
    for name, bdata in raw.get("benchmarks", {}).items():
        benchmarks[name] = BenchmarkConfig(
            name=name,
            enabled=bdata.get("enabled", True),
            dataset=bdata["dataset"],
            num_samples=bdata.get("num_samples", 50),
            split=bdata.get("split", "test"),
            subset=bdata.get("subset", ""),
        )

    categories: dict[str, list[str]] = {}
    for cat, cdata in raw.get("categories", {}).items():
        categories[cat] = cdata.get("benchmarks", [])

    sdata = raw.get("settings", {})
    settings = Settings(
        seed=sdata.get("seed", 42),
        max_tokens=sdata.get("max_tokens", 4096),
        temperature=sdata.get("temperature", 0.0),
        request_timeout=sdata.get("request_timeout", 120),
        code_exec_timeout=sdata.get("code_exec_timeout", 15),
        max_retries=sdata.get("max_retries", 3),
        retry_delay=sdata.get("retry_delay", 2.0),
        results_dir=sdata.get("results_dir", "results"),
        charts_dir=sdata.get("charts_dir", "charts"),
        hf_token_env=sdata.get("hf_token_env", "HF_TOKEN"),
    )

    return Config(
        models=models,
        benchmarks=benchmarks,
        categories=categories,
        settings=settings,
    )
