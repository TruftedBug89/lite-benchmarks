"""HuggingFace dataset loading and deterministic sampling."""

from __future__ import annotations

from datasets import load_dataset
from rich.console import Console

from .config import BenchmarkConfig, Settings

console = Console()

_cache: dict[str, list[dict]] = {}


def load_questions(bench: BenchmarkConfig, settings: Settings) -> list[dict]:
    """Load and sample questions from a HuggingFace dataset.

    Uses a fixed seed so the same subset is used across runs.
    """
    cache_key = (
        f"{bench.dataset}/{bench.subset}/{bench.split}/"
        f"{bench.revision or 'main'}/{bench.num_samples}/{settings.seed}"
    )
    if cache_key in _cache:
        return _cache[cache_key]

    label = bench.dataset + (f" ({bench.subset})" if bench.subset else "")
    console.print(f"  [dim]Loading {label}...[/dim]")

    kwargs: dict = {"path": bench.dataset, "split": bench.split}
    if bench.subset:
        kwargs["name"] = bench.subset
    if bench.revision:
        kwargs["revision"] = bench.revision
    token = settings.hf_token
    if token:
        kwargs["token"] = token

    try:
        ds = load_dataset(**kwargs)
    except Exception as e:
        if "gated" in str(e).lower() or "401" in str(e) or "token" in str(e).lower():
            console.print(
                f"  [red]Dataset {bench.dataset} requires authentication.[/red]\n"
                f"  [red]Set the {settings.hf_token_env} environment variable with a "
                f"HuggingFace token that has access to this dataset.[/red]"
            )
        raise

    available = len(ds)
    if available == 0:
        raise ValueError(f"Dataset {bench.dataset!r} split {bench.split!r} contains no examples.")

    n = min(bench.num_samples, available)
    if available <= n:
        indices = list(range(available))
    elif n == 1:
        indices = [available - 1]
    else:
        indices = [int(i * (available - 1) / (n - 1)) for i in range(n)]

    ds = ds.select(indices)

    questions = [dict(row) for row in ds]
    _cache[cache_key] = questions
    console.print(f"  [dim]Sampled {len(questions)} questions from {available} available (uniform strided)[/dim]")
    return questions

