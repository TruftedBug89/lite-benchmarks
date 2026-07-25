"""HuggingFace dataset loading and deterministic sampling."""

from __future__ import annotations

import random
from collections.abc import Callable

from datasets import load_dataset
from rich.console import Console

from .config import BenchmarkConfig, Settings

console = Console()

_cache: dict[str, list[dict]] = {}


def load_questions(
    bench: BenchmarkConfig,
    settings: Settings,
    row_filter: Callable[[dict], bool] | None = None,
) -> list[dict]:
    """Load and sample questions from a HuggingFace dataset.

    Uses a fixed seed so the same subset is used across runs. When ``row_filter``
    is given it is applied to the FULL dataset *before* sampling, so
    ``num_samples`` reflects the filtered population (e.g. SuperGPQA's "hard"
    subset) instead of silently shrinking after the fact.
    """
    cache_key = (
        f"{bench.name}/{bench.dataset}/{bench.subset}/{bench.split}/"
        f"{bench.revision or 'main'}/{bench.num_samples}/{settings.seed}/"
        f"{'filtered' if row_filter is not None else 'all'}"
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

    if row_filter is not None:
        pool = [i for i in range(available) if row_filter(ds[i])]
        if not pool:
            raise ValueError(
                f"Dataset {bench.dataset!r} has no rows matching the benchmark filter."
            )
    else:
        pool = list(range(available))

    pool_size = len(pool)
    n = min(bench.num_samples, pool_size)
    if pool_size <= n:
        indices = pool
    else:
        rng = random.Random(settings.seed)
        indices = sorted(rng.sample(pool, n))

    ds = ds.select(indices)

    questions = [dict(row) for row in ds]
    _cache[cache_key] = questions
    if row_filter is not None:
        console.print(
            f"  [dim]Sampled {len(questions)} questions from {pool_size} matching rows "
            f"({available} total, seeded random sample)[/dim]"
        )
    else:
        console.print(
            f"  [dim]Sampled {len(questions)} questions from {available} available (seeded random sample)[/dim]"
        )
    return questions
