"""Results persistence, atomic writing, schema v2, and aggregation."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import datasets
import litellm
from rich.console import Console

from .config import Config

console = Console()

SCHEMA_VERSION = 2

FATAL_PATTERNS = (
    "invalid_api_key",
    "invalid api key",
    "authentication_error",
    "authenticationerror",
    "unauthorized",
    "account_deactivated",
    "model_not_found",
    "not_found_error",
    "payment_required",
    "insufficient_quota",
    "credit balance is too low",
)


def is_fatal_error(exc: Exception) -> bool:
    """Return True if an exception indicates an unrecoverable model/account failure."""
    err_str = str(exc).lower()
    return any(pattern in err_str for pattern in FATAL_PATTERNS)


def _get_git_sha() -> str | None:
    try:
        import subprocess

        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None


def compute_question_hash(questions: list[dict]) -> str:
    """Compute sha1 hash of question prompts/ids to track exact benchmark sample state."""
    raw_str = "|".join(
        str(
            q.get("question")
            or q.get("prompt")
            or q.get("problem")
            or q.get("problem_text")
            or q.get("problem_description_main")
            or q.get("problem_id")
            or q.get("problemid")
            or q.get("id")
            or i
        )
        for i, q in enumerate(questions)
    )
    return hashlib.sha1(raw_str.encode("utf-8")).hexdigest()[:12]


def aggregate(mdata: dict, config: Config) -> None:
    """Recompute summary rollups (scores, token totals) for a model entry in-place."""
    enabled_names = set(config.enabled_benchmarks().keys())
    # Benchmarks that were attempted (have a score field, even if null) — used
    # for token/cost/throughput rollups.
    attempted = [
        v for k, v in mdata.items()
        if k in enabled_names and isinstance(v, dict) and "score" in v
    ]
    # Only benchmarks with a real numeric score count toward averages. A
    # fully-failed benchmark has score=None and is EXCLUDED (not a hard 0), so a
    # provider outage for one benchmark no longer drags down the overall score.
    bench_scores = {
        k: mdata[k]["score"]
        for k in mdata
        if k in enabled_names
        and isinstance(mdata[k], dict)
        and isinstance(mdata[k].get("score"), (int, float))
    }
    overall = config.overall_score(bench_scores)

    summary: dict = {
        "overall_score": round(overall, 4) if overall is not None else None,
        "completed_benchmarks": len(bench_scores),
        "total_input_tokens": sum(r.get("input_tokens", 0) for r in attempted),
        "total_output_tokens": sum(r.get("output_tokens", 0) for r in attempted),
        "total_thinking_tokens": sum(r.get("thinking_tokens", 0) for r in attempted),
        "total_all_tokens": sum(r.get("total_tokens", 0) for r in attempted),
    }

    total_cost = sum(r.get("total_cost_usd") for r in attempted if r.get("total_cost_usd") is not None)
    summary["total_cost_usd"] = round(total_cost, 6) if total_cost > 0 else None

    for cat in config.categories:
        cat_s = config.category_score(bench_scores, cat)
        summary[f"{cat}_score"] = round(cat_s, 4) if cat_s is not None else None

    tps_vals = [r.get("avg_tokens_per_second") for r in attempted if r.get("avg_tokens_per_second") is not None]
    summary["avg_tokens_per_second"] = round(sum(tps_vals) / len(tps_vals), 2) if tps_vals else None

    time_vals = [r.get("avg_time_ms") for r in attempted if r.get("avg_time_ms") is not None]
    summary["avg_time_ms"] = round(sum(time_vals) / len(time_vals), 1) if time_vals else None

    mdata["summary"] = summary


def _get_pkg_version(mod: any, name: str) -> str:
    v = getattr(mod, "__version__", None) or getattr(mod, "version", None)
    if v:
        return str(v)
    try:
        import importlib.metadata

        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


def save_results(
    results: dict,
    config: Config,
    results_dir: str = "results",
    filename: str = "latest.json",
) -> str:
    """Save results atomically using temp file and atomic replace."""
    os.makedirs(results_dir, exist_ok=True)
    target_path = Path(results_dir) / filename

    payload = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python_version": sys.version.split()[0],
            "litellm_version": _get_pkg_version(litellm, "litellm"),
            "datasets_version": _get_pkg_version(datasets, "datasets"),
            "git_sha": _get_git_sha(),
        },
        "settings": {
            "seed": config.settings.seed,
            "max_tokens": config.settings.max_tokens,
            "temperature": config.settings.temperature,
            "request_timeout": config.settings.request_timeout,
            "max_retries": config.settings.max_retries,
            "max_concurrency": config.settings.max_concurrency,
            "max_concurrent_models": config.settings.max_concurrent_models,
        },
        "models": results,
    }

    # Write atomically: temp file in the target dir + os.replace (same volume,
    # so no cross-drive rename failure on Windows). Clean up the temp file on any
    # error, and retry the replace a few times in case Windows has the target
    # briefly locked (editor/antivirus/indexer) rather than aborting the run.
    dir_path = target_path.parent
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=dir_path, delete=False, encoding="utf-8") as tf:
            json.dump(payload, tf, indent=2, ensure_ascii=False)
            temp_name = tf.name

        last_err: Exception | None = None
        for _ in range(5):
            try:
                os.replace(temp_name, target_path)
                temp_name = None  # moved successfully; nothing left to clean up
                break
            except PermissionError as e:
                last_err = e
                time.sleep(0.2)
        else:
            if last_err is not None:
                raise last_err
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
    return str(target_path)


HISTORY_FILENAME = "history.json"


def _history_path(results_dir: str) -> Path:
    return Path(results_dir) / HISTORY_FILENAME


def append_run_history(results: dict, config: Config, results_dir: str = "results") -> None:
    """Snapshot the current run's per-model summaries into history.json.

    Each entry records a timestamp and a compact per-model score map so the
    leaderboard can show a range (min–max / ±) across repeated runs of the
    same model.  Old runs are never deleted."""
    os.makedirs(results_dir, exist_ok=True)
    hp = _history_path(results_dir)

    history: list[dict] = []
    if hp.is_file():
        try:
            data = json.loads(hp.read_text(encoding="utf-8"))
            history = data if isinstance(data, list) else data.get("runs", [])
        except (json.JSONDecodeError, OSError):
            history = []

    enabled_names = set(config.enabled_benchmarks().keys())
    models_snapshot: dict[str, dict] = {}
    for mname, mdata in results.items():
        if not isinstance(mdata, dict):
            continue
        bench_scores: dict[str, float] = {}
        for k in mdata:
            if k in enabled_names and isinstance(mdata[k], dict) and isinstance(mdata[k].get("score"), (int, float)):
                bench_scores[k] = mdata[k]["score"]
        summary = mdata.get("summary", {})
        overall = summary.get("overall_score")
        if overall is None and bench_scores:
            overall = config.overall_score(bench_scores)
        models_snapshot[mname] = {
            "overall_score": round(overall, 4) if overall is not None else None,
            "benchmarks": {k: round(v, 4) for k, v in bench_scores.items()},
        }

    if not models_snapshot:
        return

    history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": models_snapshot,
    })

    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=hp.parent, delete=False, encoding="utf-8") as tf:
            json.dump(history, tf, indent=2, ensure_ascii=False)
            tmp_name = tf.name
        os.replace(tmp_name, hp)
        tmp_name = None
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def load_run_history(results_dir: str = "results") -> list[dict]:
    """Return the list of historical run snapshots (newest last)."""
    hp = _history_path(results_dir)
    if not hp.is_file():
        return []
    try:
        data = json.loads(hp.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("runs", [])
    except (json.JSONDecodeError, OSError):
        return []


def load_latest_results(config: Config, path: str | Path = "results/latest.json") -> dict:
    """Load latest results JSON, preserving all configured benchmark keys while dropping unknown zombie keys."""
    p = Path(path)
    if not p.is_file():
        return {}

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to parse {path}: {e}[/yellow]")
        return {}

    if not isinstance(data, dict):
        console.print(f"[yellow]Warning: {path} is not a results object; ignoring it.[/yellow]")
        return {}

    sv = data.get("schema_version")
    if sv != SCHEMA_VERSION:
        console.print(f"[dim]Note: {path} schema_version is {sv} (current is {SCHEMA_VERSION}).[/dim]")

    models = data.get("models", {})
    if not isinstance(models, dict):
        return {}

    all_benchmarks = set(config.benchmarks.keys())
    cleaned_models: dict = {}

    for mname, mdata in models.items():
        if not isinstance(mdata, dict):
            continue
        cleaned_mdata: dict = {}
        for k, v in mdata.items():
            if k in ("model_id", "thinking_effort", "summary"):
                cleaned_mdata[k] = v
            elif k in all_benchmarks:
                cleaned_mdata[k] = v
            else:
                console.print(f"[dim]Dropping stale benchmark key {k!r} from model {mname!r}[/dim]")
        
        # Re-aggregate
        aggregate(cleaned_mdata, config)
        cleaned_models[mname] = cleaned_mdata

    return cleaned_models
