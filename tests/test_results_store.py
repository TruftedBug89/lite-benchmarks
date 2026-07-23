from __future__ import annotations

import json
from pathlib import Path

from lite_bench.config import BenchmarkConfig, Config, Settings
from lite_bench.results_store import SCHEMA_VERSION, load_latest_results, save_results


def test_results_store_schema_v2_and_zombie_key_dropping(tmp_path: Path):
    settings = Settings(results_dir=str(tmp_path))
    benchmarks = {
        "gpqa": BenchmarkConfig(name="gpqa", enabled=True, dataset="nichenshun/gpqa_diamond", num_samples=50)
    }
    config = Config(benchmarks=benchmarks, settings=settings)

    raw_data = {
        "schema_version": 1,
        "models": {
            "Test Model": {
                "model_id": "test/model",
                "gpqa": {"score": 0.8, "correct": 4, "total": 5},
                "arc": {"score": 0.5, "correct": 5, "total": 10},  # Zombie key (not in config)
                "gsm8k": {"score": 0.9, "correct": 9, "total": 10},  # Zombie key
            }
        },
    }

    file_path = tmp_path / "latest.json"
    file_path.write_text(json.dumps(raw_data), encoding="utf-8")

    # Load should drop zombie keys (arc, gsm8k)
    cleaned = load_latest_results(config, path=file_path)

    assert "Test Model" in cleaned
    assert "gpqa" in cleaned["Test Model"]
    assert "arc" not in cleaned["Test Model"]
    assert "gsm8k" not in cleaned["Test Model"]

    # Save results should include schema_version: 2
    saved_path = save_results(cleaned, config, results_dir=str(tmp_path), filename="saved.json")
    saved_json = json.loads(Path(saved_path).read_text(encoding="utf-8"))

    assert saved_json["schema_version"] == SCHEMA_VERSION
    assert "environment" in saved_json
    assert "python_version" in saved_json["environment"]
