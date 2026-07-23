from __future__ import annotations

from unittest.mock import MagicMock, patch

from lite_bench.config import BenchmarkConfig, Settings
from lite_bench.datasets import load_questions


@patch("lite_bench.datasets.load_dataset")
def test_seeded_sampling_determinism(mock_load_ds):
    mock_ds = MagicMock()
    mock_ds.__len__.return_value = 100
    mock_ds.select.side_effect = lambda idxs: [({"id": i, "question": f"Q{i}"}) for i in idxs]
    mock_load_ds.return_value = mock_ds

    bench = BenchmarkConfig(
        name="test_bench", enabled=True, dataset="test/ds", num_samples=10
    )

    settings1 = Settings(seed=42)
    qs1 = load_questions(bench, settings1)

    settings2 = Settings(seed=42)
    qs2 = load_questions(bench, settings2)

    assert qs1 == qs2
    assert len(qs1) == 10

    settings3 = Settings(seed=999)
    # Clear internal cache for test
    from lite_bench.datasets import _cache

    _cache.clear()
    qs3 = load_questions(bench, settings3)

    assert qs1 != qs3
    assert len(qs3) == 10
