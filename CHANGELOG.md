# Changelog

## [0.2.0] - 2026-07-23

### ⚠️ Scoring v2 (Breaking Changes)
- **Seeded Random Sampling**: Replaced deterministic uniform strided dataset indexing with true seeded random sampling (`random.Random(seed).sample(...)`). Questions sampled differ from v0.1.0; results are not directly comparable to pre-v0.2.0 runs.
- **Strict HLE Scorer**: Removed substring free-pass matching; exact boxed, single-letter, or numerical extraction is required. Multimodal questions are filtered out at load time.
- **Tau-Bench Tool Matching**: Evaluates precise function name AND argument dictionary match.

### Added
- **Unified Engine (`lite_bench/engine.py`)**: Extracted all execution loops into a single engine with callback support.
- **Results Schema v2 (`lite_bench/results_store.py`)**: Stores Python, LiteLLM, Datasets versions, git SHA, per-benchmark question hashes, and full prompt/response text. Automatically drops stale benchmark keys from legacy runs.
- **Cost Tracking**: Integrated LiteLLM cost calculation per question (`total_cost_usd`) and added "Est. Cost" column to token usage tables.
- **Per-Model `max_tokens`**: Configurable per model in `config.yaml` or Web UI (defaults to 16,384 for reasoning models).
- **Wilson Confidence Intervals**: Per-benchmark tables in README show 95% Wilson score interval half-widths (±pp).
- **Web-First Dashboard**: Web UI (`web_app.py`) is the sole execution interface; removed deprecated CLI and terminal dashboard scripts.
- **Security Hardening**: Binding defaults to `127.0.0.1`, removed POST `env_vars` injection, sanitized `/api/config` responses, and added strict path traversal protection.

### Removed
- Removed `run_benchmark.py` CLI and `run_dashboard.py` terminal dashboard.
- Cleaned up tracked binaries (`archive.tar`, `files.zip`) and leftover test data fixtures.
