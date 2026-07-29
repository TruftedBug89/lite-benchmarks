"""Diagnostic: fetch 10 BigCodeBench-Hard questions, call the model, and log
raw output vs what _strip_code_blocks produces, plus the AST scan result."""

import sys
import textwrap

sys.path.insert(0, ".")

from lite_bench.benchmarks import BigCodeBenchHardBenchmark, _strip_code_blocks
from lite_bench.config import BenchmarkConfig, Settings
from lite_bench.providers import generate
from lite_bench.config import ModelConfig
from lite_bench.sandbox import scan_code

settings = Settings(allow_unsafe_code_execution=True)
bench_cfg = BenchmarkConfig(
    name="bigcodebench_hard",
    enabled=True,
    dataset="bigcode/bigcodebench-hard",
    num_samples=10,
    split="v0.1.4",
)
bench = BigCodeBenchHardBenchmark(bench_cfg, settings)
questions = bench.load()[:10]

model = ModelConfig(id="gemini/gemma-4-26b-a4b-it", name="gemma-4-26b")

for i, q in enumerate(questions):
    prompt = bench.format_prompt(q)
    print(f"\n{'='*80}")
    print(f"QUESTION {i}: {q.get('entry_point', '?')}")
    print(f"{'='*80}")

    try:
        result = generate(model, prompt, settings)
        raw = result.text
    except Exception as e:
        print(f"  PROVIDER ERROR: {e}")
        continue

    print(f"\n--- RAW RESPONSE (first 1500 chars) ---")
    print(raw[:1500])

    stripped = _strip_code_blocks(raw)
    print(f"\n--- AFTER _strip_code_blocks (first 800 chars) ---")
    print(stripped[:800])

    violations = scan_code(stripped)
    if violations:
        print(f"\n--- SANDBABY VIOLATIONS: {violations}")
    else:
        print(f"\n--- SANDBABY: PASS (no violations)")

    dedented = textwrap.dedent(stripped)
    violations2 = scan_code(dedented)
    if violations2 != violations:
        print(f"--- AFTER textwrap.dedent: {violations2 if violations2 else 'PASS'}")

print("\nDone.")
