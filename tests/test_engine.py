from __future__ import annotations

from unittest.mock import MagicMock, patch

from lite_bench.config import ModelConfig, Settings
from lite_bench.engine import FatalModelError, process_question


def test_process_question_eval_error_handling():
    bench = MagicMock()
    bench.name = "mock_bench"
    bench.format_prompt.return_value = "Test prompt"
    bench.evaluate.side_effect = AttributeError("Unexpected list type in dependencies")

    model = ModelConfig(id="test/model", name="Test Model")
    settings = Settings(max_retries=2)

    gen_mock = MagicMock()
    gen_mock.text = "Model answer"
    gen_mock.input_tokens = 10
    gen_mock.output_tokens = 20
    gen_mock.thinking_tokens = 0
    gen_mock.total_tokens = 30
    gen_mock.total_time_ms = 100.0
    gen_mock.tokens_per_second = 200.0
    gen_mock.cost_usd = 0.001

    with patch("lite_bench.engine.generate", return_value=gen_mock) as mock_generate:
        result = process_question(0, {"q": 1}, bench, model, settings)

        # Scorer exception must score 0.0, record eval_error, and call generate exactly ONCE (no provider retry)
        assert mock_generate.call_count == 1
        assert result["status"] == "eval_error"
        assert result["score"] == 0.0
        assert "Unexpected list type" in result["error_msg"]


def test_process_question_fatal_error():
    bench = MagicMock()
    bench.format_prompt.return_value = "Test prompt"

    model = ModelConfig(id="test/model", name="Test Model")
    settings = Settings(max_retries=3)

    with patch("lite_bench.engine.generate", side_effect=ValueError("invalid_api_key")):
        try:
            process_question(0, {"q": 1}, bench, model, settings)
            raise AssertionError("Expected FatalModelError")
        except FatalModelError:
            pass


def _ok_gen(text: str = "Model answer") -> MagicMock:
    gen = MagicMock()
    gen.text = text
    gen.input_tokens = 10
    gen.output_tokens = 20
    gen.thinking_tokens = 0
    gen.total_tokens = 30
    gen.total_time_ms = 100.0
    gen.tokens_per_second = 200.0
    gen.cost_usd = 0.001
    return gen


def test_process_question_waits_for_good_response():
    """max_retries=0 (default): transient errors retry until a good response
    arrives — the question never advances on an error."""
    bench = MagicMock()
    bench.name = "mock_bench"
    bench.format_prompt.return_value = "Test prompt"
    bench.evaluate.return_value = 1.0

    model = ModelConfig(id="test/model", name="Test Model")
    settings = Settings(max_retries=0)
    transient = ConnectionError("upstream 503 service unavailable")
    retries: list[dict] = []

    with (
        patch(
            "lite_bench.engine.generate",
            side_effect=[transient, transient, _ok_gen()],
        ) as mock_generate,
        patch("lite_bench.engine._interruptible_sleep", return_value=False),
    ):
        result = process_question(
            0, {"q": 1}, bench, model, settings,
            on_retry=lambda m, b, info: retries.append(info),
        )

        assert mock_generate.call_count == 3
        assert result["status"] == "success"
        assert result["score"] == 1.0
        assert len(retries) == 2
        assert retries[0]["attempt"] == 1
        assert retries[1]["attempt"] == 2
        assert retries[0]["max_attempts"] is None  # unlimited


def test_process_question_permanent_error_never_waits():
    """Permanent errors (context length, content filter) can never succeed, so
    they give up immediately even with unlimited retries enabled."""
    bench = MagicMock()
    bench.name = "mock_bench"
    bench.format_prompt.return_value = "Test prompt"

    model = ModelConfig(id="test/model", name="Test Model")
    settings = Settings(max_retries=0)

    with (
        patch(
            "lite_bench.engine.generate",
            side_effect=ValueError("context_length_exceeded: too many tokens"),
        ) as mock_generate,
        patch("lite_bench.engine._interruptible_sleep", return_value=False) as mock_sleep,
    ):
        result = process_question(0, {"q": 1}, bench, model, settings)

        assert mock_generate.call_count == 1
        assert mock_sleep.call_count == 0
        assert result["status"] == "error"
        assert result["score"] == 0.0


def test_process_question_persistent_connection_error_never_scores_zero():
    """Regression: identical transient errors (e.g. litellm.APIError connection
    errors) must keep retrying forever with max_retries=0 — the "same error 3x
    in a row" bailout only applies to the engine's own semantic RuntimeErrors
    (empty completion, no text content), never to provider/network failures."""
    bench = MagicMock()
    bench.name = "mock_bench"
    bench.format_prompt.return_value = "Test prompt"
    bench.evaluate.return_value = 1.0

    model = ModelConfig(id="test/model", name="Test Model")
    settings = Settings(max_retries=0)
    conn_error = ValueError("litellm.APIError: APIError: OpenAIException - Connection error.")
    retries: list[dict] = []

    with (
        patch(
            "lite_bench.engine.generate",
            side_effect=[conn_error, conn_error, conn_error, conn_error, _ok_gen()],
        ) as mock_generate,
        patch("lite_bench.engine._interruptible_sleep", return_value=False),
    ):
        result = process_question(
            0, {"q": 1}, bench, model, settings,
            on_retry=lambda m, b, info: retries.append(info),
        )

        assert mock_generate.call_count == 5  # 4 identical connection errors, never bailed
        assert result["status"] == "success"
        assert result["score"] == 1.0
        assert len(retries) == 4


def test_process_question_same_runtime_error_three_times_gives_up():
    """The 3x bailout still protects against the engine's own semantic
    RuntimeErrors (empty completion) repeating identically forever."""
    bench = MagicMock()
    bench.name = "mock_bench"
    bench.format_prompt.return_value = "Test prompt"

    model = ModelConfig(id="test/model", name="Test Model")
    settings = Settings(max_retries=0)

    with (
        patch(
            "lite_bench.engine.generate",
            side_effect=[
                RuntimeError("The provider returned an empty completion."),
                RuntimeError("The provider returned an empty completion."),
                RuntimeError("The provider returned an empty completion."),
                RuntimeError("The provider returned an empty completion."),
            ],
        ) as mock_generate,
        patch("lite_bench.engine._interruptible_sleep", return_value=False),
    ):
        result = process_question(0, {"q": 1}, bench, model, settings)

        assert mock_generate.call_count == 3  # bailed after 3 identical RuntimeErrors
        assert result["status"] == "error"


def test_process_question_capped_retries():
    """A positive max_retries still caps attempts on transient errors."""
    bench = MagicMock()
    bench.name = "mock_bench"
    bench.format_prompt.return_value = "Test prompt"

    model = ModelConfig(id="test/model", name="Test Model")
    settings = Settings(max_retries=2)

    with (
        patch(
            "lite_bench.engine.generate",
            side_effect=TimeoutError("request timed out"),
        ) as mock_generate,
        patch("lite_bench.engine._interruptible_sleep", return_value=False),
    ):
        result = process_question(0, {"q": 1}, bench, model, settings)

        assert mock_generate.call_count == 2
        assert result["status"] == "error"
        assert "timed out" in result["error_msg"]


def test_process_question_stop_during_backoff_cancels():
    """Force Stop during the retry wait returns 'cancelled', not 'error'."""
    bench = MagicMock()
    bench.name = "mock_bench"
    bench.format_prompt.return_value = "Test prompt"

    model = ModelConfig(id="test/model", name="Test Model")
    settings = Settings(max_retries=0)

    with (
        patch(
            "lite_bench.engine.generate",
            side_effect=ConnectionError("connection reset"),
        ),
        patch("lite_bench.engine._interruptible_sleep", return_value=True),
    ):
        result = process_question(
            0, {"q": 1}, bench, model, settings, should_stop=lambda: False
        )

        assert result["status"] == "cancelled"
        assert "error_msg" not in result


def test_process_question_evaluates_exactly_once():
    """When a benchmark provides evaluate_detailed (every real one does), the
    engine must use it as the single source of truth and NOT also call
    evaluate() — code-execution benchmarks run the sandbox per call, so a
    second invocation doubles cost and can collide on bound ports."""

    class CountingBench:
        name = "counting"

        def format_prompt(self, q):
            return "p"

        def evaluate(self, q, response):
            raise AssertionError("evaluate() must not run when evaluate_detailed returns a dict")

        def evaluate_detailed(self, q, response):
            return {
                "score": 1.0,
                "expected_answer": "gold",
                "extracted_answer": "pred",
                "judge_response": "single pass",
            }

    model = ModelConfig(id="test/model", name="Test Model")
    settings = Settings(max_retries=1)

    with patch("lite_bench.engine.generate", return_value=_ok_gen("pred")):
        result = process_question(0, {"q": 1}, CountingBench(), model, settings)

    assert result["status"] == "success"
    assert result["score"] == 1.0
    assert result["expected_answer"] == "gold"
    assert result["extracted_answer"] == "pred"
    assert result["judge_response"] == "single pass"
