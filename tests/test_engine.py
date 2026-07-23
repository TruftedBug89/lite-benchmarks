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
