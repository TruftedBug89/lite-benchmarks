from __future__ import annotations

import logging

from lite_bench.logging_utils import (
    get_logger,
    log_ref,
    open_run_log,
    run_log_path,
    run_log_rel,
    scrub,
)


def test_scrub_redacts_secrets():
    assert scrub("api_key=sk-abc123") == "[REDACTED]"
    assert scrub("Bearer eyJhbGciOi") == "[REDACTED]"
    assert scrub("plain message") == "plain message"


def test_open_run_log_writes_details(tmp_path, monkeypatch):
    from lite_bench import logging_utils

    monkeypatch.setattr(logging_utils, "LOG_DIR", tmp_path)
    path = open_run_log("test")
    assert path.is_file()

    logger = get_logger("test_mod")
    logger.info("short console line")
    logger.debug("full detail line")

    content = path.read_text(encoding="utf-8")
    assert "short console line" in content
    assert "full detail line" in content
    assert "test_mod" in content  # logger name recorded


def test_refs_point_at_open_log(tmp_path, monkeypatch):
    from lite_bench import logging_utils

    monkeypatch.setattr(logging_utils, "LOG_DIR", tmp_path)
    open_run_log("refs")

    assert run_log_path() is not None
    assert run_log_rel() == f"logs/{run_log_path().name}"
    assert log_ref() == f" → logs/{run_log_path().name}"
    assert run_log_path().name.startswith("run_")


def test_open_run_log_switches_file(tmp_path, monkeypatch):
    from lite_bench import logging_utils

    monkeypatch.setattr(logging_utils, "LOG_DIR", tmp_path)
    first = open_run_log("first")
    get_logger("x").info("in first")
    second = open_run_log("second")
    assert first != second

    get_logger("x").info("in second")
    first_content = first.read_text(encoding="utf-8")
    second_content = second.read_text(encoding="utf-8")
    assert "in first" in first_content
    assert "in first" not in second_content
    assert "in second" in second_content


def test_console_handler_is_error_level_only(tmp_path, monkeypatch):
    from lite_bench import logging_utils

    monkeypatch.setattr(logging_utils, "LOG_DIR", tmp_path)
    open_run_log("levels")
    logger = get_logger("levels")
    parent = logging.getLogger("litebench")
    assert parent.level == logging.DEBUG
    assert logger.level == logging.NOTSET  # inherits DEBUG from parent
    console_handlers = [
        h for h in parent.handlers if getattr(h, "_lb_console", False)
    ]
    assert len(console_handlers) == 1
    assert console_handlers[0].level == logging.ERROR
