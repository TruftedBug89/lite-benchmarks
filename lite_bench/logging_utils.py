"""Run logging: short console lines + full-detail per-run log files with refs.

Design
------
- One detailed log file per benchmark run (and one for the web server itself):
  ``logs/run_<timestamp>[_<suffix>].log``, written at DEBUG level by every
  module via :func:`get_logger`. Every line carries timestamp, level, thread,
  logger name and ``module.py:line`` so anything printed on the console can be
  traced back to the exact code path.
- Console output stays short: Rich prints (engine/UI) are the primary console
  stream and carry a ``→ logs/<file>`` suffix pointing at the detail file;
  the stdlib handler here only surfaces ERROR-level records (with the same
  suffix) so nothing critical is ever file-only.
- Secret safety: loggers never receive API key values (modules only pass
  env-var *names*); :func:`scrub` can be applied to free-text before logging.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT_DIR / "logs"

_NS = "litebench"
_RUN_LOG: Path | None = None

_SECRET_PATTERNS = re.compile(
    r"Bearer\s+\S+|sk-\S+|api[_-]?key[=:]\s*\S+|token[=:]\s*\S+",
    re.IGNORECASE,
)


def scrub(text: str) -> str:
    """Redact api-key-looking substrings before a message touches a log file."""
    return _SECRET_PATTERNS.sub("[REDACTED]", str(text))


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger (``litebench.<name>``) with the shared file
    handler attached by :func:`open_run_log`."""
    return logging.getLogger(f"{_NS}.{name}")


def run_log_path() -> Path | None:
    """Absolute path of the currently open run log, or None."""
    return _RUN_LOG


def run_log_rel() -> str:
    """Short relative pointer (``logs/run_....log``) for console lines, or ''."""
    return f"logs/{_RUN_LOG.name}" if _RUN_LOG is not None else ""


def log_ref() -> str:
    """Suffix to append to short console lines, e.g. `` → logs/run_x.log``."""
    rel = run_log_rel()
    return f" → {rel}" if rel else ""


def _console_handler() -> logging.Handler:
    """Static console handler: ERROR+ only, each line ends with the run-log ref
    so a critical line on the console always says where the details live."""
    handler = logging.StreamHandler()
    handler.setLevel(logging.ERROR)
    handler.setFormatter(
        logging.Formatter(
            "%(levelname)s %(message)s" + log_ref(),
        )
    )
    handler._lb_console = True  # type: ignore[attr-defined]
    return handler


def open_run_log(suffix: str = "") -> Path:
    """(Re)point the ``litebench`` file handler at a fresh detail log file.

    Called once at web-server startup (``suffix="server"``) and again for every
    benchmark run (``suffix="r<N>"``). Only the file handler is replaced; the
    console handler is registered once at first import.
    """
    global _RUN_LOG
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"run_{stamp}" + (f"_{suffix}" if suffix else "") + ".log"
    path = LOG_DIR / name
    _RUN_LOG = path

    logger = logging.getLogger(_NS)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for h in list(logger.handlers):
        if getattr(h, "_lb_console", False):
            continue
        logger.removeHandler(h)
        h.close()

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler._lb_file = True  # type: ignore[attr-defined]
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-5s | %(threadName)-12s | "
            "%(name)s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    if not any(getattr(h, "_lb_console", False) for h in logger.handlers):
        logger.addHandler(_console_handler())
    return path
