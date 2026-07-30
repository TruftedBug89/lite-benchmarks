"""Tests for the code-execution sandbox (lite_bench/sandbox.py)."""

from __future__ import annotations

import pytest

from lite_bench.sandbox import execute_sandboxed, scan_code

# ---------------------------------------------------------------------------
# Static scan: dangerous code must be rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_code",
    [
        # os/shutil imports are allowed at the AST layer (the runtime shim
        # confines them), but their dangerous ACTIONS stay rejected here because
        # the scan still flags process-control calls and blocked builtins.
        "import os\nos.system('whoami')",
        "import subprocess",
        "import psutil",
        "shutil.rmtree('/')",
        "import ctypes",
        "import pickle",
        "import importlib",
        "import sys\nprint(sys.modules)",
        "import multiprocessing",
        "import asyncio",
        "import winreg",
        "import ftplib",
        "import paramiko",
        "eval('1+1')",
        "exec('pass')",
        "compile('pass', '<s>', 'exec')",
        "__import__('os')",
        "getattr(x, 'y')",
        "setattr(x, 'y', 1)",
        "globals()",
        "input()",
        "f = eval",
        "y = (1).__class__",
        "y = ().__class__.__base__.__subclasses__()",
        "f = func.__globals__",
        "x = {}.__dict__",
        "obj.__builtins__",
        "frame.f_globals",
        "gen.gi_frame",
        "something.system('dir')",
        "mod.popen('calc')",
    ],
)
def test_scan_rejects_dangerous_code(bad_code: str):
    assert scan_code(bad_code), f"expected violation for: {bad_code}"


# ---------------------------------------------------------------------------
# Static scan: legitimate benchmark solutions must pass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "good_code",
    [
        "def add(a, b):\n    return a + b",
        "import math\ndef area(r):\n    return math.pi * r * r",
        "from collections import Counter\ndef mode(xs):\n    return Counter(xs).most_common(1)[0][0]",
        "import itertools, heapq, bisect, re, functools",
        "import numpy as np\ndef f(x):\n    return np.array(x).mean()",
        "import pandas as pd\ndef f(df):\n    return df.eval('a + b')",
        "from scipy.optimize import minimize\ndef f():\n    return minimize(lambda x: x*x, 0).x",
        "import sympy as sp",
        "class Node:\n    def __init__(self, v):\n        self.v = v\n    def __repr__(self): return str(self.v)",
        "class A:\n    pass\nclass B(A):\n    def f(self):\n        return super().f()",
        "def f(xs):\n    return [x*x for x in xs if x > 0]",
        "from dataclasses import dataclass\n@dataclass\nclass P:\n    x: int",
        "from typing import List\ndef f(xs: List[int]) -> int:\n    return sum(xs)",
        "import datetime, random, statistics, string, json, copy",
        "def f(s):\n    return type(s).__name__",
        "import unittest\nclass T(unittest.TestCase):\n    pass",
        "x = float('inf')\nimport threading",
        "from functools import lru_cache\n@lru_cache(maxsize=None)\ndef f(n):\n    return n",
        # Confined-at-runtime modules: allowed by the AST scan because the
        # in-child shim restricts what they can DO (file I/O -> sandbox dir,
        # sockets -> loopback, os/shutil process control -> blocked). Blocking
        # them made BigCodeBench unpassable (it forces these imports).
        "import os",
        "import shutil",
        "import socket",
        "import ssl, select",
        "import requests",
        "from urllib import request",
        "import io",
        "import pathlib",
        "import tempfile",
        "import glob",
        "import zipfile, tarfile, gzip, bz2, lzma",
        "import sqlite3",
        "import smtplib",
        "import codecs",
        "import ntpath, posixpath, stat",
        "open('data.txt')",
        "x = open",
        "with open('out.txt', 'w') as f:\n    f.write('hi')",
    ],
)
def test_scan_allows_legitimate_code(good_code: str):
    assert scan_code(good_code) == [], f"false positive for: {good_code}"


def test_scan_syntax_error_is_violation():
    assert scan_code("def broken(:\n")


# ---------------------------------------------------------------------------
# Execution: behavior of the sandboxed subprocess
# ---------------------------------------------------------------------------


def test_execute_valid_solution_passes():
    untrusted = "def add(a, b):\n    return a + b"
    trusted = "assert add(2, 3) == 5\nassert add(-1, 1) == 0"
    ok, violations = execute_sandboxed(untrusted, trusted, timeout=15, allow_execution=True)
    assert ok
    assert violations == []


def test_execute_wrong_solution_fails():
    untrusted = "def add(a, b):\n    return a - b"
    trusted = "assert add(2, 3) == 5"
    ok, _ = execute_sandboxed(untrusted, trusted, timeout=15, allow_execution=True)
    assert not ok


def test_execute_rejects_malicious_code_without_running():
    untrusted = "import os\nos.system('echo pwned')"
    ok, violations = execute_sandboxed(untrusted, "assert True", timeout=15, allow_execution=True)
    assert not ok
    assert violations  # rejected by the scan, never executed


def test_execute_env_is_scrubbed(monkeypatch):
    """API keys in the parent env must not leak into the child process."""
    monkeypatch.setenv("LITEBENCH_TEST_SECRET_KEY", "supersecret")
    trusted = (
        "import os\n"
        "assert os.environ.get('LITEBENCH_TEST_SECRET_KEY') is None, 'env leaked'"
    )
    ok, _ = execute_sandboxed("x = 1", trusted, timeout=15, allow_execution=True)
    assert ok


def test_execute_cwd_is_sandbox_dir():
    """The child must run inside its temp sandbox, not the repo."""
    trusted = "import os\nassert 'litebench_sbx_' in os.getcwd()"
    ok, _ = execute_sandboxed("x = 1", trusted, timeout=15, allow_execution=True)
    assert ok


def test_execute_timeout_kills_runaway_code():
    untrusted = "while True:\n    pass"
    ok, violations = execute_sandboxed(untrusted, "assert True", timeout=2, allow_execution=True)
    assert not ok
    assert violations == ["execution timed out"]


def test_execute_numpy_available():
    """Scientific stack must keep working inside the sandbox."""
    pytest.importorskip("numpy")
    untrusted = "import numpy as np\ndef f():\n    return float(np.mean([1, 2, 3]))"
    trusted = "assert f() == 2.0"
    ok, violations = execute_sandboxed(untrusted, trusted, timeout=30, allow_execution=True)
    assert ok, violations


def test_execute_fails_closed_without_opt_in():
    """No allow_execution flag => code never runs, regardless of content."""
    ok, violations = execute_sandboxed("def add(a, b):\n    return a + b", "assert add(1,1)==2", timeout=15)
    assert not ok
    assert violations == ["code execution is disabled (allow_unsafe_code_execution not enabled)"]


def test_scan_rejects_name_based_dunder_lookup():
    # getattr/hasattr are blocked outright, which closes the
    # name-based dunder escape (getattr(o, "__subclasses__")) at the root.
    assert scan_code("hasattr(obj, '__subclasses__')")
    assert scan_code("hasattr(obj, '__class__')")
    assert scan_code("getattr(o, '__globals__')")
    # Any call to getattr is rejected, dunder or not.
    assert scan_code("getattr(o, 'x')")


def test_scan_rejects_reflection_modules():
    for mod in ("import gc", "import ast", "import pkgutil", "import _thread", "import windows_sandbox"):
        assert scan_code(mod), f"expected violation for: {mod}"


# ---------------------------------------------------------------------------
# Runtime confinement shim (sandbox_child.py): allowed names, confined actions
# ---------------------------------------------------------------------------


def test_shim_allows_writes_inside_sandbox():
    """File I/O within the sandbox cwd (where tests create temp files) works."""
    untrusted = (
        "with open('out.txt', 'w') as f:\n"
        "    f.write('hi')\n"
        "assert open('out.txt').read() == 'hi'"
    )
    ok, violations = execute_sandboxed(untrusted, "assert True", timeout=20, allow_execution=True)
    assert ok, violations


def test_shim_blocks_read_outside_sandbox():
    """Reading a host file outside the sandbox must fail (PermissionError)."""
    host = _host_file()
    untrusted = f"open(r'{host}').read()"
    ok, _ = execute_sandboxed(untrusted, "assert True", timeout=20, allow_execution=True)
    assert not ok


def test_shim_blocks_write_outside_sandbox():
    untrusted = (
        "import tempfile, os\n"
        "p = os.path.join(os.path.dirname(tempfile.gettempdir()), 'evil.txt')\n"
        "open(p, 'w').write('x')"
    )
    ok, _ = execute_sandboxed(untrusted, "assert True", timeout=20, allow_execution=True)
    assert not ok


def test_shim_allows_loopback_socket():
    """A localhost client/server round-trip must succeed (BigCodeBench echo task)."""
    untrusted = (
        "import socket, threading\n"
        "srv = socket.socket(); srv.bind(('127.0.0.1', 0)); srv.listen(1)\n"
        "port = srv.getsockname()[1]\n"
        "def acc():\n"
        "    c, _ = srv.accept(); c.recv(8); c.send(b'pong'); c.close()\n"
        "t = threading.Thread(target=acc); t.start()\n"
        "cl = socket.socket(); cl.connect(('127.0.0.1', port)); cl.send(b'ping')\n"
        "assert cl.recv(8) == b'pong'\n"
        "cl.close(); t.join(); srv.close()"
    )
    ok, violations = execute_sandboxed(untrusted, "assert True", timeout=20, allow_execution=True)
    assert ok, violations


def test_shim_blocks_nonloopback_socket_fast():
    """A routed (non-loopback) connect must fail immediately, not hang."""
    untrusted = (
        "import socket\n"
        "s = socket.socket()\n"
        "s.connect(('8.8.8.8', 53))"  # raises OSError -> nonzero exit
    )
    ok, _ = execute_sandboxed(untrusted, "assert True", timeout=20, allow_execution=True)
    assert not ok


def test_shim_confines_sqlite_to_sandbox():
    inside = (
        "import sqlite3\n"
        "con = sqlite3.connect('t.db')\n"
        "con.execute('create table x(a)'); con.commit(); con.close()"
    )
    ok, violations = execute_sandboxed(inside, "assert True", timeout=20, allow_execution=True)
    assert ok, violations
    host = _host_file()
    outside = f"import sqlite3\nsqlite3.connect(r'{host}')"
    ok2, _ = execute_sandboxed(outside, "assert True", timeout=20, allow_execution=True)
    assert not ok2


def test_shim_allows_os_file_ops_inside_sandbox():
    """os file calls within the sandbox dir must work (BigCodeBench forces os)."""
    untrusted = (
        "import os\n"
        "os.makedirs('sub', exist_ok=True)\n"
        "with open(os.path.join('sub', 'x.txt'), 'w') as f:\n"
        "    f.write('hi')\n"
        "assert os.path.exists(os.path.join('sub', 'x.txt'))\n"
        "assert os.listdir('sub') == ['x.txt']"
    )
    ok, violations = execute_sandboxed(untrusted, "assert True", timeout=20, allow_execution=True)
    assert ok, violations


def test_shim_blocks_os_file_ops_outside_sandbox():
    host = _host_file()
    untrusted = f"import os\nos.remove(r'{host}')"
    ok, _ = execute_sandboxed(untrusted, "assert True", timeout=20, allow_execution=True)
    assert not ok


def test_shim_blocks_os_process_control():
    """os is importable but process-spawning actions are blocked at runtime."""
    for stmt in ("os.system('echo hi')", "os.popen('echo hi')", "os.kill(1, 9)"):
        untrusted = f"import os\n{stmt}"
        ok, _ = execute_sandboxed(untrusted, "assert True", timeout=20, allow_execution=True)
        assert not ok, f"expected runtime block for: {stmt}"


def test_shim_blocks_shutil_outside_sandbox():
    untrusted = f"import shutil\nshutil.rmtree(r'{_host_dir()}')"
    ok, _ = execute_sandboxed(untrusted, "assert True", timeout=20, allow_execution=True)
    assert not ok


def _host_file() -> str:
    """A real file guaranteed to live outside any sandbox dir."""
    import os
    import sys

    return os.path.join(os.path.dirname(sys.executable), "python.exe")


def _host_dir() -> str:
    import os
    import sys

    return os.path.dirname(sys.executable)
