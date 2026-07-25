"""Tests for the Windows in-process sandbox (windows_sandbox.py).

Skipped on non-Windows (the module raises RuntimeError at import on other
platforms). Kept offline and fast: no real network, no long sleeps.
"""

from __future__ import annotations

import os

import pytest

if os.name != "nt":
    pytest.skip("windows_sandbox requires a Windows host", allow_module_level=True)

from windows_sandbox import (  # noqa: E402 (import after the allow_module_level skip)
    SandboxPolicy,
    SandboxSecurityError,
    create_child_job,
    run_in_sandbox,
    run_source_in_sandbox,
)


def _policy(root: str, scrub_env: bool = False) -> SandboxPolicy:
    os.makedirs(root, exist_ok=True)
    return SandboxPolicy(root_dir=root, scrub_env=scrub_env)


def test_run_source_passes(tmp_path):
    res = run_source_in_sandbox(
        "def f():\n    return 1\nassert f() == 1",
        timeout=15,
        policy=_policy(str(tmp_path)),
    )
    assert res.ok, res.notes
    assert res.exception is None


def test_run_source_systemexit_zero_is_ok(tmp_path):
    # BigCodeBench-style harnesses signal pass via sys.exit(0).
    res = run_source_in_sandbox("import sys\nsys.exit(0)", timeout=15, policy=_policy(str(tmp_path)))
    assert res.ok
    assert res.exception is None


def test_run_source_systemexit_nonzero_is_not_ok(tmp_path):
    res = run_source_in_sandbox("import sys\nsys.exit(2)", timeout=15, policy=_policy(str(tmp_path)))
    assert not res.ok
    assert res.exception is None  # controlled exit, not a sandbox fault


def test_run_source_assertion_failure_is_not_ok(tmp_path):
    res = run_source_in_sandbox("assert 1 == 2", timeout=15, policy=_policy(str(tmp_path)))
    assert not res.ok
    assert isinstance(res.exception, AssertionError)


def test_run_source_env_is_scrubbed(monkeypatch, tmp_path):
    """scrub_env=True (default for run_source_in_sandbox) hides secrets."""
    monkeypatch.setenv("LITEBENCH_TEST_SECRET", "supersecret")
    res = run_source_in_sandbox(
        "import os\nv = os.environ.get('LITEBENCH_TEST_SECRET')\nassert v is None, v",
        timeout=15,
        policy=_policy(str(tmp_path), scrub_env=True),
    )
    assert res.ok, res.notes


def test_run_in_sandbox_blocks_file_escape(tmp_path):
    def evil():
        with open(os.path.join(os.path.expanduser("~"), "pwned.txt"), "w") as f:
            f.write("nope")

    res = run_in_sandbox(evil, timeout=10, policy=_policy(str(tmp_path)))
    assert not res.ok
    assert isinstance(res.exception, SandboxSecurityError)


def test_run_in_sandbox_trusted_compute_succeeds(tmp_path):
    root = str(tmp_path)

    def work():
        scratch = os.path.join(root, "out.txt")
        with open(scratch, "w") as f:
            f.write(str(sum(i * i for i in range(1000))))
        with open(scratch) as f:
            return int(f.read())

    res = run_in_sandbox(work, timeout=10, policy=_policy(root))
    assert res.ok, res.notes
    assert res.value == sum(i * i for i in range(1000))


def test_create_child_job_succeeds():
    cj = create_child_job()
    assert cj.handle, f"expected a job handle, got note: {cj.note!r}"
    cj.close()


def test_self_escape_import_is_blocked(tmp_path):
    """Importing the sandbox module inside the sandbox must be vetoed - it
    would hand untrusted code our own Win32 handles."""
    from windows_sandbox import _DEFAULT_BLOCKED_MODULES

    assert "windows_sandbox" in _DEFAULT_BLOCKED_MODULES

    def evil():
        import windows_sandbox  # noqa: F401
        return windows_sandbox.kernel32  # type: ignore[attr-defined]

    res = run_in_sandbox(evil, timeout=10, policy=_policy(str(tmp_path)))
    assert not res.ok
    assert isinstance(res.exception, SandboxSecurityError)