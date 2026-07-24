"""Sandboxed execution of model-generated code.

Two layers of defense, designed to not interfere with legitimate benchmark
solutions (which are pure computation and need no OS/file/network access):

1. AST static scan of the *untrusted* (model-generated) portion. Dangerous
   imports, builtin calls, and dunder escape hatches are rejected before
   anything runs. The dataset-provided test harness is trusted and unscanned.
2. Hardened subprocess. The child gets a minimal allow-listed environment
   (no API keys, tokens, or credentials), a fresh temporary working
   directory that is deleted afterwards, bytecode writes disabled, and a
   wall-clock timeout.

This is not a perfect security boundary against a determined adversary
writing handcrafted exploit code, but it reliably prevents the realistic
failure mode: an uncensored model emitting destructive commands
(rm -rf style deletion, file writes outside the sandbox, environment
exfiltration, subprocess spawning, network calls) as part of a "solution".
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# Layer 1: AST static scan
# ---------------------------------------------------------------------------

# Modules granting access to the OS, filesystem, network, processes,
# dynamic imports, or object-graph/frame escape hatches.
_BLOCKED_MODULES = frozenset(
    {
        # OS / filesystem
        "os", "sys", "pathlib", "shutil", "glob", "fileinput", "tempfile",
        "io", "mmap", "stat",
        # processes / shells
        "subprocess", "pty", "signal", "multiprocessing", "concurrent",
        "_winapi", "msvcrt", "fcntl", "termios", "tty",
        # raw memory / FFI (full interpreter escape)
        "ctypes", "_ctypes", "cffi", "_cffi_backend",
        # network
        "socket", "ssl", "select", "requests", "urllib", "http", "ftplib",
        "telnetlib", "smtplib", "imaplib", "poplib", "xmlrpc", "asyncio",
        "websocket", "aiohttp", "httpx", "paramiko",
        # dynamic import / code loading / introspection escapes
        "importlib", "runpy", "code", "codeop", "inspect", "traceback",
        "pdb", "bdb", "builtins", "linecache",
        # (de)serialization that can execute code or touch files
        "pickle", "shelve", "marshal", "dill", "dbm", "sqlite3",
        # archives (direct file writes)
        "zipfile", "tarfile", "gzip", "bz2", "lzma",
        # environment / system info / side effects
        "platform", "psutil", "getpass", "webbrowser", "dotenv",
        "winreg", "win32api", "win32com", "wmi", "pyautogui", "keyboard",
        "pynput", "pyperclip",
        # GUI
        "tkinter", "turtle",
    }
)

# Builtins that must never be called or referenced by untrusted code:
# file handles, dynamic execution, and attribute/global-namespace tricks
# used to bypass the import blocklist.
_BLOCKED_BUILTINS = frozenset(
    {
        "open", "eval", "exec", "compile", "__import__", "input",
        "getattr", "setattr", "delattr", "globals", "locals", "vars",
        "breakpoint", "exit", "quit",
    }
)

# Attribute names used in classic Python sandbox escapes
# (e.g. ().__class__.__base__.__subclasses__()) and frame/global access.
_BLOCKED_ATTRS = frozenset(
    {
        "__class__", "__base__", "__bases__", "__subclasses__", "__mro__",
        "__globals__", "__builtins__", "__builtin__", "__import__",
        "__loader__", "__spec__", "__code__", "__reduce__", "__reduce_ex__",
        "__getattribute__", "__dict__", "__init__", "__func__", "__self__",
        "gi_frame", "gi_code", "cr_frame", "cr_code", "ag_frame", "ag_code",
        "f_globals", "f_locals", "f_builtins", "f_code", "tb_frame", "mro",
    }
)

# Destructive/shell-ish attribute calls, blocked regardless of the object
# (defense in depth in case a module object is ever smuggled through).
_BLOCKED_ATTR_CALLS = frozenset(
    {
        "system", "popen", "popen2", "popen3", "popen4",
        "execl", "execle", "execlp", "execv", "execve", "execvp",
        "spawnl", "spawnv", "spawnlp", "spawnvp",
        "rmtree", "kill", "killpg", "terminate",
    }
)


def scan_code(code: str) -> list[str]:
    """Return a list of policy violations found in untrusted code.

    An empty list means the code passed the static scan.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e}"]

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _BLOCKED_MODULES:
                    violations.append(f"import of blocked module '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in _BLOCKED_MODULES:
                    violations.append(f"import of blocked module '{node.module}'")
        elif isinstance(node, ast.Attribute):
            if node.attr in _BLOCKED_ATTRS:
                violations.append(f"access to blocked attribute '{node.attr}'")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BLOCKED_BUILTINS:
                violations.append(f"call to blocked builtin '{func.id}()'")
            elif isinstance(func, ast.Attribute) and func.attr in _BLOCKED_ATTR_CALLS:
                # Attribute calls are only blocked for destructive shell-ish
                # names (os.system-style). eval/exec as attributes are left
                # alone: reaching the real builtins requires __builtins__
                # access (blocked above), and pandas legitimately uses
                # df.eval(...).
                violations.append(f"call to blocked function '{func.attr}()'")
        elif isinstance(node, ast.Name):
            # Block aliasing tricks like `f = eval; f("...")`.
            if node.id in _BLOCKED_BUILTINS and isinstance(node.ctx, ast.Load):
                violations.append(f"reference to blocked builtin '{node.id}'")
    return violations


# ---------------------------------------------------------------------------
# Layer 2: hardened subprocess
# ---------------------------------------------------------------------------

# Environment variables the sandboxed child is allowed to inherit.
# Everything else — API keys, tokens, credentials, PYTHONPATH, HF tokens —
# is dropped by construction.
# NOTE: APPDATA/LOCALAPPDATA must pass through: on Windows the user
# site-packages directory (where numpy, litellm, etc. live) is resolved
# from APPDATA, and dropping it would break third-party imports.
_ENV_ALLOWLIST = (
    "SystemRoot", "PATH", "PATHEXT", "COMSPEC", "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER",
    "APPDATA", "LOCALAPPDATA",
    "LANG", "LC_ALL", "TZ",
)


def _sandbox_env(sandbox_dir: str) -> dict[str, str]:
    env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
    env.update(
        {
            # Redirect all scratch locations into the sandbox dir.
            "TEMP": sandbox_dir,
            "TMP": sandbox_dir,
            "TMPDIR": sandbox_dir,
            "HOME": sandbox_dir,
            "USERPROFILE": sandbox_dir,
            "HOMEDRIVE": os.path.splitdrive(sandbox_dir)[0] or sandbox_dir,
            "HOMEPATH": os.path.splitdrive(sandbox_dir)[1] or "\\",
            # No __pycache__ writes into site-packages or the repo.
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            # Headless plotting, caches inside the sandbox.
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": os.path.join(sandbox_dir, "mpl"),
            "NUMBA_CACHE_DIR": os.path.join(sandbox_dir, "numba"),
            "XDG_CACHE_HOME": os.path.join(sandbox_dir, "cache"),
            # Belt-and-suspenders: no Hub network access from child.
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return env


def execute_sandboxed(
    untrusted_code: str, trusted_code: str, timeout: int
) -> tuple[bool, list[str]]:
    """Run untrusted model code + trusted test harness in a sandbox.

    Returns (passed, violations). ``passed`` is True only if the static
    scan found no violations and the combined script exited with code 0.
    ``violations`` lists the scan rejections (empty when execution ran).
    """
    violations = scan_code(untrusted_code)
    if violations:
        return False, violations

    script = (
        untrusted_code + "\n\n" + trusted_code
        if trusted_code.strip()
        else untrusted_code
    )

    sandbox_dir = tempfile.mkdtemp(prefix="litebench_sbx_")
    try:
        script_path = os.path.join(sandbox_dir, "run.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            timeout=timeout,
            text=True,
            cwd=sandbox_dir,
            env=_sandbox_env(sandbox_dir),
        )
        return result.returncode == 0, []
    except subprocess.TimeoutExpired:
        return False, ["execution timed out"]
    except Exception as e:
        return False, [f"sandbox error: {e}"]
    finally:
        shutil.rmtree(sandbox_dir, ignore_errors=True)
