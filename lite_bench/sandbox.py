"""Sandboxed execution of model-generated code.

Three layers of defense, designed to not interfere with legitimate benchmark
solutions (which are pure computation and need no OS/file/network access):

1. AST static scan of the *untrusted* (model-generated) portion. Dangerous
   imports, builtin calls, dunder-escape attribute access, and dunder-string
   attribute lookups are rejected before anything runs. The dataset-provided
   test harness is trusted and unscanned. THIS is the layer that stops the
   object-graph crawl (().__class__.__subclasses__()) which would otherwise
   reach the Windows sandbox's own Win32 handles (Layer 3).
2. Hardened subprocess. The child gets a minimal allow-listed environment
   (no API keys, tokens, or credentials), a fresh temporary working directory
   that is deleted afterwards, bytecode writes disabled, and a wall-clock
   timeout. The opt-in gate (allow_execution) is enforced here so direct
   callers can never bypass it.
3. Windows Job Object confinement (windows_sandbox.py). On Windows the spawned
   child is assigned to a fresh Job Object whose ActiveProcessLimit blocks
   grandchildren and whose UI restrictions block clipboard/desktop access - an
   OS-level backstop in case a model slips something past the AST scan. On
   other platforms only Layers 1+2 apply.

This is not a perfect security boundary against a determined adversary
writing handcrafted exploit code, but it reliably prevents the realistic
failure mode: an uncensored model emitting destructive commands
(rm -rf style deletion, file writes outside the sandbox, environment
exfiltration, subprocess spawning, network calls) as part of a "solution".
"""

from __future__ import annotations

import ast
import os
import secrets
import shutil
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# Layer 1: AST static scan
# ---------------------------------------------------------------------------

# Modules granting access to the OS, filesystem, network, processes,
# dynamic imports, or object-graph/frame escape hatches. operator/types/copy
# are intentionally NOT blocked (common in scientific code) - the dunder crawl
# they'd enable is stopped at the attribute layer (_BLOCKED_ATTRS) below instead.
_BLOCKED_MODULES = frozenset(
    {
        # OS / filesystem
        "os", "sys", "pathlib", "shutil", "glob", "fileinput", "tempfile",
        "io", "mmap", "stat", "nt", "posix", "_io", "ntpath", "posixpath",
        "genericpath",
        # processes / shells
        "subprocess", "pty", "signal", "multiprocessing", "concurrent",
        "_winapi", "msvcrt", "fcntl", "termios", "tty",
        # raw memory / FFI (full interpreter escape)
        "ctypes", "_ctypes", "cffi", "_cffi_backend",
        # network
        "socket", "ssl", "select", "requests", "urllib", "http", "ftplib",
        "telnetlib", "smtplib", "imaplib", "poplib", "xmlrpc", "asyncio",
        "websocket", "aiohttp", "httpx", "paramiko", "_socket", "_ssl",
        # dynamic import / code loading / introspection escapes
        "importlib", "runpy", "code", "codeop", "inspect", "traceback",
        "pdb", "bdb", "builtins", "linecache",
        "gc", "ast", "pkgutil", "zipimport", "site", "imp", "_thread",
        "codecs", "_codecs", "_warnings", "_abc", "_imp", "_frozen_importlib",
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
        # the harness itself + the Windows sandbox module: importing either
        # hands untrusted code live Win32 handles / the scan rules -> self-escape.
        "lite_bench", "windows_sandbox",
    }
)

# Builtins that must never be called or referenced by untrusted code:
# file handles, dynamic execution, and attribute/global-namespace tricks
# used to bypass the import blocklist. Note getattr/hasattr/setattr/delattr/
# vars are blocked OUTRIGHT - any call is rejected - which also closes the
# name-based dunder escape (getattr(o, "__subclasses__")) at the root, since
# the attribute is never resolved through them in the first place.
_BLOCKED_BUILTINS = frozenset(
    {
        "open", "eval", "exec", "compile", "__import__", "input",
        "getattr", "hasattr", "setattr", "delattr", "globals", "locals",
        "vars", "breakpoint", "exit", "quit", "__builtins__",
    }
)

# Attribute names used in classic Python sandbox escapes
# (e.g. ().__class__.__base__.__subclasses__()) and frame/global access.
_BLOCKED_ATTRS = frozenset(
    {
        "__class__", "__base__", "__bases__", "__subclasses__", "__mro__",
        "__globals__", "__builtins__", "__builtin__", "__import__",
        "__loader__", "__spec__", "__code__", "__reduce__", "__reduce_ex__",
        "__getattribute__", "__dict__", "__func__", "__self__",
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
        "attrgetter", "methodcaller", "itemgetter",
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
    shadowed_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            shadowed_names.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            shadowed_names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            shadowed_names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            shadowed_names.add(node.name)
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
            if (
                node.id in _BLOCKED_BUILTINS
                and isinstance(node.ctx, ast.Load)
                and node.id not in shadowed_names
            ):
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
            "PYTHONHASHSEED": "0",
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


def _run_child(
    script_path: str,
    sandbox_dir: str,
    env: dict[str, str],
    timeout: int,
    sentinel: str,
) -> tuple[bool, list[str]]:
    """Spawn the sandboxed interpreter, confined by a Windows Job Object when
    available. Process isolation (own cwd/env/exit code, clean kill on timeout)
    is preserved on every platform; the Job Object only adds OS-level
    grandchild-process and UI containment on Windows as defense in depth. Any
    failure to create/assign the job degrades gracefully to a plain
    subprocess - never fatal to the benchmark run.
    """
    child_job = None
    if os.name == "nt":
        try:
            from windows_sandbox import create_child_job

            child_job = create_child_job()
        except Exception:
            child_job = None
    try:
        proc = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
            errors="replace",
            cwd=sandbox_dir,
            env=env,
        )
        if child_job and child_job.handle:
            # Assign BEFORE the child can do real work. There is a tiny race
            # window, but Python startup (~tens of ms) dominates, and the AST
            # scan already blocks the spawn modules a model realistically emits.
            handle = getattr(proc, "_handle", None)
            if handle:
                child_job.assign(handle)
        try:
            stdout, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.communicate(timeout=5)
            except Exception:
                pass
            return False, ["execution timed out"]
        if sentinel:
            passed = proc.returncode == 0 and sentinel in (stdout or "")
        else:
            passed = proc.returncode == 0
        return passed, []
    finally:
        if child_job is not None:
            child_job.close()


def execute_sandboxed(
    untrusted_code: str,
    trusted_code: str,
    timeout: int,
    *,
    allow_execution: bool = False,
) -> tuple[bool, list[str]]:
    """Run untrusted model code + trusted test harness in a sandbox.

    The opt-in gate lives HERE, at the sandbox layer: ``allow_execution`` must
    be True (threaded from ``settings.allow_unsafe_code_execution``) or nothing
    runs. This means a direct ``evaluate()`` call - e.g. from a test or a future
    caller that skips the engine's bench-level skip - can never execute model
    code by accident; it fails closed instead.

    Returns (passed, violations). ``passed`` is True only if execution was
    permitted, the static scan found no violations, and the combined script
    exited with code 0. ``violations`` lists the scan rejections or the
    opt-out reason (empty when execution actually ran).
    """
    if not allow_execution:
        return False, ["code execution is disabled (allow_unsafe_code_execution not enabled)"]

    violations = scan_code(untrusted_code)
    if violations:
        return False, violations

    if not trusted_code.strip():
        return False, ["no trusted test harness provided"]

    sentinel = secrets.token_hex(8)
    script = (
        untrusted_code
        + "\n\n"
        + trusted_code
        + f'\n\nprint("{sentinel}", end="")'
    )

    sandbox_dir = tempfile.mkdtemp(prefix="litebench_sbx_")
    try:
        script_path = os.path.join(sandbox_dir, "run.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        return _run_child(
            script_path, sandbox_dir, _sandbox_env(sandbox_dir), timeout, sentinel
        )
    except subprocess.TimeoutExpired:
        return False, ["execution timed out"]
    except Exception as e:
        return False, [f"sandbox error: {e}"]
    finally:
        shutil.rmtree(sandbox_dir, ignore_errors=True)
