"""Sandboxed execution of model-generated code.

Four layers of defense, designed to not interfere with legitimate benchmark
solutions while still confining them:

1. AST static scan of the *untrusted* (model-generated) portion. Dangerous
   imports, builtin calls, dunder-escape attribute access, and dunder-string
   attribute lookups are rejected before anything runs. The dataset-provided
   test harness is trusted and unscanned. THIS is the layer that stops the
   object-graph crawl (().__class__.__subclasses__()) which would otherwise
   reach the Windows sandbox's own Win32 handles (Layer 4).
2. Hardened subprocess. The child gets a minimal allow-listed environment
   (no API keys, tokens, or credentials), a fresh temporary working directory
   that is deleted afterwards, bytecode writes disabled, and a wall-clock
   timeout. The opt-in gate (allow_execution) is enforced here so direct
   callers can never bypass it.
3. Runtime confinement shim (sandbox_child.py), prepended to the child script
   as trusted code. It wraps open/io/os.open/sqlite3.connect so file I/O may
   only touch the sandbox directory (reads also allow the Python install so
   imports work), and wraps socket so connections may only target loopback.
   This is what lets practical benchmarks (BigCodeBench file-I/O and network
   tasks) run at all: those modules are NOT AST-blocked, but the shim stops
   them reaching the host filesystem or the internet. Non-loopback connects
   fail fast (OSError) so port scanners report "closed" instead of hanging.
4. Windows Job Object confinement (windows_sandbox.py). On Windows the spawned
   child is assigned to a fresh Job Object whose ActiveProcessLimit blocks
   grandchildren and whose UI restrictions block clipboard/desktop access - an
   OS-level backstop in case a model slips something past the earlier layers.
   On other platforms only Layers 1-3 apply.

This is not a perfect security boundary against a determined adversary
writing handcrafted exploit code, but it reliably prevents the realistic
failure mode: an uncensored model emitting destructive commands
(rm -rf style deletion, file writes outside the sandbox, environment
exfiltration, subprocess spawning, outbound network calls) as part of a
"solution".
"""

from __future__ import annotations

import ast
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time

from .logging_utils import get_logger

log = get_logger("sandbox")

# Source of the in-child confinement shim, prepended to every sandboxed script.
# Loaded once at import time so the child never has to read it from disk. The
# shim defines install() at module scope; we call it immediately, then delete
# the name so untrusted code cannot re-invoke or introspect it.
_SHIM_PATH = os.path.join(os.path.dirname(__file__), "sandbox_child.py")
with open(_SHIM_PATH, encoding="utf-8") as _f:
    _SHIM_SOURCE = _f.read()
_SHIM_PREAMBLE = _SHIM_SOURCE + "\ninstall()\ndel install\n"

# ---------------------------------------------------------------------------
# Layer 1: AST static scan
# ---------------------------------------------------------------------------

# Modules granting access to the OS, filesystem, network, processes,
# dynamic imports, or object-graph/frame escape hatches. operator/types/copy
# are intentionally NOT blocked (common in scientific code) - the dunder crawl
# they'd enable is stopped at the attribute layer (_BLOCKED_ATTRS) below instead.
#
# NOTE on the "confined at runtime" modules: several file/network modules that
# practical benchmarks (BigCodeBench) legitimately require are NOT blocked here
# because the in-child confinement shim (sandbox_child.py) restricts what they
# can actually DO: open/io/tempfile/pathlib/glob/archives may only touch the
# sandbox directory, and socket/ssl/requests/urllib/http may only reach
# loopback. Blocking them outright made ~23% of BigCodeBench-Hard unpassable
# (every file-I/O and network task). The genuinely dangerous modules below stay
# blocked because no runtime confinement can make them safe.
_BLOCKED_MODULES = frozenset(
    {
        # Raw OS filesystem access modules that are NOT runtime-confined. NOTE:
        # `os` and `shutil` are deliberately NOT here — BigCodeBench forces them
        # in ~56 task preambles, so the confinement shim (sandbox_child.py)
        # allows the import but blocks process-spawning and confines every
        # filesystem call to the sandbox dir. The path helpers (ntpath,
        # posixpath, genericpath, stat) and codecs are pure string/encoding
        # utilities with no side channels, so they are allowed too.
        "sys", "fileinput", "mmap", "nt", "posix", "_io",
        # processes / shells (genuinely unconfineable: their whole purpose is
        # spawning processes; the Job Object is the only backstop, so they stay
        # blocked at the AST layer for cross-platform safety)
        "subprocess", "pty", "signal", "multiprocessing", "concurrent",
        "_winapi", "msvcrt", "fcntl", "termios", "tty",
        # raw memory / FFI (full interpreter escape)
        "ctypes", "_ctypes", "cffi", "_cffi_backend",
        # network clients that bypass the loopback shim or speak raw protocols.
        # socket/ssl/select/smtplib/requests/urllib/http are allowed (confined
        # to loopback at runtime); these lower-level/async ones stay blocked.
        "ftplib", "telnetlib", "imaplib", "poplib", "xmlrpc", "asyncio",
        "websocket", "aiohttp", "httpx", "paramiko",
        # dynamic import / code loading / introspection escapes
        "importlib", "runpy", "code", "codeop", "inspect", "traceback",
        "pdb", "bdb", "builtins", "linecache",
        "gc", "ast", "pkgutil", "zipimport", "site", "imp", "_thread",
        "_codecs", "_warnings", "_abc", "_imp", "_frozen_importlib",
        # (de)serialization that can execute code on load
        "pickle", "shelve", "marshal", "dill", "dbm",
        # environment / system info / side effects
        "platform", "psutil", "webbrowser", "dotenv",
        "winreg", "win32api", "win32com", "wmi", "pyautogui", "keyboard",
        "pynput", "pyperclip",
        # GUI
        "tkinter", "turtle",
        # the harness itself + the sandbox modules: importing either hands
        # untrusted code live Win32 handles / the confinement shim / the scan
        # rules -> self-escape.
        "lite_bench", "sandbox_child", "windows_sandbox",
    }
)

# Builtins that must never be called or referenced by untrusted code:
# dynamic execution and attribute/global-namespace tricks used to bypass the
# import blocklist. Note getattr/hasattr/setattr/delattr/vars are blocked
# OUTRIGHT - any call is rejected - which also closes the name-based dunder
# escape (getattr(o, "__subclasses__")) at the root, since the attribute is
# never resolved through them in the first place.
#
# `open` is deliberately NOT here: practical benchmarks need file I/O, and the
# in-child confinement shim (sandbox_child.py) restricts open() to the sandbox
# directory at runtime, so it is safe to reference while remaining confined.
_BLOCKED_BUILTINS = frozenset(
    {
        "eval", "exec", "compile", "__import__", "input",
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
            # Confinement shim (sandbox_child.py) reads this to know the only
            # directory untrusted file I/O may write to.
            "LITEBENCH_SANDBOX_ROOT": sandbox_dir,
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
        log.debug(
            f"sandbox: rejected {len(violations)} violation(s): "
            f"{'; '.join(violations[:5])}"
        )
        return False, violations

    if not trusted_code.strip():
        return False, ["no trusted test harness provided"]

    sentinel = secrets.token_hex(8)
    # The confinement shim runs first (trusted, unscanned), then the untrusted
    # model code, then the trusted test harness, then the success sentinel.
    script = (
        _SHIM_PREAMBLE
        + "\n"
        + untrusted_code
        + "\n\n"
        + trusted_code
        + f'\n\nprint("{sentinel}", end="")'
    )

    sandbox_dir = tempfile.mkdtemp(prefix="litebench_sbx_")
    started = time.perf_counter()
    try:
        script_path = os.path.join(sandbox_dir, "run.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        passed, issues = _run_child(
            script_path, sandbox_dir, _sandbox_env(sandbox_dir), timeout, sentinel
        )
        log.debug(
            f"sandbox: verdict={'PASS' if passed else 'FAIL'} "
            f"in {(time.perf_counter() - started) * 1000:.0f}ms "
            f"code_chars={len(untrusted_code)} "
            f"issues={'; '.join(issues[:5]) if issues else 'none'}"
        )
        return passed, issues
    except subprocess.TimeoutExpired:
        log.debug(
            f"sandbox: TIMEOUT after {timeout}s, code_chars={len(untrusted_code)}"
        )
        return False, ["execution timed out"]
    except Exception as e:
        log.debug(f"sandbox: error {type(e).__name__}: {e}")
        return False, [f"sandbox error: {e}"]
    finally:
        shutil.rmtree(sandbox_dir, ignore_errors=True)
