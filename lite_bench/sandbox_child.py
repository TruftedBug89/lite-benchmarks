"""Runtime confinement shim executed INSIDE the sandboxed child process.

This is Layer 2.5 of the sandbox: the AST scan (Layer 1) decides which names
untrusted model code may even *reference*; this shim decides what those names
may actually *do* at runtime. It is prepended to the child script as trusted
code (never AST-scanned) and installs precise, directory- and network-scoped
policy that the static scan cannot express:

  * File I/O confinement. ``builtins.open`` / ``io.open`` / ``io.FileIO`` /
    ``io.open_code`` / ``os.open`` are wrapped so that:
      - WRITES/DELETES are only permitted inside the sandbox root, and
      - READS are permitted inside the sandbox root OR the Python installation
        (stdlib + site-packages + user site + cwd) so ``import numpy`` and
        friends keep working.
    Anything else raises ``PermissionError``. The sandbox root comes from the
    ``LITEBENCH_SANDBOX_ROOT`` env var (set by ``sandbox.py``); ``TEMP``/``TMP``
    already point inside it, so ``tempfile``-based test harnesses work as-is.

  * Loopback-only networking. ``socket.socket.connect`` / ``bind`` /
    ``connect_ex`` and ``socket.create_connection`` are wrapped so connections
    may only target loopback addresses (``127.0.0.0/8``, ``::1``, ``localhost``;
    ``bind`` additionally allows ``INADDR_ANY``). Any non-loopback target raises
    ``OSError`` *immediately* — a port scanner therefore reports "closed"
    instead of hanging on a routed address, and no outbound internet connection
    can ever be established. ``ssl`` simply wraps an already-confined socket.

  * ``sqlite3.connect`` path confinement (it opens its database file at the C
    level, bypassing the Python ``open`` hooks above).

This is not a perfect boundary against a determined adversary (the untrusted
code shares the interpreter and could in principle crawl object graphs), but
the AST scan blocks the import/attribute routes that would reach the originals,
and the Windows Job Object (Layer 3) is the OS-level backstop. Together they
make every legitimately-answerable benchmark question runnable without giving
model code access to the host filesystem or the internet.
"""

from __future__ import annotations

import builtins
import errno
import io
import os
import sys
import threading

# ---------------------------------------------------------------------------
# Path confinement
# ---------------------------------------------------------------------------

# Env var carrying the sandbox root (set by sandbox._sandbox_env).
_ROOT_ENV = "LITEBENCH_SANDBOX_ROOT"


class _SandboxViolation(PermissionError):
    """Raised when confined I/O tries to leave its allowed directories."""


class _PathGate:
    """Decides whether a filesystem path may be read or written.

    Canonicalization mirrors the in-process sandbox (windows_sandbox.py):
    reject NT device/UNC namespaces and alternate data streams, then
    absolutize + ``realpath`` (collapses ``..``, symlinks, junctions), then
    case-fold and prefix-match against the allow-list with a path-separator
    boundary so ``C:\\root2`` never matches root ``C:\\root``.
    """

    def __init__(self, root: str):
        self._busy = threading.local()
        root_norm = self._normalize(root, create=True)
        self._write_roots = {root_norm}

        read_roots = {root_norm}
        for p in {
            sys.prefix,
            sys.exec_prefix,
            getattr(sys, "base_prefix", sys.prefix),
            os.path.dirname(sys.executable),
        }:
            if p:
                read_roots.add(self._normalize(p))
        try:
            import site

            for p in list(site.getsitepackages()) + [site.getusersitepackages()]:
                if p and os.path.isdir(p):
                    read_roots.add(self._normalize(p))
        except Exception:
            pass
        try:
            read_roots.add(self._normalize(os.getcwd()))
        except Exception:
            pass
        self._read_roots = read_roots

    @staticmethod
    def _normalize(path, create: bool = False) -> str:
        if hasattr(path, "__fspath__"):
            path = os.fspath(path)
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        s = str(path)
        if not s:
            raise _SandboxViolation("sandbox: empty path rejected")
        if s.startswith(("\\\\?\\", "\\\\.\\", "\\\\")):
            raise _SandboxViolation(f"sandbox: device/UNC path rejected: {s!r}")
        body = s[2:] if (len(s) >= 2 and s[1] == ":") else s
        if ":" in body:
            raise _SandboxViolation(f"sandbox: alternate data stream rejected: {s!r}")
        if create:
            os.makedirs(s, exist_ok=True)
        return os.path.normcase(os.path.realpath(os.path.abspath(s)))

    def _within(self, norm: str, roots) -> bool:
        for r in roots:
            if norm == r or norm.startswith(r + os.sep):
                return True
        return False

    def check(self, path, write: bool = False) -> None:
        # Re-entrancy guard: realpath may itself call the hooked os.stat while
        # we are resolving; let those inner calls through (same path anyway).
        if getattr(self._busy, "active", False):
            return
        self._busy.active = True
        try:
            n = self._normalize(path)
        except _SandboxViolation:
            raise
        except Exception as e:  # unresolvable path => fail closed
            raise _SandboxViolation(f"sandbox: cannot canonicalize {path!r}: {e}") from e
        finally:
            self._busy.active = False

        if write:
            if self._within(n, self._write_roots):
                return
            raise _SandboxViolation(f"sandbox: WRITE outside sandbox blocked: {path!r}")
        if self._within(n, self._read_roots) or self._within(n, self._write_roots):
            return
        raise _SandboxViolation(f"sandbox: READ outside allowed dirs blocked: {path!r}")


def _mode_is_write(mode) -> bool:
    try:
        m = str(mode).lower()
    except Exception:
        return True  # unparseable mode => fail closed
    return any(c in m for c in "wax+")


# ---------------------------------------------------------------------------
# Loopback-only networking
# ---------------------------------------------------------------------------

_LOOPBACK_NAMES = {"localhost", "ip6-localhost", "ip6-loopback"}


def _is_loopback_host(host: str) -> bool:
    """True iff every address ``host`` resolves to is a loopback address."""
    import ipaddress

    h = str(host).strip().lower()
    if not h or h in ("0.0.0.0", "::", "*"):
        # INADDR_ANY: only meaningful for bind(); treat as non-loopback so the
        # connect path rejects it, while bind() whitelists it explicitly.
        return False
    if h in _LOOPBACK_NAMES:
        return True
    # Strip an IPv6 zone id ("fe80::1%eth0") before parsing.
    try:
        return ipaddress.ip_address(h.split("%", 1)[0]).is_loopback
    except ValueError:
        pass
    # A hostname: resolve it and require ALL results to be loopback.
    try:
        infos = _ORIG_GETADDRINFO(h, None)
    except Exception:
        return False
    if not infos:
        return False
    for _fam, _typ, _proto, _canon, sockaddr in infos:
        addr = sockaddr[0]
        try:
            if not ipaddress.ip_address(addr.split("%", 1)[0]).is_loopback:
                return False
        except ValueError:
            return False
    return True


def _host_of(address) -> str:
    """Extract the host component from a socket address tuple/string."""
    if isinstance(address, tuple) and address:
        return str(address[0])
    return str(address)


# Filled in by install(); the pristine resolvers used by the loopback check.
_ORIG_GETADDRINFO = None


def install() -> None:
    """Install every confinement hook. Idempotent within one process."""
    global _ORIG_GETADDRINFO

    root = os.environ.get(_ROOT_ENV) or os.getcwd()
    gate = _PathGate(root)

    # ---- file I/O ---------------------------------------------------------
    orig_open = builtins.open

    def guarded_open(file, mode="r", *a, **k):
        if not isinstance(file, int):
            gate.check(file, write=_mode_is_write(mode))
        return orig_open(file, mode, *a, **k)

    builtins.open = guarded_open
    io.open = guarded_open

    orig_open_code = getattr(io, "open_code", None)
    if orig_open_code is not None:
        def guarded_open_code(path, *a, **k):
            gate.check(path, write=False)
            return orig_open_code(path, *a, **k)

        io.open_code = guarded_open_code

    orig_FileIO = io.FileIO

    class GuardedFileIO(orig_FileIO):  # type: ignore[misc, valid-type]
        def __init__(self, file, mode="r", *a, **k):
            if not isinstance(file, int):
                gate.check(file, write=_mode_is_write(mode))
            super().__init__(file, mode, *a, **k)

    io.FileIO = GuardedFileIO

    orig_os_open = os.open

    def guarded_os_open(path, flags, mode=0o777, *a, **k):
        if k.get("dir_fd") is not None:
            raise _SandboxViolation("sandbox: dir_fd-relative open blocked")
        wr = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC))
        gate.check(path, write=wr)
        return orig_os_open(path, flags, mode, *a, **k)

    os.open = guarded_os_open

    # ---- sqlite3 (opens its DB file at C level, bypassing open hooks) -----
    try:
        import sqlite3

        orig_sqlite_connect = sqlite3.connect

        def guarded_sqlite_connect(database, *a, **k):
            if not isinstance(database, str) or database not in (":memory:", ""):
                gate.check(database, write=True)
            return orig_sqlite_connect(database, *a, **k)

        sqlite3.connect = guarded_sqlite_connect
    except Exception:
        pass

    # ---- loopback-only sockets -------------------------------------------
    try:
        import socket

        _ORIG_GETADDRINFO = socket.getaddrinfo
        orig_socket_cls = socket.socket
        orig_create_connection = socket.create_connection

        class GuardedSocket(orig_socket_cls):  # type: ignore[misc, valid-type]
            def connect(self, address):
                if not _is_loopback_host(_host_of(address)):
                    raise OSError(
                        errno.ECONNREFUSED,
                        f"sandbox: non-loopback connect blocked: {_host_of(address)!r}",
                    )
                return super().connect(address)

            def connect_ex(self, address):
                if not _is_loopback_host(_host_of(address)):
                    return errno.ECONNREFUSED
                return super().connect_ex(address)

            def bind(self, address):
                host = _host_of(address)
                # bind may target loopback or INADDR_ANY (0.0.0.0 / :: / "").
                if not (_is_loopback_host(host) or str(host).strip().lower() in ("0.0.0.0", "::", "*", "")):
                    raise OSError(
                        errno.EACCES,
                        f"sandbox: non-loopback bind blocked: {host!r}",
                    )
                return super().bind(address)

        def guarded_create_connection(address, *a, **k):
            if not _is_loopback_host(_host_of(address)):
                raise OSError(
                    errno.ECONNREFUSED,
                    f"sandbox: non-loopback connect blocked: {_host_of(address)!r}",
                )
            return orig_create_connection(address, *a, **k)

        socket.socket = GuardedSocket
        socket.create_connection = guarded_create_connection
        # The SocketType alias some code imports; keep it consistent.
        if hasattr(socket, "SocketType"):
            socket.SocketType = GuardedSocket
    except Exception:
        pass

    # ---- os / shutil confinement -----------------------------------------
    # BigCodeBench forces `import os` / `import shutil` in ~56 task preambles,
    # so the model MUST emit them. Rather than block these modules (which would
    # make every such task guaranteed-fail), we allow the import but confine the
    # actions: process-spawning is blocked outright and every filesystem call is
    # restricted to the sandbox directory. This mirrors the proven policy in
    # windows_sandbox.py's in-process hook layer. Process creation that somehow
    # slips past the Python block is still stopped by the Windows Job Object.
    _KILL_NAMES = (
        "system", "popen", "popen2", "popen3", "popen4",
        "execl", "execle", "execlp", "execlpe",
        "execv", "execve", "execvp", "execvpe", "_execvpe",
        "spawnl", "spawnle", "spawnlp", "spawnlpe",
        "spawnv", "spawnve", "spawnvp", "spawnvpe", "_spawnvef",
        "posix_spawn", "posix_spawnp", "fork", "forkpty",
        "kill", "killpg", "abort", "_exit", "startfile",
    )
    _WRITE_ONE = ("remove", "unlink", "rmdir", "mkdir", "makedirs",
                  "chmod", "utime", "truncate", "removedirs")
    _READ_ONE = ("stat", "lstat", "listdir", "scandir", "readlink", "access")
    _TWO_PATH = ("rename", "renames", "replace")

    def _raiser(label):
        def blocked(*a, **k):
            raise _SandboxViolation(f"sandbox: {label} is blocked")
        return blocked

    def _path_guard(fn, write):
        def wrapper(path, *a, **k):
            if k.get("dir_fd") is not None:
                raise _SandboxViolation("sandbox: dir_fd-relative operations are blocked")
            if not isinstance(path, int):
                gate.check(path, write=write)
            return fn(path, *a, **k)
        return wrapper

    def _two_path_guard(fn):
        def wrapper(src, dst, *a, **k):
            if k.get("src_dir_fd") is not None or k.get("dst_dir_fd") is not None:
                raise _SandboxViolation("sandbox: dir_fd-relative operations are blocked")
            gate.check(src, write=True)
            gate.check(dst, write=True)
            return fn(src, dst, *a, **k)
        return wrapper

    def _access_guard(fn):
        def wrapper(path, mode, *a, **k):
            if k.get("dir_fd") is not None:
                raise _SandboxViolation("sandbox: dir_fd-relative operations are blocked")
            gate.check(path, write=bool(mode & os.W_OK))
            return fn(path, mode, *a, **k)
        return wrapper

    try:
        import nt  # type: ignore  # raw builtin module behind os on Windows
        _os_namespaces = [os, nt]
    except ImportError:
        _os_namespaces = [os]

    for mod in _os_namespaces:
        for name in _KILL_NAMES:
            if hasattr(mod, name):
                setattr(mod, name, _raiser(f"os.{name} (process control)"))
        for name in _WRITE_ONE:
            if hasattr(mod, name):
                setattr(mod, name, _path_guard(getattr(mod, name), True))
        for name in _READ_ONE:
            if hasattr(mod, name):
                if name == "access":
                    setattr(mod, name, _access_guard(getattr(mod, name)))
                else:
                    setattr(mod, name, _path_guard(getattr(mod, name), False))
        for name in _TWO_PATH:
            if hasattr(mod, name):
                setattr(mod, name, _two_path_guard(getattr(mod, name)))

    try:
        import shutil

        for name in ("copy", "copy2", "copyfile", "copymode", "copystat"):
            if hasattr(shutil, name):
                setattr(shutil, name, _two_path_guard(getattr(shutil, name)))
        if hasattr(shutil, "move"):
            shutil.move = _two_path_guard(shutil.move)
        if hasattr(shutil, "rmtree"):
            shutil.rmtree = _path_guard(shutil.rmtree, True)
        for name in ("make_archive", "unpack_archive"):
            if hasattr(shutil, name):
                setattr(shutil, name, _raiser(f"shutil.{name} (archive)"))
    except Exception:
        pass
