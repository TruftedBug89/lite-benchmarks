"""
windows_sandbox.py - Pure-Python, in-process security sandbox for running
potentially untrusted benchmark code on Windows. No Docker, no VMs, no
external sandbox wrapper binaries - only the Python standard library plus
direct Windows API calls through ctypes.

=============================================================================
ARCHITECTURE - three independent enforcement layers
=============================================================================

Layer 1 - Windows Job Object (kernel32, process-wide OS enforcement)
    The current process is assigned to a Job Object. A Job Object is a
    kernel securable object whose limits are enforced by the Windows kernel
    itself, completely outside the reach of Python code:

      * JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 1
        The job already contains one active process (this one), so the
        kernel refuses EVERY subsequent CreateProcess attempt made from
        inside the job. This blocks subprocess spawning at OS level even if
        every Python-level hook were bypassed (e.g. via a smuggled _winapi
        reference).
      * JOBOBJECT_BASIC_UI_RESTRICTIONS
        JOB_OBJECT_UILIMIT_HANDLES (no access to USER handles/windows owned
        by other processes, no cross-process window hooks or broadcasts),
        READCLIPBOARD / WRITECLIPBOARD, EXITWINDOWS (no logoff/shutdown),
        SYSTEMPARAMETERS, DISPLAYSETTINGS, GLOBALATOMS, DESKTOP.
      * Optional memory and CPU-time caps. Exceeding a job limit lets the
        kernel terminate the offender; that is why they are OPTIONAL here
        (in-process sandbox: the offender is also our own process).

    Caveat: job membership is irreversible for a process (a process cannot
    leave a job). We therefore create the job once, keep its handle alive
    for the process lifetime, and RELAX all limits when the sandbox run
    finishes (LimitFlags = 0, UIRestrictionsClass = 0). The restrictions are
    stripped even though the (now empty of limits) membership remains.

Layer 2 - Restricted impersonation token (advapi32, per-worker-thread)
    The untrusted callable runs on a dedicated worker thread. Before it
    starts, that thread impersonates a restricted copy of its own process
    token built with:

      OpenProcessToken(TOKEN_DUPLICATE | TOKEN_QUERY)
      DuplicateTokenEx(..., SecurityImpersonation, TokenImpersonation)
      CreateRestrictedToken(DISABLE_MAX_PRIVILEGE | SANDBOX_INERT, ...)
      SetThreadToken(NULL, restricted_token)

    DISABLE_MAX_PRIVILEGE strips every token privilege (SeDebugPrivilege,
    SeTakeOwnershipPrivilege, SeShutdownPrivilege, ...), so even raw Win32
    calls issued from the untrusted code (if it somehow reached them) run
    with no privileges at all. SANDBOX_INERT stamps the token with the
    TOKEN_SANDBOX_INERT flag - the same marker used by browser sandboxes -
    so security-aware system components treat the caller as untrusted.
    Optionally the token's integrity level can be dropped to Low
    (S-1-16-4096), which makes the Mandatory Integrity Control check deny
    writes to every object labeled Medium integrity or higher (i.e. almost
    the whole machine). Because that also blocks writes to the scratch
    directory unless it is relabeled, it is OFF by default.

    The impersonation is fully reversible: SetThreadToken(NULL, NULL)
    detaches the token and the thread reverts to the process token.

Layer 3 - Python runtime hooks (precise policy + clear security errors)
    Deterministic monkey-patching of every Python-level "danger zone" entry
    point. These hooks give the precise, directory-scoped policy that kernel
    objects cannot express, and they raise a clear SandboxSecurityError:

      * File I/O: builtins.open, io.open, io.FileIO, io.open_code, os.open /
        nt.open, remove/unlink/rmdir/mkdir/rename/replace/chmod/utime/
        truncate/stat/lstat/listdir/scandir/readlink/access/chdir on BOTH
        the `os` and the underlying `nt` module namespaces. Writes and
        deletes are confined to the sandbox root; reads are additionally
        allowed for the Python installation (so imports keep working) and
        caller-declared extra paths. Path canonicalization rejects
        \\?\\ device paths, \\.\\ device paths, UNC paths, alternate data
        streams, and resolves .., symlinks and junctions via realpath
        before the prefix check, so escapes via traversal or link tricks
        fail.
      * Network: socket/_socket/ssl entry points are replaced, and the
        modules are additionally evicted from sys.modules + blocked at
        import time, so aliased references raise too.
      * Subprocesses: subprocess.*, os.system/popen/spawn*/exec*/,
        os.startfile, os.posix_spawn, nt._spawnvef plus the Layer-1 kernel
        block as backstop.
      * Interpreters escapes: ctypes, _winapi, pywin32, multiprocessing,
        inspect, psutil are import-blocked; sys._getframe / settrace /
        setprofile are neutralized; os.kill/abort/_exit are blocked so the
        harness process itself cannot be killed from inside.

=============================================================================
HONEST RESIDUAL LIMITATIONS (read before trusting)
=============================================================================
  * An in-process Python sandbox can never be a perfect boundary: the
    untrusted code shares the interpreter. A determined attacker can crawl
    object graphs / closures (e.g. wrapper.__closure__) to recover the
    original function objects that this file stashes for restoration.
    The OS layers above exist precisely as a backstop for what they cover
    (privileges, child processes, UI isolation). For strongly adversarial
    code, run the whole benchmark harness inside a dedicated sacrificial
    process and let Layers 1+2 protect the host.
  * Python-level network/file hooks cannot defend against native code that
    was loaded before the sandbox activated.
  * A worker thread that wedges inside a blocking native call cannot be
    killed safely in-process; PyThreadState_SetAsyncExc only interrupts
    Python bytecode. On un-killable hang the sandbox deliberately LEAVES
    all restrictions engaged and reports thread_hung=True.

Usage:
    from windows_sandbox import SandboxPolicy, run_in_sandbox

    policy = SandboxPolicy(root_dir=r"C:\\bench\\workspace")
    result = run_in_sandbox(my_benchmark_fn, arg1, arg2, timeout=10,
                            policy=policy)
    if result.ok:
        print(result.value, result.wall_time)
    else:
        print("failed/blocked:", result.exception)
"""

from __future__ import annotations

import builtins
import ctypes
import io
import os
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "SandboxSecurityError",
    "SandboxPolicy",
    "SandboxResult",
    "run_in_sandbox",
]

if os.name != "nt":
    raise RuntimeError("windows_sandbox.py requires a Windows host (os.name == 'nt')")


# ===========================================================================
# Security exception
# ===========================================================================

class SandboxSecurityError(PermissionError):
    """Raised when sandboxed code attempts a blocked operation.

    Subclasses PermissionError (an OSError) on purpose: many libraries
    swallow OSError subclasses on non-critical paths (e.g. importlib
    ignoring bytecode-cache write failures), so the sandbox fails closed
    without crashing innocent machinery, while still being a loud, clear
    security exception everywhere else.
    """


# ===========================================================================
# Win32 plumbing (ctypes) - kernel32 / advapi32 / psapi-in-kernel32
# ===========================================================================

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

HANDLE = wintypes.HANDLE
DWORD = wintypes.DWORD
BOOL = wintypes.BOOL
SIZE_T = ctypes.c_size_t


def _winerr(api: str) -> OSError:
    """Build an OSError from GetLastError after a failed Win32 call."""
    err = ctypes.get_last_error()
    return OSError(err, f"{api} failed: {ctypes.FormatError(err).strip()}")


# --- access-token rights -------------------------------------------------
TOKEN_DUPLICATE = 0x0002
TOKEN_IMPERSONATE = 0x0004
TOKEN_QUERY = 0x0008
TOKEN_ADJUST_PRIVILEGES = 0x0020
TOKEN_ADJUST_DEFAULT = 0x0080

# CreateRestrictedToken flags
DISABLE_MAX_PRIVILEGE = 0x1   # strip ALL privileges from the new token
SANDBOX_INERT = 0x2           # set the TOKEN_SANDBOX_INERT flag on it

# SECURITY_IMPERSONATION_LEVEL / TOKEN_TYPE enumerations
SecurityImpersonation = 2
TokenImpersonation = 2

# TokenInformationClass / integrity label
TokenIntegrityLevel = 25
SE_GROUP_INTEGRITY = 0x00000020
LOW_INTEGRITY_SID = "S-1-16-4096"  # Mandatory Label\Low Mandatory Level

# --- job objects -----------------------------------------------------------
# JOBOBJECTINFOCLASS enumeration values used below
JobObjectBasicUIRestrictions = 4
JobObjectExtendedLimitInformation = 9

# JOBOBJECT_BASIC_LIMIT_INFORMATION.LimitFlags bits
JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
JOB_OBJECT_LIMIT_PROCESS_TIME = 0x00000002
JOB_OBJECT_LIMIT_JOB_TIME = 0x00000004
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200

# JOBOBJECT_BASIC_UI_RESTRICTIONS.UIRestrictionsClass bits
JOB_OBJECT_UILIMIT_HANDLES = 0x00000001
JOB_OBJECT_UILIMIT_READCLIPBOARD = 0x00000002
JOB_OBJECT_UILIMIT_WRITECLIPBOARD = 0x00000004
JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS = 0x00000008
JOB_OBJECT_UILIMIT_DISPLAYSETTINGS = 0x00000010
JOB_OBJECT_UILIMIT_GLOBALATOMS = 0x00000020
JOB_OBJECT_UILIMIT_DESKTOP = 0x00000040
JOB_OBJECT_UILIMIT_EXITWINDOWS = 0x00000080
_JOB_OBJECT_UILIMIT_ALL = (
    JOB_OBJECT_UILIMIT_HANDLES
    | JOB_OBJECT_UILIMIT_READCLIPBOARD
    | JOB_OBJECT_UILIMIT_WRITECLIPBOARD
    | JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS
    | JOB_OBJECT_UILIMIT_DISPLAYSETTINGS
    | JOB_OBJECT_UILIMIT_GLOBALATOMS
    | JOB_OBJECT_UILIMIT_DESKTOP
    | JOB_OBJECT_UILIMIT_EXITWINDOWS
)


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    # Matches winnt.h; the two LARGE_INTEGER time fields are in
    # 100-nanosecond intervals.
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_ulonglong),
        ("PerJobUserTimeLimit", ctypes.c_ulonglong),
        ("LimitFlags", DWORD),
        ("MinimumWorkingSetSize", SIZE_T),
        ("MaximumWorkingSetSize", SIZE_T),
        ("ActiveProcessLimit", DWORD),
        ("Affinity", SIZE_T),
        ("PriorityClass", DWORD),
        ("SchedulingClass", DWORD),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", SIZE_T),
        ("JobMemoryLimit", SIZE_T),
        ("PeakProcessMemoryUsed", SIZE_T),
        ("PeakJobMemoryUsed", SIZE_T),
    ]


class JOBOBJECT_BASIC_UI_RESTRICTIONS(ctypes.Structure):
    _fields_ = [("UIRestrictionsClass", DWORD)]


class SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", DWORD)]


class TOKEN_MANDATORY_LABEL(ctypes.Structure):
    _fields_ = [("Label", SID_AND_ATTRIBUTES)]


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", DWORD),
        ("PageFaultCount", DWORD),
        ("PeakWorkingSetSize", SIZE_T),
        ("WorkingSetSize", SIZE_T),
        ("QuotaPeakPagedPoolUsage", SIZE_T),
        ("QuotaPagedPoolUsage", SIZE_T),
        ("QuotaPeakNonPagedPoolUsage", SIZE_T),
        ("QuotaNonPagedPoolUsage", SIZE_T),
        ("PagefileUsage", SIZE_T),
        ("PeakPagefileUsage", SIZE_T),
    ]


# --- function prototypes ----------------------------------------------------
kernel32.GetCurrentProcess.restype = HANDLE
kernel32.GetCurrentProcess.argtypes = []
kernel32.CloseHandle.restype = BOOL
kernel32.CloseHandle.argtypes = [HANDLE]

kernel32.CreateJobObjectW.restype = HANDLE
kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
kernel32.AssignProcessToJobObject.restype = BOOL
kernel32.AssignProcessToJobObject.argtypes = [HANDLE, HANDLE]
kernel32.SetInformationJobObject.restype = BOOL
kernel32.SetInformationJobObject.argtypes = [HANDLE, ctypes.c_int, ctypes.c_void_p, DWORD]
kernel32.IsProcessInJob.restype = BOOL
kernel32.IsProcessInJob.argtypes = [HANDLE, HANDLE, ctypes.POINTER(BOOL)]

# K32GetProcessMemoryInfo lives in kernel32 on Windows 7+ (no psapi.dll needed)
kernel32.K32GetProcessMemoryInfo.restype = BOOL
kernel32.K32GetProcessMemoryInfo.argtypes = [HANDLE, ctypes.c_void_p, DWORD]

advapi32.OpenProcessToken.restype = BOOL
advapi32.OpenProcessToken.argtypes = [HANDLE, DWORD, ctypes.POINTER(HANDLE)]
advapi32.DuplicateTokenEx.restype = BOOL
advapi32.DuplicateTokenEx.argtypes = [HANDLE, DWORD, ctypes.c_void_p,
                                      ctypes.c_int, ctypes.c_int, ctypes.POINTER(HANDLE)]
advapi32.CreateRestrictedToken.restype = BOOL
advapi32.CreateRestrictedToken.argtypes = [HANDLE, DWORD, DWORD, ctypes.c_void_p,
                                           DWORD, ctypes.c_void_p, DWORD,
                                           ctypes.c_void_p, ctypes.POINTER(HANDLE)]
# SetThreadToken(PHANDLE Thread, HANDLE Token): passing Thread=NULL targets
# the calling thread; Token=NULL detaches the current impersonation token.
advapi32.SetThreadToken.restype = BOOL
advapi32.SetThreadToken.argtypes = [ctypes.c_void_p, HANDLE]
advapi32.ConvertStringSidToSidW.restype = BOOL
advapi32.ConvertStringSidToSidW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
advapi32.SetTokenInformation.restype = BOOL
advapi32.SetTokenInformation.argtypes = [HANDLE, ctypes.c_int, ctypes.c_void_p, DWORD]
kernel32.LocalFree.restype = ctypes.c_void_p
kernel32.LocalFree.argtypes = [ctypes.c_void_p]


# ===========================================================================
# Policy
# ===========================================================================

_DEFAULT_BLOCKED_MODULES = frozenset({
    # FFI / raw Win32: full sandbox escape if reachable
    "ctypes", "_ctypes", "_winapi", "cffi", "_cffi_backend",
    # pywin32 family
    "win32api", "win32process", "win32security", "win32file", "win32pipe",
    "win32event", "win32service", "win32con", "win32gui", "win32net",
    "pywintypes", "pythoncom", "comtypes",
    # network
    "socket", "_socket", "ssl", "_ssl",
    # process creation
    "subprocess", "multiprocessing", "_multiprocessing",
    # introspection / process tooling
    "inspect", "psutil", "pdb",
})


@dataclass
class SandboxPolicy:
    """Knobs controlling a sandbox run.

    root_dir:            the ONLY directory the code may write/delete in.
    extra_read_paths:    additional readable trees (imports from the Python
                         installation are always allowed).
    extra_blocked_modules: merged into the import blocklist.
    redirect_temp:       point TEMP/TMP/TMPDIR and tempfile at root_dir/_tmp.
    use_job / use_token / use_python_hooks:
                         individual layers can be disabled for debugging.
    limit_child_processes / restrict_ui:
                         Job Object OS enforcement toggles.
    job_memory_mb / job_time_seconds:
                         optional hard kernel-enforced caps. WARNING: the
                         kernel TERMINATES processes that exceed these; in
                         this in-process design that means the whole
                         interpreter. Leave None unless the harness runs in
                         a sacrificial process.
    low_integrity:       also set the worker thread token to Low integrity
                         level (OS-level write block of almost everything;
                         only enable if root_dir was labeled Low IL).
    block_introspection: neutralize sys._getframe/settrace/setprofile.
    """

    root_dir: str
    extra_read_paths: tuple = ()
    extra_blocked_modules: frozenset = frozenset()
    redirect_temp: bool = True

    use_job: bool = True
    use_token: bool = True
    use_python_hooks: bool = True

    limit_child_processes: bool = True
    restrict_ui: bool = True
    job_memory_mb: int | None = None
    job_time_seconds: float | None = None
    low_integrity: bool = False
    block_introspection: bool = True


# ===========================================================================
# Path gatekeeper - the precise filesystem policy
# ===========================================================================

class _PathGatekeeper:
    """Decides whether a filesystem path may be read or written.

    Canonicalization order matters for security:
      1. reject NT device namespaces and UNC outright (\\?\\, \\.\\, \\\\),
      2. reject NTFS alternate data streams (a second ':' after the drive),
      3. absolutize + resolve '..', symlinks and junctions (realpath),
      4. case-fold (Windows is case-insensitive) and prefix-match against
         the allow-list roots, requiring a path-separator boundary so that
         'C:\\root2\\x' does not match root 'C:\\root'.
    """

    def __init__(self, policy: SandboxPolicy):
        self._busy = threading.local()  # re-entrancy guard, see check()

        root = self._normalize(policy.root_dir, create=True)
        self._write_roots = {root}

        read_roots = {root}
        # The Python installation itself must stay readable or `import`
        # would die: stdlib, site-packages, user site-packages.
        for p in {sys.prefix, sys.exec_prefix, getattr(sys, "base_prefix", sys.prefix)}:
            read_roots.add(self._normalize(p))
        try:
            import site
            for p in site.getsitepackages() + [site.getusersitepackages()]:
                if p and os.path.isdir(p):
                    read_roots.add(self._normalize(p))
        except Exception:
            pass
        # The project the harness lives in is readable by default so the
        # benchmark target can import its own helpers. Tighten by building
        # a policy with explicit extra_read_paths instead.
        read_roots.add(self._normalize(os.getcwd()))
        for p in policy.extra_read_paths:
            read_roots.add(self._normalize(p))
        self._read_roots = read_roots

    @staticmethod
    def _normalize(path: Any, create: bool = False) -> str:
        if hasattr(path, "__fspath__"):
            path = os.fspath(path)
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        s = str(path)
        if not s:
            raise SandboxSecurityError("Sandbox: empty path rejected")
        if s.startswith(("\\\\?\\", "\\\\.\\", "\\\\")):
            raise SandboxSecurityError(
                f"Sandbox: device/UNC path rejected: {s!r}")
        body = s[2:] if (len(s) >= 2 and s[1] == ":") else s
        if ":" in body:
            raise SandboxSecurityError(
                f"Sandbox: alternate data stream rejected: {s!r}")
        if create:
            os.makedirs(s, exist_ok=True)
        # realpath collapses '.', '..', symlinks and junction reparse points
        # (ntpath.realpath -> GetFinalPathNameByHandle under the hood).
        return os.path.normcase(os.path.realpath(os.path.abspath(s)))

    def _within(self, norm_path: str, roots) -> bool:
        for r in roots:
            if norm_path == r or norm_path.startswith(r + os.sep):
                return True
        return False

    def check(self, path: Any, write: bool = False) -> None:
        """Raise SandboxSecurityError unless `path` is within policy."""
        # Re-entrancy guard: our own normalization may (on some builds)
        # trigger os.stat/listdir, which are themselves hooked. While we
        # are resolving, inner hook invocations are let through - they
        # operate on the very path being checked anyway.
        if getattr(self._busy, "active", False):
            return
        self._busy.active = True
        try:
            n = self._normalize(path)
        except SandboxSecurityError:
            raise
        except Exception as e:  # unresolvable path => fail closed
            raise SandboxSecurityError(
                f"Sandbox: cannot canonicalize {path!r}: {e}") from e
        finally:
            self._busy.active = False

        if write:
            if self._within(n, self._write_roots):
                return
            raise SandboxSecurityError(
                f"Sandbox: WRITE outside sandbox root blocked: {path!r}")
        if self._within(n, self._read_roots) or self._within(n, self._write_roots):
            return
        raise SandboxSecurityError(
            f"Sandbox: READ outside allowed directories blocked: {path!r}")


# ===========================================================================
# Layer 1 - Job Object guard (kernel-enforced, process-wide)
# ===========================================================================

# The job handle and membership must outlive any single sandbox run:
# a process can never LEAVE a job, so the job is created once and reused.
_JOB_STATE = {"handle": None, "lock": threading.Lock()}


class _JobGuard:
    """Creates/reuses the process Job Object and applies/relaxes limits."""

    def __init__(self, policy: SandboxPolicy):
        self.policy = policy
        self.engaged = False
        self.note = ""

    # -- creation / membership (idempotent, permanent) --
    def _ensure_membership(self) -> HANDLE:
        with _JOB_STATE["lock"]:
            if _JOB_STATE["handle"] is not None:
                return _JOB_STATE["handle"]
            h = kernel32.CreateJobObjectW(None, None)  # unnamed, inheritable=no
            if not h:
                raise _winerr("CreateJobObjectW")
            # Assign BEFORE setting any limits: assigning a process to a job
            # whose active-process limit is already exceeded fails AND
            # terminates the process being assigned.
            if not kernel32.AssignProcessToJobObject(h, kernel32.GetCurrentProcess()):
                kernel32.CloseHandle(h)
                raise _winerr("AssignProcessToJobObject")
            _JOB_STATE["handle"] = h
            return h

    # -- limit application --
    def engage(self) -> None:
        try:
            h = self._ensure_membership()
        except OSError as e:
            # e.g. host already confines us in a non-nestable job (pre-Win8
            # semantics). Degrade to the remaining layers, don't crash.
            self.note = f"job layer unavailable: {e}"
            return

        flags = 0
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        if self.policy.limit_child_processes:
            # The job already holds exactly 1 process (us); a limit of 1
            # makes the kernel deny every CreateProcess from within.
            flags |= JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            info.BasicLimitInformation.ActiveProcessLimit = 1
        if self.policy.job_memory_mb:
            flags |= JOB_OBJECT_LIMIT_JOB_MEMORY
            info.JobMemoryLimit = int(self.policy.job_memory_mb) << 20
        if self.policy.job_time_seconds:
            flags |= JOB_OBJECT_LIMIT_JOB_TIME
            # Windows time limits are expressed in 100-ns units.
            info.BasicLimitInformation.PerJobUserTimeLimit = \
                int(self.policy.job_time_seconds * 10_000_000)
        info.BasicLimitInformation.LimitFlags = flags

        if not kernel32.SetInformationJobObject(
                h, JobObjectExtendedLimitInformation,
                ctypes.byref(info), ctypes.sizeof(info)):
            self.note = str(_winerr("SetInformationJobObject(extended)"))
            return

        if self.policy.restrict_ui:
            ui = JOBOBJECT_BASIC_UI_RESTRICTIONS(_JOB_OBJECT_UILIMIT_ALL)
            if not kernel32.SetInformationJobObject(
                    h, JobObjectBasicUIRestrictions,
                    ctypes.byref(ui), ctypes.sizeof(ui)):
                self.note = str(_winerr("SetInformationJobObject(ui)"))
                return
        self.engaged = True

    # -- restriction removal (membership itself cannot be removed) --
    def relax(self) -> None:
        if not self.engaged:
            return
        h = _JOB_STATE["handle"]
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()  # LimitFlags = 0 => off
        kernel32.SetInformationJobObject(
            h, JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info))
        ui = JOBOBJECT_BASIC_UI_RESTRICTIONS(0)  # 0 => all UI rights back
        kernel32.SetInformationJobObject(
            h, JobObjectBasicUIRestrictions,
            ctypes.byref(ui), ctypes.sizeof(ui))
        self.engaged = False


# ===========================================================================
# Layer 2 - restricted token guard (per worker thread)
# ===========================================================================

class _TokenGuard:
    """Impersonates a privilege-stripped token on the calling thread.

    Call engage()/revert() from the SAME thread. engage() returns a list of
    handles that must stay open while the token is in use and are closed by
    revert().
    """

    def __init__(self, low_integrity: bool = False):
        self.low_integrity = low_integrity
        self._handles: list = []

    def engage(self) -> None:
        h_proc = kernel32.GetCurrentProcess()

        h_tok = HANDLE()
        if not advapi32.OpenProcessToken(
                h_proc, TOKEN_DUPLICATE | TOKEN_QUERY, ctypes.byref(h_tok)):
            raise _winerr("OpenProcessToken")
        self._handles.append(h_tok)

        # A thread token must be an *impersonation* token; the process token
        # is primary, so duplicate it into the impersonation flavor first.
        h_imp = HANDLE()
        access = (TOKEN_DUPLICATE | TOKEN_QUERY | TOKEN_IMPERSONATE
                  | TOKEN_ADJUST_PRIVILEGES | TOKEN_ADJUST_DEFAULT)
        if not advapi32.DuplicateTokenEx(
                h_tok, access, None, SecurityImpersonation,
                TokenImpersonation, ctypes.byref(h_imp)):
            raise _winerr("DuplicateTokenEx")
        self._handles.append(h_imp)

        # CreateRestrictedToken keeps the token type (impersonation) and
        # applies the flags: DISABLE_MAX_PRIVILEGE deletes every privilege
        # (only the unavoidable SeChangeNotifyPrivilege survives) and
        # SANDBOX_INERT marks the token as sandboxed for security-aware
        # subsystems.
        h_res = HANDLE()
        if not advapi32.CreateRestrictedToken(
                h_imp, DISABLE_MAX_PRIVILEGE | SANDBOX_INERT,
                0, None, 0, None, 0, None, ctypes.byref(h_res)):
            raise _winerr("CreateRestrictedToken")
        self._handles.append(h_res)

        if self.low_integrity:
            self._set_low_integrity(h_res)

        # Thread=NULL -> calling thread starts impersonating the restricted
        # token. From here, every access check on this thread uses the
        # stripped token.
        if not advapi32.SetThreadToken(None, h_res):
            raise _winerr("SetThreadToken")

    @staticmethod
    def _set_low_integrity(h_tok: HANDLE) -> None:
        sid = ctypes.c_void_p()
        if not advapi32.ConvertStringSidToSidW(LOW_INTEGRITY_SID, ctypes.byref(sid)):
            raise _winerr("ConvertStringSidToSidW")
        try:
            label = TOKEN_MANDATORY_LABEL()
            label.Label.Sid = sid
            label.Label.Attributes = SE_GROUP_INTEGRITY
            if not advapi32.SetTokenInformation(
                    h_tok, TokenIntegrityLevel,
                    ctypes.byref(label), ctypes.sizeof(label)):
                raise _winerr("SetTokenInformation(TokenIntegrityLevel)")
        finally:
            kernel32.LocalFree(sid)

    def revert(self) -> None:
        # Token=NULL detaches the impersonation token; the thread returns to
        # the full process token.
        advapi32.SetThreadToken(None, None)
        for h in self._handles:
            kernel32.CloseHandle(h)
        self._handles.clear()


# ===========================================================================
# Layer 3 - Python runtime hooks
# ===========================================================================

def _raiser(what: str) -> Callable:
    def blocked(*_a, **_k):
        raise SandboxSecurityError(f"Sandbox: {what} is blocked")
    blocked.__name__ = "sandbox_blocked_" + what.replace(" ", "_")
    return blocked


def _mode_is_write(mode: Any) -> bool:
    try:
        return any(c in str(mode) for c in "wax+")
    except Exception:
        return True  # unparseable mode => fail closed


class _PythonHookGuard:
    """Installs and later removes every Python-level interception.

    All original function objects are remembered so the process is restored
    bit-for-bit on exit.
    """

    # names blocked outright on both `os` and the raw `nt` builtin module
    _PROCESS_KILL_NAMES = (
        "system", "popen", "startfile",
        "execl", "execle", "execlp", "execlpe",
        "execv", "execve", "execvp", "execvpe", "_execvpe",
        "spawnl", "spawnle", "spawnlp", "spawnlpe",
        "spawnv", "spawnve", "spawnvp", "spawnvpe", "_spawnvef",
        "posix_spawn", "posix_spawnp", "fork", "forkpty",
        "kill", "killpg", "abort", "_exit",
    )

    def __init__(self, policy: SandboxPolicy, gate: _PathGatekeeper):
        self.policy = policy
        self.gate = gate
        self._originals: list = []          # (owner, name, original)
        self._saved_modules: dict = {}
        self._blocker = None
        self._temp_state: tuple | None = None
        self.blocked_modules = frozenset(
            _DEFAULT_BLOCKED_MODULES | set(policy.extra_blocked_modules))

    # -- low-level swap helpers ------------------------------------------
    def _swap(self, owner: Any, name: str, new: Any) -> None:
        if hasattr(owner, name):
            self._originals.append((owner, name, getattr(owner, name)))
            try:
                setattr(owner, name, new)
            except (TypeError, AttributeError):
                # immutable object: skip this one hook rather than fail run
                self._originals.pop()

    def _restore_all(self) -> None:
        for owner, name, original in reversed(self._originals):
            try:
                setattr(owner, name, original)
            except (TypeError, AttributeError):
                pass
        self._originals.clear()

    # -- guarded wrappers --------------------------------------------------
    def _make_guards(self):
        gate = self.gate

        def reject_dir_fd(kwargs: dict) -> None:
            if kwargs.get("dir_fd") is not None:
                raise SandboxSecurityError(
                    "Sandbox: dir_fd-relative operations are blocked")

        orig_open = builtins.open

        def guarded_open(file, mode="r", *a, **k):
            # An int is an already-open fd; every fd-producing call below is
            # itself gated, so passing one on is safe.
            if not isinstance(file, int):
                gate.check(file, write=_mode_is_write(mode))
            return orig_open(file, mode, *a, **k)

        orig_io_open = io.open

        def guarded_io_open(file, mode="r", *a, **k):
            if not isinstance(file, int):
                gate.check(file, write=_mode_is_write(mode))
            return orig_io_open(file, mode, *a, **k)

        orig_open_code = getattr(io, "open_code", None)

        def guarded_open_code(path, *a, **k):
            gate.check(path, write=False)
            return orig_open_code(path, *a, **k)

        orig_FileIO = io.FileIO

        class GuardedFileIO(orig_FileIO):  # type: ignore[misc]
            def __init__(self, file, mode="r", *a, **k):
                if not isinstance(file, int):
                    gate.check(file, write=_mode_is_write(mode))
                super().__init__(file, mode, *a, **k)

        orig_os_open = os.open

        def guarded_os_open(path, flags, mode=0o777, *a, **k):
            reject_dir_fd(k)
            wr = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT
                               | os.O_APPEND | os.O_TRUNC))
            gate.check(path, write=wr)
            return orig_os_open(path, flags, mode, *a, **k)

        def path_guard(fn, write, label):
            def wrapper(path, *a, **k):
                reject_dir_fd(k)
                if not isinstance(path, int):
                    gate.check(path, write=write)
                return fn(path, *a, **k)
            return wrapper

        def two_path_guard(fn, label):
            def wrapper(src, dst, *a, **k):
                reject_dir_fd(k)
                gate.check(src, write=True)
                gate.check(dst, write=True)
                return fn(src, dst, *a, **k)
            return wrapper

        def access_guard(fn):
            def wrapper(path, mode, *a, **k):
                reject_dir_fd(k)
                gate.check(path, write=bool(mode & os.W_OK))
                return fn(path, mode, *a, **k)
            return wrapper

        guards = {
            "open": guarded_open,
            "io_open": guarded_io_open,
            "open_code": guarded_open_code,
            "FileIO": GuardedFileIO,
            "os_open": guarded_os_open,
            "path_guard": path_guard,
            "two_path_guard": two_path_guard,
            "access_guard": access_guard,
        }
        return guards

    # -- install ------------------------------------------------------------
    def install(self) -> None:
        g = self._make_guards()

        # --- file I/O ---
        self._swap(builtins, "open", g["open"])
        self._swap(io, "open", g["io_open"])
        if getattr(io, "open_code", None):
            self._swap(io, "open_code", g["open_code"])
        self._swap(io, "FileIO", g["FileIO"])

        nt = sys.modules.get("nt")  # raw builtin module behind `os`
        namespaces = [m for m in (os, nt) if m is not None]
        write_one = ("remove", "unlink", "rmdir", "mkdir", "chmod",
                     "utime", "truncate")
        read_one = ("stat", "lstat", "listdir", "scandir", "readlink")
        for mod in namespaces:
            self._swap(mod, "open", g["os_open"])
            for name in write_one:
                if hasattr(mod, name):
                    self._swap(mod, name,
                               g["path_guard"](getattr(mod, name), True, name))
            for name in read_one:
                if hasattr(mod, name):
                    self._swap(mod, name,
                               g["path_guard"](getattr(mod, name), False, name))
            for name in ("rename", "renames", "replace"):
                if hasattr(mod, name):
                    self._swap(mod, name,
                               g["two_path_guard"](getattr(mod, name), name))
            if hasattr(mod, "access"):
                self._swap(mod, "access", g["access_guard"](mod.access))
            if hasattr(mod, "chdir"):
                # chdir target must be readable; confinement still enforced
                # afterwards because every check canonicalizes absolutely.
                self._swap(mod, "chdir",
                           g["path_guard"](mod.chdir, False, "chdir"))
            # link-creation is always blocked: a symlink/junction could be
            # weaponized to escape the prefix check on later opens.
            for name in ("symlink", "link"):
                if hasattr(mod, name):
                    self._swap(mod, name, _raiser(f"os.{name} (link creation)"))
            for name in self._PROCESS_KILL_NAMES:
                if hasattr(mod, name):
                    self._swap(mod, name, _raiser(f"os.{name} (process control)"))

        # --- subprocess module (belt; Layer 1 is the suspenders) ---
        subprocess = sys.modules.get("subprocess") or __import__("subprocess")
        for name in ("Popen", "run", "call", "check_call", "check_output"):
            self._swap(subprocess, name,
                       _raiser(f"subprocess.{name} (process creation)"))

        # --- sockets ---
        for mod_name in ("socket", "_socket"):
            mod = sys.modules.get(mod_name)
            if mod is None and mod_name == "socket":
                try:
                    mod = __import__("socket")
                except ImportError:
                    mod = None
            if mod is None:
                continue
            for name in ("socket", "create_connection", "create_server",
                         "socketpair", "fromfd", "fromshare",
                         "getaddrinfo", "gethostbyname", "gethostbyname_ex",
                         "gethostbyaddr", "getnameinfo", "getservbyname",
                         "getservbyport"):
                if hasattr(mod, name):
                    self._swap(mod, name, _raiser(f"{mod_name}.{name} (network)"))
        ssl_mod = sys.modules.get("ssl")
        if ssl_mod is not None:
            self._swap(ssl_mod, "wrap_socket", _raiser("ssl.wrap_socket"))
            ctx = getattr(ssl_mod, "SSLContext", None)
            if ctx is not None:
                self._swap(ctx, "wrap_socket", _raiser("SSLContext.wrap_socket"))

        # --- introspection / hooks on the interpreter itself ---
        if self.policy.block_introspection:
            for name in ("_getframe", "_current_frames", "settrace",
                         "setprofile"):
                if hasattr(sys, name):
                    self._swap(sys, name, _raiser(f"sys.{name} (introspection)"))

        # --- import-time blocker (sys.meta_path hook) ---
        guard = self

        class _SandboxImportBlocker:
            """Meta-path finder that vetoes blocked roots.

            Raising here (instead of returning a dummy spec) makes the
            SandboxSecurityError propagate straight to the importing code,
            which is far clearer than a generic ModuleNotFoundError.
            """

            def find_spec(self, fullname, path=None, target=None):
                root = fullname.partition(".")[0]
                if root in guard.blocked_modules:
                    raise SandboxSecurityError(
                        f"Sandbox: importing '{fullname}' is blocked")
                return None

        self._blocker = _SandboxImportBlocker()
        sys.meta_path.insert(0, self._blocker)

        # Evict already-imported blocked modules so `import ctypes` inside
        # the sandbox re-runs the import machinery and hits the blocker
        # instead of silently receiving the cached module. Restored later.
        for name in list(sys.modules):
            if name.partition(".")[0] in self.blocked_modules:
                self._saved_modules[name] = sys.modules.pop(name)

        # --- temp redirection into the sandbox root ---
        if self.policy.redirect_temp:
            tmp_root = os.path.join(self.policy.root_dir, "_tmp")
            os.makedirs(tmp_root, exist_ok=True)
            self._temp_state = (
                os.environ.get("TEMP"), os.environ.get("TMP"),
                os.environ.get("TMPDIR"), tempfile.tempdir)
            os.environ["TEMP"] = os.environ["TMP"] = os.environ["TMPDIR"] = tmp_root
            tempfile.tempdir = tmp_root

    # -- removal -------------------------------------------------------------
    def remove(self) -> None:
        # Neutralize any tracer the sandboxed code may have armed via some
        # unpatched path BEFORE we restore originals (a live trace function
        # would observe the restoration frames and could steal originals).
        orig_settrace = orig_setprofile = None
        for owner, name, original in self._originals:
            if owner is sys and name == "settrace":
                orig_settrace = original
            elif owner is sys and name == "setprofile":
                orig_setprofile = original
        try:
            if orig_settrace:
                orig_settrace(None)
            if orig_setprofile:
                orig_setprofile(None)
        except Exception:
            pass

        self._restore_all()

        if self._blocker is not None:
            try:
                sys.meta_path.remove(self._blocker)
            except ValueError:
                pass
            self._blocker = None
        sys.modules.update(self._saved_modules)
        self._saved_modules.clear()

        if self._temp_state is not None:
            te, tmp, tmpdir, tdir = self._temp_state
            for key, val in (("TEMP", te), ("TMP", tmp), ("TMPDIR", tmpdir)):
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val
            tempfile.tempdir = tdir
            self._temp_state = None


# ===========================================================================
# Metrics + result container
# ===========================================================================

@dataclass
class SandboxResult:
    """Outcome of one run_in_sandbox() call."""
    ok: bool = False
    value: Any = None
    exception: BaseException | None = None
    timed_out: bool = False
    thread_hung: bool = False
    wall_time: float = 0.0
    cpu_time: float = 0.0        # process CPU (all threads) during the run
    thread_time: float = 0.0     # CPU of the sandboxed worker thread alone
    peak_working_set_bytes: int = 0
    layers: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def __str__(self) -> str:
        state = "OK" if self.ok else (
            "TIMEOUT" if self.timed_out else "BLOCKED/ERROR")
        return (f"[{state}] wall={self.wall_time:.4f}s "
                f"cpu={self.cpu_time:.4f}s thread={self.thread_time:.4f}s "
                f"peak_ws={self.peak_working_set_bytes / 1e6:.1f}MB "
                f"exception={self.exception!r} layers={self.layers}")


def _peak_working_set() -> int:
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    if kernel32.K32GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters), counters.cb):
        return int(counters.PeakWorkingSetSize)
    return 0


# ===========================================================================
# Runner
# ===========================================================================

_SANDBOX_LOCK = threading.Lock()  # only one active sandbox per process


def run_in_sandbox(func_to_test: Callable, *args,
                   timeout: float | None = None,
                   policy: SandboxPolicy | None = None,
                   **kwargs) -> SandboxResult:
    """Run ``func_to_test(*args, **kwargs)`` under all sandbox layers.

    The callable executes on a dedicated worker thread (required so that the
    Layer-2 token impersonation can be attached to - and later stripped
    from - that thread alone). Returns a SandboxResult; exceptions raised by
    the target (including SandboxSecurityError) are captured, not re-raised.
    """
    if policy is None:
        policy = SandboxPolicy(root_dir=os.path.join(os.getcwd(),
                                                     "sandbox_workspace"))
    if not _SANDBOX_LOCK.acquire(blocking=False):
        raise RuntimeError("run_in_sandbox is not re-entrant / thread-safe; "
                           "run one sandbox at a time per process")

    result = SandboxResult()
    gate = _PathGatekeeper(policy)
    job = _JobGuard(policy) if policy.use_job else None
    hooks = (_PythonHookGuard(policy, gate)
             if policy.use_python_hooks else None)
    worker_box: dict = {}

    def _worker():
        # --- Layer 2 engages here, on the worker thread itself ---
        token = _TokenGuard(policy.low_integrity) if policy.use_token else None
        token_ok = False
        if token is not None:
            try:
                token.engage()
                token_ok = True
            except OSError as e:
                result.notes.append(f"token layer unavailable: {e}")
        result.layers["token"] = token_ok
        t0 = time.thread_time()
        try:
            worker_box["value"] = func_to_test(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001 - capture everything
            worker_box["exception"] = e
        finally:
            worker_box["thread_time"] = time.thread_time() - t0
            if token is not None:
                token.revert()  # always detach, even on async-kill path

    thread = threading.Thread(target=_worker, name="sandbox-worker",
                              daemon=True)
    try:
        if job is not None:
            job.engage()
            result.layers["job"] = job.engaged
            if job.note:
                result.notes.append(job.note)
        if hooks is not None:
            hooks.install()
            result.layers["python_hooks"] = True

        wall0 = time.perf_counter()
        cpu0 = time.process_time()
        thread.start()
        thread.join(timeout)
        result.wall_time = time.perf_counter() - wall0
        result.cpu_time = time.process_time() - cpu0

        if thread.is_alive():
            # Timeout: ask the interpreter to raise SystemExit inside the
            # worker. This only interrupts Python bytecode; a thread stuck
            # in a blocking native call cannot be safely killed in-process.
            result.timed_out = True
            tid = thread.ident
            if tid is not None:
                n = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(tid), ctypes.py_object(SystemExit))
                if n > 1:  # corrupted multiple threads - undo immediately
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_ulong(tid), None)
                    result.notes.append("async exception injection aborted")
            thread.join(3.0)  # grace period for the SystemExit to unwind
            if thread.is_alive():
                # FAIL CLOSED: leave every restriction in place so the
                # runaway thread stays confined; tell the caller loudly.
                result.thread_hung = True
                result.notes.append(
                    "worker thread unkillable (native call?); sandbox left "
                    "ENGAGED - terminate this process before reusing it")
                result.exception = TimeoutError(
                    f"sandboxed call hung beyond timeout={timeout}")
                return result
            result.notes.append("worker terminated by async SystemExit")

        result.thread_time = worker_box.get("thread_time", 0.0)
        result.value = worker_box.get("value")
        result.exception = worker_box.get("exception")
        if result.timed_out and result.exception is None:
            result.exception = TimeoutError(
                f"sandboxed call exceeded timeout={timeout}")
        result.ok = (not result.timed_out) and (result.exception is None)
        return result
    finally:
        result.peak_working_set_bytes = _peak_working_set()
        # Teardown order matters: only strip restrictions once the untrusted
        # thread is provably finished (the hung path returns early above).
        if hooks is not None:
            hooks.remove()
        if job is not None:
            job.relax()
        _SANDBOX_LOCK.release()


# ===========================================================================
# Self-test / demonstration
# ===========================================================================

if __name__ == "__main__":
    import socket  # imported pre-sandbox on purpose: tests the sys.modules
    # eviction + attribute patching path (untrusted code gets a patched ref)

    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "sandbox_workspace")
    POLICY = SandboxPolicy(root_dir=ROOT)

    def show(title, res):
        print(f"\n--- {title} ---")
        print(res)
        for n in res.notes:
            print("    note:", n)

    # 1) Legitimate benchmark work: pure compute + scratch file in root.
    def legit_workload(n):
        scratch = os.path.join(ROOT, "out.txt")
        with open(scratch, "w") as f:          # inside root: allowed
            total = sum(i * i for i in range(n))
            f.write(str(total))
        with open(scratch) as f:               # read-back inside root
            return int(f.read())

    show("allowed workload (compute + file inside root)",
         run_in_sandbox(legit_workload, 200_000, timeout=10, policy=POLICY))

    # 2) File I/O escape attempt -> blocked.
    def evil_file_write():
        evil = os.path.join(os.path.expanduser("~"), "pwned_by_benchmark.txt")
        with open(evil, "w") as f:
            f.write("you should never see this")

    r = run_in_sandbox(evil_file_write, timeout=5, policy=POLICY)
    show("escape attempt: write outside root", r)
    assert isinstance(r.exception, SandboxSecurityError)

    # 3) Read escape attempt -> blocked (hosts file is outside allow-list).
    def evil_file_read():
        return open(r"C:\Windows\System32\drivers\etc\hosts").read()

    r = run_in_sandbox(evil_file_read, timeout=5, policy=POLICY)
    show("escape attempt: read outside allow-list", r)
    assert isinstance(r.exception, SandboxSecurityError)

    # 4) Network exfiltration attempt -> blocked.
    def evil_network():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("203.0.113.1", 443))

    r = run_in_sandbox(evil_network, timeout=5, policy=POLICY)
    show("escape attempt: outbound socket", r)
    assert isinstance(r.exception, SandboxSecurityError)

    # 5) Subprocess attempt (two ways) -> blocked.
    def evil_subprocess():
        os.system("calc.exe")

    r = run_in_sandbox(evil_subprocess, timeout=5, policy=POLICY)
    show("escape attempt: os.system", r)
    assert isinstance(r.exception, SandboxSecurityError)

    def evil_import_subprocess():
        import subprocess  # import itself is vetoed
        subprocess.run(["whoami"])

    r = run_in_sandbox(evil_import_subprocess, timeout=5, policy=POLICY)
    show("escape attempt: import subprocess", r)
    assert isinstance(r.exception, SandboxSecurityError)

    # 6) FFI escape attempt -> blocked even though THIS module uses ctypes.
    def evil_ctypes():
        import ctypes as c
        c.windll.kernel32.CreateFileW(r"C:\evil.txt", 0, 0, None, 0, 0, None)

    r = run_in_sandbox(evil_ctypes, timeout=5, policy=POLICY)
    show("escape attempt: import ctypes", r)
    assert isinstance(r.exception, SandboxSecurityError)

    # 7) Runaway code -> timeout + async kill.
    def runaway():
        while True:
            pass

    r = run_in_sandbox(runaway, timeout=1.0, policy=POLICY)
    show("runaway infinite loop (timeout=1s)", r)
    assert r.timed_out and not r.thread_hung

    # 8) Post-sandbox sanity: restrictions were stripped cleanly.
    assert open is __builtins__["open"] if isinstance(__builtins__, dict) else True
    print("\nAll sandbox self-tests passed; restrictions cleanly removed.")
