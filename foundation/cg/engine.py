"""Single per-process loader for the native cabt engine.

Both the agent (main.py) and the SDK shim (cg/api.py) go through here so the
library is loaded and GameInitialize is called exactly once per process.
"""
from __future__ import annotations

import ctypes
import os
import platform

_DIR = os.path.dirname(os.path.abspath(__file__))
_lib = None


def lib_path():
    os_name = platform.system()
    if os_name == "Windows":
        return os.path.join(_DIR, "cg.dll")
    if os_name == "Darwin":
        return os.path.join(_DIR, "libcg.dylib")
    if platform.machine() in ("arm64", "aarch64"):
        return os.path.join(_DIR, "libcg-arm64.so")
    return os.path.join(_DIR, "libcg.so")


def _configure(lib):
    """Declare the agent/search ABI even when the SDK's sim shim is minimal.

    Recent kaggle-environments releases configure only the live-battle calls
    in ``cg.sim``.  Without these declarations ctypes treats ``AllCard`` as an
    integer-returning function and local search/training fails before game 1.
    Re-declaring an existing ctypes signature is harmless.
    """
    lib.AllCard.restype = ctypes.c_char_p
    lib.AllCard.argtypes = []
    lib.AllAttack.restype = ctypes.c_char_p
    lib.AllAttack.argtypes = []
    lib.AgentStart.restype = ctypes.c_void_p
    lib.AgentStart.argtypes = []
    lib.SearchBegin.restype = ctypes.c_char_p
    lib.SearchBegin.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int] + \
        [ctypes.POINTER(ctypes.c_int)] * 6 + [ctypes.c_int]
    lib.SearchStep.restype = ctypes.c_char_p
    lib.SearchStep.argtypes = [ctypes.c_void_p, ctypes.c_longlong,
                               ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.SearchEnd.restype = None
    lib.SearchEnd.argtypes = [ctypes.c_void_p]
    lib.SearchRelease.restype = None
    lib.SearchRelease.argtypes = [ctypes.c_void_p, ctypes.c_longlong]
    return lib


def get_lib():
    """Return the initialized engine handle.

    Prefers the official cg.sim module (which already loads the library, calls
    GameInitialize once, and sets every argtype). Falls back to loading it
    ourselves if sim.py is unavailable. GameInitialize is NOT idempotent -- a
    second call on the same loaded library raises -- so routing everything
    through this one function is what keeps a process from double-initializing.
    """
    global _lib
    if _lib is not None:
        return _lib
    try:
        from .sim import lib as _simlib       # official bindings, already inited
        _lib = _configure(_simlib)
        return _lib
    except Exception:
        pass
    lib = ctypes.cdll.LoadLibrary(lib_path())
    try:
        lib.GameInitialize()
    except OSError:
        # the same library instance was already initialized in this process
        pass

    _lib = _configure(lib)
    return _lib
