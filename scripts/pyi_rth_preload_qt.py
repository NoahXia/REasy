"""Load the bundled Qt runtime before third-party Qt DLLs on Windows.

Some developer tools add their own Qt directory to the system PATH.  A frozen
REasy process must bind PySide6's extension modules to the Qt and shiboken DLLs
collected by PyInstaller, not to an older installation found on PATH.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys


_DLL_DIRECTORIES = []
_DLL_HANDLES = []


def _preload_bundled_qt() -> None:
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return

    bundle_root = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    pyside_dir = bundle_root / "PySide6"
    shiboken_dir = bundle_root / "shiboken6"
    if not pyside_dir.is_dir() or not shiboken_dir.is_dir():
        return

    os.environ["PATH"] = os.pathsep.join(
        (str(pyside_dir), str(shiboken_dir), os.environ.get("PATH", ""))
    )
    if hasattr(os, "add_dll_directory"):
        _DLL_DIRECTORIES.extend(
            (
                os.add_dll_directory(str(pyside_dir)),
                os.add_dll_directory(str(shiboken_dir)),
            )
        )

    system_icu = (
        Path(os.environ.get("SystemRoot", r"C:\Windows"))
        / "System32"
        / "icuuc.dll"
    )
    for dll in (
        system_icu,
        shiboken_dir / "shiboken6.abi3.dll",
        pyside_dir / "Qt6Core.dll",
        pyside_dir / "pyside6.abi3.dll",
    ):
        if dll.is_file():
            _DLL_HANDLES.append(ctypes.WinDLL(str(dll)))


_preload_bundled_qt()
