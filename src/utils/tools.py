"""Locating the external command-line tools of the environment.

FFmpeg comes from the conda environment (``environment.yml`` pins the GPL
build), not from a system-wide installation. On Windows conda puts it in
``<env>/Library/bin``, which lands on PATH only after ``conda activate``.
Scripts here are also started by calling the environment's interpreter
directly, and then PATH knows nothing about the environment -- so the binary is
looked up next to the running interpreter first, and only then on PATH.

Without this the tools appear to be missing in exactly the setup the project
normally runs in, and the failure surfaces as a bare ``FileNotFoundError`` from
``subprocess`` several layers down.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

#: how the environment is put back together, quoted in the error messages
INSTALL_HINT = 'conda install -n wideo -c conda-forge "ffmpeg=*=gpl*"'


def _candidates(name: str) -> list[Path]:
    """Where the binary of the running environment could sit."""
    prefix = Path(sys.prefix)
    suffix = ".exe" if sys.platform == "win32" else ""
    return [prefix / "Library" / "bin" / f"{name}{suffix}",   # conda on Windows
            prefix / "bin" / f"{name}{suffix}",               # conda elsewhere
            prefix / "Scripts" / f"{name}{suffix}"]


def find(name: str) -> str | None:
    """Path of the tool, or ``None``; the environment's copy wins over PATH."""
    for candidate in _candidates(name):
        if candidate.exists():
            return str(candidate)
    return shutil.which(name)


def require(name: str) -> str:
    """Path of the tool, or a message saying how to put it back."""
    found = find(name)
    if found is None:
        raise RuntimeError(
            f"{name} not found - neither in the environment ({sys.prefix}) nor on "
            f"PATH. It belongs to the conda environment: {INSTALL_HINT}")
    return found


def ffmpeg() -> str:
    return require("ffmpeg")


def ffprobe() -> str:
    return require("ffprobe")
