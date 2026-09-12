"""Common startup for notebooks and quick checks.

Collects what would otherwise be repeated at the top of every notebook: working
directory, GPU memory limit, time and peak VRAM measurement, releasing models,
and saving measurements in a form that can be copied into a table.

    from src.utils.notebook import setup, measure, vram, release, save_measurement

    setup()
    with measure("loading OpenCLIP") as p:
        model = ...
    save_measurement("openclip", p)
    release("model")
"""

from __future__ import annotations

import gc
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

# src/utils/notebook.py -> the repository root is the directory two levels up
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MEASUREMENTS_DIR = RESULTS_DIR / "measurements"

GIB = 1024**3

# The allocator refuses to exceed this fraction of GPU memory and raises a
# clean OutOfMemoryError. Without it the driver can silently spill the excess
# into system RAM -- instead of an error we get a tenfold-plus slowdown with
# no hint of what happened.
MEMORY_FRACTION = 0.95


def _torch_with_gpu():
    """The ``torch`` module if it is installed and sees a GPU; otherwise ``None``.

    One conditional import, so the rest only needs ``if torch is not None`` -- which
    also narrows the type for static analysis.
    """
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


# ---------------------------------------------------------- Notebook startup
def setup(memory_fraction: float = MEMORY_FRACTION, quiet: bool = False) -> dict:
    """Prepares the notebook environment and returns the machine metadata.

    Adds the repository root to the import path, switches the working directory
    there (so ``data/raw/vatex`` means the same in a notebook and in a script),
    enables autoreload of ``src/`` and applies the GPU memory limit.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if Path.cwd().resolve() != ROOT:
        os.chdir(ROOT)

    _enable_autoreload()

    info = metadata()

    torch = _torch_with_gpu()
    if torch is not None:
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)

    if not quiet:
        print(f"working directory : {ROOT}")
        print(f"Python            : {info['python']}")
        print(f"PyTorch           : {info.get('torch', 'missing')}  (CUDA {info.get('cuda', '-')})")
        print(f"GPU               : {info.get('gpu', 'unavailable')}")
        if info.get("architectures"):
            mark = "OK" if "sm_120" in info["architectures"] else "MISSING sm_120"
            print(f"kernels           : {mark}  ({', '.join(info['architectures'])})")
        if info.get("vram_total_gib"):
            print(f"memory limit      : {memory_fraction:.0%} of "
                  f"{info['vram_total_gib']:.1f} GiB")
        vram("state at start")
    return info


def _enable_autoreload() -> None:
    try:
        from IPython import get_ipython  # type: ignore

        shell = get_ipython()
        if shell is None:
            return
        # reload_ext loads the extension when it is absent and does not warn when it already is
        shell.run_line_magic("reload_ext", "autoreload")
        shell.run_line_magic("autoreload", "2")
    except Exception:
        pass


def metadata() -> dict:
    """Description of the machine -- goes into every saved measurement."""
    # The annotation matters: without it the dict type is inferred from the
    # first three values (all strings), and adding a number or a list further
    # down then looks like an error.
    info: dict[str, object] = {
        "python": platform.python_version(),
        "system": f"{platform.system()} {platform.release()}",
        "processor": platform.processor(),
    }
    try:
        import torch
    except ImportError:
        return info

    info["torch"] = torch.__version__
    # torch.version.cuda can be None (build without CUDA) -- getattr instead of
    # direct access, because the `version` attribute itself is not always
    # visible to the editor
    info["cuda"] = getattr(torch.version, "cuda", None)
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        info["gpu"] = properties.name
        info["vram_total_gib"] = round(properties.total_memory / GIB, 2)
        info["architectures"] = list(torch.cuda.get_arch_list())
    return info


# ---------------------------------------------------------------- GPU memory
def gpu_state() -> dict:
    """How much GPU memory is free -- counting processes outside this notebook."""
    torch = _torch_with_gpu()
    if torch is None:
        return {"available": False}

    free, total = torch.cuda.mem_get_info()
    return {
        "available": True,
        "free_gib": round(free / GIB, 2),
        "total_gib": round(total / GIB, 2),
        "used_gib": round((total - free) / GIB, 2),
        "torch_allocated_gib": round(torch.cuda.memory_allocated() / GIB, 2),
        "torch_reserved_gib": round(torch.cuda.memory_reserved() / GIB, 2),
    }


def vram(label: str = "") -> dict:
    """Prints the GPU memory state and returns it as a dict."""
    state = gpu_state()
    if not state["available"]:
        print("GPU unavailable")
        return state
    prefix = f"[{label}] " if label else ""
    print(
        f"{prefix}free {state['free_gib']:.2f} GiB of {state['total_gib']:.2f} GiB"
        f"  (torch holds {state['torch_reserved_gib']:.2f} GiB)"
    )
    return state


def nvidia_smi() -> str:
    """Reading from ``nvidia-smi`` -- also shows the memory taken by the desktop."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,"
             "driver_version,display_active", "--format=csv"],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return result.stdout.strip()
    except Exception as error:
        return f"nvidia-smi unavailable: {error}"


@contextmanager
def measure(label: str, quiet: bool = False):
    """Measures time and peak GPU memory usage within a block.

    Returns a dict filled in on exit -- ready to pass to :func:`save_measurement`.
    """
    result: dict = {"label": label}
    torch = _torch_with_gpu()

    if torch is not None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        free_before, _ = torch.cuda.mem_get_info()
        result["free_before_gib"] = round(free_before / GIB, 2)

    start = time.perf_counter()
    try:
        yield result
    finally:
        if torch is not None:
            torch.cuda.synchronize()
        result["time"] = round(time.perf_counter() - start, 3)
        if torch is not None:
            result["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / GIB, 2)
            result["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / GIB, 2)
            free_after, _ = torch.cuda.mem_get_info()
            result["free_after_gib"] = round(free_after / GIB, 2)
        if not quiet:
            text = f"{label}: {result['time']:.2f} s"
            if "peak_allocated_gib" in result:
                text += (f", peak {result['peak_allocated_gib']:.2f} GiB "
                         f"(reserved {result['peak_reserved_gib']:.2f} GiB)")
            print(text)


def release(*names: str) -> dict:
    """Drops the given objects and gives GPU memory back.

    ``release("model", "processor")`` rebinds those names to ``None`` in the
    notebook namespace and clears IPython's output history (``Out``, ``_``,
    ``_12``), which can hold a reference to the model and keep a dozen gigabytes
    alive. Works for notebook-level variables only.

    The names are rebound rather than DELETED, and the difference matters in a
    notebook. A cell that loads a model earlier, uses it here and releases it at
    the end is the normal shape of these checks -- and deleting the name made that
    cell impossible to run twice: the second run raised ``NameError`` on its own
    guard (``if detector is not None``) instead of skipping the section. ``None``
    keeps the guard working, says the same thing, and frees exactly as much: the
    reference is gone either way, and the collection below is what returns the
    memory.
    """
    frame = sys._getframe(1)
    for name in names:
        for namespace in (frame.f_locals, frame.f_globals):
            if name in namespace:
                try:
                    namespace[name] = None
                except Exception:
                    pass

    try:
        from IPython import get_ipython  # type: ignore

        shell = get_ipython()
        if shell is not None:
            shell.run_line_magic("reset", "-f out")
    except Exception:
        pass

    gc.collect()
    torch = _torch_with_gpu()
    if torch is not None:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return vram("after release")


# ------------------------------------------------------------- Test material
def find_clip(directories: list[Path] | None = None) -> Path | None:
    """The first video file found in the data -- for quick checks."""
    directories = directories or [
        DATA_DIR / "raw" / "vatex",
        DATA_DIR / "processed",
        DATA_DIR / "raw",
        DATA_DIR,
    ]
    for directory in directories:
        if not directory.exists():
            continue
        for extension in ("*.mp4", "*.mkv", "*.avi", "*.webm"):
            for file in sorted(directory.rglob(extension)):
                return file
    return None


def test_frame(path: Path | str | None = None, position: float = 0.5):
    """A frame from a video as ``PIL.Image`` (``position`` = fraction of length)."""
    from PIL import Image

    path = Path(path) if path else find_clip()
    if path is not None and Path(path).exists():
        import cv2

        reader = cv2.VideoCapture(str(path))
        try:
            count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
            if count > 1:
                reader.set(cv2.CAP_PROP_POS_FRAMES, int(count * position))
            ok, frame = reader.read()
            if ok:
                return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            reader.release()

    import numpy as np

    print("no videos in data/ -- using a placeholder image")
    grid = np.zeros((480, 640, 3), dtype=np.uint8)
    grid[..., 0] = np.linspace(0, 255, 640, dtype=np.uint8)
    grid[..., 1] = np.linspace(0, 255, 480, dtype=np.uint8)[:, None]
    return Image.fromarray(grid)


# ------------------------------------------------------- Saving measurements
def save_measurement(name: str, data: dict) -> Path:
    """Saves a measurement to ``results/measurements/`` with the machine metadata.

    Numbers entered into the thesis come from these files: every record carries the
    date, library versions and GPU name.
    """
    MEASUREMENTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    record = {
        "name": name,
        "when": now.isoformat(timespec="seconds"),
        "machine": metadata(),
        "data": data,
    }
    file = MEASUREMENTS_DIR / f"{name}_{now.strftime('%Y%m%d_%H%M%S')}.json"
    file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {file.relative_to(ROOT)}")
    return file


def load_measurements(name: str | None = None) -> list[dict]:
    """All saved measurements, newest last."""
    if not MEASUREMENTS_DIR.exists():
        return []
    pattern = f"{name}_*.json" if name else "*.json"
    return [
        json.loads(file.read_text(encoding="utf-8"))
        for file in sorted(MEASUREMENTS_DIR.glob(pattern))
    ]


def summary() -> None:
    """Overview of saved measurements -- one line per run."""
    records = load_measurements()
    if not records:
        print("no saved measurements")
        return
    print(f"{'measurement':<28}{'when':<21}{'time [s]':>10}{'peak [GiB]':>14}")
    print("-" * 73)
    for record in records:
        data = record.get("data", {})
        elapsed = data.get("time")
        peak = data.get("peak_allocated_gib")
        print(
            f"{record['name']:<28}"
            f"{record['when'].replace('T', ' '):<21}"
            f"{(f'{elapsed:.2f}' if isinstance(elapsed, (int, float)) else '-'):>10}"
            f"{(f'{peak:.2f}' if isinstance(peak, (int, float)) else '-'):>14}"
        )
