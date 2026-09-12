"""Giving the card back between components.

Nothing here belongs to a single component: it is the one place that knows how to
make a model that has just finished actually release its memory, so the next one
can load. Two things have to happen and neither is enough on its own -- Python
has to drop the last reference (``gc``), and the caching allocator has to hand
the freed blocks back to the driver (``empty_cache``).
"""

from __future__ import annotations


def free() -> dict:
    """Releases what the previous component left behind; reports what is left.

    Returns ``{"free_gb", "total_gb"}``, or an empty dict when there is no GPU.
    """
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free_bytes, total = torch.cuda.mem_get_info()
    return {"free_gb": free_bytes / 1024**3, "total_gb": total / 1024**3}


def report(label: str = "") -> str:
    """One ASCII line about the state of the card, for a progress log."""
    state = free()
    if not state:
        return f"{label}no GPU"
    return (f"{label}{state['free_gb']:.1f} GB free of {state['total_gb']:.1f} GB")
