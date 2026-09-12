"""Logging a loop that may run over thousands of items.

A cache loop over eighteen episodes should say what it did with each of them; the
same loop over 2489 VATEX clips should not, or the lines that matter drown. One
helper decides that, so every loop makes the same choice.
"""

from __future__ import annotations

from collections.abc import Callable

#: below this many items every one is worth a line
EVERY = 100


def progress(log: Callable[[str], None], total: int | None = None,
             every: int = EVERY) -> Callable[[str], None]:
    """A log for per-item messages that thins itself out when there are many.

    With ``total`` at or below ``every`` -- a series, always -- nothing is
    thinned and the output is what it has always been. Above it only every
    ``every``-th message and the last one get through, so a VATEX collection
    reports about two dozen lines instead of a few thousand.

    ``total`` is not in the plan's signature; without it a run over six episodes
    would print nothing at all.
    """
    if total is not None and total <= every:
        return log

    seen = {"n": 0}

    def emit(message: str) -> None:
        seen["n"] += 1
        if seen["n"] % every == 0 or seen["n"] == total:
            log(f"  [{seen['n']}/{total if total is not None else '?'}] {message}")

    return emit
