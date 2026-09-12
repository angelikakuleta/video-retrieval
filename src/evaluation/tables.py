"""Formatting the result tables of chapter 6 for reading in a notebook.

Numbers are printed the way they are written in the thesis: Recall@K as a
percentage, differences in percentage points, comma as the decimal separator. It
stays a plain table, not a LaTeX row -- the thesis is written by hand and
generated markup only hides where a number came from.

Column headers stay ASCII, because a Windows console renders them under whatever
code page it happens to have.
"""

from __future__ import annotations

from collections.abc import Sequence

#: what a missing value looks like in a table
DASH = "-"


def _decimal(value: float, digits: int, sign: bool = False) -> str:
    text = f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"
    return text.replace(".", ",")


def percent(value: float | None, digits: int = 1) -> str:
    """A share (0..1) as a percentage: ``0.473`` -> ``47,3``."""
    return DASH if value is None else _decimal(100 * value, digits)


def points(value: float | None, digits: int = 1, sign: bool = True) -> str:
    """A difference of shares as percentage points: ``0.021`` -> ``+2,1``."""
    return DASH if value is None else _decimal(100 * value, digits, sign)


def interval(entry: dict | None, digits: int = 1) -> str:
    """A difference with its 95% interval: ``+2,1 [-0,4; 4,6]``.

    An interval that could not be computed shows the bare difference with the count,
    so it is never mistaken for a wide one.
    """
    if entry is None or entry.get("mean") is None:
        return DASH
    mean = points(entry["mean"], digits)
    if entry.get("low") is None:
        return f"{mean} (n={entry.get('n', 0)})"
    return (f"{mean} [{points(entry['low'], digits, sign=False)}; "
            f"{points(entry['high'], digits, sign=False)}]")


def number(value: float | None, digits: int = 1) -> str:
    """A plain quantity: seconds, milliseconds, a count of fragments."""
    if value is None:
        return DASH
    return _decimal(value, digits) if digits else f"{round(value)}"


def render(columns: Sequence[str], rows: Sequence[Sequence[str] | str],
           align: str | None = None) -> str:
    """An aligned text table; a row given as a plain string becomes a subheading.

    ``align`` is one character per column, ``l`` or ``r``.
    """
    align = align or ("l" + "r" * (len(columns) - 1))
    cells = [[str(c) for c in row] for row in rows if not isinstance(row, str)]
    widths = [max(len(columns[i]), *(len(row[i]) for row in cells)) if cells
              else len(columns[i]) for i in range(len(columns))]

    def line(values):
        return "  ".join(v.ljust(w) if a == "l" else v.rjust(w)
                         for v, w, a in zip(values, widths, align)).rstrip()

    out = [line(columns), "-" * len(line(columns))]
    for row in rows:
        if isinstance(row, str):
            out.append("")
            out.append(row)
        else:
            out.append(line([str(c) for c in row]))
    return "\n".join(out)


def show(title: str, columns: Sequence[str], rows: Sequence, note: str = "",
         align: str | None = None) -> None:
    """Prints one table under its heading, with an optional note below it."""
    print(f"\n{title}")
    print("=" * len(title))
    print(render(columns, rows, align))
    if note:
        print(f"\n{note}")
