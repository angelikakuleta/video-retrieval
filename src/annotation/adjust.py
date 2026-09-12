"""Fitting an imported annotation to the masks of its episode.

TVR windows are wide and often open before the action becomes visible, so an
annotation quite often starts inside a mask. What to do depends on the side:

* it starts inside a mask and continues past it -- the marker opened too early,
  so the whole interval is SHIFTED to the far edge with its length kept.
  Trimming instead would leave a scrap: ``tbbt_s03e02_d88919`` has 0.93 s of
  its 1.43 s inside the animation and would end up below the length limit;
* it ends inside a mask -- that tail is TVR padding, not the event, so the
  interval is TRIMMED to the near edge, however much is cut.

An annotation lying entirely inside a mask is rejected. One spanning a whole
mask keeps its times and is flagged: a visible event cannot continue across a
scene change, so it is a mis-timed marker for a human to look at.

Both classes of mask count: the structural/transition distinction matters to
the content axis, not to an annotation -- neither belongs to the corpus.

Propagation treats the shifts as a measurement: if the markers of an episode
are systematically early, a shift measured where an annotation happened to hit
a mask also describes annotations that did not. Whether it helps depends on the
episode, so BOTH variants are produced for every episode and the choice is made
by hand, per episode.
"""

from __future__ import annotations

#: interval moved to the far edge of a mask, length kept
FLAG_SHIFTED = "shifted"
#: interval cut back to the near edge of a mask
FLAG_TRIMMED = "trimmed"
#: the shift had nowhere to land -- original times kept, for a human to decide
FLAG_SHIFT_FAILED = "shift_failed"
#: a whole mask lies inside the interval -- a mis-timed marker
FLAG_SPANS_MASK = "spans_mask"

#: how a shift measured on one annotation is carried to the others
#: how a measured shift is carried to the rest of the episode. Both are
#: generated for every episode; the choice is made by hand, per episode.
PROPAGATION_MODES = ("off", "forward")

EPS = 1e-9


def _inside_a_range(start: float, end: float, ranges) -> bool:
    return any(a - EPS <= start and end <= b + EPS for a, b in ranges)


def _clear_of_masks(start: float, end: float, masks) -> bool:
    return not any(start < b - EPS and a < end - EPS for a, b in masks)


def fits(start: float, end: float, masks, ranges) -> bool:
    """Whether the interval touches no mask and lies inside one corpus range."""
    return _clear_of_masks(start, end, masks) and _inside_a_range(start, end, ranges)


def fit(start: float, end: float, masks, ranges):
    """One annotation -> ``(start, end, flags, offset)``, or ``None``.

    ``None`` means the interval lies entirely inside a mask. ``offset`` is how far
    it was moved [s] -- zero unless the shift succeeded, and what propagation
    carries over.
    """
    duration = end - start
    if any(a - EPS <= start and end <= b + EPS for a, b in masks):
        return None

    flags: list[str] = []

    lead = next((m for m in masks if m[0] - EPS <= start < m[1] - EPS), None)
    offset = 0.0
    if lead is not None:
        moved_start, moved_end = lead[1], lead[1] + duration
        if fits(moved_start, moved_end, masks, ranges):
            offset = moved_start - start
            start, end = moved_start, moved_end
            flags.append(FLAG_SHIFTED)
        else:
            return start, end, [FLAG_SHIFT_FAILED], 0.0

    tail = next((m for m in masks if m[0] + EPS < end <= m[1] + EPS), None)
    if tail is not None and start < tail[0] - EPS:
        end = tail[0]
        flags.append(FLAG_TRIMMED)

    if any(start < a - EPS and b < end + EPS for a, b in masks):
        flags.append(FLAG_SPANS_MASK)

    return round(start, 2), round(end, 2), flags, round(offset, 3)


def fit_episode(items, masks, ranges, propagation: str = "off") -> dict:
    """Fits every annotation of one episode -> ``id -> (start, end, flags, offset)``.

    ``items`` is ``(id, start, end)`` in any order; a rejected annotation maps to
    ``None``. In ``off`` every annotation is fitted on its own; in ``forward`` a
    measured shift of ``x`` is added to every LATER annotation of the episode
    before it is fitted, so a systematically early set of markers is recovered as
    a whole.
    """
    if propagation not in PROPAGATION_MODES:
        raise ValueError(f"unknown propagation mode: {propagation!r} - "
                         f"expected one of {PROPAGATION_MODES}")

    ordered = sorted(items, key=lambda i: (float(i[1]), float(i[2])))
    if propagation == "off":
        return {i[0]: fit(float(i[1]), float(i[2]), masks, ranges) for i in ordered}

    out, carried = {}, 0.0
    for key, start, end in ordered:
        fitted = fit(float(start) + carried, float(end) + carried, masks, ranges)
        out[key] = fitted
        if fitted and FLAG_SHIFTED in fitted[2]:
            carried += fitted[3]
    return out
