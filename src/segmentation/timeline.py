"""The content time axis of one episode.

Two clocks run at once:

* file time -- seconds from the start of the .mp4. Canonical: masks,
  annotations, fragments and every report use it.
* content time -- seconds of material inside one corpus range, counted with the
  transition masks removed. Used only here and by segmentation (window length,
  the 3-15 s correction).

Masks are of two kinds. A structural one (logo, opening titles, credits) cuts
the episode into corpus ranges. A transition one (the 1.3 s atom animation in
TBBT) is absorbed instead: the range runs across it and only the seconds it
occupies stop counting. Reason for E1 -- a range edge is a free fragment
boundary for every strategy, so turning scene transitions into range edges
would hand the fixed-window baseline the boundaries it is supposed to be unable
to find.

Skips (black stretches from blackdetect) are not a mask: they change neither
the corpus nor content time, they only mark frames no model should be fed.

All intervals are half-open ``[start, end)``.
"""

from __future__ import annotations

from src.annotation.ranges import merge

#: tolerance for comparing times [s] -- far below one frame at any frame rate
EPS = 1e-9


# ------------------------------------------------------- Interval arithmetic
def clip(spans: list[tuple[float, float]], start: float, end: float) -> list[tuple[float, float]]:
    """The part of ``spans`` lying inside ``[start, end)``."""
    out = [(max(a, start), min(b, end)) for a, b in spans]
    return [(a, b) for a, b in out if b - a > EPS]


def subtract(spans: list[tuple[float, float]],
             holes: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """``spans`` with every hole cut out; both lists sorted and disjoint."""
    out = []
    for a, b in spans:
        position = a
        for ha, hb in holes:
            if hb <= position or ha >= b:
                continue
            if ha > position:
                out.append((position, ha))
            position = max(position, hb)
            if position >= b:
                break
        if b - position > EPS:
            out.append((position, b))
    return out


def total(spans: list[tuple[float, float]]) -> float:
    """Total length of a list of spans [s]."""
    return sum(b - a for a, b in spans)


# ------------------------------------------------------------------ Timeline
class Timeline:
    """The two clocks of one episode.

    ``ranges`` are the corpus ranges (the complement of the structural masks),
    ``holes`` the transition masks, ``skips`` the black stretches. Holes and skips
    are clipped to the ranges and merged on construction, so every method below may
    assume them disjoint and inside the corpus.
    """

    def __init__(self,
                 ranges: list[tuple[float, float]],
                 holes: list[tuple[float, float]] = (),
                 skips: list[tuple[float, float]] = ()) -> None:
        self.ranges = [(float(a), float(b)) for a, b in merge([tuple(r) for r in ranges])]
        self.holes = self._inside_corpus(holes)
        self.skips = self._inside_corpus(skips)
        #: hole-free pieces of every range, in time order -- the content axis
        self.pieces = [subtract([r], self.holes) for r in self.ranges]

    def _inside_corpus(self, spans) -> list[tuple[float, float]]:
        """The parts of ``spans`` that fall inside the corpus ranges."""
        merged = merge([(float(a), float(b)) for a, b in spans]) if spans else []
        out: list[tuple[float, float]] = []
        for start, end in self.ranges:
            out += clip(merged, start, end)
        return merge(out)

    # ----------------------------------------------------------- ranges

    @classmethod
    def from_ranges(cls, ranges: list[tuple[float, float]]) -> "Timeline":
        """A timeline without holes or skips -- content time equals file time."""
        return cls(ranges)

    def range_index(self, time: float) -> int | None:
        """Index of the range holding ``time``, or ``None`` outside the corpus."""
        for index, (a, b) in enumerate(self.ranges):
            if a <= time < b:
                return index
        return None

    def range_of(self, time: float) -> tuple[float, float] | None:
        index = self.range_index(time)
        return self.ranges[index] if index is not None else None

    # ------------------------------------------------- the content axis

    def in_hole(self, time: float) -> bool:
        return any(a <= time < b for a, b in self.holes)

    def in_skip(self, time: float) -> bool:
        return any(a <= time < b for a, b in self.skips)

    def is_content(self, time: float) -> bool:
        """Inside the corpus and not inside a transition mask."""
        return self.range_index(time) is not None and not self.in_hole(time)

    def is_usable(self, time: float) -> bool:
        """Content, and not a frame of black picture -- may be fed to a model."""
        return self.is_content(time) and not self.in_skip(time)

    def to_content(self, time: float) -> float | None:
        """File time -> seconds of content since the start of its range.

        ``None`` inside a transition mask or outside the corpus: such a moment has no
        content coordinate, and the nearest one would hide the case the caller has to
        handle.
        """
        index = self.range_index(time)
        if index is None or self.in_hole(time):
            return None
        return round(total(clip(self.pieces[index], self.ranges[index][0], time)), 6)

    def advance(self, start: float, amount: float) -> float:
        """File time reached after ``amount`` seconds of content from ``start``.

        A position landing exactly on the edge of a hole is reported AFTER the hole, so
        a fragment never begins with material that is not there.
        """
        range_ = self.range_of(start)
        if range_ is None:
            return start + amount
        remaining = amount
        for a, b in subtract([(start, range_[1])], self.holes):
            if remaining < (b - a) - EPS:
                return a + remaining
            remaining -= b - a
        return range_[1]

    def to_file(self, range_index: int, content: float) -> float:
        """Content coordinate inside one range -> file time."""
        return self.advance(self.ranges[range_index][0], content)

    # ------------------------------------------------------- fragments

    def content_pieces(self, start: float, end: float) -> list[tuple[float, float]]:
        """``[start, end)`` restricted to the corpus, without transition masks."""
        return subtract(clip(self.ranges, start, end), self.holes)

    def usable_pieces(self, start: float, end: float) -> list[tuple[float, float]]:
        """Content pieces without the black stretches -- what a model may see.

        Falls back to the content pieces when the black filter would leave nothing at
        all: an empty representation is worse than a black one.
        """
        pieces = self.content_pieces(start, end)
        usable = subtract(pieces, self.skips)
        return usable or pieces or [(start, end)]

    def content_length(self, start: float, end: float) -> float:
        """Length of ``[start, end)`` measured in content seconds."""
        return round(total(self.content_pieces(start, end)), 6)

    # ------------------------------------------------------- sampling

    def select(self, times) -> list[bool]:
        """Which of the given frame times may be fed to a model (list of bools)."""
        return [self.is_usable(float(t)) for t in times]

    def sample_content(self, start: float, end: float, count: int) -> list[float]:
        """``count`` times spread evenly over the usable content of a fragment.

        Spacing is measured on the content axis, so a transition mask or a black
        stretch inside the fragment shifts the samples instead of being sampled.
        """
        pieces = self.usable_pieces(start, end)
        length = total(pieces)
        step = length / count if count else 0.0
        out, position, offset = [], 0.0, 0.0
        for a, b in pieces:
            while offset < position + (b - a) - EPS and len(out) < count:
                out.append(a + (offset - position))
                offset += step
            position += b - a
        while len(out) < count:                      # rounding at the tail
            out.append(pieces[-1][1] - EPS)
        return out

    def frame_indices(self, start: float, end: float, count: int, fps: float,
                      last_frame: int | None = None) -> list[int]:
        """:meth:`sample_content` expressed as frame numbers of the source file."""
        indices = [int(round(t * fps)) for t in self.sample_content(start, end, count)]
        if last_frame is not None:
            indices = [min(i, last_frame) for i in indices]
        return indices
