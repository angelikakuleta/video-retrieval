"""How much the result depends on the weights the signals are fused with (6.7.5).

The pipeline gives every active signal the same weight, 1/|A(q)|. That is a
choice, and the question this module answers is how much of the result rests on
it: if a plausible reweighting moves Recall@10 by a tenth of a point the choice
carries nothing, and if it moves it by five the thesis is about the weights.

Nothing is recomputed on the recordings. A run stores its standardized signal
matrices (``signals.npz``), so a grid point is a weighted sum over values that
already exist -- seconds instead of hours, and the same numbers the run itself
scored, not a re-extraction that might differ.

Two things in here are less obvious than they look.

**The point w = 1 of a gated signal is undefined, and keeping it would FLATTER
the result rather than depress it.** At w = 1 every other signal has weight zero,
so a query the signal is inactive for gets a total weight of zero and
``fusion.query_weights`` hands it a row of zeros over the whole collection. A
rank here is the number of fragments scoring HIGHER, plus one -- and when every
score is zero, nothing scores higher than anything: every fragment ties at rank
one, the correct one included. That query enters the average as a PERFECT HIT at
every k, having been scored on no information at all. A zero would at least look
wrong; this looks like a result. The point is marked and left out of the curve
and the range.

**"Full without k" at non-uniform weights rescales the rest proportionally.**
Literally ``query_weights`` with the base vector of the grid point and the mask
with k zeroed. Falling back to uniform weights would change two things at once --
the signal removed AND the weighting of the others -- and would report the sum of
the two as the contribution of the one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.evaluation.compare import mean_of_differences
from src.retrieval.fusion import query_weights
from src.utils.experiments import ADDED_SIGNALS, BASE_SIGNAL

#: the grid of chapter 4: eleven points from nothing to everything. The uniform
#: weight 1/|A| joins it per dataset, and lands on it only when it happens to be
#: a tenth. Neither collection in play is such a case: a series carries six
#: signals (1/6) and VATEX four (1/4 = 0,25, and the grid steps by a tenth), so
#: every row of every dataset gets a twelfth point of its own.
GRID: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(11))

#: the metric the analysis is read at, the one chapter 6 reports
METRIC_K = 10

#: signal weights are stored as float16, so a reproduced number is close, never
#: equal. Above this difference in Recall@10 the edge control has found a real
#: disagreement rather than the rounding of the store.
EDGE_TOLERANCE = 0.005


def grid_for(n_signals: int) -> list[float]:
    """The grid of one row: the eleven points plus the uniform weight.

    Twelve points whenever the uniform weight is not itself a tenth, which is
    the case for both collections here -- 1/6 for a series and 1/4 for VATEX --
    and eleven when it is. That is where the 72 reweightings of a series
    (6 x 12) and the 48 of VATEX (4 x 12) come from.
    """
    if n_signals < 1:
        raise ValueError("a collection scored by no signal has no weights to vary")
    uniform = round(1.0 / n_signals, 6)
    points = set(GRID) | {uniform}
    return sorted(points)


def weight_vector(n_signals: int, row: int, w: float) -> np.ndarray:
    """``w`` on the signal under test, ``1 - w`` shared equally by the others."""
    if n_signals > 1:
        base = np.full(n_signals, (1.0 - w) / (n_signals - 1), dtype=np.float64)
    else:
        base = np.zeros(n_signals, dtype=np.float64)
    base[row] = w
    return base


def scores_at(values: np.ndarray, active: np.ndarray, base: np.ndarray) -> np.ndarray:
    """The (n_queries, n_fragments) score matrix at one point of the grid."""
    weights = query_weights(active, base)
    total = np.zeros(values.shape[1:], dtype=np.float64)
    for matrix, row in zip(values, weights):
        total += row[:, None] * matrix
    return total


def per_query_hits(scores: np.ndarray, fragments, relevant,
                   k: int = METRIC_K) -> tuple[list[int], list[float]]:
    """``(rows, hits)`` for the queries that have a correct fragment at all.

    ``rows`` are their positions in the score matrix and ``hits`` their Recall@k,
    zero or one. A query with nothing correct in the collection is left out of
    both, so which queries are counted depends on the RELEVANCE alone and not on
    the weights -- that is what lets two grid points be subtracted query by query.

    Rank by counting the fragments that score higher, which is what
    ``metrics._ranks_from_matrix`` does -- a full sort per query would be the
    expensive part of a grid of seventy-two points.
    """
    index = {name: column for column, name in enumerate(fragments)}
    rows, hits = [], []
    for position, (row, wanted) in enumerate(zip(scores, relevant)):
        columns = [index[name] for name in wanted if name in index]
        if not columns:
            continue                       # nothing correct is in the collection
        best = min(int((row > row[column]).sum()) + 1 for column in columns)
        rows.append(position)
        hits.append(1.0 if best <= k else 0.0)
    return rows, hits


def recall_of(scores: np.ndarray, fragments, relevant, k: int = METRIC_K) -> float | None:
    """Recall@k over the queries that have a correct fragment in the collection.

    A LEVEL, so it is the mean over queries -- the curve and its range are read
    this way throughout the thesis. Differences between two points are a separate
    matter and go through :func:`mean_of_differences`.
    """
    _, hits = per_query_hits(scores, fragments, relevant, k)
    return (sum(hits) / len(hits)) if hits else None


def marginal_contributions(values, active, base, fragments, relevant,
                           names: list[str], k: int = METRIC_K,
                           units: list[str] | None = None) -> dict[str, float]:
    """What each added signal is worth AT THIS POINT of the grid.

    ``full - full without k``, where "without k" keeps the weights of the point
    and only zeroes k's row of the activity mask, so ``query_weights`` shares its
    weight out proportionally. That is one change, not two.

    A contribution is a difference between two configurations, so it is read at
    the UNIT OF INFERENCE: ``units`` names the episode of every query, in the
    order of the score matrix, and the per-query differences are averaged inside
    each episode before being averaged across episodes
    (:func:`compare.mean_of_differences`). ``None`` leaves the query as the unit,
    which is what a VATEX clip is. The ordering this feeds
    (:func:`order_is_kept`) is therefore the ordering of the same quantity the
    contribution tables of chapter 6 print, and not of a second one.
    """
    rows, full = per_query_hits(scores_at(values, active, base), fragments, relevant, k)
    at_unit = None if units is None else [units[row] for row in rows]
    out = {}
    for row, name in enumerate(names):
        if name == BASE_SIGNAL:
            continue
        mask = active.copy()
        mask[row] = False
        other, without = per_query_hits(scores_at(values, mask, base), fragments,
                                        relevant, k)
        if other != rows:
            raise ValueError(
                "the two grid points count different queries, which cannot happen: "
                "a query is counted when the collection holds something correct "
                "for it, and reweighting does not change that")
        out[name] = mean_of_differences([a - b for a, b in zip(full, without)],
                                        at_unit)
    return out


def order_is_kept(reference: dict[str, float], found: dict[str, float],
                  signal: str | None = None) -> bool:
    """Whether the contributions still order the way the uniform weights did.

    Ties count as kept: the question is whether an order CONSISTENT with the
    reference exists, so the comparison is ``>=`` down the reference order rather
    than equality of positions. Two signals that contribute the same amount are
    not evidence that their order changed.

    With ``signal`` only that one signal's position is checked -- everything the
    reference puts above it must still be at least as high, and everything below
    at most as low. Without it the whole order is checked, which is what the
    baseline row asks: the base has no position of its own among the signals it
    is being weighed against.
    """
    usable = [name for name in reference
              if reference[name] is not None and found.get(name) is not None]
    order = sorted(usable, key=lambda name: -reference[name])
    if signal is not None:
        if signal not in usable:
            return True                    # nothing to keep or lose
        at = order.index(signal)
        return (all(found[name] >= found[signal] for name in order[:at])
                and all(found[signal] >= found[name] for name in order[at + 1:]))
    return all(found[a] >= found[b] for a, b in zip(order, order[1:]))


def load_signals(run_dir: Path | str) -> dict:
    """``names``, ``desc_ids``, ``fragments``, ``active``, ``values`` of one run."""
    with np.load(Path(run_dir) / "signals.npz", allow_pickle=False) as data:
        return {"names": [str(name) for name in data["names"]],
                "desc_ids": [int(i) for i in data["desc_ids"]],
                "fragments": [str(name) for name in data["fragments"]],
                "active": np.asarray(data["active"], dtype=bool),
                "values": np.asarray(data["values"], dtype=np.float64)}


def sweep(signals: dict, relevance: dict[int, set[str]], k: int = METRIC_K,
          units: list[str] | None = None) -> dict:
    """The whole grid of one dataset: one row per signal, the base included.

    Returns ``{signal: {"range", "curve", "order_kept", "undefined_points"}}``.

    UNIT: ``range`` and the Recall of every point of ``curve`` are SHARES of one,
    like every other Recall in this repository -- 0,047 means 4,7 percentage
    points. Section 12 of the plan defines the table COLUMN in percentage points
    and that is what the table prints (``tables.points``); storing one number of
    a payload in a different unit from the rest is how a result ends up wrong by
    a factor of a hundred.

    UNIT OF INFERENCE: the curve, and with it ``range``, is a LEVEL and stays a
    mean over queries. ``order_kept`` rests on marginal contributions, which are
    DIFFERENCES between two configurations, so ``units`` -- the episode of every
    query, in the order of the score matrix -- makes them means of the
    per-episode differences instead. ``None`` leaves the query as the unit, which
    is what a VATEX clip is.
    """
    names = signals["names"]
    values, active = signals["values"], signals["active"]
    fragments = signals["fragments"]
    wanted = [relevance.get(desc_id, set()) for desc_id in signals["desc_ids"]]
    grid = grid_for(len(names))

    uniform = marginal_contributions(values, active, None, fragments, wanted,
                                     names, k, units)

    out: dict[str, dict] = {}
    for row, name in enumerate(names):
        curve, undefined, kept = [], [], True
        for w in grid:
            base = weight_vector(len(names), row, w)
            # A query the signal is inactive for has no weight left at w = 1,
            # so its whole score row is zero -- and then nothing outranks
            # anything, every fragment ties at rank one, and the query counts as
            # a perfect hit. The point does not lower the curve; it lifts it.
            if w >= 1.0 and not active[row].all():
                undefined.append(w)
                continue
            value = recall_of(scores_at(values, active, base), fragments, wanted, k)
            if value is None:
                undefined.append(w)
                continue
            curve.append([w, value])
            if w >= 1.0:
                continue        # every contribution but this signal's is zero here
            found = marginal_contributions(values, active, base, fragments,
                                           wanted, names, k, units)
            kept = kept and order_is_kept(uniform, found,
                                          None if name == BASE_SIGNAL else name)
        reached = [value for _, value in curve]
        out[name] = {
            "range": (max(reached) - min(reached)) if reached else None,
            "curve": curve,
            "order_kept": (kept if len(uniform) > 1 and reached else None),
            "undefined_points": undefined,
        }
    return out


def edge_of(name: str) -> float:
    """The weight at which a row has to reproduce a run of its own.

    ``w = 0`` for an added signal -- it is switched off there and the point has
    to be ``full_no_<signal>`` -- and ``w = 1`` for the base row, which is then
    alone and has to be the baseline.
    """
    return 1.0 if name == BASE_SIGNAL else 0.0


def edge_difference(signals: dict, relevance: dict[int, set[str]], name: str,
                    reference: dict[int, float], k: int = METRIC_K,
                    units: list[str] | None = None) -> float | None:
    """The edge point of one row against the run it has to reproduce.

    A difference between two configurations, so it is read at the unit of
    inference like every other one: ``units`` names the episode of every query,
    in the order of the score matrix, and ``None`` leaves the query as the unit.
    ``reference`` is that run's Recall@k per ``desc_id``.

    ``None`` when there is no reference run, or when the edge is the undefined
    ``w = 1`` of a gated signal -- the point that would count every query as a
    perfect hit and is left out of the curve for the same reason.
    """
    names = signals["names"]
    row = names.index(name)
    edge = edge_of(name)
    if edge >= 1.0 and not signals["active"][row].all():
        return None
    wanted = [relevance.get(desc_id, set()) for desc_id in signals["desc_ids"]]
    scores = scores_at(signals["values"], signals["active"],
                       weight_vector(len(names), row, edge))
    rows, hits = per_query_hits(scores, signals["fragments"], wanted, k)
    shared = [(position, hit) for position, hit in zip(rows, hits)
              if signals["desc_ids"][position] in reference]
    if not shared:
        return None
    return mean_of_differences(
        [hit - reference[signals["desc_ids"][position]] for position, hit in shared],
        None if units is None else [units[position] for position, _ in shared])


def edge_checks(swept: dict, expected: dict[str, float | None],
                differences: dict[str, float | None] | None = None) -> dict:
    """Does ``w = 0`` reproduce ``full_no_<signal>``, and ``w = 1`` the baseline?

    Both are the same computation reached from two directions, so a disagreement
    means the stored signals are not the ones the run scored. float16 makes exact
    equality impossible, so the difference travels with the verdict instead of
    being asserted away.

    ``computed`` and ``expected`` are LEVELS and stay means over queries. What
    separates them is a difference between two configurations, so the caller
    passes it in ``differences``, read at the unit of inference
    (:func:`edge_difference`). Without it the query-level subtraction is used,
    which is the right reading only where the query is the unit.
    """
    out = {}
    for name, entry in swept.items():
        at = {round(w, 6): value for w, value in entry["curve"]}
        edge = edge_of(name)
        computed, target = at.get(edge), expected.get(name)
        if differences is None:
            difference = (None if computed is None or target is None
                          else computed - target)
        else:
            difference = differences.get(name)
        out[name] = {"edge": edge, "computed": computed, "expected": target,
                     "difference": difference,
                     "close": (None if difference is None
                               else abs(difference) <= EDGE_TOLERANCE)}
    return out


def added_signals_of(names) -> list[str]:
    """The signals a row of the table exists for, in the order of chapter 6."""
    return [name for name in ADDED_SIGNALS if name in set(names)]
