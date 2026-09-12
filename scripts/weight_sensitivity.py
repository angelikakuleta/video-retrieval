"""Sensitivity of Recall@10 to the weights the signals are fused with (6.7.5).

    python scripts/weight_sensitivity.py --split test

Reads the full pipeline's run of every dataset and re-weighs the signal matrices
it stored. Nothing is recomputed on the recordings and no model is loaded, so a
grid of seventy-two reweightings for a series and forty-eight for VATEX takes
seconds.

Writes results/measurements/weight_sensitivity_<date>_<time>.json through
save_measurement:

    {dataset: {signal: {"range": float, "curve": [[w, r10], ...],
                        "order_kept": bool | null, "undefined_points": [w, ...]}}}

UNIT: ``range`` and the Recall of every point of the curve are SHARES of one --
0,047 means 4,7 percentage points. The table of the thesis prints the range in
percentage points and converts on the way (``tables.points``); the payload keeps
one unit throughout, because the reader of this JSON is a notebook written next
week.

Two controls travel with the measurement. ``w = 0`` of a signal's row has to
reproduce the run of ``full_no_<signal>`` and ``w = 1`` of the base row the run
of the baseline -- reached from the other direction, so a disagreement means the
stored matrices are not the ones that were scored. The signals are float16, so
the check reports the difference rather than asserting equality.

UNIT OF INFERENCE: a LEVEL of Recall@10 -- every point of a curve, and so the
range -- is a mean over queries. A DIFFERENCE between two configurations -- the
marginal contributions the ordering rests on, and the gap between an edge point
and the run it reproduces -- is read at the unit of inference of the dataset: per
episode for a series, per clip for VATEX. That is the rule the whole of chapter 6
is read by, and the episode of every query travels into the sweep for it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import runs as runs_module, sensitivity
from src.utils import experiments as exp
from src.utils.notebook import save_measurement


def relevance_of(run: dict) -> dict[int, set[str]]:
    """The relevance the run scored against, read from its own ``relevance.csv``.

    From the run and not recomputed: the sensitivity analysis re-weighs that
    run's own matrices, so it has to be judged against that run's own answer key.
    """
    import csv
    from collections import defaultdict

    out: dict[int, set[str]] = defaultdict(set)
    path = Path(run["dir"]) / "relevance.csv"
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            out[int(row["desc_id"])].add(row["fragment_id"])
    return out


def edge_labels(names: list[str]) -> dict[str, str]:
    """Which run each row has to reproduce at its edge.

    One map, read twice -- once for the LEVEL the edge is compared against and
    once for the per-query values the DIFFERENCE is formed from -- so the two
    halves of the control cannot end up pointing at different runs.
    """
    out = {name: exp.no_label(name) for name in sensitivity.added_signals_of(names)}
    out[exp.BASE_SIGNAL] = exp.BASE
    return out


def expected_edges(runs: dict, dataset: str, names: list[str],
                   metric: str) -> dict[str, float | None]:
    """Recall@10 the edge of each row should reproduce, from the runs themselves."""
    from src.evaluation.compare import query_mean

    def value(label):
        run = runs.get(label, {}).get(dataset)
        return None if run is None else query_mean(run["per_query"], metric)

    return {name: value(label) for name, label in edge_labels(names).items()}


def reference_recall(runs: dict, dataset: str, names: list[str],
                     metric: str) -> dict[str, dict[int, float]]:
    """The same runs read per query: ``{row: {desc_id: Recall@10}}``.

    The edge control is a difference between two configurations, and a difference
    at the unit of inference cannot be formed from two collection averages -- it
    needs the queries themselves, to be grouped into episodes.
    """
    out = {}
    for name, label in edge_labels(names).items():
        run = runs.get(label, {}).get(dataset)
        if run is None:
            continue
        out[name] = {int(record["desc_id"]): float(record[metric])
                     for record in run["per_query"]}
    return out


def units_of(run: dict, desc_ids: list[int], dataset: str) -> list[str] | None:
    """The episode of every query, in the order of the score matrix.

    ``None`` where the clip is the unit of inference (VATEX), which is what tells
    the sweep to average differences over queries instead of over episodes.
    """
    from src.evaluation.compare import inference_unit

    if inference_unit(dataset) != "episode":
        return None
    episodes = {int(record["desc_id"]): record["episode"]
                for record in run["per_query"]}
    missing = [desc_id for desc_id in desc_ids if desc_id not in episodes]
    if missing:
        raise ValueError(
            f"{len(missing)} queries of signals.npz are not in per_query.jsonl "
            f"(e.g. {missing[:3]}): the episode of every query has to be known "
            "before a difference can be read per episode")
    return [episodes[desc_id] for desc_id in desc_ids]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test",
                        help="the part the full pipeline was run on")
    parser.add_argument("--datasets", nargs="+", default=list(exp.DATASETS))
    parser.add_argument("--metric", default=f"recall@{sensitivity.METRIC_K}")
    args = parser.parse_args()

    runs = runs_module.load_runs(args.split)
    if exp.FULL not in runs:
        raise SystemExit(
            f"no run of {exp.FULL!r} on split {args.split!r}: the sensitivity "
            "analysis re-weighs the full pipeline's own signal matrices, so it "
            "has nothing to work on until that run exists.")

    data, controls = {}, {}
    for dataset in args.datasets:
        run = runs[exp.FULL].get(dataset)
        if run is None:
            print(f"{dataset}: no run of {exp.FULL!r} - skipped")
            continue
        signals = sensitivity.load_signals(run["dir"])
        relevance = relevance_of(run)
        units = units_of(run, signals["desc_ids"], dataset)
        swept = sensitivity.sweep(signals, relevance, units=units)
        data[dataset] = swept

        per_query = reference_recall(runs, dataset, signals["names"], args.metric)
        differences = {
            name: sensitivity.edge_difference(signals, relevance, name,
                                              values, units=units)
            for name, values in per_query.items()}
        controls[dataset] = sensitivity.edge_checks(
            swept, expected_edges(runs, dataset, signals["names"], args.metric),
            differences)

        points = sum(len(entry["curve"]) + len(entry["undefined_points"])
                     for entry in swept.values())
        unit = "episode" if units else "query"
        print(f"{dataset}: {len(signals['names'])} signals, {points} reweightings, "
              f"differences per {unit}")
        for name, entry in swept.items():
            check = controls[dataset][name]
            mark = {True: "ok", False: "DIVERGES", None: "no reference run"}[
                check["close"]]
            print(f"  {name:16} range {entry['range']}  order_kept="
                  f"{entry['order_kept']}  undefined={entry['undefined_points']}  "
                  f"edge w={check['edge']}: {mark}")

    if not data:
        raise SystemExit("no dataset had a run to analyse")
    save_measurement("weight_sensitivity", {
        "grid": data, "edge_controls": controls, "split": args.split,
        "metric": args.metric,
        # in the payload itself, not only in this file: whoever opens the JSON in
        # a month reads it there and nowhere else
        "unit": "Recall as a share of one; range is max - min of the curve, also "
                "a share (0.047 = 4.7 percentage points)",
        "inference_unit": "levels (curve, range) are means over queries; "
                          "differences (the marginal contributions order_kept "
                          "rests on, and the edge control) are means of the "
                          "per-episode differences for the series and of the "
                          "per-clip ones for VATEX",
        "undefined_points": "w at which a gated signal leaves a query with no "
                            "weight at all: every fragment then ties at rank one "
                            "and the query would count as a PERFECT hit, so the "
                            "point is excluded from the curve and from the range",
    })


if __name__ == "__main__":
    main()
