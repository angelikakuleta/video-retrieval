# -*- coding: utf-8 -*-
"""How incomplete annotation would have to be to change the ordering of contributions.

    python scripts/pool_sensitivity.py
    python scripts/pool_sensitivity.py --share tbbt=0.11 office=0.05 --draws 200

This is NOT the manual assessment of the pool and does not stand in for it. The
pool is built (data/interim/<dataset>/work/<dataset>_pool_judgments_new.csv)
and unjudged; what this
script does instead is answer the question the assessment was there to answer --
"could missing relevant fragments overturn the ordering of the contributions?" --
by assumption rather than by judging.

For a stated share of pooled queries it marks one or two of their pooled
fragments relevant, drawn at random, recomputes Recall@10 of every configuration
over the pooled queries, and reports whether the ordering of the contributions
survives. Every quantity it prints is conditional on that assumption, and the
sweep says how far the assumption has to be pushed before the ordering breaks.

Writes results/measurements/pool_sensitivity_<date>_<time>.json.

UNIT OF INFERENCE: a contribution is a difference between two configurations and
is read the way chapter 6 reads every one of them -- for the series as the mean
of the per-episode differences, not of the per-query ones. Only Recall@10 LEVELS
stay means over queries.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import compare, pool, runs as runs_module  # noqa: E402
from src.utils import experiments as exp                       # noqa: E402
from src.utils.notebook import save_measurement                # noqa: E402

SPLIT = "test"

#: the contributions whose ordering is under test: signal added to the baseline.
#: Kept to the five signals of tab:wklady-zbiorczo, in its order.
INDIVIDUAL = (("opisy scen", "E3-C"), ("obiekty", "E4-B"), ("ruch", "E2-C"),
              ("twarze", "E5-B"), ("tozsamosc", "E6-B"))


def parse_shares(pairs: list[str]) -> dict[str, float]:
    out = {}
    for item in pairs:
        name, _, value = item.partition("=")
        out[name] = float(value)
    return out


def draw_judgments(rows, share: float, per_query: tuple[int, int],
                   rng: random.Random) -> list[dict]:
    """Mark a share of the pooled queries as having 1-2 relevant fragments.

    The draw is over QUERIES first and fragments second, because that is the
    shape the assumption is stated in: "for this share of queries the judge
    found one or two". Drawing over fragments instead would spread the same
    count over more queries and understate the effect.
    """
    by_query: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_query[str(row["desc_id"])].append(row)

    queries = sorted(by_query)
    chosen = rng.sample(queries, round(share * len(queries)))
    # every pooled fragment appears, most of them rejected: pool_ids() takes the
    # query set from the judgments, so a list holding only the accepted ones
    # would silently narrow the recomputation to the queries that gained something
    judgments = [{"desc_id": row["desc_id"], "fragment": row["fragment"],
                  "relevant": "no"} for row in rows]
    accepted = set()
    for desc_id in chosen:
        available = by_query[desc_id]
        wanted = min(len(available), rng.randint(*per_query))
        for row in rng.sample(available, wanted):
            accepted.add((desc_id, str(row["fragment"])))
    for row in judgments:
        if (str(row["desc_id"]), str(row["fragment"])) in accepted:
            row["relevant"] = "yes"
    return judgments


def ordering(values: dict[str, float]) -> tuple[str, ...]:
    """Signal names sorted by contribution, best first."""
    return tuple(name for name, _ in sorted(values.items(), key=lambda kv: -kv[1]))


def episodes_of(run) -> dict[int, str]:
    """``{desc_id: episode}`` of one run, for grouping differences."""
    return {int(record["desc_id"]): record["episode"] for record in run["per_query"]}


def contributions(runs, dataset, judgments) -> dict[str, float] | None:
    """Delta of every individual contribution over the pooled queries.

    A difference between two configurations, so it is read at the unit of
    inference of the dataset: for a series the per-query differences are averaged
    inside each episode first (``compare.mean_of_differences``), exactly as every
    contribution in chapter 6 is. Reading it per query here and per episode in
    the contribution table would put the ordering under test and the ordering
    being defended one step apart, which is the one thing this script must not do.
    """
    base = runs.get(exp.BASE, {}).get(dataset)
    if base is None:
        return None
    reference = pool.recomputed_hits(base, judgments)
    if not reference:
        return None
    episodes = (episodes_of(base) if compare.inference_unit(dataset) == "episode"
                else None)
    out = {}
    for name, label in INDIVIDUAL:
        run = runs.get(label, {}).get(dataset)
        if run is None:
            continue
        after = pool.recomputed_hits(run, judgments)
        shared = sorted(set(after) & set(reference))
        if not shared:
            continue
        out[name] = 100 * compare.mean_of_differences(
            [after[desc_id][1] - reference[desc_id][1] for desc_id in shared],
            None if episodes is None else [episodes[desc_id] for desc_id in shared])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--share", nargs="+", default=["tbbt=0.11", "office=0.05"],
                        help="assumed share of pooled queries with a missed "
                             "relevant fragment, per dataset")
    parser.add_argument("--per-query", type=int, nargs=2, default=(1, 2),
                        help="how many fragments such a query gains")
    parser.add_argument("--draws", type=int, default=200,
                        help="random draws per share; the spread over draws says "
                             "whether the answer depends on which fragments")
    parser.add_argument("--sweep", type=float, nargs="+",
                        default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
                        help="shares for the breaking-point sweep")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    shares = parse_shares(args.share)
    per_query = tuple(args.per_query)
    runs = runs_module.load_runs(SPLIT)
    report = {"split": SPLIT, "per_query": list(per_query), "draws": args.draws,
              "seed": args.seed, "assumed_share": shares, "datasets": {}}

    for dataset in sorted(shares):
        rows = pool.load_judgments(pool.pool_file(dataset, SPLIT))
        if not rows:
            print(f"{dataset}: no pool on disk - run test_summary section 7 first")
            continue
        queries = {str(row["desc_id"]) for row in rows}
        print(f"\n=== {dataset}: pool of {len(rows)} fragments over "
              f"{len(queries)} queries ===")

        # "no assumption" still needs the pooled query set, so the whole pool
        # goes in rejected: adding nothing, but naming the queries to score on
        zero = [{"desc_id": row["desc_id"], "fragment": row["fragment"],
                 "relevant": "no"} for row in rows]
        baseline = contributions(runs, dataset, zero)
        if not baseline:
            print(f"  {dataset}: no baseline run, skipped")
            continue
        order0 = ordering(baseline)
        print("  without any assumption: "
              + ", ".join(f"{name} {baseline[name]:+.1f}" for name in order0))

        entry = {"pool_fragments": len(rows), "pool_queries": len(queries),
                 "baseline": baseline, "baseline_order": list(order0), "sweep": []}

        for share in sorted({*args.sweep, shares[dataset]}):
            rng = random.Random(args.seed)
            kept, flips = [], 0
            for _ in range(args.draws):
                judgments = draw_judgments(rows, share, per_query, rng)
                values = contributions(runs, dataset, judgments)
                if not values:
                    continue
                kept.append(values)
                flips += ordering(values) != order0
            if not kept:
                continue
            mean = {name: sum(v[name] for v in kept) / len(kept) for name in baseline}
            # the exact permutation is a brittle test where two contributions sit
            # within a rounding of each other, and two of them do. What the claim
            # actually rests on is coarser: does any signal cross zero, and does
            # the best one stay best.
            signs = sum(any((v[n] > 0) != (baseline[n] > 0) for n in baseline)
                        for v in kept)
            tops = sum(ordering(v)[0] != order0[0] for v in kept)
            shift = max(abs(v[n] - baseline[n]) for v in kept for n in baseline)
            mark = "  <- zalozenie" if abs(share - shares[dataset]) < 1e-9 else ""
            print(f"  share {share:5.0%}: znak {signs / len(kept):5.1%}, "
                  f"najlepszy {tops / len(kept):5.1%}, "
                  f"permutacja {flips / len(kept):5.1%}, max |zmiana| {shift:4.1f}; "
                  + ", ".join(f"{name} {mean[name]:+.1f}" for name in order0) + mark)
            entry["sweep"].append({"share": share, "draws": len(kept),
                                   "order_changed": flips / len(kept),
                                   "sign_changed": signs / len(kept),
                                   "best_changed": tops / len(kept),
                                   "max_abs_shift": round(shift, 2),
                                   "mean_contributions": mean})
        report["datasets"][dataset] = entry

    save_measurement("pool_sensitivity", report)


if __name__ == "__main__":
    main()
