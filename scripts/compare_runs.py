"""Compares experiment variants across ready run directories. Development only.

    python scripts/compare_runs.py results/runs/E1-*_dev_* \
        --reference E1-A --simplicity E1-A E1-B E1-C \
        --contrast requirements:wymaga_scenerii \
        --out results/reports/e1_decision

Runs are grouped by their ``experiment`` label and narrowed to the datasets every
label covers; the loading and that narrowing live in
:mod:`src.evaluation.runs`, which is the only place either happens. The console
gets the main measure and the verdict; <out>.json carries the rest - per-dataset
means, paired differences with 95% intervals, the pooled interval and every
contrast as a difference of differences.

The DEVELOPMENT phase is all this script is for. A verdict and a simplicity order
are decisions about which variant to freeze, and the test phase has none to make:
its configuration is already frozen and its tables are assembled by the notebooks.

Intervals are computed for --metric alone; secondary measures are stored as plain
effect sizes. --contrast may be repeated; each takes ``requirements:<tag>``,
``complexity:<P|Z>``, ``kinetics_vocab:<yes|no>`` or ``identities:>=1``/``>=2``.
--control names variants reported but kept out of the decision.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import compare, runs as runs_module

#: the split this script works on. A verdict belongs to the development phase.
SPLIT = "dev"


def formatted(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def interval(entry) -> str:
    if entry["low"] is None:
        return f"{formatted(entry['mean'])} (n={entry['n']} - no interval)"
    return (f"{formatted(entry['mean'])} "
            f"[{formatted(entry['low'])}; {formatted(entry['high'])}] n={entry['n']}")


def expand(patterns: list[str]) -> list[str]:
    """Resolves shell wildcards in the run paths.

    PowerShell does not expand wildcards for a native program, so a pattern
    reaches this script unchanged; bash expands it beforehand and the paths
    arrive already resolved. Handling both here keeps one documented command
    working in either shell.

    A wildcard means "the runs that match", so a directory left behind by an
    interrupted run is reported and skipped. An explicitly named path is meant
    literally and still fails.
    """
    out: list[str] = []
    for pattern in patterns:
        if not any(char in pattern for char in "*?["):
            out.append(pattern)
            continue
        found = sorted(glob.glob(pattern))
        if not found:
            raise SystemExit(f"no run directory matches {pattern}")
        ready = [p for p in found if (Path(p) / "config.resolved.yaml").exists()]
        for path in sorted(set(found) - set(ready)):
            print(f"WARNING: {path} has no config.resolved.yaml"
                  " - interrupted run, skipped")
        if not ready:
            raise SystemExit(f"no finished run matches {pattern}")
        out += ready
    return out


def contrast_subsets(spec: str, datasets: list[str]) -> tuple[dict, dict, list[str]]:
    """The two sides of one contrast, per dataset, and where it can be run.

    A dataset whose tag matches nothing drops out ALONE. VATEX carries neither
    `wymaga_osoby` nor `wymaga_mimiki` by design, and dropping the whole contrast
    because of that would throw away the two series it is actually a claim about.
    """
    matching, rest, usable = {}, {}, []
    for dataset in datasets:
        found, other = compare.subset_ids(dataset, SPLIT, spec)
        if not found:
            print(f"WARNING: no query carries {spec!r} in {dataset}"
                  " - that dataset takes no part in this contrast")
            continue
        matching[dataset], rest[dataset] = found, other
        usable.append(dataset)
    return matching, rest, usable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="run directories to compare")
    parser.add_argument("--reference", required=True,
                        help="experiment label of the reference variant (e.g. E1-A)")
    parser.add_argument("--metric", default="recall@10")
    parser.add_argument("--simplicity", nargs="+", required=True,
                        help="the deciding labels, from the simplest to the most complex")
    parser.add_argument("--control", nargs="*", default=[],
                        help="labels reported but kept out of the decision "
                             "(e.g. E2-Ap, which only splits the effect in two)")
    parser.add_argument("--contrast", action="append", default=[],
                        help="confirmatory contrast; may be repeated")
    parser.add_argument("--out", default=None, help="output path without extension")
    args = parser.parse_args()

    by_label = runs_module.load_runs(SPLIT, expand(args.runs))
    if not by_label:
        raise SystemExit(f"no finished run of split {SPLIT!r} among the given paths - "
                         "this script compares development runs only")
    if args.reference not in by_label:
        raise SystemExit(f"no runs for the reference {args.reference!r}")

    controls = set(args.control)
    missing = set(by_label) - set(args.simplicity) - controls
    if missing:
        raise SystemExit("--simplicity must list every deciding label (pass --control "
                         f"for the rest); missing: {sorted(missing)}")

    labels, datasets, skipped = runs_module.available(by_label, sorted(by_label))
    for label, reason in skipped:
        print(f"note: {label} - {reason}")
    if args.reference not in labels or not datasets:
        raise SystemExit(f"the reference {args.reference!r} shares no dataset with "
                         "the other labels - nothing to compare")
    by_label = runs_module.select(by_label, labels, datasets)
    references = by_label[args.reference]
    candidates = {label: group for label, group in by_label.items()
                  if label != args.reference}

    results = {label: compare.compare_variant(group, references, args.metric)
               for label, group in candidates.items()}
    # a control variant is reported but does not decide anything: it is there to
    # split the measured effect, not to be chosen (chapter 6, experiment E2)
    deciding = {label: r for label, r in results.items() if label not in controls}
    verdict = compare.decide(deciding, [s for s in args.simplicity if s in by_label])

    contrasts: dict[str, dict] = {}
    for spec in args.contrast:
        matching, rest, usable = contrast_subsets(spec, datasets)
        if not usable:
            print(f"WARNING: {spec!r} matches no query in any dataset - contrast skipped\n")
            continue
        contrasts[spec] = {
            label: compare.contrast_variant(
                {d: group[d] for d in usable},
                {d: references[d] for d in usable},
                args.metric, matching, rest)
            for label, group in candidates.items()}

    # ---- report ------------------------------------------------------------
    # The console gets the main measure and the verdict, nothing else; the full
    # tables live in the JSON, which is what the notebook reads.
    print(f"\nsplit {SPLIT}, datasets: {', '.join(datasets)}")
    for label in list(args.simplicity) + sorted(controls):
        if label == args.reference:
            values = "  ".join(
                f"{ds}: {formatted(compare.query_mean(references[ds]['per_query'], args.metric))}"
                for ds in datasets)
            print(f"{label:<7} {values}   (reference)")
            continue
        if label not in results:
            continue
        values = "  ".join(f"{ds}: {formatted(results[label]['per_dataset'][ds]['value'])}"
                           for ds in datasets)
        control = "  (control)" if label in controls else ""
        print(f"{label:<7} {values}   pooled delta "
              f"{interval(results[label]['pooled'])}{control}")

    print(f"\nverdict: {verdict['winner']} - {verdict['reason']}")
    if verdict["provisional"]:
        print(f"provisional: the pooled interval rests on {verdict['series']} "
              "series, the rule needs two")
    for spec, entries in contrasts.items():
        print(f"contrast {spec}:")
        for label, entry in entries.items():
            mark = "confirmed" if compare.confirmed(entry) else "not confirmed"
            print(f"  {label:<7} {interval(entry['pooled'])}  {mark}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"metric": args.metric, "split": SPLIT,
                   "reference": args.reference, "contrast": args.contrast,
                   "control": sorted(controls), "datasets": datasets,
                   "skipped": [{"label": label, "reason": reason}
                               for label, reason in skipped],
                   "runs": {label: {ds: str(r["dir"]) for ds, r in group.items()}
                            for label, group in by_label.items()},
                   "results": results,
                   # keyed by contrast specification, because --contrast may be repeated
                   "contrasts": contrasts,
                   "verdict": verdict}
        Path(str(out) + ".json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved -> {out}.json")


if __name__ == "__main__":
    main()
