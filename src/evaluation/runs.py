"""Finding the run directories a comparison needs. The only loader there is.

Both the notebook and ``scripts/compare_runs.py`` used to carry their own copy
of "group by label and dataset, newest wins, intersect the datasets", and the
two copies had already drifted: the script refused to work when one label
covered fewer datasets than the reference, the notebook narrowed to the
intersection instead. Which of the two a number came from was not visible in
the number.

This module loads and selects. It computes nothing -- no metric, no difference,
no verdict.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from src.data import datasets as datasets_registry
from src.evaluation.compare import load_run
from src.utils.experiments import DATASETS


def _started(run: dict) -> str:
    """Sort key for picking the newest of several runs of one (label, dataset).

    Compared only within one such group, where the directory names share their
    prefix and therefore sort by the timestamp that closes them. ``created`` is
    written when a run FINISHES, so it only breaks ties.
    """
    return f"{Path(run['dir']).name}|{run.get('created', '')}"


def load_runs(split: str, run_dirs: list[str | Path] | None = None,
              log=print) -> dict[str, dict[str, dict]]:
    """``{label: {dataset: run}}`` for one split; the newest directory wins.

    Without ``run_dirs`` the whole of ``results/runs`` is read, which is what a
    notebook wants. With it only those directories are, which is what the script
    wants when its command line names them -- either way the grouping happens
    here and only here.

    A directory without ``metrics.json`` is an interrupted run: it is named and
    skipped, never half-read.
    """
    if run_dirs is None:
        run_dirs = sorted(datasets_registry.runs_dir().glob("*"))

    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        if not (run_dir / "metrics.json").exists():
            log(f"skipped {run_dir.name}: no metrics.json - interrupted run")
            continue
        run = load_run(run_dir)
        if run["split"] != split:
            continue
        grouped[run["experiment"]][run["dataset"]].append(run)

    out: dict[str, dict[str, dict]] = {}
    for label in sorted(grouped):
        out[label] = {}
        for dataset in sorted(grouped[label]):
            found = sorted(grouped[label][dataset], key=_started)
            out[label][dataset] = found[-1]
            for older in found[:-1]:
                log(f"skipped older run of {label}/{dataset}: {Path(older['dir']).name}")
    return out


def available(runs: dict[str, dict[str, dict]], labels) -> tuple[list[str], list[str],
                                                                list[tuple[str, str]]]:
    """Which of ``labels`` can be compared, on which datasets, and what fell out.

    Returns ``(labels, datasets, skipped)``: the labels that have runs, the
    datasets ALL of them cover, and ``(label, reason)`` for each one left out.
    Narrowing to the intersection rather than refusing is the point -- a table
    of two series is a result; an exception because VATEX has not been run yet
    is not.

    ``runs`` is what :func:`load_runs` returned. The plan writes this signature
    as ``available(labels)``; the mapping has to come from somewhere, and passing
    it keeps the function free of any idea where the runs live.
    """
    wanted = list(labels)
    skipped = [(label, "no run for this split") for label in wanted if label not in runs]
    present = [label for label in wanted if label in runs]
    if not present:
        return [], [], skipped

    shared = set.intersection(*(set(runs[label]) for label in present))
    if not shared:
        return [], [], skipped + [
            (label, f"covers only {sorted(runs[label])}, no dataset is shared")
            for label in present]

    # the order of the thesis first, anything unexpected after it
    ordered = [d for d in DATASETS if d in shared] + sorted(shared - set(DATASETS))
    for label in present:
        missing = sorted(set(runs[label]) - shared)
        if missing:
            skipped.append((label, f"has {missing} beyond the shared datasets, "
                                   "those runs take no part"))
    return present, ordered, skipped


def select(runs: dict[str, dict[str, dict]], labels: list[str],
           datasets: list[str]) -> dict[str, dict[str, dict]]:
    """The runs of exactly those labels on exactly those datasets."""
    return {label: {dataset: runs[label][dataset] for dataset in datasets}
            for label in labels}
