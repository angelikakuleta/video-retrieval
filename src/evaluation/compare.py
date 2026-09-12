"""Comparing runs: paired differences with intervals over the right unit.

Chapter 4: queries of one episode are correlated, so for the series the EPISODE
is the unit of inference -- the metric is averaged per episode in each run, the
paired per-episode differences are formed, and a t interval is computed
separately per series and on their pooled episodes. VATEX clips are independent,
so there the interval comes from a bootstrap over clips and VATEX never joins the
pool.

Decision rule: a candidate wins when the direction agrees in every series AND the
pooled interval excludes zero. Otherwise the simpler variant wins, which is an
engineering decision and not a claim of equivalence.

A confirmatory contrast asks something narrower -- whether the effect
concentrates on the tagged queries -- so it is a difference of differences with
its own interval (:func:`contrast_variant`).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import yaml

CONFIDENCE = 0.95
#: bootstrap replications for the clip-level interval, as in chapter 4
REPLICATIONS = 10_000
#: datasets whose observations are independent clips rather than episodes
CLIP_UNIT_DATASETS = ("vatex",)


def inference_unit(dataset: str) -> str:
    """``"clip"`` for VATEX, ``"episode"`` for the series."""
    return "clip" if dataset in CLIP_UNIT_DATASETS else "episode"


# -------------------------------------------------------------- Reading runs
def load_run(run_dir: Path | str) -> dict:
    """Loads the artifacts of one run directory."""
    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    per_query = [json.loads(line) for line
                 in (run_dir / "per_query.jsonl").read_text(encoding="utf-8").splitlines()]
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return {"dir": run_dir, "config": config, "per_query": per_query,
            "metrics": metrics, "experiment": config["experiment"],
            "dataset": config["dataset"], "split": config["split"]}


def episode_means(per_query: list[dict], metric: str,
                  desc_ids: set[int] | None = None) -> dict[str, float]:
    """Mean of the metric per episode, optionally restricted to some queries."""
    groups: dict[str, list[float]] = defaultdict(list)
    for record in per_query:
        if desc_ids is not None and record["desc_id"] not in desc_ids:
            continue
        groups[record["episode"]].append(record[metric])
    return {episode: sum(values) / len(values) for episode, values in groups.items()}


def query_mean(per_query: list[dict], metric: str,
               desc_ids: set[int] | None = None) -> float | None:
    values = [r[metric] for r in per_query
              if desc_ids is None or r["desc_id"] in desc_ids]
    return sum(values) / len(values) if values else None


# --------------------------------------------- Selecting a subset of queries
#: how a subset may be named. `identities` counts the profiled characters a
#: query calls up, which after the author's decision is exactly what the
#: `wymaga_osoby` tag means -- so E6 needs no conjunction of specifications.
SUBSET_KINDS = ("requirements", "complexity", "kinetics_vocab", "identities")


def subset_ids(dataset: str, split: str, spec: str) -> tuple[set[int], set[int]]:
    """Splits the queries of a dataset in two according to ``spec``.

    ``requirements:<tag>``, ``complexity:<P|Z>``, ``kinetics_vocab:<yes|no>``
    (VATEX only) or ``identities:>=1`` / ``identities:>=2``; a bare tag name is
    read as a requirement tag. A value outside the vocabulary is refused rather
    than quietly matching nothing.

    An EMPTY matching side is a legitimate answer and not an error: VATEX carries
    no `wymaga_osoby` and no `wymaga_mimiki`, because a one-sentence clip caption
    settles neither. The caller decides whether an empty subset means "skip this
    dataset" or "the tag file is not filled in yet".
    """
    from src.data.queries import load_queries
    from src.utils.vocabulary import COMPLEXITY_VALUES, REQUIREMENT_TAGS

    kind, _, value = spec.partition(":")
    if not value:
        kind, value = "requirements", kind
    if kind not in SUBSET_KINDS:
        raise ValueError(f"unknown subset {spec!r}: expected one of "
                         f"{[f'{k}:' for k in SUBSET_KINDS]}")

    queries = load_queries(dataset, split)
    everything = {q.desc_id for q in queries}

    if kind == "requirements":
        _check_value(kind, value, REQUIREMENT_TAGS)
        matching = {q.desc_id for q in queries if value in (q.requirements or [])}
    elif kind == "complexity":
        _check_value(kind, value, COMPLEXITY_VALUES)
        matching = {q.desc_id for q in queries if q.complexity == value}
    elif kind == "kinetics_vocab":
        _check_value(kind, value, ["yes", "no"])
        if dataset != "vatex":
            raise ValueError("kinetics_vocab applies to VATEX only, not "
                             f"{dataset!r}: only a clip carries a Kinetics class")
        from src.utils import vatex

        known = vatex.k400_membership(split)
        matching = {q.desc_id for q in queries
                    if known.get(q.vid_name) is (value == "yes")}
    else:
        _check_value(kind, value, [">=1", ">=2"])
        least = int(value[2:])
        matching = {q.desc_id for q in queries if len(q.identities or []) >= least}
    return matching, everything - matching


def _check_value(kind: str, value: str, allowed) -> None:
    if value not in allowed:
        raise ValueError(f"unknown {kind} value {value!r}: expected one of {list(allowed)}")


# ---------------------------------------------------------------- Statistics
def t_interval(diffs: list[float], confidence: float = CONFIDENCE) -> dict:
    """Mean difference with a t interval; the episode is the cluster unit."""
    n = len(diffs)
    mean = sum(diffs) / n if n else None
    if n < 2:
        return {"n": n, "mean": mean, "low": None, "high": None, "method": "t"}
    from scipy.stats import t

    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    half = t.ppf(0.5 + confidence / 2, n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "low": mean - half, "high": mean + half, "method": "t"}


def bootstrap_interval(diffs: list[float], confidence: float = CONFIDENCE,
                       replications: int = REPLICATIONS, seed: int = 1234) -> dict:
    """Percentile bootstrap over independent observations (VATEX clips).

    ``diffs`` are the paired per-clip differences, so a replication draws the same
    clips for both configurations.
    """
    n = len(diffs)
    mean = sum(diffs) / n if n else None
    if n < 2:
        return {"n": n, "mean": mean, "low": None, "high": None, "method": "bootstrap"}
    import numpy as np

    values = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, n, size=(replications, n))].mean(axis=1)
    tail = (1.0 - confidence) / 2 * 100
    return {"n": n, "mean": mean,
            "low": float(np.percentile(draws, tail)),
            "high": float(np.percentile(draws, 100 - tail)),
            "method": "bootstrap"}


def interval(diffs: list[float], unit: str = "episode") -> dict:
    """The interval appropriate for the unit of inference."""
    return bootstrap_interval(diffs) if unit == "clip" else t_interval(diffs)


def two_sample_interval(a: list[float], b: list[float],
                        confidence: float = CONFIDENCE) -> dict:
    """Difference of two INDEPENDENT group means, pooled variance, df = na + nb - 2.

    Used where the two sides are different queries rather than the same query
    twice -- research question PB3 compares groups, not paired runs, so the
    paired interval of :func:`t_interval` would be the wrong instrument.

    Deliberately NOT Welch: the thesis states one pooled degrees-of-freedom
    figure, and a per-comparison df that moves with the variance ratio would not
    be the number it reports. Where the group variances differ sharply this
    interval is the optimistic one, and that is a known, stated choice.
    """
    na, nb = len(a), len(b)
    empty = {"n": na + nb, "n_a": na, "n_b": nb, "mean": None,
             "low": None, "high": None, "method": "t2"}
    if na < 2 or nb < 2:
        return empty
    mean_a, mean_b = sum(a) / na, sum(b) / nb
    var_a = sum((x - mean_a) ** 2 for x in a) / (na - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (nb - 1)
    df = na + nb - 2
    pooled = ((na - 1) * var_a + (nb - 1) * var_b) / df
    error = math.sqrt(pooled * (1.0 / na + 1.0 / nb))
    difference = mean_a - mean_b
    if error == 0.0:
        return {**empty, "mean": difference, "low": difference, "high": difference,
                "df": df}
    from scipy.stats import t

    half = t.ppf(0.5 + confidence / 2, df) * error
    return {"n": na + nb, "n_a": na, "n_b": nb, "mean": difference,
            "low": difference - half, "high": difference + half,
            "df": df, "method": "t2"}


# --------------------------------------------------------------- Differences
def _episode_diff(candidate: dict, reference: dict, metric: str,
                  desc_ids: set[int] | None = None) -> dict[str, float]:
    """Per-episode differences candidate - reference (shared episodes only)."""
    a = episode_means(candidate["per_query"], metric, desc_ids)
    b = episode_means(reference["per_query"], metric, desc_ids)
    return {episode: a[episode] - b[episode] for episode in sorted(set(a) & set(b))}


def _query_diff(candidate: dict, reference: dict, metric: str,
                desc_ids: set[int] | None = None) -> list[float]:
    """Per-query differences candidate - reference (shared queries only)."""
    b = {r["desc_id"]: r[metric] for r in reference["per_query"]}
    return [r[metric] - b[r["desc_id"]] for r in candidate["per_query"]
            if r["desc_id"] in b and (desc_ids is None or r["desc_id"] in desc_ids)]


#: the measure the transition table is built on. Recall@10 of one query is
#: zero or one by construction (chapter 4: one query, one event), which is what
#: makes the four groups exhaustive and disjoint.
TRANSITION_METRIC = "recall@10"


def transitions(candidate: dict, reference: dict, desc_ids: set[int] | None = None,
                metric: str = TRANSITION_METRIC) -> dict[str, int]:
    """How many queries the signal fixed, broke and left alone.

    ``{"0->1", "1->0", "1->1", "0->0"}`` over the queries both runs answered. A
    mean difference says how MUCH a signal helps; this says WHOM, and with the
    closed-vocabulary signals being rare and sharp (a single detection carries a
    fragment to the top) the two are not the same question:
    ``Delta_ind = (0->1 - 1->0) / n`` can be small because little moved or
    because a lot moved both ways.

    A query one run answered and the other did not is REFUSED, not dropped. The
    four groups are the appendix table and their sum is its ``n`` column; a
    silently shorter sum would be a smaller n that still looked like the whole
    collection. Narrow the scope with ``desc_ids`` when that is what is meant.
    """
    before = {r["desc_id"]: r[metric] for r in reference["per_query"]}
    after = {r["desc_id"]: r[metric] for r in candidate["per_query"]}
    scope = (set(before) | set(after)) if desc_ids is None else set(desc_ids)
    only_candidate = sorted(scope - set(before))
    only_reference = sorted(scope - set(after))
    if only_candidate or only_reference:
        raise ValueError(
            f"the two runs answer different queries: {len(only_candidate)} only in "
            f"the candidate, {len(only_reference)} only in the reference "
            f"(e.g. {(only_candidate + only_reference)[:3]}). The transition counts "
            "are the n of a table, so pass desc_ids to say which queries are meant.")

    counts = {"0->1": 0, "1->0": 0, "1->1": 0, "0->0": 0}
    for record in candidate["per_query"]:
        desc_id = record["desc_id"]
        if desc_id not in scope:
            continue
        was, now = before[desc_id], record[metric]
        if was not in (0, 1) or now not in (0, 1):
            raise ValueError(
                f"{metric!r} is not binary for query {desc_id} ({was} -> {now}); "
                "the transition table counts queries, not partial credit")
        counts[f"{int(was)}->{int(now)}"] += 1
    return counts


#: ranks counted as lying AT the cutoff of the transition metric: K = 10 with
#: three places of margin either way. Wider and "borderline" would stop meaning
#: borderline; narrower and a query that moved from 9 to 11 would fall outside
#: the band that exists to catch exactly that move.
BORDERLINE_BAND: tuple[int, int] = (7, 14)


def borderline(candidate: dict, reference: dict, desc_ids: set[int] | None = None,
               metric: str = TRANSITION_METRIC,
               band: tuple[int, int] = BORDERLINE_BAND) -> int:
    """How many of the changed queries only crossed the cutoff, not the ranking.

    Counts the queries of ``0->1`` and ``1->0`` whose relevant fragment sits
    inside ``band`` in BOTH runs. Such a query changed status because the cutoff
    of the measure happens to fall between two nearly identical rankings, not
    because the candidate found something the reference did not.

    This is the second half of what :func:`transitions` says. A comparison with
    many changed queries and few borderline ones moved the ranking; the reverse
    means the difference would largely disappear at another K, which is what
    makes the number worth reporting next to a decision.

    ``rank`` is the rank of the first relevant fragment and is NOT clipped to
    the top ten -- clipping it would make a rank of 11 indistinguishable from a
    rank of 500 and every changed query would look borderline. A query with no
    relevant fragment in either run has no rank and never counts here.
    """
    low, high = band
    before = {r["desc_id"]: r for r in reference["per_query"]}
    found = 0
    for record in candidate["per_query"]:
        other = before.get(record["desc_id"])
        if other is None or (desc_ids is not None and record["desc_id"] not in desc_ids):
            continue
        if (record[metric] > 0) == (other[metric] > 0):
            continue          # the status did not change; nothing to explain
        ranks = (record.get("rank"), other.get("rank"))
        if all(rank is not None and low <= rank <= high for rank in ranks):
            found += 1
    return found


def activation_share(run: dict, desc_ids: set[int] | None = None) -> dict[str, float]:
    """``{signal: share of queries the signal spoke about}``, from signals.npz.

    A gated signal is left out of a query's weighting entirely when it has
    nothing to say, so how OFTEN it speaks is half of what a contribution means:
    a signal active for one query in ten cannot move a collection average far,
    however well it does when it does speak.

    ``desc_ids`` narrows to a subset -- the "docelowe" column of the activation
    table is this same share over the queries whose tag belongs to the signal's
    experiment (``vocabulary.TAG_EXPERIMENT``).
    """
    import numpy as np

    with np.load(Path(run["dir"]) / "signals.npz", allow_pickle=False) as data:
        names = [str(name) for name in data["names"]]
        ids = data["desc_ids"]
        active = np.asarray(data["active"], dtype=bool)

    keep = (np.ones(len(ids), dtype=bool) if desc_ids is None
            else np.fromiter((int(i) in desc_ids for i in ids), bool, len(ids)))
    if not keep.any():
        return {name: float("nan") for name in names}
    return {name: float(active[row][keep].mean()) for row, name in enumerate(names)}


def active_ids(run: dict, signal: str) -> set[int]:
    """The ``desc_id`` of the queries one signal was active for, from signals.npz.

    :func:`activation_share` answers "how often", which is what a table column
    needs; a ROW of chapter 6 asks something else -- what the two face
    mechanisms do on the queries BOTH of them spoke about -- and for that the
    identifiers themselves are needed, not their count. Reading the file twice
    for two questions is cheaper than a notebook opening it by hand.

    A signal the run does not carry has an empty set, not an error: the block
    that asks is the one that reports the absence.
    """
    import numpy as np

    with np.load(Path(run["dir"]) / "signals.npz", allow_pickle=False) as data:
        names = [str(name) for name in data["names"]]
        ids = data["desc_ids"]
        active = np.asarray(data["active"], dtype=bool)

    if signal not in names:
        return set()
    row = active[names.index(signal)]
    return {int(desc_id) for desc_id, on in zip(ids, row) if on}


def paired_differences(candidate: dict, reference: dict, metric: str,
                       desc_ids: set[int] | None = None,
                       unit: str = "episode") -> list[float]:
    """Paired differences at the unit of inference of the dataset."""
    if unit == "clip":
        return _query_diff(candidate, reference, metric, desc_ids)
    return list(_episode_diff(candidate, reference, metric, desc_ids).values())


def mean_of_differences(differences: list[float],
                        units: list[str] | None = None) -> float | None:
    """Mean of per-query differences at the unit of inference.

    :func:`paired_differences` is this same quantity read off two run
    directories. This one starts from differences computed some other way -- a
    reweighted grid point of the sensitivity analysis, a Recall recomputed after
    the judgment pool -- where there are no ``per_query`` records to group by.

    ``units`` names the unit every difference belongs to, in the same order.
    ``None`` means the QUERY is the unit, which is what a VATEX clip is;
    otherwise the differences are averaged inside each unit first and those means
    averaged in turn, exactly as the episode branch above does. That second step
    is the whole point: without it an episode carrying thirty queries would
    outweigh one carrying three, and the unit of inference would be the query
    again under another name.
    """
    if not differences:
        return None
    if units is None:
        return sum(differences) / len(differences)
    if len(units) != len(differences):
        raise ValueError(
            f"{len(differences)} differences and {len(units)} units: the unit of "
            "every difference has to be named, or none of them")
    groups: dict[str, list[float]] = defaultdict(list)
    for unit, value in zip(units, differences):
        groups[unit].append(value)
    means = [sum(values) / len(values) for values in groups.values()]
    return sum(means) / len(means)


def compare_variant(candidates: dict[str, dict], references: dict[str, dict],
                    metric: str, desc_ids: dict[str, set[int]] | None = None) -> dict:
    """One candidate variant vs the reference, per dataset and pooled.

    Computed on the datasets BOTH sides cover. A candidate that has not been run
    on every dataset of the reference used to raise ``KeyError`` here, which
    turned "VATEX is still downloading" into a broken notebook; the honest answer
    is a comparison over what exists, and ``pooled_datasets`` says what that was.

    Only datasets whose unit is the episode enter the pool.
    """
    per_dataset, pooled = {}, []
    for dataset in sorted(set(references) & set(candidates)):
        subset = desc_ids.get(dataset) if desc_ids else None
        unit = inference_unit(dataset)
        diffs = paired_differences(candidates[dataset], references[dataset],
                                   metric, subset, unit)
        per_dataset[dataset] = {
            **interval(diffs, unit),
            "unit": unit,
            "value": query_mean(candidates[dataset]["per_query"], metric, subset),
            "reference_value": query_mean(references[dataset]["per_query"],
                                          metric, subset),
        }
        if unit == "episode":
            pooled += diffs
    return {"per_dataset": per_dataset, "pooled": t_interval(pooled),
            "pooled_datasets": sorted(d for d in per_dataset
                                      if inference_unit(d) == "episode")}


def contrast_variant(candidates: dict[str, dict], references: dict[str, dict],
                     metric: str, matching: dict[str, set[int]],
                     rest: dict[str, set[int]]) -> dict:
    """Confirmatory contrast: is the effect larger on the tagged queries?

    Per episode, the difference on the tagged queries minus the difference on the
    rest, with an interval. An episode without queries on either side drops out. The
    two sides are reported as plain means alongside -- the first two columns of the
    contrast tables.
    """
    per_dataset, pooled = {}, []
    for dataset in sorted(set(references) & set(candidates)):
        unit = inference_unit(dataset)
        if unit == "clip":
            entry = _clip_contrast(candidates[dataset], references[dataset], metric,
                                   matching.get(dataset), rest.get(dataset))
        else:
            a = _episode_diff(candidates[dataset], references[dataset], metric,
                              matching.get(dataset))
            b = _episode_diff(candidates[dataset], references[dataset], metric,
                              rest.get(dataset))
            diffs = [a[episode] - b[episode] for episode in sorted(set(a) & set(b))]
            entry = t_interval(diffs)
            pooled += diffs
        per_dataset[dataset] = {
            **entry,
            "unit": unit,
            "matching": query_mean(candidates[dataset]["per_query"], metric,
                                   matching.get(dataset)),
            "matching_reference": query_mean(references[dataset]["per_query"], metric,
                                             matching.get(dataset)),
            "rest": query_mean(candidates[dataset]["per_query"], metric,
                               rest.get(dataset)),
            "rest_reference": query_mean(references[dataset]["per_query"], metric,
                                         rest.get(dataset)),
            "matching_delta": _mean_or_none(paired_differences(
                candidates[dataset], references[dataset], metric,
                matching.get(dataset), unit)),
            "rest_delta": _mean_or_none(paired_differences(
                candidates[dataset], references[dataset], metric,
                rest.get(dataset), unit)),
        }
    return {"per_dataset": per_dataset, "pooled": t_interval(pooled),
            "pooled_datasets": sorted(d for d in per_dataset
                                      if inference_unit(d) == "episode")}


def _mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _clip_contrast(candidate: dict, reference: dict, metric: str,
                   matching: set[int] | None, rest: set[int] | None,
                   replications: int = REPLICATIONS, seed: int = 1234) -> dict:
    """Difference of the two group means, with a bootstrap over clips.

    One replication resamples the clips once and recomputes both group means from
    that draw, so the sides stay tied to the same resampled collection.
    """
    import numpy as np

    by_id = {r["desc_id"]: r[metric] for r in reference["per_query"]}
    diffs, side = [], []
    for record in candidate["per_query"]:
        desc_id = record["desc_id"]
        if desc_id not in by_id:
            continue
        if matching is not None and desc_id in matching:
            side.append(True)
        elif rest is not None and desc_id in rest:
            side.append(False)
        else:
            continue
        diffs.append(record[metric] - by_id[desc_id])

    values, flags = np.asarray(diffs, dtype=np.float64), np.asarray(side, dtype=bool)
    if flags.sum() < 2 or (~flags).sum() < 2:
        return {"n": int(flags.sum()), "mean": None, "low": None, "high": None,
                "method": "bootstrap"}

    observed = float(values[flags].mean() - values[~flags].mean())
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replications):
        pick = rng.integers(0, values.size, size=values.size)
        drawn, drawn_flags = values[pick], flags[pick]
        if not drawn_flags.any() or drawn_flags.all():
            continue          # a draw with one side empty carries no contrast
        draws.append(drawn[drawn_flags].mean() - drawn[~drawn_flags].mean())
    tail = (1.0 - CONFIDENCE) / 2 * 100
    return {"n": int(flags.sum()), "mean": observed,
            "low": float(np.percentile(draws, tail)),
            "high": float(np.percentile(draws, 100 - tail)),
            "method": "bootstrap"}


# -------------------------------------------------------- Secondary measures

#: reported next to the main measure, in this order. Chapter 4: inference runs on
#: the MAIN measure alone; these are here to show whether the conclusion depends
#: on the cut-off, and carry no intervals and no claim of significance --
#: comparing on several measures at once would inflate the chance that a
#: difference is only apparent.
SECONDARY_METRICS = ("recall@1", "recall@5", "recall@10", "mAP")

#: how metrics.json spells them versus how a per-query record does
_PER_QUERY = {"mAP": "ap"}


def effect_sizes(candidates: dict[str, dict], references: dict[str, dict],
                 metrics: tuple[str, ...] = SECONDARY_METRICS) -> dict:
    """Values and plain differences of every measure, per dataset.

    ``{dataset: {metric: {"value", "reference", "delta", "unit"}}}``. No
    intervals and no verdict on purpose: this is a robustness check of the
    decision taken on the main measure, not a second decision.

    ``value`` and ``reference`` are means over QUERIES; ``delta`` is the mean of
    the paired differences at the unit of inference of the dataset, so it is not
    in general ``value - reference`` for a series. The two answer different
    questions and the difference between them is exactly the clustering the
    episode unit exists to respect.
    """
    out: dict[str, dict] = {}
    for dataset, run in candidates.items():
        reference = references[dataset]
        unit = inference_unit(dataset)
        entry = {}
        for metric in metrics:
            key = _PER_QUERY.get(metric, metric)
            value = query_mean(run["per_query"], key)
            base = query_mean(reference["per_query"], key)
            # the LEVELS are means over queries, like every Recall reported here,
            # but the DELTA is a difference between two configurations and so is
            # read at the unit of inference -- per episode for a series. A check
            # of a decision has to be read the same way the decision was.
            entry[metric] = {"value": value, "reference": base,
                             "delta": _mean_or_none(paired_differences(
                                 run, reference, key, None, unit)),
                             "unit": unit}
        out[dataset] = entry
    return out


# --------------------------------------------------------- The decision rule
def wins(result: dict) -> bool:
    """Consistent direction in every pooled dataset, and a pooled CI above zero."""
    pooled = result["pooled"]
    if pooled["low"] is None or pooled["mean"] is None:
        return False
    directions = [result["per_dataset"][d]["mean"] > 0
                  for d in result.get("pooled_datasets", result["per_dataset"])
                  if result["per_dataset"][d]["mean"] is not None]
    return bool(directions) and all(directions) and pooled["low"] > 0


def confirmed(contrast: dict) -> bool:
    """Whether a confirmatory contrast holds: the effect really is concentrated.

    Three conditions at once: the pooled interval excludes zero, its sign agrees
    with the hypothesis (the tagged queries gain MORE, so the difference of
    differences is positive), and each series says the same on its own.

    Not :func:`wins`, although the shape is close: ``wins`` walks
    ``pooled_datasets`` of a comparison, and a contrast is a claim about SERIES
    material -- VATEX carries neither `wymaga_osoby` nor `wymaga_mimiki`, and its
    clip-level interval never joins the pool anyway. Reusing ``wins`` here would
    read a dataset the claim is not about.
    """
    pooled = contrast.get("pooled", {})
    if pooled.get("low") is None or pooled.get("mean") is None:
        return False
    series = contrast.get("pooled_datasets") or [
        name for name, entry in contrast["per_dataset"].items()
        if entry.get("unit") == "episode"]
    means = [contrast["per_dataset"][name].get("mean") for name in series]
    if not means or any(mean is None for mean in means):
        return False
    return pooled["low"] > 0 and all(mean > 0 for mean in means)


def decide(results: dict[str, dict], simplicity: list[str]) -> dict:
    """Chooses the winning variant among reference + candidates.

    ``simplicity`` orders ALL deciding labels from simplest to most complex, the
    reference included. The verdict carries ``series``: the number of series the
    pooled interval rests on, and ``provisional`` when that is fewer than two.
    """
    qualified = [label for label, result in results.items() if wins(result)]
    if not qualified:
        winner, reason = simplicity[0], "no candidate beat the reference - the simplest variant wins"
    elif len(qualified) == 1:
        winner, reason = qualified[0], "the only candidate that beat the reference"
    else:
        winner = min(qualified, key=simplicity.index)
        reason = ("several candidates beat the reference - resolve their direct "
                  "comparison; provisionally the simplest of them")
    series = max((len(r.get("pooled_datasets", [])) for r in results.values()),
                 default=0)
    return {"winner": winner, "qualified": qualified, "reason": reason,
            "series": series, "provisional": series < 2}
