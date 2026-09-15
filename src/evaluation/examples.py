"""Qualitative examples: one query, two rankings, real frames.

A contribution measured as a number does not show what the system returns. Each
example answers that for one query: the shown top-K of the baseline beside the
shown top-K of the variant, and three frames of every fragment. There is one
example per kind of information a query can require.

**Nothing here computes a ranking.** Every value comes out of a run directory
that already exists -- ``per_query.jsonl`` for the order, ``relevance.csv`` for
the correct fragments, the segmentation cache for the time span of a fragment.
The figure therefore shows the very ranking the measurements were computed
from, and no model is loaded to draw it.

Two things are kept apart on purpose.

:data:`CASES` holds the CLASS of each example -- which requirement tag, which
complexity, which variant, which direction it illustrates. That is what the set
of examples is meant to cover, so it lives in code.

``results/reports/examples/selection.csv`` holds the CHOICE -- which query
illustrates that class, on which dataset, and how many ranked positions its
figure shows. That is data, editable in a spreadsheet without touching Python,
and it is the only place the choice exists.

The two files of that directory share one format (:data:`ROW_COLUMNS`), so a row
can be copied from the population into the selection unchanged. Columns taken
from the query file carry the names that file gives them.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from src.data import datasets
from src.utils.experiments import BASE, LABEL_COMPONENT
from src.utils.queries import Query

#: Default number of ranked positions a figure shows, used where a case has no
#: choice of its own yet. Four, because in `object` the correct fragment sits at
#: position 4 in BOTH variants, which is the whole point of that example; every
#: case can say otherwise in the `top` column of the selection file.
TOPK = 4

#: frames sampled per fragment: beginning, middle and end of its CONTENT
FRAMES_PER_FRAGMENT = 3

#: Fields of ``<dataset>_queries_<split>.jsonl``, spelled as that file spells
#: them. Copying the names rather than inventing new ones is what lets a row of
#: either file be checked against the annotation without a lookup table.
QUERY_COLUMNS: tuple[str, ...] = (
    "desc_id", "desc", "vid_name", "ts", "source", "requirements",
    "complexity", "identities", "event_id")

#: The one format ``selection.csv`` and ``candidates_*.csv`` share: which case
#: and dataset the row belongs to, how deep its figure goes, the whole query
#: record, and what the two configurations did with it. Nothing else -- the
#: helper columns an earlier version carried (a score, the episode count, the
#: shortest fragment) were sorting aids, and the order of the rows says the same
#: thing without a column of its own.
ROW_COLUMNS: tuple[str, ...] = (
    "stem", "dataset", "top", *QUERY_COLUMNS,
    "rank_base", "rank_variant", "correct_in_top_base", "correct_in_top_variant")

#: Names of the variants as they are DRAWN on a figure, with the diacritics
#: Polish has. Console output may not carry them (CLAUDE.md) and
#: ``LABEL_COMPONENT`` feeds both, so the two are kept apart exactly as
#: :mod:`src.evaluation.figures` keeps ``FIGURE_SIGNAL_NAMES`` apart from
#: ``SIGNAL_NAMES``; only the entries that actually differ are listed.
FIGURE_COMPONENT: dict[str, str] = {
    "E6-B": "tożsamość",
    "recommended": "L + ID",
}

#: What "Dodanie ..." takes as its object in the one-sentence comment, in the
#: GENITIVE. Polish needs the case, and every display name in the repository is
#: nominative ("regiony twarzy", "tozsamosc"), so a sentence built from those
#: would be ungrammatical.
COMMENT_SUBJECT: dict[str, str] = {
    "E2-C": "sygnału ruchu (SlowFast)",
    "E3-B": "opisów scen (BLIP)",
    "E3-C": "opisów scen (LLaVA-1.5)",
    "E4-B": "sygnału obiektowego (YOLO11)",
    "E4-C": "sygnału obiektowego (YOLOE-11)",
    "E5-B": "sygnału regionów twarzy",
    "E5-C": "sygnału mimiki (HSEmotion)",
    "E6-B": "sygnału tożsamości",
    "recommended": "opisów scen i sygnału tożsamości",
}

#: the direction a case is meant to illustrate
DIRECTIONS = ("improvement", "degradation", "none")


@dataclass(frozen=True)
class Spec:
    """The CLASS one example illustrates: tag, complexity, variant, direction.

    Carries no query and no dataset: which of them illustrates this class is a
    choice, and the choice lives in ``selection.csv``.
    """

    stem: str                      # file name of the case, e.g. "face"
    kind: str                      # name of this kind of query, for reports
    tags: tuple[str, ...]          # requirement tags every candidate carries
    complexity: str                # "P" or "Z"
    variant: str                   # the label added to the base
    direction: str                 # one of DIRECTIONS

    @property
    def variant_label(self) -> str:
        """``BAZA + regiony twarzy (E5-B)`` -- the heading above the variant."""
        component = FIGURE_COMPONENT.get(self.variant) \
            or LABEL_COMPONENT.get(self.variant, self.variant)
        code = f" ({self.variant})" if self.variant.startswith("E") else ""
        return f"BAZA + {component}{code}"


@dataclass(frozen=True)
class Pick:
    """The CHOICE for one case: which query, on which dataset, how deep."""

    stem: str
    dataset: str
    desc_id: int
    top: int


#: The six examples. Which query illustrates each one is in ``selection.csv``.
CASES: tuple[Spec, ...] = (
    Spec("person", "Osoba / tożsamość", ("wymaga_osoby",), "P",
         "E6-B", "improvement"),
    Spec("object", "Obiekt", ("wymaga_obiektu",), "P",
         "E4-B", "none"),
    Spec("motion", "Ruch / czynność", ("wymaga_ruchu",), "P",
         "E2-C", "degradation"),
    Spec("face", "Mimika", ("wymaga_mimiki",), "P",
         "E5-B", "degradation"),
    Spec("scenery", "Sceneria", ("wymaga_scenerii",), "P",
         "E3-C", "improvement"),
    Spec("complex", "Złożone: osoba + obiekt + czynność",
         ("wymaga_osoby", "wymaga_obiektu", "wymaga_ruchu"), "Z",
         "recommended", "improvement"),
)

def case_of(stem: str) -> Spec:
    """The specification of one case by its file name."""
    for spec in CASES:
        if spec.stem == stem:
            return spec
    raise ValueError(f"unknown example: {stem!r}; expected one of "
                     f"{[s.stem for s in CASES]}")


@dataclass(frozen=True)
class Hit:
    """One entry of a shown ranking, with everything a figure row needs."""

    rank: int
    fragment: str                  # s09e22_026
    episode: str                   # s09e22
    video_file: str                # tbbt_s09e22.mp4
    start: float
    end: float
    score: float
    relevant: bool

    @property
    def span(self) -> str:
        """The interval written the way ``ts`` is written in the query file.

        A dot and two decimals, exactly as the file spells it: every number on
        the figure can then be looked up in ``data/annotations/`` without
        converting anything.
        """
        return f"[{self.start:.2f}, {self.end:.2f}]"


@dataclass(frozen=True)
class Panel:
    """One configuration's shown ranking, and how it did on this query."""

    label: str                     # "BAZA" / "BAZA + regiony twarzy (E5-B)"
    heading: str                   # one line: what this configuration did
    hits: tuple[Hit, ...]
    rank_first_correct: int | None  # in the WHOLE collection, not only in hits


@dataclass(frozen=True)
class Case:
    """Everything one example is made of."""

    spec: Spec
    pick: Pick
    query: Query
    reference: Panel
    candidate: Panel
    n_relevant: int
    candidates_total: int          # how many queries the class held

    @property
    def dataset(self) -> str:
        return self.pick.dataset

    @property
    def fragments(self) -> tuple[Hit, ...]:
        """Every shown fragment, once, in the order the figures need them."""
        seen, out = set(), []
        for hit in (*self.reference.hits, *self.candidate.hits):
            if hit.fragment not in seen:
                seen.add(hit.fragment)
                out.append(hit)
        return tuple(out)

    @property
    def comment(self) -> str:
        """The one sentence that says what the variant did with this query.

        Exactly this and no more: what adding the component did to the position
        of the correct fragment, with no further analysis.

        A figure whose rows hold no correct fragment needs one more clause, and
        it has to name WHICH side is missing it -- saying only "position 19,
        outside the shown top-4" after a sentence that already reads "from 19 to
        1" would suggest the fragment is still at 19.
        """
        before = self.reference.rank_first_correct
        after = self.candidate.rank_first_correct
        if after < before:
            move = (f"przesunęło poprawny fragment z pozycji {before} "
                    f"na {after}")
        elif after > before:
            move = (f"pogorszyło pozycję poprawnego fragmentu z {before} "
                    f"na {after}")
        else:
            move = ("przestawiło kolejność wyników, ale poprawny fragment "
                    f"został na pozycji {before}")
        sentence = f"Dodanie {COMMENT_SUBJECT[self.spec.variant]} {move}."

        top = self.pick.top
        if before > top:
            sentence += (f" W BAZIE poprawny fragment był poza pokazanym "
                         f"top-{top} (pozycja {before}).")
        if after > top:
            sentence += (f" Po dodaniu sygnału poprawny fragment jest poza "
                         f"pokazanym top-{top} (pozycja {after}).")
        return sentence

    @property
    def meta_line(self) -> str:
        """Requirement tags and complexity, for the line typed under the query."""
        level = ("zapytanie proste" if self.query.complexity == "P"
                 else "zapytanie złożone")
        return "  ·  ".join([", ".join(self.query.requirements or []), level])


# ------------------------------------------------------------- reading runs
def newest_run(label: str, dataset: str, split: str = "test") -> Path | None:
    """The newest finished run directory of one (label, dataset, split)."""
    found = sorted(directory for directory
                   in datasets.runs_dir().glob(f"{label}_{dataset}_{split}_*")
                   if (directory / "metrics.json").exists())
    return found[-1] if found else None


def _per_query(run_dir: Path) -> dict[int, dict]:
    import json

    lines = (run_dir / "per_query.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    return {record["desc_id"]: record for record in records}


def _fragments(dataset: str, strategy: str) -> dict[str, dict]:
    """``fragment id -> its segmentation row``, for the time spans of a figure."""
    from src.evaluation.relevance import fragment_id
    from src.segmentation import segments as seg

    rows = seg.load(datasets.segments_csv(strategy, dataset))
    return {fragment_id(row["episode"], row["segment_id"]): row for row in rows}


def _strategy_of(run_dir: Path) -> str:
    """The segmentation the run used, read from its own resolved configuration.

    Not the frozen verdict: a figure has to measure the fragments of the run it
    illustrates, and E1 variants exist precisely because that can differ.
    """
    import yaml

    config = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(
        encoding="utf-8"))
    return config["segmentation"]["strategy"]


def _heading(spec: Spec, label: str, rank: int | None, is_reference: bool,
             top: int) -> str:
    """The line typed above one figure, phrased for what that side did.

    The rank is named even when it lies beyond the rows the figure draws, and
    that is the point of the sentence: a reader who sees no correct frame needs
    to know whether the fragment fell to position 6 or to position 146.
    """
    if rank is None:                      # no correct fragment ranked at all
        return f"{label} — nic poprawnego w top-{top}"
    if is_reference:
        return f"{label} — trafienie na pozycji {rank}"
    if spec.direction == "degradation":
        return f"{label} — spadek na pozycję {rank}"
    if spec.direction == "none":
        return f"{label} — trafienie nadal na pozycji {rank}"
    return f"{label} — trafienie na pozycji {rank}"


def _panel(spec: Spec, label: str, record: dict, relevant: set[str],
           fragments: dict[str, dict], is_reference: bool, top: int) -> Panel:
    hits = []
    for position, entry in enumerate(record["top"][:top], 1):
        fragment = entry["fragment"]
        row = fragments.get(fragment)
        if row is None:
            raise KeyError(
                f"fragment {fragment!r} of query {record['desc_id']} is not in "
                "the segmentation cache - the run and the cache disagree, so "
                "rebuild the cache or read the run that produced it")
        hits.append(Hit(rank=position, fragment=fragment, episode=row["episode"],
                        video_file=row["video_file"], start=row["start"],
                        end=row["end"], score=float(entry["score"]),
                        relevant=fragment in relevant))
    rank = record.get("rank")
    return Panel(label=label,
                 heading=_heading(spec, label, rank, is_reference, top),
                 hits=tuple(hits), rank_first_correct=rank)


def build(spec: Spec, pick: Pick, split: str = "test") -> Case:
    """Assembles one example out of the two run directories it compares."""
    from src.data.queries import load_queries
    from src.evaluation.compare import load_relevance

    reference_dir = newest_run(BASE, pick.dataset, split)
    candidate_dir = newest_run(spec.variant, pick.dataset, split)
    for label, directory in ((BASE, reference_dir), (spec.variant, candidate_dir)):
        if directory is None:
            raise FileNotFoundError(
                f"no finished {label} run for {pick.dataset}/{split} - the "
                f"example {spec.stem} cannot be drawn until that run exists")

    fragments = _fragments(pick.dataset, _strategy_of(candidate_dir))
    before, after = _per_query(reference_dir), _per_query(candidate_dir)
    relevant = load_relevance(candidate_dir)[pick.desc_id]
    if not relevant:
        raise ValueError(
            f"{spec.stem}: query {pick.desc_id} has no correct fragment in "
            f"{candidate_dir.name} - it cannot illustrate anything")

    queries = {query.desc_id: query for query in load_queries(pick.dataset, split)}
    query = queries[pick.desc_id]

    return Case(
        spec=spec, pick=pick, query=query,
        reference=_panel(spec, "BAZA", before[pick.desc_id], relevant,
                         fragments, True, pick.top),
        candidate=_panel(spec, spec.variant_label, after[pick.desc_id], relevant,
                         fragments, False, pick.top),
        n_relevant=len(relevant),
        candidates_total=len(candidates(spec, pick.dataset, split, pick.top)))


# ------------------------------------------------------------- the one format
def _yes(value: bool) -> str:
    """Coded cell values are English, like every other one in the repository."""
    return "yes" if value else "no"


def _row(spec: Spec, dataset: str, top: int, query: Query,
         rank_base: int, rank_variant: int) -> dict:
    """One row of the shared format."""
    return {
        "stem": spec.stem,
        "dataset": dataset,
        "top": top,
        "desc_id": query.desc_id,
        "desc": query.desc,
        "vid_name": query.vid_name,
        # written as the query file writes it, values included
        "ts": "[" + ", ".join(str(value) for value in query.ts) + "]",
        "source": query.source,
        "requirements": ",".join(query.requirements or []),
        "complexity": query.complexity or "",
        "identities": ",".join(query.identities or []),
        "event_id": query.event_id,
        "rank_base": rank_base,
        "rank_variant": rank_variant,
        "correct_in_top_base": _yes(rank_base <= top),
        "correct_in_top_variant": _yes(rank_variant <= top),
    }


def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROW_COLUMNS),
                                delimiter=";", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def reports_dir() -> Path:
    """Where the selection and the audited populations live. Versioned."""
    return datasets.ROOT / "results" / "reports" / "examples"


def selection_file() -> Path:
    """One row per case: the chosen query and how deep its figure goes."""
    return reports_dir() / "selection.csv"


def candidates_file(stem: str) -> Path:
    return reports_dir() / f"candidates_{stem}.csv"


# ------------------------------------------------------------- the selection
def load_picks(path: Path | None = None) -> dict[str, Pick]:
    """``stem -> Pick`` from the selection file. It is the only source there is.

    Exactly one row per case, so there is no column to mark and nothing to
    resolve: the row IS the choice. A case named twice, or named at all without
    belonging to :data:`CASES`, is refused rather than silently resolved --
    drawing a different query than the one written down would be the worst
    failure this module could have.
    """
    path = path or selection_file()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing, and it is where the choice of queries lives. "
            "Rebuild it from the registry with examples.write_selection(), then "
            "edit it.")

    stems = {spec.stem for spec in CASES}
    picks: dict[str, Pick] = {}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for line, row in enumerate(csv.DictReader(handle, delimiter=";"), 2):
            stem = (row.get("stem") or "").strip()
            if not stem:
                continue
            if stem not in stems:
                raise ValueError(f"{path.name}, line {line}: {stem!r} is not one "
                                 f"of the cases {sorted(stems)}")
            if stem in picks:
                raise ValueError(f"{path.name}, line {line}: {stem!r} appears "
                                 "twice - the file holds one row per case")
            picks[stem] = Pick(stem=stem, dataset=(row["dataset"] or "").strip(),
                               desc_id=int(row["desc_id"]),
                               top=int(row.get("top") or TOPK))
    missing = sorted(stems - set(picks))
    if missing:
        raise ValueError(f"{path.name}: no row for {missing}")
    return picks


def selected(split: str = "test") -> list[tuple[Spec, Pick]]:
    """The six cases paired with their chosen query, in the order of CASES."""
    picks = load_picks()
    return [(spec, picks[spec.stem]) for spec in CASES]


def write_selection(picks: dict[str, Pick] | None = None,
                    split: str = "test", log=print) -> Path:
    """Writes the selection file in the shared format; rebuilds it if it is lost.

    Without ``picks`` the current file is re-read and rewritten, which is how a
    hand edit gets its ``rank_base``, ``rank_variant`` and the two
    ``correct_in_top_*`` columns brought back in step with the chosen ``top``.
    Those four are derived, never authoritative: the row says which query and
    how deep, and everything else follows from the runs.
    """
    from src.data.queries import load_queries

    picks = picks or load_picks()
    rows = []
    for spec in CASES:
        pick = picks[spec.stem]
        reference = newest_run(BASE, pick.dataset, split)
        candidate = newest_run(spec.variant, pick.dataset, split)
        if reference is None or candidate is None:
            log(f"{spec.stem}: no runs for {pick.dataset}/{split} - row skipped")
            continue
        before, after = _per_query(reference), _per_query(candidate)
        query = next(q for q in load_queries(pick.dataset, split)
                     if q.desc_id == pick.desc_id)
        rows.append(_row(spec, pick.dataset, pick.top, query,
                         before[pick.desc_id]["rank"],
                         after[pick.desc_id]["rank"]))
    path = _write(selection_file(), rows)
    log(f"{path.name}: {len(rows)} rows, one per case")
    return path


# ----------------------------------------------------- the audited population
def _score(spec: Spec, rank_base: int, rank_variant: int,
           base: list[str], variant: list[str], top: int) -> float:
    """How well one query would illustrate this case; bigger is better.

    Used to ORDER the population file and nowhere else, which is why it is not a
    column: the order says the same thing, and a number invites being read as a
    measurement.

    For a case showing an improvement or a degradation: how high the correct
    fragment reached and how many places it moved, with a bonus when it stays
    visible on BOTH sides, because then the figure shows an order changing
    rather than a hit disappearing. For a case showing NO change the amplitude
    is zero by construction, so the measure becomes the reshuffling around a
    fragment that did not move.
    """
    best = min(rank_base, rank_variant)
    if spec.direction != "none":
        visible = 6.0 if max(rank_base, rank_variant) <= top else 0.0
        return visible + 10.0 / best + 2.0 * math.log2(
            1 + abs(rank_base - rank_variant))
    churn = len(set(variant) - set(base))
    if best > top:
        return 0.5 * churn                     # invisible on the figure anyway
    above = sum(1 for i in range(best - 1) if base[i] != variant[i])
    below = sum(1 for i in range(best, top) if base[i] != variant[i])
    position = {1: 0.5, 2: 1.0, 3: 3.0, 4: 3.0, 5: 1.5}.get(best, 1.0)
    return position + 3.0 * above + 1.5 * below + 0.5 * churn


def candidates(spec: Spec, dataset: str, split: str = "test",
               top: int = TOPK) -> list[dict]:
    """Every query of this case's class on one dataset, most useful first.

    The WHOLE class, not a shortlist: the file is what makes a hand-picked
    example auditable, so a reader can see the population it came from.
    """
    from src.data.queries import load_queries
    from src.evaluation.compare import load_relevance

    reference_dir = newest_run(BASE, dataset, split)
    candidate_dir = newest_run(spec.variant, dataset, split)
    if reference_dir is None or candidate_dir is None:
        return []

    before, after = _per_query(reference_dir), _per_query(candidate_dir)
    relevance = load_relevance(candidate_dir)

    scored = []
    for query in load_queries(dataset, split):
        if not all(tag in (query.requirements or []) for tag in spec.tags):
            continue
        if query.complexity != spec.complexity:
            continue
        if query.desc_id not in before or query.desc_id not in after:
            continue
        rank_base = before[query.desc_id].get("rank")
        rank_variant = after[query.desc_id].get("rank")
        if rank_base is None or rank_variant is None:
            continue

        base = [e["fragment"] for e in before[query.desc_id]["top"][:top]]
        variant = [e["fragment"] for e in after[query.desc_id]["top"][:top]]
        direction = ("improvement" if rank_variant < rank_base
                     else "degradation" if rank_variant > rank_base else "none")
        row = _row(spec, dataset, top, query, rank_base, rank_variant)
        scored.append((direction == spec.direction,
                       _score(spec, rank_base, rank_variant, base, variant, top),
                       row))

    # the direction the case illustrates first, and inside it the clearest
    scored.sort(key=lambda item: (not item[0], -item[1]))
    return [row for _, _, row in scored]


def write_candidates(split: str = "test", log=print) -> list[Path]:
    """One file per case, covering BOTH series, in the shared format.

    Both series, because the dataset is part of the choice: an example belongs
    on the series where its effect measured most clearly, and deciding that
    needs the other series in view.
    """
    from src.utils.experiments import SERIES

    picks = load_picks()
    written = []
    for spec in CASES:
        top = picks[spec.stem].top
        rows = []
        for dataset in SERIES:
            rows += candidates(spec, dataset, split, top)
        if not rows:
            log(f"{spec.stem}: no runs for {split} - skipped")
            continue
        path = _write(candidates_file(spec.stem), rows)
        chosen = next((index for index, row in enumerate(rows, 1)
                       if row["desc_id"] == picks[spec.stem].desc_id
                       and row["dataset"] == picks[spec.stem].dataset), None)
        visible = sum(1 for row in rows
                      if row["correct_in_top_base"] == "yes"
                      and row["correct_in_top_variant"] == "yes")
        log(f"{path.name}: {len(rows)} queries, {visible} with the hit visible "
            f"on both sides, top-{top}, chosen one is row {chosen}")
        written.append(path)
    return written


def selection_report(log=print) -> None:
    """Prints the current choice: which query per case, and how deep."""
    for spec, pick in selected():
        log(f"  {spec.stem:<11} {pick.dataset:<7} desc_id {pick.desc_id:<9} "
            f"top-{pick.top}")
