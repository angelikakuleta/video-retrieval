"""VATEX adapter: download report + descriptions -> query records and clip paths.

VATEX is a set of short independent clips, ten English descriptions each. The
experiments use one description per clip (the first), the implementation
correctness check all ten as separate queries, following the reference works.

Two clip sets, deliberately different:

* ``ok_clips`` -- every downloaded and verified clip (status ``ok``). Only the
  correctness check runs on them: there comparability with published values
  matters, not cleanliness with respect to training.
* ``test_split_clips`` -- ``ok_clips`` minus the recordings present in the
  Kinetics-400 training set (leak), from ``vatex_split_test.csv``
  (``scripts/prepare_data/vatex_leak_filter.py``). This is the basis of ALL
  experiments.

This module BUILDS only the test part: query records, tag rows and the clip
sets all carry the ``_test`` suffix, because the development part gets its query
file from ``vatex_01_dev.ipynb`` instead, which is standard-library-only by
design and cannot import anything from here. Two functions do take a part:
:func:`load_split` and :func:`k400_membership`, because the leak filter writes
one split file per part and the runner reads whichever part it evaluates.

The clip name encodes the boundaries on the source recording
(``G9zN5TTuGO4_000179_000189`` -> 179-189 s), but the file is already that
excerpt, so on its own axis ``ts = (0, duration)``.

The two sets are also numbered differently, and on purpose: the experiment
queries take their ``desc_id`` from the clip name, because the hand-filled tag
file is keyed by it, while the check queries keep the positional numbering the
published comparison of chapter 6 was run on.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from src.utils import settings
from src.utils.queries import Query

ROOT = Path(__file__).resolve().parents[2]
# Data layout (variant B): source material in data/raw, intermediate files that
# produce the annotations in data/interim, ready annotations in data/annotations.
# The test clips live under test/, the development ones under dev/. Section 03
# recorded the test part as lying flat and argued against moving 2560 files; they
# were moved anyway, and this constant follows the disk. Nothing computed changes
# -- the correctness check's result (R@1 46,5) is in the thesis and its index is
# cached; the point is that the check can be repeated at all.
CLIPS_DIR = ROOT / "data" / "raw" / "vatex" / "test"
INTERIM_DIR = ROOT / "data" / "interim" / "vatex"
# Intermediate files of a single procedure sit in work/; the report and the
# split stay one level up, because they carry a decision or a result.
WORK_DIR = INTERIM_DIR / "work"
REPORT_FILE = INTERIM_DIR / "vatex_report_test.csv"
DESCRIPTIONS_FILE = WORK_DIR / "vatex_descriptions_test.csv"
SPLIT_FILE = INTERIM_DIR / "vatex_split_test.csv"

SOURCE = "vatex"
# This module builds only the test part, so the split of every query is the same one.
SPLIT = "test"
# In the download report the status "ok" marks a clip downloaded and verified.
STATUS_OK = "ok"
# Leak flag in vatex_split_test.csv: "yes" = the recording is in the Kinetics-400
# training set (scripts/vatex_leak_filter.py).
LEAK_YES = "yes"
# Which description becomes the query -- one definition for both parts,
# see src.utils.settings.
EXPERIMENT_DESC = settings.VATEX_EXPERIMENT_DESC
# Decimal digits of an identifier derived from the clip name. Twelve keep the
# chance of two of a few thousand clips colliding around one in a million, and
# `build_queries` raises rather than lets a collision through.
DESC_ID_DIGITS = 12

_NAME = re.compile(r"_(\d+)_(\d+)$")


def split_file(part: str) -> Path:
    """``vatex_split_<part>.csv`` -- the leak filter writes one per part."""
    return INTERIM_DIR / f"vatex_split_{part}.csv"


def descriptions_file(part: str) -> Path:
    """``work/vatex_descriptions_<part>.csv`` -- the export of one part."""
    return WORK_DIR / f"vatex_descriptions_{part}.csv"


def duration_from_name(vid_name: str) -> float:
    """Clip duration in seconds read from its name (end - start)."""
    match = _NAME.search(vid_name)
    if not match:
        raise ValueError(f"clip name contains no time markers: {vid_name}")
    start, end = (int(g) for g in match.groups())
    return float(end - start)


def clip_path(vid_name: str, directory: Path = CLIPS_DIR) -> Path:
    """Path of the clip file for the given identifier."""
    return directory / f"{vid_name}.mp4"


def load_report(file: Path = REPORT_FILE) -> pd.DataFrame:
    """Clip download report; ``;`` separator, BOM."""
    return pd.read_csv(file, sep=";", encoding="utf-8-sig", dtype=str)


def load_descriptions(file: Path = DESCRIPTIONS_FILE) -> pd.DataFrame:
    """English descriptions: ``videoID``, ``desc_no``, ``desc_en``."""
    descriptions = pd.read_csv(file, sep=";", encoding="utf-8-sig")
    descriptions["desc_no"] = descriptions["desc_no"].astype(int)
    return descriptions


def load_split(file: Path | None = None, part: str = SPLIT) -> pd.DataFrame:
    """Dataset split from ``vatex_split_<part>.csv``; the test part by default.

    Columns: ``videoID``, ``class_k600``, ``leak_k400`` (yes = the recording is in
    the Kinetics-400 training set), ``class_in_k400_vocab``.

    ``file`` stays the first argument because the correctness check passes the
    path straight from its own configuration.
    """
    path = split_file(part) if file is None else Path(file)
    if not path.exists():
        raise FileNotFoundError(
            f"no {path.name} - run scripts/prepare_data/vatex_leak_filter.py "
            f"--split {'dev' if part == 'dev' else 'validation'} first ({path})")
    return pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)


def k400_membership(part: str = SPLIT) -> dict[str, bool]:
    """``clip -> is its Kinetics-600 class among the 400 SlowFast was trained on``.

    The cut the motion signal is read against: outside those 400 names there is
    nothing for a matched action phrase to land on, so a clip on either side of
    this line is asking a different question of the same signal.
    """
    frame = load_split(part=part)
    return {str(row["videoID"]): str(row["class_in_k400_vocab"]).strip().lower() == "yes"
            for _, row in frame.iterrows()}


def ok_clips(report: pd.DataFrame, directory: Path = CLIPS_DIR) -> list[str]:
    """Identifiers of clips with status ``ok`` whose file exists.

    Intersecting with the directory contents protects against drift; the order is
    sorted, so query identifiers are reproducible.
    """
    ok = report.loc[report["status"] == STATUS_OK, "videoID"]
    present = {p.stem for p in directory.glob("*.mp4")}
    return sorted(vid for vid in ok if vid in present)


def test_split_clips(
    report: pd.DataFrame,
    split: pd.DataFrame,
    directory: Path = CLIPS_DIR,
) -> list[str]:
    """Test split: ``ok_clips`` after filtering out the K400 leak.

    For a clip present in the training of the motion models there is no telling
    whether the event was recognized or recalled. Requires the split to cover the
    whole ``ok`` set -- otherwise it raises, so a stale ``vatex_split_test.csv`` cannot
    silently trim the split. Surplus split rows are ignored.
    """
    ok = ok_clips(report, directory)
    missing = set(ok) - set(split["videoID"])
    if missing:
        raise ValueError(
            f"{len(missing)} clips with status 'ok' are absent from vatex_split_test.csv "
            f"-- recompute scripts/vatex_leak_filter.py. "
            f"Examples: {sorted(missing)[:3]}"
        )
    leak = set(split.loc[split["leak_k400"] == LEAK_YES, "videoID"])
    return [vid for vid in ok if vid not in leak]


def desc_id_for(vid_name: str) -> int:
    """Query identifier of a clip, derived from its name.

    Numbering by position would move every identifier the moment one clip leaves
    the set -- a clip dropped from the download report, a change of the leak
    filter -- and the hand-filled tag file is keyed by ``desc_id``: the tags
    would land on other queries without a word. A digest of the clip name does
    not move, so the identifier stays with the recording it was assigned to.
    """
    digest = hashlib.blake2b(vid_name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 10 ** DESC_ID_DIGITS


def _record(row: pd.Series, desc_id: int) -> Query:
    """One description row as a query record.

    A clip is a self-contained event, so ``event_id`` is its identifier and the
    correct answer is that very clip.
    """
    vid = row["videoID"]
    return Query(
        desc_id=desc_id,
        desc=str(row["desc_en"]).strip(),
        vid_name=vid,
        ts=(0.0, duration_from_name(vid)),
        source=SOURCE,
        event_id=vid,
    )


def build_queries(vids: list[str], descriptions: pd.DataFrame) -> list[Query]:
    """Experiment queries: the first description of every given clip.

    Identifiers come from the clip name (:func:`desc_id_for`), so they survive a
    change of the clip set -- this is the file the tag file is keyed against.
    """
    selected = descriptions[descriptions["videoID"].isin(set(vids))]
    selected = selected[selected["desc_no"] == EXPERIMENT_DESC]
    selected = selected.sort_values(["videoID", "desc_no"])

    queries = [_record(row, desc_id_for(row["videoID"]))
               for _, row in selected.iterrows()]
    repeated = {i for i, n in Counter(q.desc_id for q in queries).items() if n > 1}
    if repeated:
        clashing = sorted(q.vid_name for q in queries if q.desc_id in repeated)
        raise ValueError(
            f"derived desc_id collides for {len(clashing)} clips "
            f"({', '.join(clashing[:4])}) -- widen DESC_ID_DIGITS"
        )
    return queries


def build_check_queries(vids: list[str], descriptions: pd.DataFrame) -> list[Query]:
    """Check queries: all ten descriptions of every given clip.

    Numbered by position in the sorted set, and it stays that way. Nothing
    hand-written is keyed by these identifiers -- the file is read only by the
    implementation correctness check, whose comparison with the published values
    was run on exactly this numbering.
    """
    selected = descriptions[descriptions["videoID"].isin(set(vids))]
    selected = selected.sort_values(["videoID", "desc_no"])
    return [_record(row, desc_id)
            for desc_id, (_, row) in enumerate(selected.iterrows())]


def tag_source_rows(queries: list[Query]) -> list[dict]:
    """Query records in the row shape the tag file is built from.

    The file is shared with the series, where those rows come from the register;
    VATEX has no register, so the query records stand in. ``episode`` holds the
    clip identifier -- that is what pins a ``desc_id`` to a recording and lets
    :func:`src.annotation.tags.misaligned` notice if the two ever come apart.
    """
    return [{"desc_id": str(query.desc_id), "episode": query.vid_name,
             "split": SPLIT, "desc": query.desc} for query in queries]
