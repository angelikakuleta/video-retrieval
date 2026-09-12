"""E4-D: the object signal asked at QUERY TIME (section 08).

Every other object variant detects once, into a cache, against a vocabulary fixed
before any query exists; a fragment holding a thing the vocabulary has no name
for is a fragment the signal cannot speak about. E4-D turns that around: the
phrases of the query become the detector's classes, so the vocabulary is the
query. That is the whole point of the variant, and it is why no threshold applies
here -- there is no class name to match a phrase against, the phrase IS the class.

A SCORING STAGE, not a signal. It re-ranks the fifty best fragments of the base
and leaves the rest of the collection alone, so it has no row in ``signals.npz``,
no weight in the fusion of the pipeline and no ``matching.jsonl``. The validator
in :mod:`src.utils.config` keeps it that way: nothing else may be switched on
beside it.

Three decisions in here are not obvious and each one has a reason.

**A candidate the detector looked at and found nothing in scores a RAW ZERO, not
"no value".** With fifty candidates and one detection, forty-nine neutral values
give a column of near-zero deviation, which :mod:`src.retrieval.fusion` zeroes
outright -- and E4-D becomes the base, exactly. "Nothing found" is evidence here,
because the detector was asked for these phrases and no other: the same evidence
as a fragment full of other classes in the indexed mode. Only a candidate with no
usable frame at all -- entirely inside a mask, or in black -- has nothing to say
and gets NaN.

**The scene values are standardized again inside the fifty.** They arrive
standardized over the whole collection, where fifty top-ranked fragments are all
above the mean and barely differ; re-standardizing inside the subset is what lets
the base and the detector actually compete over the re-ranking.

**Detection runs per query, never over the union of the prompts.** The NMS of
ultralytics runs with ``multi_label=False``, so a box keeps its single best
class: adding another query's prompt can take a detection away from the phrase of
this one, and the results would not be the ones the variant is defined by. The
cost of that is real -- a series is around 240 000 inferences -- and accepted.

Decoding, not detection, is the bottleneck, so the loop is inverted: episode by
episode, sequentially, detecting for every query that pointed at a fragment of it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from src.features.objects import BATCH, CONFIDENCE, DETECTORS
from src.retrieval.fusion import combine, standardize
from src.segmentation.frames import STEP, _sample_times

#: how many fragments of the base ranking the stage re-ranks
CANDIDATES = 50

#: the promptable checkpoint. The prompt-free one used by the indexed mode
#: cannot take text at all; this is a different set of weights, and the text
#: encoder (mobileclip_blt.ts) comes down with it on the first `set_classes`.
#: Named in one place with every other set of detector weights.
PROMPTABLE = DETECTORS["yoloe_prompted"]

#: the two columns of the subset fusion: the detector and the base, equally
WEIGHTS = (0.5, 0.5)


def prompts_of(query) -> list[tuple[str, ...]]:
    """The object phrases of one query, each as its own tuple of forms.

    Both forms of every phrase, because the detector is asked for the phrase and
    a name may match either the cleaned phrase or its bare head. No phrases means
    the stage has nothing to ask, so it stays inactive and the base ranking
    stands -- not a ranking scored on an empty prompt list.
    """
    from src.retrieval.phrases import phrases_for

    return [tuple(phrase.forms) for phrase in phrases_for(query)["object"]]


def texts_of(prompts: Sequence[Sequence[str]]) -> tuple[list[str], list[list[int]]]:
    """``(distinct texts, the phrases each text belongs to)``.

    Distinct, because two phrases can share a form and a class list with the same
    name twice would make the detector's answer ambiguous. The ownership map is
    what puts a detection back on the right phrase afterwards.
    """
    texts: list[str] = []
    owners: list[list[int]] = []
    seen: dict[str, int] = {}
    for phrase, forms in enumerate(prompts):
        for form in forms:
            if form not in seen:
                seen[form] = len(texts)
                texts.append(form)
                owners.append([])
            owners[seen[form]].append(phrase)
    return texts, owners


def candidates_of(row: np.ndarray, k: int = CANDIDATES) -> np.ndarray:
    """The columns of the ``k`` best fragments of one base score row, in its order."""
    row = np.asarray(row)
    return np.argsort(-row, kind="stable")[:k]


def frame_numbers(times: Sequence[float], fps: float, frame_count: int) -> np.ndarray:
    """Grid moments to frame numbers, by the rule ``frames.iter_frames`` uses.

    Written out rather than imported because ``iter_frames`` is a generator over
    a file and this needs the numbers alone -- but it is the same arithmetic, and
    a test asserts the two agree frame for frame.
    """
    numbers = np.minimum(np.round(np.asarray(times, dtype=np.float64) * fps),
                         frame_count - 1)
    return numbers.astype(np.int64)


def grid_times(frame_count: int, fps: float, step: float = STEP) -> list[float]:
    """The sampling grid of a file, the same one the indexing pass walked."""
    return _sample_times(frame_count / fps if fps > 0 else 0.0, step)


def value_of(per_phrase: Sequence[float] | None) -> float:
    """One candidate's value: the mean over its phrases, or NaN.

    ``per_phrase`` holds, for each phrase, the best confidence any of its forms
    reached on any usable frame of the fragment -- zero when none did. ``None``
    means the candidate had no usable frame, which is the only case that is
    genuinely "no value".
    """
    if per_phrase is None or not len(per_phrase):
        return float("nan")
    return float(np.mean(np.asarray(per_phrase, dtype=np.float64)))


def fuse(detection: np.ndarray, scene: np.ndarray,
         weights: Sequence[float] = WEIGHTS) -> np.ndarray:
    """The subset score: both columns standardized INSIDE the fifty, then summed.

    Through the ordinary :mod:`src.retrieval.fusion`, on a 2 x 50 matrix, so the
    treatment of a missing value is the pipeline's own: NaN is excluded from the
    mean and the deviation and comes back as the neutral zero.
    """
    detection, scene = np.asarray(detection, float), np.asarray(scene, float)
    if detection.shape != scene.shape:
        raise ValueError("the detector and the base must score the same candidates")
    matrices = [standardize(detection[None, :]), standardize(scene[None, :])]
    active = np.ones((2, 1), dtype=bool)
    return combine(matrices, active, np.asarray(weights, dtype=np.float64))[0]


def reorder(order: Sequence[int], candidates: Sequence[int],
            fused: np.ndarray) -> np.ndarray:
    """The candidates by their fused score, then the rest of the collection.

    Ties keep the order of the base: the sort is stable over the candidates,
    which arrive in base order. A fragment outside the fifty does not move.
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    ranked = candidates[np.argsort(-np.asarray(fused, float), kind="stable")]
    chosen = set(candidates.tolist())
    rest = [column for column in order if column not in chosen]
    return np.concatenate([ranked, np.asarray(rest, dtype=np.int64)])


class PromptedDetector:
    """YOLOE-11 asked for the phrases of one query.

    The text embeddings are computed once per distinct tuple of strings: the
    encoder is 572 MB and most queries of an episode ask for the same handful of
    everyday things.
    """

    def __init__(self, weights: str = PROMPTABLE, confidence: float = CONFIDENCE,
                 batch: int = BATCH, log: Callable[[str], None] = print) -> None:
        self.weights = weights
        self.confidence = confidence
        self.batch = batch
        self._log = log
        self._model = None
        self._embeddings: dict[tuple[str, ...], object] = {}

    def model(self):
        if self._model is None:
            from src.data import datasets
            from ultralytics import YOLOE

            path = datasets.ROOT / self.weights
            self._log(f"loading promptable YOLOE ({self.weights})")
            self._model = YOLOE(str(path) if path.exists() else self.weights)
        return self._model

    def _classes(self, texts: Sequence[str]) -> None:
        key = tuple(texts)
        net = self.model()
        if key not in self._embeddings:
            self._embeddings[key] = net.model.get_text_pe(list(key))
        net.set_classes(list(key), self._embeddings[key])

    def detect(self, frames: list, texts: Sequence[str]) -> list[dict[int, float]]:
        """Best confidence per prompt on each frame: ``[{class index: score}]``.

        Frames are PIL images. Never numpy arrays: the ultralytics loader
        converts a PIL image to BGR itself but takes an array as ALREADY BGR, so
        an array would feed the detector swapped channels -- the defect section
        08 opens with, on the other side of the pipeline.
        """
        if not frames or not texts:
            return [{} for _ in frames]
        self._classes(texts)
        net = self.model()
        out = []
        # In batches, like the indexed mode: a query's fifty candidates carry a
        # few hundred frames at SOURCE resolution, and handing them over in one
        # call asks the card for several gigabytes at once.
        for start in range(0, len(frames), self.batch):
            results = net.predict(frames[start:start + self.batch],
                                  verbose=False, conf=self.confidence)
            for result in results:
                best: dict[int, float] = {}
                boxes = getattr(result, "boxes", None)
                if boxes is not None:
                    for class_id, score in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                        index = int(class_id)
                        best[index] = max(best.get(index, 0.0), float(score))
                out.append(best)
        return out


def per_phrase_maxima(detections: Sequence[dict[int, float]],
                      owners: Sequence[Sequence[int]],
                      n_phrases: int) -> list[float]:
    """Best confidence each phrase reached, over its forms and over the frames.

    Zero for a phrase nothing reached the threshold for: the detector was asked
    and answered no.
    """
    best = [0.0] * n_phrases
    for frame in detections:
        for index, score in frame.items():
            for phrase in owners[index]:
                best[phrase] = max(best[phrase], score)
    return best


def episode_frames(rows: list[dict], episode: str, row_of: dict[str, int],
                   timeline, frame_count: int, fps: float,
                   step: float = STEP) -> dict[int, list[int]]:
    """``collection row -> frame numbers`` of that fragment's usable grid points.

    Usable is what the rest of the pipeline means by it: a grid moment inside the
    fragment that the timeline does not exclude, so a frame in a transition mask
    or in black never reaches the detector.
    """
    from src.runners.components import fragment_of_times

    times = np.asarray(grid_times(frame_count, fps, step), dtype=np.float64)
    assigned = fragment_of_times(rows, episode, times, row_of, timeline)
    numbers = frame_numbers(times, fps, frame_count)

    out: dict[int, list[int]] = {}
    for column in sorted(set(assigned.tolist()) - {-1}):
        out[int(column)] = sorted(set(numbers[assigned == column].tolist()))
    return out


def probe(path: Path | str) -> tuple[float, int]:
    """``(fps, frame count)`` of a recording."""
    import cv2

    reader = cv2.VideoCapture(str(path))
    try:
        fps = reader.get(cv2.CAP_PROP_FPS) or 0.0
        count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        reader.release()
    if fps <= 0 or count <= 0:
        raise ValueError(f"could not read the video: {path}")
    return float(fps), count


def base_rank_of(order: Sequence[int], vids: Sequence[str],
                 relevant: set[str] | None) -> int | None:
    """Where the base put the first correct fragment.

    Recorded because Recall@50 of the base is the CEILING of this stage: nothing
    outside the fifty can be reached, however good the detector is. Reading it
    off the base ranking of the same run makes the ceiling checkable against the
    baseline's own numbers instead of assumed.
    """
    if not relevant:
        return None
    for position, column in enumerate(order, 1):
        if vids[column] in relevant:
            return position
    return None


def rescore(scores: np.ndarray, vids: Sequence[str], queries: Sequence,
            rows: list[dict], row_of: dict[str, int], timelines: dict,
            video_of: Callable[[str], Path], relevant: Sequence[set[str]] | None = None,
            detector: "PromptedDetector | None" = None, step: float = STEP,
            k: int = CANDIDATES,
            log: Callable[[str], None] = print
            ) -> tuple[list[np.ndarray], list[dict], list[np.ndarray]]:
    """The whole stage: base scores in, re-ranked orders out.

    Returns ``(orders, records, values)``. The record is exactly the three fields
    section 08 fixes for ``per_query.jsonl`` and nothing more; the detection
    values come back beside it because the cost measurement and the smoke test
    have to be able to count what the detector found without re-running it.

    The loop is INVERTED -- episode by episode rather than query by query. The
    naive order opens the same recording once per query, and with a few hundred
    queries pointing at six episodes that is the cost of the whole variant;
    decoding, not detection, is the bottleneck.

    Within one episode the frames of all of one query's candidates go to the
    detector in a single call, because they share that query's prompts. Across
    queries they do not: two queries seeing the same frame have it detected
    twice, which is the price of asking per query rather than over the union.
    """
    from collections import defaultdict

    from PIL import Image

    from src.evaluation.relevance import fragment_id
    from src.features.xclip import collect_frames

    scores = np.asarray(scores, dtype=np.float64)
    detector = detector or PromptedDetector(log=log)
    prompts = [prompts_of(query) for query in queries]
    candidates = [candidates_of(scores[i], k) for i in range(len(queries))]
    orders = [np.argsort(-scores[i], kind="stable") for i in range(len(queries))]

    episode_of_column = {}
    for row in rows:
        column = row_of.get(fragment_id(row["episode"], row["segment_id"]))
        if column is not None:
            episode_of_column[int(column)] = row["episode"]

    wanted: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, found in enumerate(prompts):
        if not found:
            continue                    # no phrases: the stage says nothing
        for column in candidates[i]:
            episode = episode_of_column.get(int(column))
            if episode is not None:
                wanted[episode][int(column)].append(i)

    detection = [np.full(len(candidates[i]), np.nan) for i in range(len(queries))]
    for episode in sorted(wanted):
        path = video_of(episode)
        fps, count = probe(path)
        per_column = episode_frames(rows, episode, row_of, timelines.get(episode),
                                    count, fps, step)
        needed = sorted({number for column in wanted[episode]
                         for number in per_column.get(column, [])})
        decoded = collect_frames(path, needed, height=None)

        by_query: dict[int, list[int]] = defaultdict(list)
        for column, asking in wanted[episode].items():
            for i in asking:
                by_query[i].append(column)

        for i, columns in sorted(by_query.items()):
            texts, owners = texts_of(prompts[i])
            frames, spans = [], []
            for column in columns:
                numbers = [n for n in per_column.get(column, []) if n in decoded]
                spans.append((column, len(frames), len(numbers)))
                frames += [Image.fromarray(decoded[n]) for n in numbers]
            found = detector.detect(frames, texts) if frames else []
            for column, start, length in spans:
                at = int(np.flatnonzero(candidates[i] == column)[0])
                detection[i][at] = value_of(
                    per_phrase_maxima(found[start:start + length], owners,
                                      len(prompts[i])) if length else None)
        log(f"{episode}: {len(needed)} frames decoded for {len(by_query)} queries")

    out_orders, records = [], []
    for i, query in enumerate(queries):
        active = bool(prompts[i])
        if active:
            fused = fuse(detection[i], scores[i][candidates[i]])
            out_orders.append(reorder(orders[i], candidates[i], fused))
        else:
            out_orders.append(orders[i])
        records.append({
            "active": active,
            "n_prompts": len(prompts[i]),
            "base_rank": base_rank_of(orders[i], vids,
                                      relevant[i] if relevant is not None else None),
        })
    return out_orders, records, detection
