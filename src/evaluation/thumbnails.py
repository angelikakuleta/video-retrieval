"""Frames of a fragment, sampled on the content axis and cached as JPEG.

The only step of the qualitative examples that touches a recording. Everything
else reads run directories, so this module is what decides which moment of a
fragment a reader actually sees.

Three moments per fragment, at the MIDPOINTS of three equal parts of its usable
content: one sixth, one half, five sixths. Two properties come out of that and
both matter on the page.

The content axis, not the file axis. A fragment may hold a transition mask or a
stretch of black picture, and those seconds do not count as content
(:class:`src.segmentation.timeline.Timeline`); sampling the file linearly would
land inside the animation announcing a scene change and the figure would look
like a bug in the system. Of the fragments the six examples show, four carry
such a mask.

Midpoints rather than edges. Sampling at the start of a fragment lands on the
cut itself, which in a series is often a cross-fade or a black frame -- a
picture of nothing, in the row a reader looks at first.

Cached under ``data/cache/thumbnails/``, so it is reproducible from the
recordings and deleted like any other cache. The files are NOT redistributable
and neither are the figures built from them (see ``.gitignore``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from src.data import datasets

#: height of a cached frame in pixels. A drawn frame is 24.68 mm tall
#: (:mod:`src.evaluation.sheets`), which is 292 px at the 300 dpi the figures
#: are saved at, so 360 px never has to be upscaled.
HEIGHT = 360

#: JPEG quality. These are photographs of a television picture reduced to a
#: twentieth of their area; 90 leaves no visible artefact at that size and keeps
#: the whole cache around ten megabytes.
QUALITY = 90

#: frames per fragment, and where they sit in its content
COUNT = 3


def cache_dir(dataset: str) -> Path:
    return datasets.ROOT / "data" / "cache" / "thumbnails" / datasets.dataset_dir(dataset)


def frame_path(dataset: str, fragment: str, index: int) -> Path:
    return cache_dir(dataset) / f"{fragment}_{index}.jpg"


def frame_times(timeline, start: float, end: float, count: int = COUNT) -> list[float]:
    """Midpoints of ``count`` equal parts of the fragment's usable content.

    Offsets are measured along the content, so a masked stretch inside the
    fragment shifts the samples instead of being sampled. Returns times on the
    FILE axis, which is what a decoder seeks to.

    :meth:`Timeline.sample_content` answers a neighbouring question -- it places
    samples at the START of each part, which for three frames puts them at 0,
    1/3 and 2/3 of the fragment and leaves the last third unrepresented. Here
    the first sample must also stay off the cut that opens the fragment, so the
    midpoints are computed rather than reused.
    """
    pieces = timeline.usable_pieces(start, end)
    if not pieces:
        return [start] * count
    length = sum(b - a for a, b in pieces)
    if length <= 0:
        return [pieces[0][0]] * count

    wanted = [length * (2 * k + 1) / (2 * count) for k in range(count)]
    out, position, index = [], 0.0, 0
    for a, b in pieces:
        while index < count and position <= wanted[index] < position + (b - a):
            out.append(a + (wanted[index] - position))
            index += 1
        position += b - a
    while len(out) < count:                      # rounding at the tail
        out.append(pieces[-1][1] - 1e-6)
    return out


def ensure(dataset: str, hits: Iterable, count: int = COUNT, force: bool = False,
           log: Callable[[str], None] = print) -> dict[str, list[Path]]:
    """Decodes the missing frames of the given fragments; returns their paths.

    ``hits`` are :class:`src.evaluation.examples.Hit` records, or anything with
    ``fragment``, ``episode``, ``video_file``, ``start`` and ``end``. One
    recording is opened once and its wanted frames are read in time order, so a
    page of an example costs a couple of seconds rather than a seek per frame.
    """
    import cv2

    wanted = {hit.fragment: hit for hit in hits}
    directory = cache_dir(dataset)
    directory.mkdir(parents=True, exist_ok=True)

    paths = {fragment: [frame_path(dataset, fragment, index)
                        for index in range(count)]
             for fragment in wanted}
    missing = {fragment: hit for fragment, hit in wanted.items()
               if force or not all(path.exists() for path in paths[fragment])}
    if not missing:
        return paths

    from src.data.ranges import load_ranges, load_timelines

    timelines = load_timelines(dataset, load_ranges(dataset))

    by_recording: dict[str, list] = {}
    for hit in missing.values():
        by_recording.setdefault(hit.video_file, []).append(hit)

    log(f"{dataset}: thumbnails for {len(missing)} fragments "
        f"in {len(by_recording)} recordings")
    for video_file, group in sorted(by_recording.items()):
        path = datasets.video_path(dataset, video_file)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing - the recordings are not in the repository "
                "(see README, 'Dostepnosc danych'); the examples need them once, "
                "after which the thumbnail cache is enough")
        reader = cv2.VideoCapture(str(path))
        try:
            fps = reader.get(cv2.CAP_PROP_FPS) or 0.0
            last = int(reader.get(cv2.CAP_PROP_FRAME_COUNT) or 0) - 1
            if fps <= 0 or last < 0:
                raise ValueError(f"could not read the video: {path}")
            # in time order: a forward seek inside one file is cheap, a backward
            # one makes the decoder start from the previous keyframe again
            plan = sorted(
                ((moment, hit.fragment, index)
                 for hit in group
                 for index, moment in enumerate(frame_times(
                     timelines[hit.episode], hit.start, hit.end, count))),
                key=lambda item: item[0])
            for moment, fragment, index in plan:
                target = paths[fragment][index]
                if target.exists() and not force:
                    continue
                reader.set(cv2.CAP_PROP_POS_FRAMES, min(round(moment * fps), last))
                ok, frame = reader.read()
                if not ok or frame is None:
                    raise ValueError(
                        f"no frame at {moment:.2f} s of {path.name} "
                        f"(fragment {fragment})")
                height, width = frame.shape[:2]
                scaled = cv2.resize(
                    frame, (max(1, round(width * HEIGHT / height)), HEIGHT),
                    interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(target), scaled,
                            [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY])
        finally:
            reader.release()
    return paths
