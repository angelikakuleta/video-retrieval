"""Component caches and the signals built from them.

:mod:`src.runners.stages` guarantees the collection; this module guarantees
everything a configuration adds on top: captions, object labels, faces and what
is derived from them, motion distributions. Each is computed once per
(component, variant, episode).

What happens here and nowhere else is binding an item to its fragment. The
components work on the frame grid, which covers the whole file and knows nothing
about segmentation; :func:`fragment_of_times` maps one onto the other and applies
the timeline on the way, so a frame inside a mask or a black stretch reaches no
signal. The extraction itself is untouched by that, which is why re-drawing a
mask costs nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from src.evaluation.relevance import fragment_id
from src.segmentation.timeline import Timeline
from src.utils.config import ExperimentConfig

#: name of the vocabulary cache of each closed-vocabulary component
OBJECT_VOCABULARY = "objects_{detector}"
EXPRESSION_VOCABULARY = "expressions_hsemotion"
MOTION_VOCABULARY = "actions_kinetics400"


# ------------------------------------------------ binding items to fragments
def fragment_of_times(rows: list[dict], episode: str, times: np.ndarray,
                      row_of: dict[str, int],
                      timeline: Timeline | None) -> np.ndarray:
    """For every grid moment: the collection row it belongs to, or ``-1``.

    Outside every fragment, or inside one the timeline excludes: both mean the same
    here, that whatever was computed for that moment takes part in no signal.
    """
    own = sorted((r for r in rows if r["episode"] == episode),
                 key=lambda r: r["start"])
    out = np.full(len(times), -1, dtype=np.int32)
    if not own or not len(times):
        return out

    starts = np.asarray([r["start"] for r in own], dtype=np.float64)
    ends = np.asarray([r["end"] for r in own], dtype=np.float64)
    indices = np.asarray([row_of.get(fragment_id(r["episode"], r["segment_id"]), -1)
                          for r in own], dtype=np.int32)

    position = np.searchsorted(starts, times, side="right") - 1
    inside = (position >= 0) & (times < ends[np.clip(position, 0, len(ends) - 1)])
    if timeline is not None:
        inside &= np.asarray(timeline.select(times), dtype=bool)
    out[inside] = indices[position[inside]]
    return out


def _gather(episodes: list[str], rows: list[dict], row_of: dict[str, int],
            timelines: dict[str, Timeline],
            loader: Callable[[str], tuple[np.ndarray, np.ndarray, np.ndarray]],
            ) -> tuple[np.ndarray, np.ndarray]:
    """Collects the items of every episode into one array plus its fragment map.

    ``loader`` reads one episode's cache once and returns ``(times, frames,
    values)``. Items whose frame belongs to no fragment are dropped here.
    """
    values, fragments = [], []
    for episode in episodes:
        times, frames, payload = loader(episode)
        if not len(frames):
            continue
        assigned = fragment_of_times(rows, episode, times, row_of,
                                     timelines.get(episode))
        target = assigned[frames]
        keep = target >= 0
        if not keep.any():
            continue
        values.append(payload[keep])
        fragments.append(target[keep])
    if not values:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.concatenate(values), np.concatenate(fragments).astype(np.int32)


# -------------------------------------------- the signals of a configuration
def build_signals(
    config: ExperimentConfig,
    collection,
    rows: list[dict],
    episodes: dict[str, dict],
    timelines: dict[str, Timeline],
    encoder_factory: Callable[[], Any],
    force: bool = False,
    log: Callable[[str], None] = print,
) -> list:
    """Every signal the configuration enables, with its cache guaranteed."""
    from src.runners import stages

    signals = [stages.scene_signal(config, collection, encoder_factory())]
    components = config.components
    encoder = config.components.scene_embedding.model
    row_of = {vid: i for i, vid in enumerate(collection.vids)}
    size = collection.size
    shared = dict(dataset=config.dataset, rows=rows, row_of=row_of,
                  timelines=timelines, episodes=episodes, encoder=encoder,
                  encoder_factory=encoder_factory, size=size,
                  matching=config.matching, force=force, log=log)

    if components.caption.enabled:
        signals.append(_caption_signal(components.caption.model,
                                       config.frames.step_s, **shared))
    if components.objects.enabled:
        signals.append(_object_signal(components.objects.detector,
                                      config.frames.step_s, **shared))
    if components.face_regions.enabled:
        signals.append(_face_signal(components.face_regions.mode,
                                    components.face_regions.query,
                                    config.frames.step_s, **shared))
    if components.identity.enabled:
        signals.append(_identity_signal(config.frames.step_s, **shared))
    if components.motion.enabled:
        signals.append(_motion_signal(config.collection_strategy, **shared))
    return signals


def _caption_signal(generator, step, *, dataset, rows, row_of, timelines, episodes,
                    encoder, encoder_factory, size, matching, force, log):
    from src.features import captions
    from src.retrieval.signals import CaptionSignal

    captions.ensure_captions(dataset, episodes, generator, step, force=force, log=log)
    covered = captions.ensure_caption_embeddings(dataset, episodes, generator,
                                                 encoder, encoder_factory,
                                                 force=force, log=log)

    def load(episode):
        times, texts = captions.load_episode(generator, dataset, episode)
        # one caption per grid frame, in grid order
        return (times, np.arange(len(texts), dtype=np.int32),
                captions.load_embeddings(generator, encoder, dataset, episode))

    vectors, fragments = _gather(covered, rows, row_of, timelines, load)
    log(f"caption signal: {len(vectors)} descriptions over {size} fragments")
    return CaptionSignal(encoder_factory(), vectors, fragments, size)


def _object_signal(detector, step, *, dataset, rows, row_of, timelines, episodes,
                   encoder, encoder_factory, size, matching, force, log):
    from src.features import objects, text_vocab
    from src.retrieval.signals import ObjectSignal

    covered = objects.ensure_detections(dataset, episodes, detector, step,
                                        force=force, log=log)
    names = objects.load_vocabulary(detector)
    vocabulary = text_vocab.ensure(OBJECT_VOCABULARY.format(detector=detector),
                                   names, encoder, encoder_factory, log=log)
    def load(episode):
        data = objects.load_episode(detector, dataset, episode)
        # the class alone is not enough any more: a fragment is worth the best
        # CONFIDENCE among its detections of the matched class
        return (data["times"], data["frame"],
                np.stack([data["class_id"].astype(np.float32),
                          data["score"].astype(np.float32)], axis=1))

    detections, fragments = _gather(covered, rows, row_of, timelines, load)
    # two columns per detection; the empty case still has to split into (0,) arrays
    detections = detections.reshape(len(fragments), 2)
    log(f"object signal: {len(fragments)} detections over {size} fragments")
    return ObjectSignal(encoder_factory(), vocabulary, names,
                        OBJECT_VOCABULARY.format(detector=detector), matching, size,
                        class_of_item=detections[:, 0].astype(np.int32),
                        score_of_item=detections[:, 1],
                        fragment_of=fragments, log=log)


def _face_signal(mode, query, step, *, dataset, rows, row_of, timelines, episodes,
                 encoder, encoder_factory, size, matching, force, log):
    from src.features import expressions, faces, regions, text_vocab
    from src.retrieval.signals import ExpressionSignal, FaceRegionSignal

    covered = faces.ensure_faces(dataset, episodes, step, force=force, log=log)
    if mode == "hsemotion":
        expressions.ensure_expressions(dataset, episodes, force=force, log=log)
        names = expressions.class_names()
        vocabulary = text_vocab.ensure(EXPRESSION_VOCABULARY, names, encoder,
                                       encoder_factory, log=log)
        def load(episode):
            buffer = faces.load_episode(dataset, episode)
            return (buffer["times"], buffer["frame"],
                    expressions.load_episode(dataset, episode)["distribution"])

        distribution, fragments = _gather(covered, rows, row_of, timelines, load)
        log(f"expression signal: {len(distribution)} faces over {size} fragments")
        return ExpressionSignal(encoder_factory(), vocabulary, names,
                                EXPRESSION_VOCABULARY, matching, size,
                                distribution=distribution, fragment_of=fragments,
                                log=log)

    regions.ensure_regions(dataset, episodes, encoder, encoder_factory,
                           force=force, log=log)
    def load(episode):
        buffer = faces.load_episode(dataset, episode)
        return (buffer["times"], buffer["frame"],
                regions.load_episode(encoder, dataset, episode))

    vectors, fragments = _gather(covered, rows, row_of, timelines, load)
    log(f"face region signal: {len(vectors)} faces over {size} fragments,"
        f" asked with {query}")
    return FaceRegionSignal(encoder_factory(), vectors, fragments, size, query=query)


def _identity_signal(step, *, dataset, rows, row_of, timelines, episodes,
                     encoder, encoder_factory, size, matching, force, log):
    from src.features import faces, identity
    from src.retrieval.signals import IdentitySignal

    covered = faces.ensure_faces(dataset, episodes, step, force=force, log=log)
    identity.ensure_identity(dataset, episodes, force=force, log=log)
    profiles = identity.load_profiles(dataset)
    if not profiles:
        log("identity signal: no profiles - the component stays inactive everywhere")
    def load(episode):
        buffer = faces.load_episode(dataset, episode)
        return (buffer["times"], buffer["frame"],
                identity.load_episode(dataset, episode))

    vectors, fragments = _gather(covered, rows, row_of, timelines, load)
    log(f"identity signal: {len(profiles)} profiles, {len(vectors)} faces")
    return IdentitySignal(profiles, vectors, fragments, size)


def _motion_signal(strategy, *, dataset, rows, row_of, timelines, episodes,
                   encoder, encoder_factory, size, matching, force, log):
    from src.features import motion, text_vocab
    from src.retrieval.signals import MotionSignal

    covered = motion.ensure_motion(dataset, episodes, rows, strategy, timelines,
                                   force=force, log=log)
    names = motion.class_names()
    vocabulary = text_vocab.ensure(MOTION_VOCABULARY, names, encoder,
                                   encoder_factory, log=log)

    # NaN, not a sentinel class: a fragment the model never classified has no
    # value, and filling it with class 0 at probability 1.0 would hand the whole
    # collection a perfect score whenever the query matched that one class
    probability = np.full((size, motion.CLASSES), np.nan, dtype=np.float32)
    classified = 0
    for episode in covered:
        data = motion.load_episode(strategy, dataset, episode)
        for segment, row_values in zip(data["segment_id"], data["probability"]):
            row = row_of.get(fragment_id(episode, int(segment)))
            if row is None:
                continue
            probability[row] = row_values
            classified += 1
    log(f"motion signal: {len(covered)} episodes, {classified} of {size} fragments")
    return MotionSignal(encoder_factory(), vocabulary, names, MOTION_VOCABULARY,
                        matching, size, probability=probability, log=log)
