"""X-CLIP encoder: the video-language base representation of experiment E2.

WHICH X-CLIP: the ACTION-RECOGNITION model of Ni et al. (ECCV 2022), published as
``microsoft/xclip-*``. A second, unrelated 2022 model carries the same name -- Ma
et al. (ACM MM 2022), built for text-video retrieval -- and it is that one the
retrieval literature means by "X-CLIP". This project uses Ni et al. on purpose:
E2 compares two ways of representing MOTION (against SlowFast), not two retrieval
models, and every component here works without training on the target material.
The thesis says the same in a footnote in chapter 3.

Chapter 4: eight frames out of a 64-frame window, the same window length as
SlowFast, so the comparison is about how motion is represented and not about how
much context each model saw. Fragments longer than one window are covered by
consecutive windows and averaged.

Two variants share this encoder and one index. Without prompting the fragment is
matched against its stored vector, like any contrastive representation. With it,
the full video-specific prompting mechanism shifts the query embedding by a term
derived from the fragment being scored, which is why the query phase then grows
with the collection -- what a fragment leaves behind is therefore both the window
embedding and the patch features.
"""

from __future__ import annotations

import numpy as np

MODEL = "microsoft/xclip-base-patch32"
WINDOW_FRAMES = 64          # source frames per window (matches SlowFast 8x8)
FRAMES_PER_WINDOW = 8       # frames the model consumes, sampled every 8th
TEXT_BATCH = 64
VIDEO_BATCH = 8
#: fragments conditioned at once during the query phase -- the patch features of
#: the whole collection do not fit on the card in one go
FRAGMENT_BATCH = 256


def window_offsets(length: float, span: float, fps: float) -> list[float]:
    """Window starts of one fragment, measured from its beginning.

    Whole windows are tiled from the start; when a tail is left over, ONE more
    window is added, pushed against the END of the fragment. Every window keeps
    its full ``span`` -- the length chapter 4 ties to SlowFast 8x8 -- and the
    whole fragment is covered, at the price of the last window overlapping the
    one before it. Plain integer division used to drop the tail instead, which
    cost about a fifth of a 10 s fragment.

    ``length`` and ``span`` are in the same unit (seconds of content, or frames
    with ``fps = 1``); ``fps`` only sets how close two starts may be.
    """
    count = max(1, max(1, round(length * fps)) // max(1, round(span * fps)))
    out = [w * span for w in range(count)]
    tail = length - span
    if length > span and tail > out[-1] + 0.5 / fps:
        out.append(tail)
    return out


def window_frame_indices(start_frame: int, n_frames: int,
                         last_frame: int) -> list[list[int]]:
    """Frame indices of the consecutive windows covering one fragment.

    The plain version, where there is no timeline to consult (VATEX, tests).
    """
    step = WINDOW_FRAMES // FRAMES_PER_WINDOW
    limit = min(start_frame + n_frames - 1, last_frame)
    return [[min(start_frame + round(offset) + k * step, limit)
             for k in range(FRAMES_PER_WINDOW)]
            for offset in window_offsets(n_frames, WINDOW_FRAMES, 1.0)]



class XClipEncoder:
    """Video and text encoder sharing the interface of ``ClipEncoder``."""

    def __init__(self, model: str = MODEL, device: str | None = None,
                 text_batch: int = TEXT_BATCH, video_batch: int = VIDEO_BATCH) -> None:
        import torch
        from transformers import AutoProcessor, XCLIPModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.text_batch = text_batch
        self.video_batch = video_batch
        self.model = XCLIPModel.from_pretrained(model).eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(model)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Normalized text embeddings; the contract shared with ``ClipEncoder``.

        Used by every signal comparing a query with stored text. The scene signal does
        not go through here when prompting is on: it starts from
        :meth:`encode_texts_projected`.
        """
        embeddings = self.encode_texts_projected(texts)
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        return (embeddings / np.where(norms > 0, norms, 1.0)).astype(np.float32)

    def encode_texts_projected(self, texts: list[str]) -> np.ndarray:
        """Text embeddings BEFORE normalization, as the prompt module expects.

        The generator adds a video-conditioned term and only the sum is normalized, so
        normalizing first would change the result.
        """
        import torch

        result = []
        with torch.no_grad():
            for start in range(0, len(texts), self.text_batch):
                chunk = texts[start:start + self.text_batch]
                inputs = self.processor(text=chunk, return_tensors="pt",
                                        padding=True).to(self.device)
                pooled = self.model.text_model(**inputs)[1]
                result.append(self.model.text_projection(pooled).cpu().numpy())
        return np.concatenate(result, axis=0).astype(np.float32)

    def encode_windows_with_prompts(
        self, windows: list[list[np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(video_embeds, prompt_features)`` for the given windows.

        The normalized window representation the index stores, and the patch-level
        features the prompt module conditions the query on.
        """
        import torch

        videos, prompts = [], []
        with torch.no_grad():
            for start in range(0, len(windows), self.video_batch):
                chunk = windows[start:start + self.video_batch]
                inputs = self.processor(videos=chunk, return_tensors="pt").to(self.device)
                video, prompt = self._vision_pass(inputs["pixel_values"])
                videos.append(_normalize(video))
                prompts.append(prompt.cpu().numpy().astype(np.float32))
        return np.concatenate(videos, axis=0), np.concatenate(prompts, axis=0)

    def _vision_pass(self, pixel_values):
        """One vision pass -> (window embedding, patch features).

        Mirrors ``XCLIPModel.forward`` up to the point where both are ready, so nothing
        here reimplements the model.
        """
        batch, frames = pixel_values.shape[:2]
        outputs = self.model.vision_model(
            pixel_values=pixel_values.reshape(-1, *pixel_values.shape[2:]))
        projected = self.model.visual_projection(outputs[1])
        video = self.model.mit(projected.view(batch, frames, -1))[1]

        patches = self.model.prompts_visual_layernorm(outputs[0][:, 1:, :])
        patches = patches @ self.model.prompts_visual_projection
        patches = patches.view(batch, frames, -1, video.shape[-1]).mean(dim=1)
        return video, patches

    def encode_images(self, images: list) -> np.ndarray:
        """Encodes single images (face crops) with the same vision tower.

        The tower passes messages between the frames of a clip, so it cannot take a lone
        image: the layer wants whole clips and what it returns for one frame depends on
        the others. Each crop is therefore encoded as a still clip -- the frame repeated
        :data:`FRAMES_PER_WINDOW` times -- which makes the exchange degenerate to the
        frame with itself and the result independent of how the batch was cut.
        """
        import torch

        processor = self.processor.image_processor
        result = []
        with torch.no_grad():
            for start in range(0, len(images), self.video_batch):
                chunk = images[start:start + self.video_batch]
                pixel_values = processor(images=[np.asarray(i) for i in chunk],
                                         return_tensors="pt")["pixel_values"]
                pixel_values = pixel_values.reshape(-1, *pixel_values.shape[-3:])
                still = pixel_values.repeat_interleave(FRAMES_PER_WINDOW, dim=0)
                pooled = self.model.vision_model(pixel_values=still.to(self.device))[1]
                projected = self.model.visual_projection(pooled)
                result.append(_normalize(projected[::FRAMES_PER_WINDOW]))
        return np.concatenate(result, axis=0)

    def prompted_similarity(self, text_embeds: np.ndarray, video_embeds: np.ndarray,
                            prompt_features: np.ndarray,
                            batch: int = FRAGMENT_BATCH) -> np.ndarray:
        """Text-video match with the video-specific prompts: (n_texts, n_fragments).

        For every fragment the query is shifted by a term derived from that fragment's
        patch features and only then compared, so the cost grows with the collection.
        """
        import torch

        device = self.device
        texts = torch.as_tensor(text_embeds, device=device)
        scores = np.empty((len(text_embeds), len(video_embeds)), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(video_embeds), batch):
                stop = min(start + batch, len(video_embeds))
                video = torch.as_tensor(video_embeds[start:stop], device=device)
                patches = torch.as_tensor(prompt_features[start:stop], device=device)
                # (n_fragments, n_texts, dim): every text is conditioned on every
                # fragment, exactly as in the model's own forward pass
                expanded = texts.unsqueeze(0).expand(video.shape[0], -1, -1)
                conditioned = expanded + self.model.prompts_generator(expanded, patches)
                conditioned = conditioned / conditioned.norm(p=2, dim=-1, keepdim=True)
                video = video / video.norm(p=2, dim=-1, keepdim=True)
                block = torch.einsum("bd,bkd->bk", video, conditioned)
                scores[:, start:stop] = block.T.cpu().numpy()
        return scores


def _normalize(features) -> np.ndarray:
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)


def fragment_windows_for_episode(
    fragments: list[dict], fps: float, frame_count: int, timeline=None,
) -> tuple[list[list[list[int]]], list[int]]:
    """Window frame indices per fragment, plus the sorted set of frames needed.

    With a ``timeline`` the windows are laid out on the content axis, so no frame of
    a mask or of a black stretch reaches the model.
    """
    per_fragment = []
    needed: set[int] = set()
    for row in fragments:
        start, end = float(row["start"]), float(row["end"])
        if timeline is None:
            windows = window_frame_indices(round(start * fps),
                                           max(1, round((end - start) * fps)),
                                           frame_count - 1)
        else:
            windows = _content_windows(start, end, timeline, fps, frame_count - 1)
        per_fragment.append(windows)
        for window in windows:
            needed.update(window)
    return per_fragment, sorted(needed)


def _content_windows(start: float, end: float, timeline, fps: float,
                     last_frame: int) -> list[list[int]]:
    """Windows of one fragment, tiled and sampled on the content axis."""
    content = timeline.content_length(start, end)
    span = WINDOW_FRAMES / fps
    out = []
    for offset in window_offsets(content, span, fps):
        a = timeline.advance(start, min(offset, content))
        b = timeline.advance(start, min(offset + span, content))
        times = timeline.sample_content(a, max(b, a + 1.0 / fps), FRAMES_PER_WINDOW)
        out.append([min(round(t * fps), last_frame) for t in times])
    return out


def collect_frames(path, needed: list[int],
                   height: int | None = 224) -> dict[int, np.ndarray]:
    """Decodes the file sequentially and keeps the needed frames (RGB uint8).

    Memory stays bounded by the number of frames wanted, not by the file length.

    ``height`` resizes the shorter side, which the sequence models need; pass
    ``None`` to keep the source resolution. Resizing is not a property of
    sequential decoding -- the detector run at query time wants the same frames
    the indexing pass saw, and those are not resized.
    """
    import cv2

    reader = cv2.VideoCapture(str(path))
    wanted = set(needed)
    out: dict[int, np.ndarray] = {}
    index = 0
    try:
        while wanted:
            ok, frame = reader.read()
            if not ok:
                break
            if index in wanted:
                if height is not None:
                    h, w = frame.shape[:2]
                    size = (max(1, round(w * height / h)), height)
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                out[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                wanted.discard(index)
            index += 1
    finally:
        reader.release()
    if wanted:   # file ended early -- clamp to the last decoded frame
        last = out[max(out)] if out else None
        if last is None:
            raise ValueError(f"no frames decoded from: {path}")
        for missing in wanted:
            out[missing] = last
    return out
