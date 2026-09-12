"""Scene descriptions of the grid frames: the descriptive signal (E3).

Every grid frame gets one sentence and the fragment takes the highest similarity
to any of them (chapter 4). The maximum, rather than one description of the whole
fragment, is what lets something visible in a single frame still be found.

Both generators run greedily, so the same frame always yields the same sentence.
The text is cached separately from its embedding: the sentence depends only on
the generator, the vector also on the base representation, so switching the base
re-encodes but never regenerates -- and generating is the most expensive step of
the pipeline.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from src.data import datasets
from src.features import episode_cache
from src.utils import gpu
from src.segmentation.frames import STEP, iter_batches

#: checkpoints of the two generators, as named in chapter 5
GENERATORS = {
    "blip": "Salesforce/blip-image-captioning-large",
    "llava_1_5_7b": "llava-hf/llava-1.5-7b-hf",
}

#: what LLaVA is asked for; BLIP captions unconditionally
PROMPT = "Describe in one sentence what is happening in this image."

#: longest caption, in tokens
MAX_NEW_TOKENS = 48

#: frames captioned at once (LLaVA runs one at a time regardless)
BATCH = {"blip": 16, "llava_1_5_7b": 1}


def cache_dir(generator: str, dataset: str) -> Path:
    return (datasets.ROOT / "data" / "cache" / "captions" / generator
            / datasets.dataset_dir(dataset))


def episode_jsonl(generator: str, dataset: str, episode: str) -> Path:
    return cache_dir(generator, dataset) / f"{episode}.jsonl"


def embeddings_npz(generator: str, encoder: str, dataset: str, episode: str) -> Path:
    return (datasets.ROOT / "data" / "cache" / "captions" / generator / encoder
            / datasets.dataset_dir(dataset) / f"{episode}.npz")


def load_episode(generator: str, dataset: str, episode: str) -> tuple[np.ndarray, list[str]]:
    """``(times, captions)`` of one episode, in grid order."""
    path = episode_jsonl(generator, dataset, episode)
    times, texts = [], []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            times.append(record["time"])
            texts.append(record["text"])
    return np.asarray(times, dtype=np.float32), texts


def load_embeddings(generator: str, encoder: str, dataset: str,
                    episode: str) -> np.ndarray:
    with np.load(embeddings_npz(generator, encoder, dataset, episode)) as data:
        return data["embeddings"]


class Captioner:
    """One generator behind one method: images in, one sentence each out."""

    def __init__(self, generator: str) -> None:
        import torch

        self.generator = generator
        self.checkpoint = GENERATORS[generator]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        if generator == "blip":
            from transformers import BlipForConditionalGeneration, BlipProcessor

            self.processor = BlipProcessor.from_pretrained(self.checkpoint)
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.checkpoint, dtype=self.dtype).to(self.device).eval()
        else:
            from transformers import AutoProcessor, LlavaForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(self.checkpoint)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.checkpoint, dtype=self.dtype, device_map="auto",
                low_cpu_mem_usage=True).eval()

    def describe(self, images: list) -> list[str]:
        """One sentence per image, decoded greedily."""
        import torch

        with torch.no_grad():
            if self.generator == "blip":
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = inputs.to(self.device, self.dtype)
                output = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                             max_new_tokens=MAX_NEW_TOKENS)
                return [t.strip() for t in
                        self.processor.batch_decode(output, skip_special_tokens=True)]
            return [self._llava(image) for image in images]

    def _llava(self, image) -> str:
        import torch

        conversation = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT}]}]
        inputs = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.model.device, self.dtype)
        with torch.no_grad():
            output = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                         max_new_tokens=MAX_NEW_TOKENS)
        prompt_length = inputs["input_ids"].shape[-1]
        return self.processor.decode(output[0][prompt_length:],
                                     skip_special_tokens=True).strip()


def ensure_captions(
    dataset: str,
    episodes: dict[str, dict],
    generator: str,
    step: float = STEP,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> list[str]:
    """Generates the missing per-episode captions; returns the episodes covered."""
    if generator not in GENERATORS:
        raise ValueError(f"unknown caption generator: {generator!r}")

    box: dict = {}

    def captioner():
        if "model" not in box:
            log(f"loading {generator} ({GENERATORS[generator]})")
            box["model"] = Captioner(generator)
        return box["model"]

    def compute(episode: str, video: Path, target: Path) -> str:
        model = captioner()
        count = 0
        with target.open("w", encoding="utf-8") as handle:
            for times, frames in iter_batches(video, step, BATCH[generator]):
                for time, text in zip(times, model.describe(frames)):
                    handle.write(json.dumps({"time": time, "text": text},
                                            ensure_ascii=False) + "\n")
                    count += 1
        return f"{count} captions"

    try:
        return episode_cache.ensure(dataset, episodes,
                                    lambda ep: episode_jsonl(generator, dataset, ep),
                                    compute, f"captions ({generator})",
                                    force=force, log=log)
    finally:
        # the generator is the biggest model in the pipeline and the encoder that
        # embeds these captions loads right after it; without giving the card back
        # the second one has nowhere to go
        box.clear()
        log("  " + gpu.report("released the generator: "))


def ensure_caption_embeddings(
    dataset: str,
    episodes: dict[str, dict],
    generator: str,
    encoder: str,
    encoder_factory: Callable[[], Any],
    force: bool = False,
    log: Callable[[str], None] = print,
) -> list[str]:
    """Encodes the cached captions with the encoder of the configuration."""

    def compute(episode: str, video: Path, target: Path) -> str:
        _, texts = load_episode(generator, dataset, episode)
        embeddings = (encoder_factory().encode_texts(texts).astype(np.float32)
                      if texts else np.zeros((0, 1), np.float32))
        np.savez_compressed(target, embeddings=embeddings)
        return f"{len(embeddings)} captions encoded"

    with_captions = {ep: entry for ep, entry in episodes.items()
                     if episode_jsonl(generator, dataset, ep).exists()}
    for episode in sorted(set(episodes) - set(with_captions)):
        log(f"  {episode}: no captions - generate them first")
    return episode_cache.ensure(
        dataset, with_captions,
        lambda ep: embeddings_npz(generator, encoder, dataset, ep),
        compute, f"caption embeddings ({generator}/{encoder})",
        force=force, log=log)
