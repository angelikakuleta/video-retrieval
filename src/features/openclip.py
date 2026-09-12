"""OpenCLIP encoder, shared by the scene signal and query encoding.

Loaded once, encodes images and text into a common space. The returned vectors
are L2-normalized, so a dot product is the cosine similarity -- the FAISS index
and every component signal rely on it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

MODEL = "ViT-H-14"
WEIGHTS = "laion2b_s32b_b79k"
BATCH = 64


def _device(choice: str | None) -> str:
    if choice:
        return choice
    return "cuda" if torch.cuda.is_available() else "cpu"


class ClipEncoder:
    """Base (scene) representation -- OpenCLIP image and text encoder."""

    def __init__(
        self,
        model: str = MODEL,
        weights: str = WEIGHTS,
        device: str | None = None,
        batch: int = BATCH,
    ) -> None:
        import open_clip

        self.device = _device(device)
        self.batch = batch
        # create_model_and_transforms has no return annotation, so the model API and
        # the transform are marked as an untyped library boundary rather than given
        # an invented concrete type.
        net, _, transform = open_clip.create_model_and_transforms(
            model, pretrained=weights, device=self.device
        )
        net.eval()
        self.model: Any = net
        self.transform: Any = transform
        self.tokenizer = open_clip.get_tokenizer(model)

    @property
    def dim(self) -> int:
        return self.model.visual.output_dim

    @torch.no_grad()
    def encode_images(self, images: list) -> np.ndarray:
        """Encodes images into a matrix (n, dim).

        Takes ``PIL.Image`` objects and RGB arrays alike: the grid frames arrive
        decoded, the face crops as arrays out of their buffer, and both encoders must
        take either -- that is what makes the choice of base representation invisible to
        the components above.
        """
        result = []
        for start in range(0, len(images), self.batch):
            chunk = [_as_image(img) for img in images[start : start + self.batch]]
            batch = torch.stack([self.transform(img) for img in chunk]).to(self.device)
            features = self.model.encode_image(batch)
            result.append(_normalize(features))
        return np.concatenate(result, axis=0)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encodes a list of texts into a matrix (n, dim)."""
        result = []
        for start in range(0, len(texts), self.batch):
            chunk = texts[start : start + self.batch]
            tokens = self.tokenizer(chunk).to(self.device)
            features = self.model.encode_text(tokens)
            result.append(_normalize(features))
        return np.concatenate(result, axis=0)


def _as_image(image):
    """An RGB array becomes a ``PIL.Image``; anything else passes through."""
    if isinstance(image, np.ndarray):
        from PIL import Image

        return Image.fromarray(image.astype(np.uint8))
    return image


def _normalize(features: torch.Tensor) -> np.ndarray:
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)
