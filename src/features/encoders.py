"""The encoder of a configuration: one place that decides which one it is.

Chapter 4 has the base representation take over the encoders of the remaining
signals, so with X-CLIP as the base the captions, class names and face crops are
embedded in ITS space. The choice is a property of the configuration, not of the
component.

Both encoders expose ``encode_texts`` and ``encode_images`` returning normalized
vectors, so nothing above this module needs to know which one it got.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.features.frame_cache import OPENCLIP_MODELS

#: identifier of the video-language base representation
XCLIP = "xclip_b32"


def build(model: str) -> Any:
    """Creates the encoder of the given base representation."""
    if model == XCLIP:
        from src.features.xclip import XClipEncoder

        return XClipEncoder()
    if model in OPENCLIP_MODELS:
        from src.features.openclip import ClipEncoder

        name, weights = OPENCLIP_MODELS[model]
        return ClipEncoder(model=name, weights=weights)
    raise ValueError(f"unknown base representation: {model!r}")


def lazy(model: str) -> Callable[[], Any]:
    """A factory creating the encoder at most once, and only if it is needed.

    A run whose caches are all present loads no model at all.
    """
    box: dict = {}

    def factory():
        if "encoder" not in box:
            box["encoder"] = build(model)
        return box["encoder"]

    return factory
