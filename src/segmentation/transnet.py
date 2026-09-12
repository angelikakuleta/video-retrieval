"""Learned shot-boundary detection: TransNetV2 (variant E1-C).

The network reads 48x27 RGB frames and returns a per-frame probability of being
in a transition. Inference follows the original repository: the sequence is
padded, processed in windows of 100 frames with a stride of 50, and only the
middle 50 predictions of each window are kept. Frames above the threshold form
transition runs and the boundary is the first frame AFTER a run.
"""

from __future__ import annotations

import numpy as np

THRESHOLD = 0.5
WINDOW = 100
STRIDE = 50
PAD = 25
BATCH = 16


def frame_predictions(frames: np.ndarray, device: str = "cuda",
                      batch: int = BATCH) -> np.ndarray:
    """(N, 27, 48, 3) uint8 -> per-frame transition probabilities (N,)."""
    import torch
    from transnetv2_pytorch import TransNetV2

    model = TransNetV2().eval().to(device)

    n = len(frames)
    # Tail padding as in the original repository: enough copies of the last frame
    # for the sequence to end on a whole stride. The window count comes from the
    # PADDED length, not the frame count -- otherwise the last window runs past the
    # end whenever the count is not a multiple of the stride.
    tail = PAD + STRIDE - (n % STRIDE or STRIDE)
    padded = np.concatenate([
        np.repeat(frames[:1], PAD, axis=0),
        frames,
        np.repeat(frames[-1:], tail, axis=0),
    ])
    starts = list(range(0, len(padded) - WINDOW + 1, STRIDE))

    chunks = []
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            windows = np.stack([padded[s:s + WINDOW] for s in starts[i:i + batch]])
            inputs = torch.from_numpy(windows).to(device)
            single, _ = model(inputs)
            probs = torch.sigmoid(single).squeeze(-1)[:, PAD:PAD + STRIDE]
            chunks.append(probs.reshape(-1).cpu().numpy())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(chunks)[:n]


def boundaries_from_predictions(predictions: np.ndarray, fps: float,
                                threshold: float = THRESHOLD) -> list[float]:
    """Transition runs -> boundary times (first frame after each run)."""
    marked = predictions > threshold
    out = []
    previous = False
    for index, flag in enumerate(marked):
        if previous and not flag:
            out.append(round(index / fps, 3))
        previous = bool(flag)
    return out
