"""Procedural grain for believable scanned-print texture."""

from __future__ import annotations

import cv2
import numpy as np


def apply_grain(
    image: np.ndarray,
    *,
    strength: float = 0.028,
    size: float = 1.15,
    chroma: float = 0.22,
    seed: int | None = None,
) -> np.ndarray:
    """Overlay fine luminance-dependent grain.

    Grain makes smooth digital gradients feel more tactile and scanned, but it is
    most convincing when it remains small, irregular, and weaker in clean bright
    regions. This stage creates mostly monochrome texture with slight chromatic
    variation so the result feels like consumer film grain rather than colored
    sensor noise or an obvious overlay.
    """
    img = np.clip(image.astype(np.float32, copy=False), 0.0, 1.0)
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)

    base = rng.normal(0.0, 1.0, (h, w, 1)).astype(np.float32)
    if size > 0.0:
        base = cv2.GaussianBlur(base, ksize=(0, 0), sigmaX=max(size * 0.35, 0.01))
        if base.ndim == 2:
            base = base[..., None]
    base /= max(float(np.std(base)), 1e-6)

    color_noise = rng.normal(0.0, 1.0, (h, w, 3)).astype(np.float32)
    color_noise = cv2.GaussianBlur(color_noise, ksize=(0, 0), sigmaX=max(size * 0.25, 0.01))
    color_noise /= max(float(np.std(color_noise)), 1e-6)

    luminance = np.dot(img, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))[..., None]
    intensity = 0.55 + (1.0 - luminance) * 0.65
    grain = (base * (1.0 - chroma) + color_noise * chroma) * strength * intensity

    return np.clip(img + grain, 0.0, 1.0)
