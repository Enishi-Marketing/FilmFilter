"""Color shaping for restrained, emotionally believable film warmth."""

from __future__ import annotations

import numpy as np


def _luminance(image: np.ndarray) -> np.ndarray:
    return np.dot(image, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def apply_color(
    image: np.ndarray,
    *,
    saturation: float = 0.93,
    green_mute: float = 0.09,
    highlight_warmth: float = 0.045,
    skin_magenta_bias: float = 0.018,
) -> np.ndarray:
    """Apply a subtle consumer-film color bias.

    The target look favors warm, forgiving highlights, less electronic-looking
    greens, and a tiny magenta nudge in likely skin ranges. The operation is
    deliberately restrained: it should read as photographic color memory rather
    than a visible orange/teal grade or a preset stamped onto every image.
    """
    img = np.clip(image.astype(np.float32, copy=False), 0.0, 1.0)
    lum = _luminance(img)[..., None]

    # Restrain saturation around luminance to reduce digital harshness.
    img = lum + (img - lum) * saturation

    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]

    # Mute saturated green/yellow-green foliage without turning it gray.
    green_dominance = np.clip(g - np.maximum(r, b), 0.0, 1.0)
    img[..., 1] -= green_dominance * green_mute
    img[..., 0] += green_dominance * green_mute * 0.20

    # Warm only the brighter values so highlights feel like print film paper.
    high_mask = _smoothstep(0.55, 1.0, lum)
    img[..., 0] += high_mask[..., 0] * highlight_warmth
    img[..., 2] -= high_mask[..., 0] * highlight_warmth * 0.45

    # Approximate a skin-friendly region and bias it gently toward magenta.
    skin_mask = (
        (r > g * 0.95)
        & (r > b * 1.05)
        & (g > b * 0.85)
        & (_luminance(img) > 0.22)
        & (_luminance(img) < 0.82)
    ).astype(np.float32)[..., None]
    img[..., 0] += skin_mask[..., 0] * skin_magenta_bias
    img[..., 1] -= skin_mask[..., 0] * skin_magenta_bias * 0.35
    img[..., 2] += skin_mask[..., 0] * skin_magenta_bias * 0.35

    return np.clip(img, 0.0, 1.0)
