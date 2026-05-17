"""Tone shaping utilities for a soft consumer-film rendering."""

from __future__ import annotations

import numpy as np


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Return a smooth 0..1 transition used for gentle tonal blends."""
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def apply_tone(
    image: np.ndarray,
    *,
    lifted_black: float = 0.045,
    contrast: float = 0.92,
    highlight_rolloff: float = 0.35,
    rolloff_start: float = 0.62,
    midtone_preservation: float = 0.65,
) -> np.ndarray:
    """Shape scene contrast into a gentler print-like response.

    Consumer print scans often feel forgiving because deep shadows are not fully
    black and bright areas compress before clipping. This transform intentionally
    protects midtones while lifting the floor and rounding highlights, giving the
    image a nostalgic softness without making it look flat or artificially faded.
    """
    img = np.clip(image.astype(np.float32, copy=False), 0.0, 1.0)

    # Reduce harsh global contrast around the photographic middle gray region.
    contrasted = 0.5 + (img - 0.5) * contrast

    # Lift blacks by raising the toe of the curve while leaving midtones usable.
    lifted = contrasted + lifted_black * (1.0 - contrasted) ** 2

    # Compress the upper tonal range with a smooth shoulder rather than a clamp.
    shoulder = _smoothstep(rolloff_start, 1.0, lifted)
    compressed_highs = rolloff_start + (1.0 - rolloff_start) * (
        1.0 - np.exp(-(lifted - rolloff_start) / max(1.0 - rolloff_start, 1e-6))
    )
    rolled = lifted * (1.0 - shoulder * highlight_rolloff) + compressed_highs * (
        shoulder * highlight_rolloff
    )

    # Blend back some original midtone structure so faces and ordinary objects do
    # not become milky while shadows/highlights still receive the filmic curve.
    luminance = np.dot(img, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
    mid_mask = 1.0 - np.abs(luminance - 0.5) * 2.0
    mid_mask = np.clip(mid_mask[..., None], 0.0, 1.0) * midtone_preservation
    result = rolled * (1.0 - mid_mask) + img * mid_mask

    return np.clip(result, 0.0, 1.0)
