"""Gentle lens imperfections for consumer-camera softness."""

from __future__ import annotations

import cv2
import numpy as np


def _radial_distance(height: int, width: int) -> np.ndarray:
    y, x = np.ogrid[-1.0:1.0:complex(height), -1.0:1.0:complex(width)]
    return np.sqrt(x * x + y * y).astype(np.float32)


def _shift_channel(channel: np.ndarray, shift_x: float, shift_y: float) -> np.ndarray:
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(channel, matrix, (channel.shape[1], channel.shape[0]), borderMode=cv2.BORDER_REFLECT)


def apply_lens(
    image: np.ndarray,
    *,
    vignette: float = 0.10,
    edge_softness: float = 0.18,
    aberration: float = 0.25,
) -> np.ndarray:
    """Introduce small optical imperfections without announcing themselves.

    Disposable and toy cameras rarely render corners as crisply or as evenly as
    the center. A restrained vignette, barely softer edges, and optional subpixel
    color separation create a consumer-grade photographic feeling while avoiding
    the heavy blur and extreme dark corners associated with novelty filters.
    """
    img = np.clip(image.astype(np.float32, copy=False), 0.0, 1.0)
    h, w = img.shape[:2]
    radius = _radial_distance(h, w)
    edge_mask = np.clip((radius - 0.35) / 0.85, 0.0, 1.0)[..., None]

    if edge_softness > 0.0:
        blurred = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=0.8 + edge_softness * 2.0)
        img = img * (1.0 - edge_mask * edge_softness) + blurred * (edge_mask * edge_softness)

    if aberration > 0.0:
        px = float(aberration)
        shifted = img.copy()
        shifted[..., 0] = _shift_channel(img[..., 0], px, 0.0)
        shifted[..., 2] = _shift_channel(img[..., 2], -px, 0.0)
        img = shifted

    if vignette > 0.0:
        falloff = 1.0 - np.clip(radius / 1.35, 0.0, 1.0) ** 2 * vignette
        img = img * falloff[..., None]

    return np.clip(img, 0.0, 1.0)
