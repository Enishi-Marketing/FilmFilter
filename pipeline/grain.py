"""Procedural grain for believable scanned-print texture."""

from __future__ import annotations

import cv2
import numpy as np

from .tonal import luminance, tonal_masks


def apply_grain(
    image: np.ndarray,
    *,
    grain_amount: float = 0.024,
    grain_size: float = 1.05,
    grain_shadow_bias: float = 0.58,
    grain_chromaticity: float = 0.18,
    strength: float | None = None,
    size: float | None = None,
    chroma: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Overlay fine luminance-dependent grain.

    Real film grain is not a uniform layer pasted over the frame. It clusters at
    different scales, reads more clearly in shadows and midtones, and becomes less
    visible in dense highlights where the image should stay creamy. This stage
    blends several fine procedural noise scales, weights them by luminance, and
    adds only restrained chromatic variation so texture feels scanned rather than
    like digital sensor noise.
    """
    img = np.clip(image.astype(np.float32, copy=False), 0.0, 1.0)
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)

    # Accept the first-pass names so older presets remain usable.
    if strength is not None:
        grain_amount = strength
    if size is not None:
        grain_size = size
    if chroma is not None:
        grain_chromaticity = chroma

    def normalized_noise(shape: tuple[int, ...], sigma: float) -> np.ndarray:
        noise = rng.normal(0.0, 1.0, shape).astype(np.float32)
        if sigma > 0.01:
            noise = cv2.GaussianBlur(noise, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
            if noise.ndim == 2:
                noise = noise[..., None]
        return noise / max(float(np.std(noise)), 1e-6)

    luminance_values = luminance(img)[..., None]
    masks = tonal_masks(luminance_values[..., 0])

    fine_sigma = max(float(grain_size) * 0.22, 0.01)
    medium_sigma = max(float(grain_size) * 0.58, 0.01)
    coarse_sigma = max(float(grain_size) * 1.15, 0.01)
    fine = normalized_noise((h, w, 1), fine_sigma)
    medium = normalized_noise((h, w, 1), medium_sigma)
    coarse = normalized_noise((h, w, 1), coarse_sigma)
    mono_grain = fine * 0.58 + medium * 0.30 + coarse * 0.12
    mono_grain /= max(float(np.std(mono_grain)), 1e-6)

    color_fine = normalized_noise((h, w, 3), max(float(grain_size) * 0.18, 0.01))
    color_medium = normalized_noise((h, w, 3), max(float(grain_size) * 0.48, 0.01))
    color_grain = color_fine * 0.72 + color_medium * 0.28
    color_grain /= max(float(np.std(color_grain)), 1e-6)

    # Grain should not look like a static opacity overlay. Shadows/mids get the
    # most texture; highlights stay cleaner so the shoulder remains creamy.
    shadow_weight = (1.0 - luminance_values) ** (1.15 + np.clip(grain_shadow_bias, 0.0, 1.0) * 0.85)
    mid_weight = masks["midtones"] * 0.62 + np.exp(-((luminance_values - 0.42) ** 2) / 0.08) * 0.38
    highlight_protection = 1.0 - masks["highlights"] * 0.72
    intensity = (0.34 + shadow_weight * 0.62 + mid_weight * 0.34) * highlight_protection

    chroma_mix = np.clip(grain_chromaticity, 0.0, 1.0)
    grain = (mono_grain * (1.0 - chroma_mix) + color_grain * chroma_mix) * grain_amount * intensity

    return np.clip(img + grain, 0.0, 1.0)
