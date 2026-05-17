"""Color shaping for restrained, emotionally believable film warmth."""

from __future__ import annotations

import numpy as np

from .tonal import blend_with_mask, luminance, smoothstep, tonal_masks


def _compress_saturation(
    image: np.ndarray,
    *,
    saturation_compression: float,
    highlight_desaturation: float,
) -> np.ndarray:
    """Reduce color purity where film and print scans tend to compress color."""
    lum = luminance(image)[..., None]
    masks = tonal_masks(lum[..., 0])
    max_channel = np.max(image, axis=-1, keepdims=True)
    clipping_pressure = smoothstep(0.68, 1.0, max_channel)

    compression = (
        masks["highlights"] * (0.55 * saturation_compression + highlight_desaturation)
        + masks["shadows"] * saturation_compression * 0.14
        + clipping_pressure * saturation_compression * 0.30
    )
    midtone_protection = masks["midtones"] * 0.28
    saturation_scale = 1.0 - np.clip(compression * (1.0 - midtone_protection), 0.0, 0.85)
    return lum + (image - lum) * saturation_scale


def _apply_cross_channel_response(
    image: np.ndarray,
    *,
    crossover_strength: float,
    shadow_color_shift: float,
    highlight_warmth: float,
) -> np.ndarray:
    """Let channels influence one another in a restrained film-stock direction."""
    strength = np.clip(crossover_strength, 0.0, 1.0)
    shadow_shift = np.clip(shadow_color_shift, -1.0, 1.0)
    if strength <= 0.0 and abs(shadow_shift) <= 1e-6:
        return image

    img = image.copy()
    masks = tonal_masks(img)
    lum = luminance(img)[..., None]
    r = img[..., 0:1]
    g = img[..., 1:2]
    b = img[..., 2:3]

    # A tiny non-orthogonal matrix makes colors interact before selective rolloff.
    mixed = np.empty_like(img)
    mixed[..., 0:1] = r + (g - b) * 0.025 * strength
    mixed[..., 1:2] = g + (r - g) * 0.012 * strength + (b - g) * 0.010 * strength
    mixed[..., 2:3] = b + (g - r) * 0.018 * strength
    img = img * (1.0 - strength * 0.22) + mixed * (strength * 0.22)

    r = img[..., 0:1]
    g = img[..., 1:2]
    b = img[..., 2:3]
    max_gb = np.maximum(g, b)
    red_excess = np.clip(r - max_gb, 0.0, 1.0)
    red_pressure = smoothstep(0.66, 0.98, r)
    img[..., 0:1] -= red_excess * red_pressure * masks["highlights"] * strength * 0.22
    img[..., 1:2] += red_excess * red_pressure * masks["highlights"] * strength * 0.045

    green_dominance = np.clip(g - np.maximum(r, b), 0.0, 1.0)
    olive = img.copy()
    olive[..., 0:1] += green_dominance * strength * 0.12
    olive[..., 1:2] -= green_dominance * strength * 0.16
    olive[..., 2:3] -= green_dominance * strength * 0.055
    img = blend_with_mask(img, olive, masks["midtones"] + masks["highlights"] * 0.35, strength=0.75)

    blue_dominance = np.clip(b - np.maximum(r, g), 0.0, 1.0)
    blue_compressed = lum + (img - lum) * (1.0 - blue_dominance * masks["shadows"] * strength * 0.55)
    img = blend_with_mask(img, blue_compressed, masks["shadows"], strength=0.62)

    cyan_bias = np.array([-0.010, 0.006, 0.012], dtype=np.float32) * shadow_shift
    img += masks["shadows"] * cyan_bias * (0.45 + strength * 0.55)

    cream = np.array([1.0, 0.935, 0.825], dtype=np.float32)
    highlight_pressure = masks["highlights"] * smoothstep(0.72, 1.0, np.max(img, axis=-1, keepdims=True))
    cream_mix = highlight_pressure * (highlight_warmth * 0.85 + strength * 0.030)
    img = img * (1.0 - cream_mix) + lum * cream * cream_mix

    return img


def apply_color(
    image: np.ndarray,
    *,
    saturation: float = 0.93,
    green_mute: float = 0.09,
    highlight_warmth: float = 0.045,
    skin_magenta_bias: float = 0.018,
    saturation_compression: float = 0.22,
    highlight_desaturation: float = 0.10,
    shadow_color_shift: float = 0.28,
    crossover_strength: float = 0.34,
) -> np.ndarray:
    """Apply subtle cross-channel consumer-film color rendering.

    The target look favors warm, forgiving highlights, less electronic-looking
    greens, nonlinear saturation, and a tiny magenta nudge in likely skin ranges.
    Film and print scans do not behave like isolated RGB sliders: dye layers,
    exposure density, paper response, and scanning all let channels contaminate
    and compress each other. These small interactions make bright colors pastelize
    near clipping, cool shadows feel slightly impure, and foliage drift toward
    olive without turning the image into a visible cinematic grade.
    """
    img = np.clip(image.astype(np.float32, copy=False), 0.0, 1.0)
    lum = luminance(img)[..., None]

    # Restrain saturation around luminance to reduce digital harshness.
    img = lum + (img - lum) * saturation
    img = _apply_cross_channel_response(
        img,
        crossover_strength=crossover_strength,
        shadow_color_shift=shadow_color_shift,
        highlight_warmth=highlight_warmth,
    )
    img = _compress_saturation(
        img,
        saturation_compression=saturation_compression,
        highlight_desaturation=highlight_desaturation,
    )

    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]

    # Mute saturated green/yellow-green foliage without turning it gray.
    green_dominance = np.clip(g - np.maximum(r, b), 0.0, 1.0)
    img[..., 1] -= green_dominance * green_mute
    img[..., 0] += green_dominance * green_mute * 0.20

    # Warm only the brighter values so highlights feel like print film paper.
    high_mask = smoothstep(0.55, 1.0, lum)
    img[..., 0] += high_mask[..., 0] * highlight_warmth
    img[..., 2] -= high_mask[..., 0] * highlight_warmth * 0.45

    # Approximate a skin-friendly region and bias it gently toward magenta.
    skin_luma = luminance(img)
    skin_mask = (
        (r > g * 0.95)
        & (r > b * 1.05)
        & (g > b * 0.85)
        & (skin_luma > 0.22)
        & (skin_luma < 0.82)
    ).astype(np.float32)[..., None]
    img[..., 0] += skin_mask[..., 0] * skin_magenta_bias
    img[..., 1] -= skin_mask[..., 0] * skin_magenta_bias * 0.35
    img[..., 2] += skin_mask[..., 0] * skin_magenta_bias * 0.35

    return np.clip(img, 0.0, 1.0)
