"""Procedural media texture for believable scanned-print rendering."""

from __future__ import annotations

import cv2
import numpy as np

from .tonal import luminance, smoothstep, tonal_masks


def apply_grain(
    image: np.ndarray,
    *,
    grain_amount: float = 0.024,
    grain_size: float = 1.05,
    grain_shadow_bias: float = 0.58,
    grain_chromaticity: float = 0.18,
    micro_grain_amount: float = 0.018,
    mid_grain_amount: float = 0.008,
    density_variation_amount: float = 0.004,
    texture_scale_balance: float = 0.48,
    scanner_softness: float = 0.05,
    tonal_diffusion: float = 0.04,
    edge_softening: float = 0.04,
    chroma_instability: float = 0.018,
    density_instability: float = 0.012,
    scan_irregularity: float = 0.012,
    strength: float | None = None,
    size: float | None = None,
    chroma: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Render image content through layered photographic media texture.

    Scanned consumer film rarely separates clean image structure from texture.
    Grain, paper density, scanner softness, shadow impurity, and tiny alignment
    errors all interact at low strength. This stage keeps those cues subtle by
    layering micro grain, mid-frequency texture, and extremely low-frequency
    density variation, then coupling them to luminance, local contrast, color
    purity, and edge transitions. The intent is not visible damage or retro
    spectacle; it is a quieter sense that the picture passed through a physical
    medium before becoming pixels.
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

    def normalized_map(values: np.ndarray) -> np.ndarray:
        values = values - float(np.mean(values))
        return values / max(float(np.std(values)), 1e-6)

    def normalized_noise(shape: tuple[int, ...], sigma: float) -> np.ndarray:
        noise = rng.normal(0.0, 1.0, shape).astype(np.float32)
        if sigma > 0.01:
            noise = cv2.GaussianBlur(noise, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
            if noise.ndim == 2:
                noise = noise[..., None]
        return normalized_map(noise)

    def shift_channel(channel: np.ndarray, shift_x: float, shift_y: float) -> np.ndarray:
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        return cv2.warpAffine(channel, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    luminance_values = luminance(img)[..., None]
    masks = tonal_masks(luminance_values[..., 0])
    lum2d = luminance_values[..., 0]

    local_luma = cv2.GaussianBlur(lum2d, ksize=(0, 0), sigmaX=1.05, sigmaY=1.05)
    local_contrast = np.abs(lum2d - local_luma)[..., None]
    edge_map = smoothstep(0.018, 0.105, local_contrast)
    harsh_edge = smoothstep(0.035, 0.14, local_contrast)

    balance = np.clip(texture_scale_balance, 0.0, 1.0)
    micro = normalized_noise((h, w, 1), max(float(grain_size) * 0.16, 0.01))
    fine = normalized_noise((h, w, 1), max(float(grain_size) * 0.34, 0.01))
    mid = normalized_noise((h, w, 1), max(float(grain_size) * (1.15 + balance * 1.10), 0.01))
    low = normalized_noise((h, w, 1), max(min(h, w) * (0.038 + balance * 0.030), 1.8))

    micro_layer = normalized_map(micro * 0.72 + fine * 0.28)
    mid_layer = normalized_map(mid * 0.78 + fine * 0.22)
    density_layer = normalized_map(low)
    mono_grain = normalized_map(
        micro_layer * (0.70 - balance * 0.18)
        + mid_layer * (0.20 + balance * 0.22)
        + density_layer * (0.05 + balance * 0.02)
    )

    color_fine = normalized_noise((h, w, 3), max(float(grain_size) * 0.18, 0.01))
    color_medium = normalized_noise((h, w, 3), max(float(grain_size) * 0.72, 0.01))
    color_grain = normalized_map(color_fine * 0.66 + color_medium * 0.34)

    # Texture is exposure-aware and edge-aware. Highlights remain creamy, while
    # hard digital transitions get less additive grain and more soft integration.
    shadow_weight = (1.0 - luminance_values) ** (1.15 + np.clip(grain_shadow_bias, 0.0, 1.0) * 0.85)
    mid_weight = masks["midtones"] * 0.62 + np.exp(-((luminance_values - 0.42) ** 2) / 0.08) * 0.38
    highlight_protection = 1.0 - masks["highlights"] * 0.78
    contrast_texture = 1.0 + smoothstep(0.004, 0.05, local_contrast) * 0.16
    edge_attenuation = 1.0 - edge_map * np.clip(edge_softening, 0.0, 1.0) * 0.48
    intensity = (0.30 + shadow_weight * 0.62 + mid_weight * 0.32) * highlight_protection
    intensity = intensity * contrast_texture * edge_attenuation

    chroma_mix = np.clip(grain_chromaticity, 0.0, 1.0)
    micro_strength = np.clip(micro_grain_amount, 0.0, 0.12)
    mid_strength = np.clip(mid_grain_amount, 0.0, 0.08)
    density_strength = np.clip(density_variation_amount, 0.0, 0.04)
    legacy_strength = np.clip(grain_amount, 0.0, 0.20)

    luma_texture = (
        micro_layer * micro_strength
        + mid_layer * mid_strength * (0.72 + masks["shadows"] * 0.22)
        + mono_grain * legacy_strength * 0.55
    )
    chroma_texture = color_grain * (legacy_strength * chroma_mix * 0.70 + micro_strength * chroma_mix * 0.20)
    grain = (luma_texture * (1.0 - chroma_mix * 0.38) + chroma_texture) * intensity

    density_amount = np.clip(density_instability, 0.0, 0.08) + density_strength
    density_modulation = density_layer * density_amount * (
        0.42 + masks["shadows"] * 0.36 + masks["midtones"] * 0.18
    )
    img = img * (1.0 + density_modulation)
    img = img + grain

    # Shadow density on print scans has a soft floor and slight color impurity.
    # Keep the contamination weak and red/yellow-biased enough to avoid teal mud.
    shadow_texture = normalized_map(density_layer * 0.70 + mid_layer * 0.30)
    shadow_mask = masks["shadows"] * (1.0 - smoothstep(0.12, 0.46, luminance_values))
    soft_floor = shadow_mask * (0.0035 + density_amount * 0.020) * (1.0 + shadow_texture * 0.18)
    contamination = np.array([0.006, 0.0015, -0.0035], dtype=np.float32)
    img = img + soft_floor + shadow_mask * shadow_texture * contamination * (0.32 + chroma_mix * 0.24)

    # Scanner-like softness is frequency-selective. It trims brittle precision
    # and lets texture participate in transitions without creating global haze.
    soft = np.clip(scanner_softness, 0.0, 1.0)
    diffusion = np.clip(tonal_diffusion, 0.0, 1.0)
    edge_soft = np.clip(edge_softening, 0.0, 1.0)
    if soft > 0.0 or diffusion > 0.0 or edge_soft > 0.0:
        bilateral = cv2.bilateralFilter(
            np.clip(img, 0.0, 1.0).astype(np.float32),
            d=0,
            sigmaColor=0.010 + diffusion * 0.045,
            sigmaSpace=0.65 + soft * 1.35,
        )
        gaussian = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=0.42 + soft * 0.52, sigmaY=0.42 + soft * 0.52)
        small_detail = img - gaussian
        detail_scale = 1.0 - (0.10 * soft + 0.08 * diffusion) * (1.0 - masks["highlights"] * 0.35)
        detail_softened = gaussian + small_detail * detail_scale
        edge_mix = harsh_edge * edge_soft * 0.22 + masks["midtones"] * diffusion * 0.055
        img = detail_softened * (1.0 - edge_mix) + bilateral * edge_mix

    # Texture slightly breaks digital color purity. The modulation is strongest
    # outside likely skin ranges and fades in highlights to preserve natural faces.
    lum_after = luminance(np.clip(img, 0.0, 1.0))[..., None]
    chroma_delta = img - lum_after
    color_purity = (np.max(img, axis=-1, keepdims=True) - np.min(img, axis=-1, keepdims=True))
    r = img[..., 0:1]
    g = img[..., 1:2]
    b = img[..., 2:3]
    skin_like = (
        (r > g * 0.94)
        & (r > b * 1.04)
        & (g > b * 0.82)
        & (lum_after > 0.22)
        & (lum_after < 0.84)
    ).astype(np.float32)
    texture_chroma = np.clip(chroma_instability, 0.0, 0.12)
    saturation_mod = 1.0 + density_layer * texture_chroma * color_purity * (1.0 - masks["highlights"] * 0.70)
    saturation_mod -= mid_layer * texture_chroma * 0.22 * (1.0 - skin_like * 0.72)
    img = lum_after + chroma_delta * saturation_mod

    if texture_chroma > 0.0:
        drift_px = texture_chroma * 0.42
        drifted = img.copy()
        drifted[..., 0] = shift_channel(img[..., 0], drift_px, -drift_px * 0.35)
        drifted[..., 2] = shift_channel(img[..., 2], -drift_px * 0.55, drift_px * 0.25)
        chroma_mask = (masks["shadows"] * 0.55 + masks["midtones"] * 0.28) * (1.0 - skin_like * 0.62)
        img = img * (1.0 - chroma_mask * texture_chroma * 0.60) + drifted * (chroma_mask * texture_chroma * 0.60)

    irregularity = np.clip(scan_irregularity, 0.0, 0.10)
    if irregularity > 0.0:
        x_grid, y_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        dx = normalized_noise((h, w, 1), max(min(h, w) * 0.055, 2.0))[..., 0] * irregularity * 0.24
        dy = normalized_noise((h, w, 1), max(min(h, w) * 0.075, 2.0))[..., 0] * irregularity * 0.14
        img = cv2.remap(
            img.astype(np.float32),
            (x_grid + dx).astype(np.float32),
            (y_grid + dy).astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    return np.clip(img, 0.0, 1.0).astype(np.float32)
