"""JSON-driven orchestration for FilmFilter effects."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .color import apply_color
from .grain import apply_grain
from .halation import apply_halation
from .lens import apply_lens
from .tone import apply_tone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRESET_DIR = PROJECT_ROOT / "presets"

StageFunction = Callable[..., np.ndarray]


STAGES: dict[str, StageFunction] = {
    "tone": apply_tone,
    "color": apply_color,
    "halation": apply_halation,
    "grain": apply_grain,
    "lens": apply_lens,
}


@dataclass(frozen=True)
class FilmPipeline:
    """Sequential, composable FilmFilter processing pipeline.

    Each stage receives and returns a floating-point RGB image in the 0..1 range.
    Keeping stages pure and ordered makes the aesthetic recipe easy to inspect,
    test, and revise as the project grows beyond this first preset.
    """

    preset: dict[str, Any]

    def process(self, image: np.ndarray) -> np.ndarray:
        """Run enabled preset stages in order and return a clipped RGB image."""
        result = np.clip(image.astype(np.float32, copy=False), 0.0, 1.0)
        for stage_name in self.preset.get("pipeline", []):
            if stage_name not in STAGES:
                raise ValueError(f"Unknown pipeline stage: {stage_name}")
            config = self.preset.get("effects", {}).get(stage_name, {})
            if not config.get("enabled", True):
                continue
            params = {key: value for key, value in config.items() if key != "enabled"}
            result = STAGES[stage_name](result, **params)
        return np.clip(result, 0.0, 1.0)

    @classmethod
    def from_preset_name(cls, preset_name: str) -> "FilmPipeline":
        """Create a pipeline from a preset name or JSON file path."""
        return cls(load_preset(preset_name))


def load_preset(preset_name: str | Path) -> dict[str, Any]:
    """Load a JSON preset from presets/ or from an explicit file path."""
    preset_path = Path(preset_name)
    if preset_path.suffix != ".json":
        preset_path = PRESET_DIR / f"{preset_name}.json"
    elif not preset_path.is_absolute() and not preset_path.exists():
        preset_path = PRESET_DIR / preset_path

    if not preset_path.exists():
        raise FileNotFoundError(f"Preset not found: {preset_path}")

    with preset_path.open("r", encoding="utf-8") as handle:
        preset = json.load(handle)

    if "pipeline" not in preset or not isinstance(preset["pipeline"], list):
        raise ValueError(f"Preset {preset_path} must define a pipeline list")
    return preset


def read_image(path: str | Path) -> np.ndarray:
    """Read an image as EXIF-corrected RGB float data in the 0..1 range."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image, dtype=np.float32) / 255.0


def write_image(path: str | Path, image: np.ndarray, *, quality: int = 95) -> None:
    """Write a float RGB image to disk, creating parent directories as needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(data, mode="RGB").save(output_path, quality=quality)
