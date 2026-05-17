"""Smoke tests for the first FilmFilter pipeline foundation."""

from __future__ import annotations

import numpy as np

from pipeline.pipeline import FilmPipeline, load_preset


def test_soft_portrait_pipeline_runs_on_float_rgb_image() -> None:
    preset = load_preset("soft_portrait_400")
    preset["effects"]["grain"]["seed"] = 123
    pipeline = FilmPipeline(preset)

    gradient = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    image = np.dstack(np.meshgrid(gradient, gradient, indexing="xy") + [np.full((32, 32), 0.45, dtype=np.float32)])

    output = pipeline.process(image)

    assert output.shape == image.shape
    assert np.isfinite(output).all()
    assert output.min() >= 0.0
    assert output.max() <= 1.0
    assert not np.allclose(output, image)
