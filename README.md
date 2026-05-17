# FilmFilter

FilmFilter is a small Python image-processing project for subtle, emotionally believable film-emulation looks. It is inspired by scanned print film, disposable cameras, 90s/2000s consumer portraits, toy digital cameras, soft highlight rolloff, restrained halation, and realistic grain.

The project is intentionally **not** a physically accurate film chemistry simulator, a LUT pack, a VHS/glitch tool, or an over-stylized social media filter. The goal is a soft photographic foundation that can grow carefully over time.

## Visual Philosophy

FilmFilter aims for images that feel:

- nostalgic
- soft
- photographic
- consumer-grade
- imperfect in believable ways

The default aesthetic favors:

- soft highlight compression instead of clipped digital whites
- lifted blacks instead of crushed shadows
- reduced microcontrast instead of brittle sharpness
- warm but restrained highlight color
- muted olive greens
- subtle bloom and halation
- fine procedural grain
- gentle optical imperfections

FilmFilter avoids:

- giant fake grain
- heavy blur
- orange-and-teal grading
- fake dust overlays
- VHS artifacts
- extreme vignettes
- hardcoded preset values scattered across the codebase

## Architecture

The project is built around a JSON-driven, sequential pipeline. Images are loaded as RGB float arrays in the `0..1` range. Each module receives an image, applies one aesthetic transform, and returns a new image for the next stage.

Current stages:

1. `tone` — lifted blacks, softened contrast, and highlight rolloff while preserving midtones.
2. `color` — restrained saturation, muted greens, warm highlights, and a slight magenta skin bias.
3. `halation` — highlight isolation, Gaussian blur, warm tint, and subtle additive blending.
4. `grain` — procedural luminance-dependent grain with slight chromatic variation.
5. `lens` — subtle vignette, edge softness, and optional tiny chromatic aberration.

Preset files live in `presets/`. The first preset is `soft_portrait_400.json`, which defines both the stage order and effect strengths.

## Project Structure

```text
FilmFilter/
├── presets/
│   └── soft_portrait_400.json
├── pipeline/
│   ├── tone.py
│   ├── color.py
│   ├── halation.py
│   ├── grain.py
│   ├── lens.py
│   └── pipeline.py
├── input/
├── output/
├── tests/
├── cli.py
├── requirements.txt
├── README.md
└── AGENTS.md
```

## Installation

FilmFilter targets Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Place a source image anywhere on disk, then run:

```bash
python cli.py input.jpg --preset soft_portrait_400
```

By default, output is written to:

```text
output/input_soft_portrait_400.jpg
```

You can choose a custom destination:

```bash
python cli.py input.jpg --preset soft_portrait_400 --output output/example.jpg
```

## Presets

Presets are JSON files that define the ordered pipeline and per-stage parameters. Example:

```json
{
  "pipeline": ["tone", "color", "halation", "grain", "lens"],
  "effects": {
    "tone": {
      "enabled": true,
      "lifted_black": 0.045
    }
  }
}
```

This keeps aesthetic choices centralized, inspectable, and easy to tune without editing implementation files.

## Current Limitations

FilmFilter is an early foundation. It currently has:

- no GUI
- no web app
- no GPU path
- no machine learning or deep learning
- no batch processing command yet
- no metadata preservation beyond basic image orientation handling
- no physically accurate film-stock modeling
- no automated perceptual quality tests

## Roadmap

Possible future additions:

- batch processing
- side-by-side contact sheet generation
- more presets with carefully documented intent
- preset schema validation
- stronger automated tests for stage ranges and determinism
- optional film border/crop tools that remain subtle
- better color-space management
- per-camera/lens imperfection profiles
- CLI controls for seed, output format, and batch directories

## Development Notes

The codebase prioritizes readability, documentation, and modularity over feature count. New effects should remain subtle by default and should explain their aesthetic purpose in docstrings.
