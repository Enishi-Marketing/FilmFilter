# Repository Guidance for Coding Agents

FilmFilter is a Python 3.12+ image-processing project focused on subtle consumer-film aesthetics.

## Project Priorities

- Prefer modular architecture over feature count.
- Keep effects sequential, composable, and JSON-driven.
- Keep default strengths subtle and photographically believable.
- Write clear docstrings that explain the aesthetic reason for a transform, not only the implementation details.
- Avoid premature optimization, GPU assumptions, ML dependencies, and large dependency chains.

## Aesthetic Direction

Target:

- soft highlight compression
- lifted blacks
- reduced microcontrast
- warm skin tones
- muted olive greens
- subtle bloom/halation
- believable fine grain
- gentle optical imperfections

Avoid:

- giant fake grain
- crushed blacks
- orange-and-teal grading
- fake dust overlays
- VHS/glitch artifacts
- heavy blur
- extreme vignettes
- over-stylized Instagram-filter looks

The output should feel nostalgic, soft, photographic, consumer-grade, and imperfect in believable ways.

## Code Organization

- `pipeline/` contains individual effect modules and the JSON-driven pipeline orchestrator.
- `presets/` contains named JSON presets. Do not scatter hardcoded preset recipes through implementation files.
- `cli.py` is the first-pass command-line interface.
- `input/` and `output/` are convenience folders and should not be used for committed sample binaries unless explicitly requested.

## Implementation Guidelines

- All image-processing stages should accept and return NumPy RGB float arrays in the `0..1` range.
- Prefer pure functions for effect modules.
- Add new effect parameters through presets rather than hidden constants where practical.
- Keep defaults conservative.
- Do not add a GUI, web app, GPU requirement, or ML model unless explicitly requested.
- Never wrap imports in `try`/`except` blocks.

## Testing and Validation

- At minimum, run a CLI smoke test or module-level smoke test after changing processing code.
- Use deterministic seeds where tests need stable output.
- Confirm outputs remain finite and clipped to the `0..1` range.
