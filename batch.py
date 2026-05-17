"""Batch-process all images in an input folder through the FilmFilter pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.pipeline import FilmPipeline, read_image, write_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply film-emulation processing to every image in a folder.",
    )
    parser.add_argument("input_dir", help="Folder containing source images.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="output",
        help="Destination folder for processed images (default: output/).",
    )
    parser.add_argument(
        "--preset",
        default="soft_portrait_400",
        help="Preset name from presets/ or path to a JSON preset.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality for saved output files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise SystemExit(f"Error: '{input_dir}' is not a directory.")

    images = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        raise SystemExit(f"No supported images found in '{input_dir}'.")

    output_dir.mkdir(parents=True, exist_ok=True)
    preset_label = Path(args.preset).stem
    pipeline = FilmPipeline.from_preset_name(args.preset)

    for i, src in enumerate(images, 1):
        dest = output_dir / f"{src.stem}_{preset_label}.jpg"
        print(f"[{i}/{len(images)}] {src.name} → {dest}")
        image = read_image(src)
        processed = pipeline.process(image)
        write_image(dest, processed, quality=args.quality)

    print(f"\nDone — {len(images)} image(s) written to '{output_dir}'.")


if __name__ == "__main__":
    main()
