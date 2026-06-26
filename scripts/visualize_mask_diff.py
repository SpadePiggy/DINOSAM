#!/usr/bin/env python3
"""Visualize agreement and disagreement between two binary masks over a base image.

Produces four JPG outputs showing intersection (yellow), mask2-only (blue),
and mask1-only (red) regions overlaid with alpha blending on the base image.
"""

import argparse
import os

from PIL import Image, ImageChops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize mask agreement/disagreement regions over a base image."
    )
    parser.add_argument("--base", required=True, help="Path to base JPG image")
    parser.add_argument("--mask1", required=True, help="Path to first binary mask PNG")
    parser.add_argument("--mask2", required=True, help="Path to second binary mask PNG")
    parser.add_argument("--outdir", required=True, help="Output directory for JPG files")
    parser.add_argument("--alpha", type=float, default=0.5, help="Blend factor (default: 0.5)")
    parser.add_argument("--blend_bg", action="store_true",
                        help="Also blend non-mask regions with black overlay (darkens background)")
    return parser.parse_args()


def load_mask_binary(path: str, size: tuple[int, int]) -> Image.Image:
    """Load a PNG mask, threshold to binary (>127), and resize to match base size."""
    img = Image.open(path).convert("L")
    img = img.point(lambda p: 255 if p > 127 else 0)
    if img.size != size:
        img = img.resize(size, Image.NEAREST)
    return img.convert("1")


def make_overlay(mask: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    """Create an RGB overlay image colored where mask is 1, black elsewhere."""
    colored = Image.new("RGB", mask.size, color)
    black = Image.new("RGB", mask.size, (0, 0, 0))
    return Image.composite(colored, black, mask)


def blend_mask(base: Image.Image, overlay: Image.Image, mask: Image.Image, alpha: float) -> Image.Image:
    """Blend overlay onto base only where mask is True. Non-mask regions keep base pixels."""
    blended = Image.blend(base, overlay, alpha)
    return Image.composite(blended, base, mask)


def main() -> None:
    args = parse_args()

    # Load base image
    base = Image.open(args.base).convert("RGB")

    # Load and align masks
    mask1 = load_mask_binary(args.mask1, base.size)
    mask2 = load_mask_binary(args.mask2, base.size)

    # Compute regions with logical operations
    both = ImageChops.logical_and(mask1, mask2)
    only_mask2 = ImageChops.logical_and(ImageChops.invert(mask1), mask2)
    only_mask1 = ImageChops.logical_and(mask1, ImageChops.invert(mask2))

    # Build colored overlays
    yellow = make_overlay(both, (255, 255, 0))
    blue = make_overlay(only_mask2, (0, 0, 255))
    red = make_overlay(only_mask1, (255, 0, 0))
    full_m1 = make_overlay(mask1, (255, 0, 0))
    full_m2 = make_overlay(mask2, (0, 0, 255))

    # Combined overlay for composite (non-overlapping regions, single blend)
    combined = Image.new("RGB", base.size, (0, 0, 0))
    combined = Image.composite(yellow, combined, both)
    combined = Image.composite(blue, combined, only_mask2)
    combined = Image.composite(red, combined, only_mask1)
    any_mask = ImageChops.logical_or(both, ImageChops.logical_or(only_mask2, only_mask1))

    # Output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Save all four outputs
    blend = (lambda b, o, m: Image.blend(b, o, args.alpha)) if args.blend_bg else (
        lambda b, o, m: blend_mask(b, o, m, args.alpha))

    blend(base, yellow, both).save(
        os.path.join(args.outdir, "intersection.jpg"), "JPEG", quality=95)
    blend(base, blue, only_mask2).save(
        os.path.join(args.outdir, "mask2_only.jpg"), "JPEG", quality=95)
    blend(base, red, only_mask1).save(
        os.path.join(args.outdir, "mask1_only.jpg"), "JPEG", quality=95)
    blend(base, combined, any_mask).save(
        os.path.join(args.outdir, "composite.jpg"), "JPEG", quality=95)
    blend(base, full_m1, mask1).save(
        os.path.join(args.outdir, "mask1_full.jpg"), "JPEG", quality=95)
    blend(base, full_m2, mask2).save(
        os.path.join(args.outdir, "mask2_full.jpg"), "JPEG", quality=95)

    print(f"Saved 6 images to {args.outdir}")


if __name__ == "__main__":
    main()
