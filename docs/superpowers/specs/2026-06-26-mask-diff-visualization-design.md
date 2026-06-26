# Mask Diff Visualization Script — Design Spec

**Date:** 2026-06-26
**Status:** approved

## 1. Purpose

A CLI script that takes a base JPG image and two binary PNG mask ground-truths, and produces four labeled JPG visualization outputs highlighting the agreement and disagreement regions between the two masks.

## 2. Inputs

| Argument | Type | Description |
|----------|------|-------------|
| `--base` | path (JPG) | Background image |
| `--mask1` | path (PNG) | First ground-truth mask, binary (0 / 255) |
| `--mask2` | path (PNG) | Second ground-truth mask, binary (0 / 255) |
| `--outdir` | path (dir) | Directory for output JPGs |
| `--alpha` | float (default 0.5) | Blend factor for overlays |

All arguments are required except `--alpha`.

## 3. Processing Logic

### 3.1 Region classification (per pixel)

| Region | Condition | Display color |
|--------|-----------|---------------|
| Intersection (both agree) | mask1=255 AND mask2=255 | Yellow `(255, 255, 0)` |
| Mask2 only | mask1=0 AND mask2=255 | Blue `(0, 0, 255)` |
| Mask1 only | mask1=255 AND mask2=0 | Red `(255, 0, 0)` |

### 3.2 Blending

Alpha blend via `PIL.Image.blend(base, colored_overlay, alpha)`. The overlay is a solid-color image matching the region shape; blending at `alpha=0.5` gives a semi-transparent overlay effect so the base image remains visible underneath.

### 3.3 Mask size handling

If mask dimensions differ from the base image, masks are resized to match the base using `PIL.Image.resize` with `NEAREST` interpolation (no new values introduced for binary data).

## 4. Outputs (4 JPG files in `--outdir`)

All four outputs use the JPG base image as background.

| File | Overlaid regions |
|------|-----------------|
| `composite.jpg` | Yellow (intersection) + Blue (mask2 only) + Red (mask1 only) |
| `intersection.jpg` | Yellow (intersection) only |
| `mask2_only.jpg` | Blue (mask2 only) only |
| `mask1_only.jpg` | Red (mask1 only) only |

## 5. Implementation

- **File:** `scripts/visualize_mask_diff.py`
- **Language:** Python 3
- **Dependencies:** Pillow (PIL), argparse (stdlib)
- **Structure:** Single-file script, no classes, pure function composition
  - `parse_args()` — argparse setup
  - `load_mask_binary(path, size)` — load PNG → binary 0/1 numpy-like logic (PIL pixel access)
  - `make_overlay(mask, color, size)` — create an RGB overlay image for a given binary mask and color
  - `blend_and_save(base, overlay, alpha, outpath)` — blend and save as JPG
  - `main()` — orchestration

## 6. Edge Cases

- Output directory created if it doesn't exist (`os.makedirs`)
- Mask/base size mismatch handled via resize-to-base
- Non-binary mask pixels treated as: value > 127 → 1, else 0 (robustness against off-by-one values like 254)
