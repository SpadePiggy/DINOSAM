# Mask Diff Visualization Script — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CLI script that visualizes agreement/disagreement between two binary masks overlaid on a base image.

**Architecture:** Single-file Python script using PIL/Pillow for image manipulation and argparse for CLI. Four pure functions + main orchestration. No classes, no external dependencies beyond Pillow.

**Tech Stack:** Python 3, Pillow, argparse (stdlib)

**Spec:** `docs/superpowers/specs/2026-06-26-mask-diff-visualization-design.md`

## Global Constraints

- Mask format: binary PNG (0/255), threshold >127 → 1
- Blend mode: `Image.blend(base, overlay, alpha)` with default alpha=0.5
- Resize method: NEAREST (preserves binary values)
- Output format: JPG

---

### Task 1: Script skeleton and argument parsing

**Files:**
- Create: `scripts/visualize_mask_diff.py`

**Interfaces:**
- Produces: `parse_args() -> argparse.Namespace` — returns namespace with `.base`, `.mask1`, `.mask2`, `.outdir`, `.alpha`

- [ ] **Step 1: Write the skeleton with argparse**

```python
#!/usr/bin/env python3
"""Visualize agreement and disagreement between two binary masks over a base image.

Produces four JPG outputs showing intersection (yellow), mask2-only (blue),
and mask1-only (red) regions overlaid on the base image.
"""

import argparse
import os
import sys

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize mask agreement/disagreement regions over a base image."
    )
    parser.add_argument("--base", required=True, help="Path to base JPG image")
    parser.add_argument("--mask1", required=True, help="Path to first binary mask PNG")
    parser.add_argument("--mask2", required=True, help="Path to second binary mask PNG")
    parser.add_argument("--outdir", required=True, help="Output directory for JPG files")
    parser.add_argument("--alpha", type=float, default=0.5, help="Blend factor (default: 0.5)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"base={args.base}, mask1={args.mask1}, mask2={args.mask2}, outdir={args.outdir}, alpha={args.alpha}")
```

- [ ] **Step 2: Verify it runs (help and arg parse)**

```bash
python scripts/visualize_mask_diff.py --help
python scripts/visualize_mask_diff.py --base a.jpg --mask1 m1.png --mask2 m2.png --outdir /tmp/out
```

Expected: help text with all arguments; second command prints parsed values.

- [ ] **Step 3: Commit**

```bash
git add scripts/visualize_mask_diff.py
git commit -m "feat: add mask diff viz script skeleton with argparse"
```

---

### Task 2: Mask loading and overlay generation

**Files:**
- Modify: `scripts/visualize_mask_diff.py` — add functions below `parse_args`, update `main`

**Interfaces:**
- Consumes: `parse_args()` from Task 1
- Produces:
  - `load_mask_binary(path: str, size: tuple[int, int]) -> Image.Image` — returns mode "1" binary image resized to `size`
  - `make_overlay(mask: Image.Image, color: tuple[int, int, int]) -> Image.Image` — returns mode "RGB" image, colored where mask=1, black elsewhere

- [ ] **Step 1: Add load_mask_binary and make_overlay**

```python
def load_mask_binary(path: str, size: tuple[int, int]) -> Image.Image:
    """Load a PNG mask, threshold to binary, and resize to match base."""
    img = Image.open(path).convert("L")
    # Threshold: > 127 -> 255, else 0; then convert to mode "1"
    img = img.point(lambda p: 255 if p > 127 else 0)
    if img.size != size:
        img = img.resize(size, Image.NEAREST)
    return img.convert("1")


def make_overlay(mask: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    """Create an RGB overlay image with the given color where mask is 1, black elsewhere."""
    overlay = Image.new("RGB", mask.size, (0, 0, 0))
    for y in range(mask.height):
        for x in range(mask.width):
            if mask.getpixel((x, y)):
                overlay.putpixel((x, y), color)
    return overlay
```

- [ ] **Step 2: Verify with a quick manual test**

```python
# In Python REPL:
mask = load_mask_binary("test_m1.png", (640, 480))
print(mask.size, mask.mode)  # (640, 480), "1"
overlay = make_overlay(mask, (255, 255, 0))
print(overlay.size, overlay.mode)  # (640, 480), "RGB"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/visualize_mask_diff.py
git commit -m "feat: add mask loading and overlay generation functions"
```

---

### Task 3: Composite generation and main orchestration

**Files:**
- Modify: `scripts/visualize_mask_diff.py` — add `blend_and_save`, `main`, update `__main__` block

**Interfaces:**
- Consumes: `load_mask_binary`, `make_overlay` from Task 2; `parse_args` from Task 1
- Produces:
  - `blend_and_save(base: Image.Image, overlay: Image.Image, alpha: float, outpath: str) -> None`
  - `main() -> None` — full pipeline

- [ ] **Step 1: Add blend_and_save and main**

Replace the `if __name__ == "__main__":` block entirely:

```python
def blend_and_save(base: Image.Image, overlay: Image.Image, alpha: float, outpath: str) -> None:
    """Blend overlay onto base at given alpha, save as JPG."""
    blended = Image.blend(base, overlay, alpha)
    blended.save(outpath, "JPEG", quality=95)


def main() -> None:
    args = parse_args()

    # Load base
    base = Image.open(args.base).convert("RGB")

    # Load and align masks
    mask1 = load_mask_binary(args.mask1, base.size)
    mask2 = load_mask_binary(args.mask2, base.size)

    # Compute regions
    both = Image.new("1", base.size, 0)
    only_mask2 = Image.new("1", base.size, 0)
    only_mask1 = Image.new("1", base.size, 0)

    for y in range(base.height):
        for x in range(base.width):
            m1 = mask1.getpixel((x, y))
            m2 = mask2.getpixel((x, y))
            if m1 and m2:
                both.putpixel((x, y), 1)
            elif m2:
                only_mask2.putpixel((x, y), 1)
            elif m1:
                only_mask1.putpixel((x, y), 1)

    # Build overlays
    overlay_both = make_overlay(both, (255, 255, 0))       # Yellow
    overlay_m2 = make_overlay(only_mask2, (0, 0, 255))      # Blue
    overlay_m1 = make_overlay(only_mask1, (255, 0, 0))      # Red

    # Composite (all three regions)
    composite = Image.blend(base, overlay_both, args.alpha)
    composite = Image.blend(composite, overlay_m2, args.alpha)
    composite = Image.blend(composite, overlay_m1, args.alpha)

    # Output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Save all four outputs
    blend_and_save(base, overlay_both, args.alpha, os.path.join(args.outdir, "intersection.jpg"))
    blend_and_save(base, overlay_m2, args.alpha, os.path.join(args.outdir, "mask2_only.jpg"))
    blend_and_save(base, overlay_m1, args.alpha, os.path.join(args.outdir, "mask1_only.jpg"))
    composite.save(os.path.join(args.outdir, "composite.jpg"), "JPEG", quality=95)

    print(f"Saved 4 images to {args.outdir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script runs end-to-end**

Create test fixtures:
```bash
# Generate a test base (200x200 red square)
python -c "
from PIL import Image
Image.new('RGB', (200, 200), (128, 128, 128)).save('/tmp/test_base.jpg')
# mask1: left half white
m1 = Image.new('L', (200, 200), 0)
for y in range(200):
    for x in range(100):
        m1.putpixel((x, y), 255)
m1.save('/tmp/test_m1.png')
# mask2: top half white
m2 = Image.new('L', (200, 200), 0)
for y in range(100):
    for x in range(200):
        m2.putpixel((x, y), 255)
m2.save('/tmp/test_m2.png')
"
```

Run:
```bash
python scripts/visualize_mask_diff.py \
  --base /tmp/test_base.jpg \
  --mask1 /tmp/test_m1.png \
  --mask2 /tmp/test_m2.png \
  --outdir /tmp/mask_viz_out
```

Expected: 4 JPG files in `/tmp/mask_viz_out/` — verify visually:
- `composite.jpg`: top-left = yellow, top-right = blue, bottom-left = red, bottom-right = no overlay (base visible)
- `intersection.jpg`: only top-left yellow
- `mask2_only.jpg`: only top-right blue
- `mask1_only.jpg`: only bottom-left red

- [ ] **Step 3: Commit**

```bash
git add scripts/visualize_mask_diff.py
git commit -m "feat: complete mask diff visualization script"
```
