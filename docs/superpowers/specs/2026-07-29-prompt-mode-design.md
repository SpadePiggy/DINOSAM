# Prompt Mode Selection for Evaluation

**Date:** 2026-07-29
**Status:** draft
**Scope:** evaluate.py, DepthDataset, forward_mask

## Motivation

Evaluation currently hardcodes GT-derived centroid point prompts. This makes it
impossible to measure the model's segmentation performance under different prompt
conditions: GT-guided (oracle), noisy (simulated user click), centered (naive),
or no prompt at all (automatic segmentation).

## Design

### Four prompt modes

| Mode | Behavior | GT access |
|------|----------|-----------|
| `gt_centroid` | Connected components top-3 by area → centroid (x, y), label=1 | Yes |
| `random_circle` | Components >20px, top-3 → minEnclosingCircle → concentric circle at half radius → uniform random sample, label=1 | Yes |
| `center` | Image center (w/2, h/2), label=1 | No |
| `no_prompt` | No point prompt at all (points=None via forward_mask) | No |

All four modes are evaluation-only concerns. Training always uses `gt_centroid`.

### Files changed

#### `dinosam/dataset/depth_dataset.py`

- `DepthDataset.__init__` gains `prompt_mode: str = "gt_centroid"`
- `_get_point_prompts` dispatches to three private methods based on mode:
  - `_gt_centroid_prompts(mask_arr)` — existing logic
  - `_random_circle_prompts(mask_arr)` — new
  - `_center_prompts(mask_arr)` — new, ignores mask_arr, always returns center
- `no_prompt` uses `_center_prompts` to return a valid array (collate_fn can't handle None)

#### `dinosam/evaluate.py`

- `--no_prompt` flag replaced by `--prompt_mode` with `choices=["gt_centroid", "random_circle", "center", "no_prompt"]`, `default="gt_centroid"`
- `evaluate()` parameter: `no_prompt: bool` → `prompt_mode: str`
- `no_prompt` mode: after batch load, set `point_coords = None, point_labels = None` (reuses existing None-carrying logic in forward_mask)
- `DepthDataset(..., prompt_mode=args.prompt_mode)`

#### `dinosam/model/depth_sam.py` (bug fix)

- `forward_mask`: when `point_coords is None`, skip `prompt_encoder` entirely and construct embeddings directly:
  ```python
  sparse_emb = torch.empty(B, 0, 256, device=device)
  dense_emb = no_mask_embed.expand(B, -1, 64, 64)
  ```
- This avoids the `_get_batch_size → 1` bug where prompt_encoder returns batch=1 regardless of real batch size

### `random_circle` sampling algorithm

```
For each connected component (>20 pixels), top-3 by area:
  contour = find_contours(component_mask)
  (cx, cy), radius = cv2.minEnclosingCircle(contour)
  r' = (radius / 2) * sqrt(random.uniform(0, 1))
  θ  = random.uniform(0, 2π)
  x = cx + r' * cos(θ)
  y = cy + r' * sin(θ)
```

Uses `sqrt(U)` scaling for uniform density in the circle. Dependency: `cv2` (opencv-python).

### CLI usage

```bash
# Default (GT centroid)
python dinosam/evaluate.py ...

# Noisy simulated user click
python dinosam/evaluate.py ... --prompt_mode random_circle

# Center point only
python dinosam/evaluate.py ... --prompt_mode center

# Fully automatic (no prompt)
python dinosam/evaluate.py ... --prompt_mode no_prompt
```

### Edge cases

| Case | Behavior |
|------|----------|
| No components >20px (random_circle) | Fallback to `_center_prompts` |
| < 3 components (any GT-based mode) | Use however many exist |
| Empty GT mask (gt_centroid) | Existing fallback: center point |
| Single-sample batches | No issue (already batched) |
| Iterative eval + no_prompt | First iter no prompt, subsequent iters GT-guided corrections (same as before) |

### Non-goals

- Does not change training logic
- Does not add prompt modes to PCA visualization scripts
- Does not refactor dataset beyond what's needed for mode dispatch
