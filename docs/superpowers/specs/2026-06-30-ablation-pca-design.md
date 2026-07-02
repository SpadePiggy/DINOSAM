# DINOSAM Ablation Study & PCA Visualization Design

**Date:** 2026-06-30  
**Status:** DRAFT  
**Scope:** DINOSAM project (`/Users/lxt/Documents/SEU/DINOSAM`)

---

## Overview

Two features for DINOSAM:

1. **Configurable DPT fusion layers** — CLI parameter `--dpt_layers` controls which transformer blocks feed into the DPT head, enabling ablation experiments
2. **PCA feature visualization** — training-time optional monitoring + standalone inference CLI tool

---

## Layer Indexing Convention

All layer indices are **0-indexed** transformer block indices, matching `get_intermediate_layers` API (`dinov3/models/vision_transformer.py:273-282`). For ViT-B/16 (12 blocks): valid range is `[0, 11]`.

### Ablation Configurations

| Config Name | `--dpt_layers` | n_layers | Description |
|---|---|---|---|
| Three-layer early | `2,5,8` | 3 | Skip the last block |
| Three-layer uniform | `3,6,9` | 3 | Evenly spaced, shifted later |
| Four-layer SegDINO-style | `2,5,8,11` | 4 | Current default |
| High-level | `8,9,10,11` | 4 | Last 4 blocks only |
| All-layer | `0,1,2,3,4,5,6,7,8,9,10,11` | 12 | Every block |

---

## Part 1: CLI Parameter

### Keep existing comma-separated format (backward compatible)

Existing `trainer.py` and `evaluate.py` already use `--dpt_layers` as a comma-separated string. **Keep this format** — do not change to `nargs="+"`.

Existing code (`trainer.py:592-594`):
```python
parser.add_argument("--dpt_layers", type=str, default="2,5,8,11",
                    help="Comma-separated intermediate layer indices for DPT "
                         "(e.g., '2,5,8,11'). Only used with --adapter_type dpt_simple")
```

Parsing (`trainer.py:339-342`):
```python
dpt_layers = (
    [int(x.strip()) for x in args.dpt_layers.split(",")]
    if args.adapter_type == "dpt_simple" else None
)
```

**Change required**: Remove the `adapter_type == "dpt_simple"` guard — `--dpt_layers` should work with any adapter type now that the DPT head supports dynamic layer counts. Pass `dpt_layers` to `DepthSam` → `DINOv3MonaEncoder` unconditionally when `encoder_type == "dinov3"`.

### Usage examples

```bash
# SegDINO default (4 layers)
python -m dinosam.train.trainer --dpt_layers "2,5,8,11" ...

# Three-layer early
python -m dinosam.train.trainer --dpt_layers "2,5,8" ...

# All-layer
python -m dinosam.train.trainer --dpt_layers "0,1,2,3,4,5,6,7,8,9,10,11" ...
```

---

## Part 2: Dynamic DPTHead

### Problem

Current `DPTSimpleHead` (`dinosam/model/dpt_heads.py`) is hardcoded for exactly 4 layers: 4 projections, 4 refines, `output_conv(features*4, embed_dim)`. Ablation requires 3, 4, or 12 layers.

### Solution: replace `DPTSimpleHead` with dynamic `DPTHead` in `dpt_heads.py`

**Key constraint**: when `n_layers=4`, the new `DPTHead` must be **architecturally identical** to `DPTSimpleHead` — same parameter names, same shapes, same output — so existing checkpoints load without modification.

### Existing DPTSimpleHead architecture (reference)

```python
class DPTSimpleHead(nn.Module):
    def __init__(self, embed_dim=768, features=256):
        # 4× 1×1 Conv: embed_dim → features (uniform)
        self.projects = ModuleList([Conv2d(768, 256, 1) for _ in range(4)])
        # 4× 3×3 Conv: features → features
        self.refine_convs = ModuleList([Conv2d(256, 256, 3, padding=1) for _ in range(4)])
        # 1×1 Conv: features*4 → embed_dim
        self.output_conv = Conv2d(256*4, 768, 1)

    def forward(self, features, patch_h, patch_w):
        # each: (B,N,C) → reshape(B,C,H,W) → project → refine
        # cat → (B, features*4, H, W) → output_conv → (B, embed_dim, H, W)
```

No ConvTranspose. No upsample. All layers at same 64×64 resolution. Output is `embed_dim` (768), not `nclass`.

### New DPTHead design

```python
class DPTHead(nn.Module):
    """Dynamic DPT head supporting arbitrary number of fusion layers.

    Architecturally identical to DPTSimpleHead when n_layers=4:
    - Uniform projection: embed_dim → features for all layers
    - Uniform refine: features → features for all layers
    - No upsample (all layers at same spatial resolution)
    - Output: features*n_layers → embed_dim via 1×1 Conv

    Backward compatible: existing checkpoints trained with DPTSimpleHead
    (n_layers=4) load directly into DPTHead.
    """

    def __init__(self, embed_dim=768, features=256, n_layers=4):
        super().__init__()
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.features = features

        self.projects = nn.ModuleList([
            nn.Conv2d(embed_dim, features, kernel_size=1)
            for _ in range(n_layers)
        ])
        self.refine_convs = nn.ModuleList([
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)
            for _ in range(n_layers)
        ])
        self.output_conv = nn.Conv2d(features * n_layers, embed_dim, kernel_size=1)

    def forward(self, features, patch_h, patch_w, return_fused=False):
        """
        Args:
            features: list of n_layers tensors, each (B, N, embed_dim)
            patch_h, patch_w: spatial size of patch tokens (e.g., 64, 64)
            return_fused: if True, also return pre-output fused features

        Returns:
            (B, embed_dim, patch_h, patch_w)
            optionally: tuple of above + (B, features*n_layers, patch_h, patch_w)
        """
        out = []
        for i, x in enumerate(features):
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
            x = self.projects[i](x)
            x = self.refine_convs[i](x)
            out.append(x)

        fused = torch.cat(out, dim=1)       # (B, features*n_layers, H, W)
        result = self.output_conv(fused)     # (B, embed_dim, H, W)

        if return_fused:
            return result, fused
        return result
```

### Backward compatibility verification

With `n_layers=4, embed_dim=768, features=256`:

| Parameter | DPTSimpleHead | DPTHead | Match? |
|---|---|---|---|
| `projects[i].weight` | `(256, 768, 1, 1)` × 4 | `(256, 768, 1, 1)` × 4 | ✅ |
| `refine_convs[i].weight` | `(256, 256, 3, 3)` × 4 | `(256, 256, 3, 3)` × 4 | ✅ |
| `output_conv.weight` | `(768, 1024, 1, 1)` | `(768, 1024, 1, 1)` | ✅ |
| Output shape | `(B, 768, 64, 64)` | `(B, 768, 64, 64)` | ✅ |

Existing `phase1_best.pt` checkpoints load directly.

### Memory implications

| n_layers | fused channels | output_conv params | fused tensor (B=4, 64×64) |
|---|---|---|---|
| 3 | 768 | 768×768 = 590K | ~48 MB |
| 4 | 1024 | 768×1024 = 787K | ~64 MB |
| 12 | 3072 | 768×3072 = 2.4M | ~192 MB |

All-layer fusion triples memory at the concat stage vs 4-layer. Manageable but worth noting for batch size tuning.

### Integration with `DINOv3MonaEncoder`

Replace `DPTSimpleHead` with `DPTHead` in `dinov3_encoder.py:72-77`:

```python
if adapter_type == "dpt_simple":
    from .dpt_heads import DPTHead
    self.dpt_head = DPTHead(
        embed_dim=embed_dim,
        features=256,
        n_layers=len(self.dpt_layers),
    )
```

No other changes to the encoder forward path — `DPTHead.forward` signature matches `DPTSimpleHead.forward` (plus optional `return_fused`).

---

## Part 3: PCA Visualization Module

### New files

- `dinosam/viz/__init__.py` (empty)
- `dinosam/viz/pca.py`

Core function adapted from `dinov3/notebooks/pca.ipynb`:

```python
import numpy as np
import torch

def pca_visualize(
    features: torch.Tensor,
    mask: torch.Tensor = None,
    n_components: int = 3,
    sigmoid_scale: float = 2.0,
) -> np.ndarray:
    """PCA visualization of 2D feature map.

    Args:
        features: (B, C, H, W) feature map — backbone layer or fused output
        mask: (B, 1, H, W) optional binary mask; if provided, PCA fits only on
              mask=True patches. Default None = fit on all patches.
        n_components: number of PCA components (mapped to RGB channels)
        sigmoid_scale: multiplier before sigmoid for color vibrancy

    Returns:
        (B, H, W, 3) uint8 numpy array — PCA RGB visualization
    """
    from sklearn.decomposition import PCA  # lazy import — optional dependency

    B, C, H, W = features.shape
    results = []

    for b in range(B):
        flat = features[b].reshape(C, -1).permute(1, 0).cpu().numpy()  # (H*W, C)

        if mask is not None:
            fg = mask[b].reshape(-1).cpu().numpy() > 0.5
            fit_data = flat[fg] if fg.any() else flat
        else:
            fit_data = flat

        pca = PCA(n_components=n_components, whiten=True)
        pca.fit(fit_data)

        projected = pca.transform(flat).reshape(H, W, n_components)
        projected = torch.sigmoid(torch.from_numpy(projected) * sigmoid_scale).numpy()

        if mask is not None:
            fg = mask[b, 0].cpu().numpy() > 0.5
            projected *= fg[:, :, np.newaxis]

        results.append((projected * 255).astype(np.uint8))

    return np.stack(results)  # (B, H, W, 3)
```

### Convenience wrapper

```python
def visualize_model_features(
    model,
    images: torch.Tensor,
    device: torch.device,
    save_dir: str,
    prefix: str = "",
    mask: torch.Tensor = None,
):
    """Generate PCA visualizations for selected backbone layers + fused features."""
    from PIL import Image
    import os
    import torch.nn.functional as TF

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        if model.encoder_type == "dinov3" and model.dinov3_encoder is not None:
            encoder = model.dinov3_encoder

            # Preprocess: inline copy of encoder.forward() preprocessing
            x = TF.interpolate(
                images.to(device), size=(encoder.img_size, encoder.img_size),
                mode="bilinear", align_corners=False,
            )
            x = (x - encoder.pixel_mean) / encoder.pixel_std

            # Get intermediate features — same call as training path (no reshape/norm)
            raw_features = encoder.backbone.get_intermediate_layers(
                x, n=encoder.dpt_layers
            )
            # raw_features: list of (B, N, embed_dim) — token sequences

            patch_h = patch_w = encoder.img_size // 16

            # Per-layer PCA: reshape to spatial, then visualize
            for layer_idx, feat in zip(encoder.dpt_layers, raw_features):
                feat_2d = feat.permute(0, 2, 1).reshape(
                    feat.shape[0], feat.shape[-1], patch_h, patch_w
                )  # (B, C, H, W)
                vis = pca_visualize(feat_2d, mask=mask)
                for b in range(vis.shape[0]):
                    path = os.path.join(save_dir, f"{prefix}layer{layer_idx}_b{b}.png")
                    Image.fromarray(vis[b]).save(path)

            # Fused PCA: run through DPTHead with return_fused=True
            _, fused = encoder.dpt_head(raw_features, patch_h, patch_w, return_fused=True)
            vis = pca_visualize(fused, mask=mask)
            for b in range(vis.shape[0]):
                path = os.path.join(save_dir, f"{prefix}fused_b{b}.png")
                Image.fromarray(vis[b]).save(path)
```

### Key design decisions

1. **sklearn is lazy-imported** — only loaded when PCA visualization is actually used. Not required for training.
2. **Preprocessing is duplicated inline** (not extracted to `_preprocess`) — avoids modifying the encoder's public API. The 3-line preprocessing (interpolate + normalize) is simple enough to duplicate.
3. **`get_intermediate_layers` called without `reshape`/`norm`** — matches the training path exactly. Reshape to spatial is done manually in the PCA wrapper.
4. **`return_fused` flag** on DPTHead — enables fused PCA without running a separate forward pass.

---

## Part 4: Training Integration

### New CLI arguments in `trainer.py`

```python
parser.add_argument("--pca_viz", action="store_true",
                    help="Enable PCA visualization during validation")
parser.add_argument("--pca_interval", type=int, default=1,
                    help="Generate PCA visualizations every N epochs")
parser.add_argument("--pca_dir", type=str, default="./pca_viz",
                    help="Directory to save PCA visualization images")
parser.add_argument("--pca_samples", type=int, default=4,
                    help="Number of validation samples to visualize")
```

### Validation hook

At the end of `validate_phase1`:

```python
if args.pca_viz and epoch % args.pca_interval == 0:
    from dinosam.viz.pca import visualize_model_features
    # Use first pca_samples images from val_loader (deterministic)
    sample_batch = next(iter(val_loader))
    save_dir = os.path.join(args.pca_dir, f"epoch_{epoch:03d}")
    visualize_model_features(
        model, sample_batch["image"][:args.pca_samples],
        device, save_dir, prefix=f"ep{epoch:03d}_",
    )
```

### Output structure

```
pca_viz/
  epoch_000/
    ep000_layer2_b0.png      # Per-layer PCA: block index 2, sample 0
    ep000_layer5_b0.png
    ep000_layer8_b0.png
    ep000_layer11_b0.png
    ep000_fused_b0.png       # Fused features PCA
    ep000_layer2_b1.png
    ...
  epoch_001/
    ...
```

---

## Part 5: Inference CLI Tool

### New files

- `dinosam/scripts/__init__.py` (empty)
- `dinosam/scripts/pca_viz.py`

```bash
python -m dinosam.scripts.pca_viz \
    --checkpoint /path/to/best.pt \
    --sam_checkpoint /path/to/sam_vit_b.pth \
    --image /path/to/image.jpg \
    --output_dir ./pca_output \
    --dpt_layers "2,5,8,11" \
    --mask  # optional: use predicted mask for foreground-only PCA fitting
```

### CLI arguments

```python
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--sam_checkpoint", required=True)
parser.add_argument("--image", required=True, nargs="+")
parser.add_argument("--output_dir", default="./pca_output")
parser.add_argument("--dpt_layers", type=str, default="2,5,8,11")  # comma-separated
parser.add_argument("--mask", action="store_true",
                    help="Use model-predicted mask for foreground-only PCA fitting")
parser.add_argument("--encoder", type=str, default="sam", choices=["sam", "dinov3"])
parser.add_argument("--dinov3_checkpoint", type=str, default=None)
parser.add_argument("--adapter_type", type=str, default="mona")
parser.add_argument("--img_size", type=int, default=1024)
```

### Output per image

```
pca_output/
  image_name/
    original.png             # Input image
    mask.png                 # Predicted mask (if --mask)
    layer_2_pca.png          # Per-layer PCA
    layer_5_pca.png
    layer_8_pca.png
    layer_11_pca.png
    fused_pca.png            # Fused features PCA
```

---

## File Changes Summary

| File | Action | Description |
|---|---|---|
| `dinosam/model/dpt_heads.py` | MODIFY | Replace `DPTSimpleHead` with `DPTHead` (dynamic `n_layers`, backward compatible) |
| `dinosam/viz/__init__.py` | NEW | Empty package init |
| `dinosam/viz/pca.py` | NEW | `pca_visualize` + `visualize_model_features` |
| `dinosam/scripts/__init__.py` | NEW | Empty package init |
| `dinosam/scripts/pca_viz.py` | NEW | Standalone inference CLI |
| `dinosam/model/dinov3_encoder.py` | MODIFY | Use `DPTHead` instead of `DPTSimpleHead`; pass `n_layers` |
| `dinosam/model/depth_sam.py` | MODIFY | Pass `dpt_layers` through to encoder construction |
| `dinosam/train/trainer.py` | MODIFY | Remove `adapter_type` guard on `dpt_layers`; add PCA args + validation hook |
| `dinosam/evaluate.py` | MODIFY | Remove `adapter_type` guard on `dpt_layers` (same CLI format) |
| `README.md` | MODIFY | Document new arguments; add scikit-learn to optional dependencies |
| `requirements.txt` (if exists) | MODIFY | Add `scikit-learn` as optional dependency |

---

## Backward Compatibility

- Default `--dpt_layers "2,5,8,11"` = current behavior (4-layer fusion)
- New `DPTHead` with `n_layers=4` is **architecturally identical** to `DPTSimpleHead` — existing checkpoints load directly
- CLI format unchanged (comma-separated string)
- `--pca_viz` defaults to off — no change to existing training
- sklearn is lazy-imported — not required for normal training/evaluation

---

## Testing

```python
# tests/test_dpt_heads.py
import torch
from dinosam.model.dpt_heads import DPTHead, DPTSimpleHead

def test_dpt_head_backward_compat():
    """DPTHead(n_layers=4) must produce identical params to DPTSimpleHead."""
    old = DPTSimpleHead(embed_dim=768, features=256)
    new = DPTHead(embed_dim=768, features=256, n_layers=4)
    old_state = old.state_dict()
    new.load_state_dict(old_state)  # must not raise

def test_dpt_head_forward_compat():
    """DPTHead(n_layers=4) must produce identical output to DPTSimpleHead."""
    torch.manual_seed(42)
    old = DPTSimpleHead(embed_dim=768, features=256)
    new = DPTHead(embed_dim=768, features=256, n_layers=4)
    new.load_state_dict(old.state_dict())
    features = [torch.randn(2, 4096, 768) for _ in range(4)]
    old_out = old(features, 64, 64)
    new_out = new(features, 64, 64)
    assert torch.allclose(old_out, new_out)

def test_dpt_head_3_layers():
    head = DPTHead(embed_dim=768, features=256, n_layers=3)
    features = [torch.randn(2, 4096, 768) for _ in range(3)]
    out = head(features, 64, 64)
    assert out.shape == (2, 768, 64, 64)

def test_dpt_head_12_layers():
    head = DPTHead(embed_dim=768, features=256, n_layers=12)
    features = [torch.randn(2, 4096, 768) for _ in range(12)]
    out = head(features, 64, 64)
    assert out.shape == (2, 768, 64, 64)

def test_dpt_head_return_fused():
    head = DPTHead(embed_dim=768, features=256, n_layers=4)
    features = [torch.randn(2, 4096, 768) for _ in range(4)]
    out, fused = head(features, 64, 64, return_fused=True)
    assert out.shape == (2, 768, 64, 64)
    assert fused.shape == (2, 1024, 64, 64)  # features * n_layers

# tests/test_pca.py
import numpy as np
import torch
from dinosam.viz.pca import pca_visualize

def test_pca_visualize_no_mask():
    features = torch.randn(1, 768, 16, 16)
    vis = pca_visualize(features)
    assert vis.shape == (1, 16, 16, 3)
    assert vis.dtype == np.uint8

def test_pca_visualize_with_mask():
    features = torch.randn(1, 768, 16, 16)
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, 4:12, 4:12] = 1.0
    vis = pca_visualize(features, mask=mask)
    assert vis.shape == (1, 16, 16, 3)
    assert vis[0, 0, 0].sum() == 0  # background black

def test_pca_visualize_empty_mask():
    """All-zero mask falls back to full-image fit."""
    features = torch.randn(1, 768, 16, 16)
    mask = torch.zeros(1, 1, 16, 16)
    vis = pca_visualize(features, mask=mask)
    assert vis.shape == (1, 16, 16, 3)
```

---

## Out of Scope

1. **Phase 2 depth decoder** — ablation study targets mask branch only
2. **PCA during training forward pass** — visualization runs in validation only (no_grad)
3. **Multi-model comparison** — PCA tool visualizes one model at a time
4. **Foreground classifier** — dinov3 notebook uses a separate classifier; we use predicted mask or full image
5. **Encoder `_preprocess` extraction** — preprocessing is duplicated inline (3 lines); extracting to a shared method is future cleanup
6. **Weight migration tooling** — not needed since DPTHead is architecturally identical to DPTSimpleHead when n_layers=4

---

## Review Findings Addressed

| Severity | Finding | Resolution |
|----------|---------|------------|
| CRITICAL | DPTHead not backward-compatible with DPTSimpleHead | Redesigned: identical architecture (uniform projection, no ConvTranspose, output embed_dim=768) |
| CRITICAL | CLI format mismatch (nargs="+" vs comma-separated) | Kept existing comma-separated format |
| HIGH | `_preprocess()` doesn't exist | Preprocessing duplicated inline in `visualize_model_features` |
| HIGH | PCA `get_intermediate_layers` flags mismatch | Match training path: no reshape/norm; manual reshape in PCA wrapper |
| HIGH | DPTHead output dimension mismatch (nclass vs embed_dim) | Output is embed_dim (768), matching DPTSimpleHead |
| HIGH | `out_channels` auto-generation underspecified | Removed — all layers use uniform `features` channels |
| HIGH | Filename `dpt_head.py` vs `dpt_heads.py` | Modify existing `dpt_heads.py` |
| MEDIUM | Missing `__init__.py` for new packages | Added `viz/__init__.py` and `scripts/__init__.py` |
| MEDIUM | sklearn not in requirements | Added as optional dependency with lazy import |
| MEDIUM | evaluate.py not in file changes | Added |
| LOW | Status says APPROVED pre-review | Changed to DRAFT |
| LOW | Test uses 197 (wrong spatial size) | Changed to 4096 (64×64 patches) |
| LOW | Memory implications undocumented | Added memory table |
