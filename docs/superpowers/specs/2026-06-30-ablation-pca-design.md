# DINOSAM Ablation Study & PCA Visualization Design

**Date:** 2026-06-30  
**Status:** APPROVED  
**Scope:** DINOSAM project (`/Users/lxt/Documents/SEU/DINOSAM`)

---

## Overview

Two features for DINOSAM:

1. **Configurable DPT fusion layers** — CLI parameter `--dpt_layers` controls which transformer blocks feed into the DPT head, enabling ablation experiments
2. **PCA feature visualization** — training-time optional monitoring + standalone inference CLI tool

---

## Layer Indexing Convention

All layer indices are **0-indexed** transformer block indices, matching segdino's `get_intermediate_layers` API. For ViT-B/16 (12 blocks): valid range is `[0, 11]`.

### Ablation Configurations

| Config Name | `--dpt_layers` | n_layers | Description |
|---|---|---|---|
| Three-layer early | `2 5 8` | 3 | Skip the last block |
| Three-layer uniform | `3 6 9` | 3 | Evenly spaced, shifted later |
| Four-layer SegDINO-style | `2 5 8 11` | 4 | Current default |
| High-level | `8 9 10 11` | 4 | Last 4 blocks only |
| All-layer | `0 1 2 3 4 5 6 7 8 9 10 11` | 12 | Every block |

---

## Part 1: CLI Parameter

### `trainer.py` — new argparse argument

```python
parser.add_argument("--dpt_layers", type=int, nargs="+", default=[2, 5, 8, 11],
                    help="0-indexed transformer block indices for DPT fusion (e.g. 2 5 8 11)")
```

Pass through to model construction:

```python
model = DepthSam(
    sam,
    encoder_type=args.encoder,
    dinov3_checkpoint=args.dinov3_checkpoint,
    adapter_type=args.adapter_type,
    depth_transformer_type=args.depth_transformer_type,
    dpt_layers=args.dpt_layers,  # NEW
)
```

`DINOv3MonaEncoder.__init__` already accepts `dpt_layers` (`dinov3_encoder.py:24`). No new parameter needed on the encoder side.

### Usage examples

```bash
# SegDINO default (4 layers)
python -m dinosam.train.trainer --dpt_layers 2 5 8 11 ...

# Three-layer early
python -m dinosam.train.trainer --dpt_layers 2 5 8 ...

# All-layer
python -m dinosam.train.trainer --dpt_layers 0 1 2 3 4 5 6 7 8 9 10 11 ...
```

---

## Part 2: Dynamic DPTHead

### Problem

Current `_make_scratch` and `DPTHead.forward` are hardcoded for exactly 4 layers. Ablation requires 3, 4, or 12 layers.

### Solution: new file `dinosam/model/dpt_head.py`

```python
class DPTHead(nn.Module):
    """Dynamic DPT head supporting arbitrary number of fusion layers."""

    def __init__(
        self,
        nclass,
        in_channels,        # backbone embed_dim (768 for ViT-B)
        n_layers,            # number of fusion layers
        features=256,
        out_channels=None,   # per-layer projection channels; auto-generated if None
    ):
        # out_channels: one per layer, controls projection target channels
        # If None, default to linearly spaced channels
        # self.projects: ModuleList of n_layers 1x1 Convs (in_channels → out_channels[i])
        # self.refines: ModuleList of n_layers 3x3 Convs (out_channels[i] → features)
        # self.proj: ConvTranspose2d(features, features, 4, stride=4) for first layer
        # self.output_conv: Conv2d(features * n_layers, nclass, 1)

    def forward(self, out_features, patch_h, patch_w):
        # 1. For each layer: permute → reshape → project → refine
        # 2. First layer: ConvTranspose2d upsample 4x
        # 3. Remaining layers: F.interpolate to match first layer's spatial size
        # 4. torch.cat(all_refined, dim=1)  — dynamic n_layers
        # 5. output_conv → logits
```

### `out_channels` auto-generation

When `n_layers` varies, `out_channels` must have matching length. If not provided:

```python
if out_channels is None:
    # Linearly space from 96 to in_channels, matching DPT convention
    out_channels = [int(96 * (2 ** i)) for i in range(n_layers)]
    # e.g. n_layers=4 → [96, 192, 384, 768]
    # e.g. n_layers=3 → [96, 192, 384]
    # e.g. n_layers=12 → clip to reasonable range
```

For `n_layers > 4`, cap the maximum channel to `in_channels` (768) and distribute evenly.

### Integration with `DINOv3MonaEncoder`

The encoder's `dpt_head` initialization (`dinov3_encoder.py:72-77`) changes to use the new dynamic head:

```python
if adapter_type == "dpt_simple":
    from .dpt_head import DPTHead
    n_layers = len(self.dpt_layers)
    self.dpt_head = DPTHead(
        nclass=256,  # output dim for adapter pipeline
        in_channels=embed_dim,
        n_layers=n_layers,
        features=256,
    )
```

---

## Part 3: PCA Visualization Module

### New file `dinosam/viz/pca.py`

Core function adapted from `dinov3/notebooks/pca.ipynb`:

```python
import numpy as np
import torch
from sklearn.decomposition import PCA


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
    B, C, H, W = features.shape
    results = []

    for b in range(B):
        flat = features[b].reshape(C, -1).permute(1, 0).cpu().numpy()  # (H*W, C)

        # Fit PCA
        if mask is not None:
            fg = mask[b].reshape(-1).cpu().numpy() > 0.5
            fit_data = flat[fg] if fg.any() else flat
        else:
            fit_data = flat

        pca = PCA(n_components=n_components, whiten=True)
        pca.fit(fit_data)

        # Project all patches
        projected = pca.transform(flat).reshape(H, W, n_components)  # (H, W, 3)

        # Color mapping: sigmoid scaling for vibrant colors
        projected = torch.from_numpy(projected)
        projected = torch.sigmoid(projected * sigmoid_scale).numpy()

        # Mask background to black if mask provided
        if mask is not None:
            fg = mask[b, 0].cpu().numpy() > 0.5
            projected *= fg[:, :, np.newaxis]

        results.append((projected * 255).astype(np.uint8))

    return np.stack(results)  # (B, H, W, 3)
```

### Convenience wrapper for model features

```python
def visualize_model_features(
    model,
    images: torch.Tensor,
    device: torch.device,
    save_dir: str,
    prefix: str = "",
    mask: torch.Tensor = None,
):
    """Generate PCA visualizations for selected backbone layers + fused features.

    Saves per-layer PCA images and fused PCA image to save_dir.

    Args:
        model: DepthSam model
        images: (B, 3, H, W) input images
        device: torch device
        save_dir: output directory
        prefix: filename prefix (e.g. epoch number)
        mask: optional (B, 1, H, W) binary mask for foreground-only PCA fitting
    """
    from PIL import Image
    import os

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        # 1. Get backbone intermediate features (per-layer)
        embeddings, input_size = model.encode_images(images.to(device))

        # For DINOv3 encoder path: access backbone directly
        if model.encoder_type == "dinov3" and model.dinov3_encoder is not None:
            encoder = model.dinov3_encoder
            x = encoder._preprocess(images.to(device))
            backbone_features = encoder.backbone.get_intermediate_layers(
                x, n=encoder.dpt_layers, reshape=True, norm=True,
            )
            # backbone_features: list of (B, C, H, W) per selected layer

            # Per-layer PCA
            for i, (layer_idx, feat) in enumerate(zip(encoder.dpt_layers, backbone_features)):
                vis = pca_visualize(feat, mask=mask)
                for b in range(vis.shape[0]):
                    path = os.path.join(save_dir, f"{prefix}layer{layer_idx}_b{b}.png")
                    Image.fromarray(vis[b]).save(path)

            # Fused PCA: run through DPT head, get concat features before output_conv
            # (requires DPTHead to optionally return intermediate fused features)

    # 2. Fused features PCA
    # ... (see DPTHead.return_fused option below)
```

### DPTHead: optional fused feature return

Add a `return_fused` flag to `DPTHead.forward`:

```python
def forward(self, out_features, patch_h, patch_w, return_fused=False):
    # ... existing logic ...
    fused = torch.cat(refined, dim=1)
    out = self.output_conv(fused)
    if return_fused:
        return out, fused  # fused: (B, features*n_layers, H', W')
    return out
```

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

At the end of `validate_phase1` (and `validate_phase2` if applicable):

```python
if args.pca_viz and epoch % args.pca_interval == 0:
    from dinosam.viz.pca import visualize_model_features
    # Use first pca_samples images from val_loader
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
    ep000_layer2_b0.png      # Per-layer PCA: layer index 2, batch sample 0
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

### New file `dinosam/scripts/pca_viz.py`

```bash
python -m dinosam.scripts.pca_viz \
    --checkpoint /path/to/best.pt \
    --sam_checkpoint /path/to/sam_vit_b.pth \
    --image /path/to/image.jpg \
    --output_dir ./pca_output \
    --dpt_layers 2 5 8 11 \
    --mask  # optional: use predicted mask for foreground-only PCA fitting
```

### CLI arguments

```python
parser.add_argument("--checkpoint", required=True, help="DINOSAM model checkpoint")
parser.add_argument("--sam_checkpoint", required=True, help="SAM ViT-B checkpoint")
parser.add_argument("--image", required=True, nargs="+", help="Input image path(s)")
parser.add_argument("--output_dir", default="./pca_output")
parser.add_argument("--dpt_layers", type=int, nargs="+", default=[2, 5, 8, 11])
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
| `dinosam/model/dpt_head.py` | NEW | Dynamic DPTHead with `n_layers` parameter |
| `dinosam/viz/pca.py` | NEW | `pca_visualize` + `visualize_model_features` |
| `dinosam/scripts/pca_viz.py` | NEW | Standalone inference CLI |
| `dinosam/model/dinov3_encoder.py` | MODIFY | Use new DPTHead; pass `n_layers` from `dpt_layers` |
| `dinosam/train/trainer.py` | MODIFY | `--dpt_layers`, `--pca_viz`, `--pca_interval`, `--pca_dir`, `--pca_samples` args; PCA hook in validation |
| `dinosam/model/depth_sam.py` | MODIFY | Pass `dpt_layers` through to encoder |
| `README.md` | MODIFY | Document new arguments |

---

## Backward Compatibility

- Default `--dpt_layers 2 5 8 11` = current behavior (SegDINO-style 4-layer fusion)
- `--pca_viz` defaults to off — no change to existing training
- New DPTHead with `n_layers=4` produces identical results to current implementation

---

## Testing

```python
# tests/test_dpt_head.py
def test_dpt_head_3_layers():
    head = DPTHead(nclass=1, in_channels=768, n_layers=3)
    features = [torch.randn(2, 197, 768) for _ in range(3)]
    out = head(features, patch_h=14, patch_w=14)
    assert out.shape[0] == 2

def test_dpt_head_12_layers():
    head = DPTHead(nclass=1, in_channels=768, n_layers=12)
    features = [torch.randn(2, 197, 768) for _ in range(12)]
    out = head(features, patch_h=14, patch_w=14)
    assert out.shape[0] == 2

def test_dpt_head_return_fused():
    head = DPTHead(nclass=1, in_channels=768, n_layers=4)
    features = [torch.randn(2, 197, 768) for _ in range(4)]
    out, fused = head(features, patch_h=14, patch_w=14, return_fused=True)
    assert fused.shape[1] == 256 * 4  # features * n_layers

# tests/test_pca.py
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
    # Background should be black
    assert vis[0, 0, 0].sum() == 0
```

---

## Out of Scope

1. **Phase 2 depth decoder** — ablation study targets mask branch only
2. **PCA during training forward pass** — visualization runs in validation only (no_grad)
3. **Multi-model comparison** — PCA tool visualizes one model at a time
4. **Foreground classifier** — dinov3 notebook uses a separate classifier; we use predicted mask or full image
5. **Video/batch PCA** — inference tool processes one image at a time
