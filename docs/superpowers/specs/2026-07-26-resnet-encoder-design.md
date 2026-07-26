# ResNet-50/101 Encoder for 4-Way Backbone Comparison

**Status:** spec (awaiting implementation plan)  
**Date:** 2026-07-26  
**Context:** Add ResNet-50 and ResNet-101 as encoder backbone options, enabling a 4-way comparison experiment: SAM ViT-B vs DINOv3 ViT-B vs ResNet-50 vs ResNet-101.

## Motivation

Current `DepthSam` supports two encoder backbones (`sam`, `dinov3`), both ViT-based. Adding ResNet-50/101 enables a controlled comparison across backbone architectures (ViT vs CNN) and across ResNet depths (50 vs 101 layers), under identical training protocol.

## Design

### Architecture: ResNetEncoder

A minimal encoder class that wraps a pretrained, frozen ResNet with a single trainable conv adapter, producing embeddings compatible with SAM's mask decoder and depth decoder.

```
images (B, 3, H, W) [0, 255]
  → ResizeLongestSide.get_preprocess_shape (aspect-ratio-preserving resize)
  → F.interpolate + ImageNet normalization + F.pad (to 1024×1024)
  → backbone = Sequential(conv1, bn1, relu, maxpool, layer1, layer2, layer3)
  → (B, 1024, 64, 64)
  → Conv2d(1024, 256, kernel_size=3, padding=1)
  → (B, 256, 64, 64), input_size=(newh, neww)
```

Key design decisions:

1. **ResNet layer3 output** (1024 channels, 64×64 spatial) — exact spatial match to SAM's 64×64 embeddings, one conv adapter suffices. Layer4 (2048, 32×32) is stripped via `nn.Sequential` truncation.
2. **Aspect-ratio-preserving resize + pad** — matches `DepthSam.preprocess` semantics exactly. `input_size=(newh, neww)` (NOT fixed `(1024,1024)`), so `postprocess_masks` works without modification.
3. **ImageNet normalization** — `(x - mean*255) / (std*255)` in [0,255] input space, distinct from SAM's normalization. Applied after resize+pad, same position in pipeline as SAM's normalization.
4. **Single Conv2d adapter** — `Conv2d(1024, 256, 3, padding=1)`. No separate 1×1 projection, no multi-scale fusion, no DPT head. One conv is sufficient for a channel-only projection with minor spatial refinement.
5. **Frozen backbone, trainable adapter** — backbone params set `requires_grad=False` at `__init__`, then `backbone.eval()` called to prevent BatchNorm running stats drift. Conv adapter stays trainable.
6. **No dual-DPT support** — ResNet has no dual-head decoder path. `dinov3_encoder = None` ensures `dual_dpt` property returns `False`.

### DepthSam integration

Three minimal branches added to `depth_sam.py`:

**`__init__`** — restructure if/else to hoist `dinov3_encoder = None`:
```python
self.dinov3_encoder = None
if encoder_type == "dinov3":
    self.dinov3_encoder = DINOv3MonaEncoder(...)
elif encoder_type.startswith("resnet"):
    self.resnet_encoder = ResNetEncoder(
        model_name=encoder_type, img_size=image_size)
```
(Eliminates the redundant `else: self.dinov3_encoder = None` from current code.)

**`encode_images`** — add dispatch:
```python
elif self.encoder_type.startswith("resnet"):
    return self.resnet_encoder(images)
```

**`freeze_image_encoder`** — no ResNet branch needed. Backbone is already frozen at construction; the existing unconditional `sam.image_encoder` freeze covers SAM.

### CLI changes

**`trainer.py`** and **`evaluate.py`** and **`pca_viz.py`**:
```python
choices=["sam", "dinov3", "resnet50", "resnet101"]
```

**`trainer.py` validation** (extend existing checks):
```python
# Extend existing adapter_type warning to cover resnet
if args.encoder in ("sam", "resnet50", "resnet101") and args.adapter_type != "mona":
    print(f"Warning: --adapter_type '{args.adapter_type}' is ignored when --encoder {args.encoder}")

# Existing dpt_layers_depth guard already catches resnet (args.encoder != "dinov3"),
# but add explicit check for clearer error message:
if args.dpt_layers_depth is not None and args.encoder.startswith("resnet"):
    parser.error("--dpt_layers_depth is not supported with ResNet encoder")
```

### Backbone eval mode (BatchNorm fix)

ResNet uses BatchNorm. In PyTorch, `requires_grad=False` does NOT stop `running_mean`/`running_var` updates during `.train()`. ViT-based backbones (SAM, DINOv3) use LayerNorm and are unaffected.

Fix: after freezing backbone params in `ResNetEncoder.__init__`, call `self.backbone.eval()` — equivalent to `track_running_stats=False` on every BN module. The backbone never enters training mode regardless of the outer model state.

## Files Changed

```
NEW:
  dinosam/model/resnet_encoder.py       (~45 lines)

MODIFY:
  dinosam/model/__init__.py              +1 line (export ResNetEncoder)
  dinosam/model/depth_sam.py             +8 lines (import + 2 elif branches)
  dinosam/train/trainer.py               +6 lines (choices + warnings + guard)
  dinosam/evaluate.py                    +4 lines (choices + adapter_type warning)
  dinosam/scripts/pca_viz.py             +2 lines (choices extension)
```

Total: **1 new file, 5 modified files, ~65 lines of changes**.

## What Does NOT Change

- Training pipeline (2-phase: mask → depth)
- Loss functions (dice, iou, depth mse, boundary loss)
- Data loading (DepthDataset, collate_fn)
- SAM mask decoder and depth decoder
- Checkpoint format (`strict=False` already handles cross-encoder loading)
- Evaluation metrics (extract_eval_metrics.py works as-is)
- Freeze semantics (all backbones frozen, only adapters + decoders trainable)

## Comparison Matrix

| encoder | backbone (frozen) | adapter (trainable) | decoder (trainable) |
|---|---|---|---|
| `sam` | SAM ViT-B (~89M) | — (built-in) | mask + depth (~4M) |
| `dinov3` | DINOv3 ViT-B/16 (~89M) | DPT + Mona adapters (~15M) | mask + depth (~4M) |
| `resnet50` | ResNet-50 layer0-3 (~23M) | Conv2d 1024→256 (~2.4M) | mask + depth (~4M) |
| `resnet101` | ResNet-101 layer0-3 (~42M) | Conv2d 1024→256 (~2.4M) | mask + depth (~4M) |

All four trained with identical protocol: Phase1 (mask-only, 50 epochs) → Phase2 (depth-only, 50 epochs), batch_size=4, lr=1e-5.

## Usage Examples

```bash
# ResNet-50
python -m dinosam.train.trainer \
    --encoder resnet50 --train_dir /data3/.../1724_train \
    --val_dir /data3/.../1724_test --output_dir ./out/resnet50 \
    --train_phase both --epochs 50

# ResNet-101
python -m dinosam.train.trainer \
    --encoder resnet101 --train_dir /data3/.../1724_train \
    --val_dir /data3/.../1724_test --output_dir ./out/resnet101 \
    --train_phase both --epochs 50

# Evaluation
python -m dinosam.evaluate \
    --encoder resnet50 --checkpoint ./out/resnet50/phase2_best.pt \
    --eval_dir /data3/.../1724_test
```

## Follow-up (not in scope)

- dino-launch skill adaptation for ResNet launch templates
- Smoke tests for ResNetEncoder shape contract and BN freeze correctness
- Update usage guide documentation with ResNet options

## Review History

- Round 1: 3-agent review (code/harness/ponytail) → found `input_size` semantic mismatch, conv adapter trainability ambiguity, missing `freeze_image_encoder` branch, over-engineered 2-conv adapter
- Round 2: 3-agent review → found BatchNorm running stats drift, missing import, missing `pca_viz.py` and `__init__.py` changes, redundant `freeze_image_encoder` branch
- All round 2 issues resolved in this spec
