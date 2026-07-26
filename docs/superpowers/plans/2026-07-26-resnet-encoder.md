# ResNet-50/101 Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ResNet-50 and ResNet-101 as frozen image encoder backbones, producing (B, 256, 64, 64) embeddings compatible with SAM's mask decoder and depth decoder.

**Architecture:** New `ResNetEncoder` class in `dinosam/model/resnet_encoder.py` wraps a pretrained, frozen ResNet (conv1 through layer3) with a single trainable `Conv2d(1024→256, k3p1)` adapter. Reuses SAM's aspect-ratio-preserving resize+pad preprocessing so `input_size` semantics match exactly. Three minimal branches added to `DepthSam` (init, encode_images, freeze_image_encoder). CLI choices extended in trainer, evaluate, and pca_viz.

**Tech Stack:** PyTorch, torchvision (resnet50/resnet101 with IMAGENET1K_V1 weights), segment-anything

## Global Constraints

- ResNet backbone must be frozen at construction AND set to eval mode (BatchNorm running stats freeze)
- Conv adapter must remain trainable (not frozen by freeze_image_encoder or ResNetEncoder.__init__)
- `input_size` must return `(newh, neww)` from aspect-ratio-preserving resize (not fixed 1024×1024), matching SAM's preprocess semantics
- `startswith("resnet")` single branch handles both resnet50 and resnet101
- No new CLI parameters — only extend existing `--encoder` choices
- `strict=False` checkpoint loading must continue working for all encoder types
- `dual_dpt` property must return False for ResNet

---

### Task 1: Create ResNetEncoder class

**Files:**
- Create: `dinosam/model/resnet_encoder.py`

**Interfaces:**
- Produces: `ResNetEncoder(model_name: str, img_size: int)` — `forward(images: Tensor) -> Tuple[Tensor, Tuple[int,int]]`

- [ ] **Step 1: Write the full module**

```python
"""ResNet-50/101 image encoder with conv adapter producing SAM-compatible embeddings.

Output: (B, 256, 64, 64) for compatibility with SAM mask decoder and depth decoder.
"""

import torch
from torch import nn
from torch.nn import functional as F
from segment_anything.utils.transforms import ResizeLongestSide


class ResNetEncoder(nn.Module):
    """Pretrained, frozen ResNet backbone + trainable Conv2d adapter.

    Uses ResNet layers conv1 through layer3 (stride 16). Layer4 is stripped.
    Backbone is frozen and set to eval mode to prevent BatchNorm running stats
    from drifting during training. Only the Conv2d adapter is trainable.

    Preprocessing matches DepthSam.preprocess: aspect-ratio-preserving resize
    via ResizeLongestSide, then ImageNet normalization, then square pad to
    img_size. input_size = (newh, neww) preserves the pre-pad dimensions for
    correct postprocess_masks scaling.
    """

    def __init__(self, model_name="resnet50", img_size=1024):
        super().__init__()
        from torchvision.models import resnet50, resnet101

        if model_name == "resnet50":
            resnet = resnet50(weights="IMAGENET1K_V1")
        elif model_name == "resnet101":
            resnet = resnet101(weights="IMAGENET1K_V1")
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3,
        )
        self.adapter = nn.Conv2d(1024, 256, kernel_size=3, padding=1)

        self.img_size = img_size

        # ImageNet normalization (stored as buffers so .to(device) moves them)
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255,
        )

        # Freeze backbone AND set eval mode (prevents BatchNorm running stats drift)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def forward(self, images):
        """Encode images to (B, 256, 64, 64) embeddings.

        Args:
            images: (B, 3, H, W) raw [0, 255] range

        Returns:
            image_embeddings: (B, 256, 64, 64)
            input_size: (H, W) of resized image before padding
        """
        oldh, oldw = images.shape[2], images.shape[3]
        newh, neww = ResizeLongestSide.get_preprocess_shape(oldh, oldw, self.img_size)
        x = F.interpolate(
            images, size=(newh, neww), mode="bilinear",
            align_corners=False, antialias=True,
        )
        input_size = (newh, neww)

        x = (x - self.pixel_mean) / self.pixel_std

        padh = self.img_size - newh
        padw = self.img_size - neww
        x = F.pad(x, (0, padw, 0, padh))

        x = self.backbone(x)
        x = self.adapter(x)
        return x, input_size
```

- [ ] **Step 2: Verify import and shape with quick inline check**

```bash
cd /home/LXT/lxt/DINOSAM/DINOSAM && python -c "
import torch
from dinosam.model.resnet_encoder import ResNetEncoder

# Test resnet50 construction
enc = ResNetEncoder('resnet50')
print('resnet50 created')
print(f'  backbone params: {sum(p.numel() for p in enc.backbone.parameters()):,}')
print(f'  adapter params: {sum(p.numel() for p in enc.adapter.parameters()):,}')
print(f'  backbone requires_grad: {all(not p.requires_grad for p in enc.backbone.parameters())}')
print(f'  adapter requires_grad: {all(p.requires_grad for p in enc.adapter.parameters())}')
print(f'  backbone training: {enc.backbone.training}')  # should be False (eval mode)

# Test forward shape: square input
x = torch.randint(0, 255, (2, 3, 1024, 1024), dtype=torch.float32)
emb, input_size = enc(x)
print(f'  square input: emb={tuple(emb.shape)}, input_size={input_size}')
assert emb.shape == (2, 256, 64, 64), f'Expected (2,256,64,64), got {emb.shape}'
assert input_size == (1024, 1024), f'Expected (1024,1024), got {input_size}'

# Test forward shape: non-square input (aspect-ratio preserved)
x2 = torch.randint(0, 255, (1, 3, 800, 600), dtype=torch.float32)
emb2, input_size2 = enc(x2)
print(f'  800x600 input: emb={tuple(emb2.shape)}, input_size={input_size2}')
assert emb2.shape == (1, 256, 64, 64)
assert input_size2[0] != input_size2[1], 'Non-square input should produce non-square input_size'

# Test resnet101
enc101 = ResNetEncoder('resnet101')
emb101, _ = enc101(x)
assert emb101.shape == (2, 256, 64, 64)
print('resnet101: OK')

print('All checks passed')
"
```

- [ ] **Step 3: Verify BatchNorm stats don't drift during train()**

```bash
cd /home/LXT/lxt/DINOSAM/DINOSAM && python -c "
import torch
from dinosam.model.resnet_encoder import ResNetEncoder

enc = ResNetEncoder('resnet50')
# Capture initial BN running_mean
bn_layers = [m for m in enc.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d)]
initial_means = [bn.running_mean.clone() for bn in bn_layers]

# Run forward in train() — backbone should stay in eval (BN stats frozen)
enc.train()
x = torch.randn(4, 3, 1024, 1024) * 50 + 128  # realistic-ish image stats
for _ in range(10):
    enc(x)

for i, (bn, init_mean) in enumerate(zip(bn_layers, initial_means)):
    diff = (bn.running_mean - init_mean).abs().max().item()
    assert diff == 0.0, f'BN layer {i} running_mean drifted by {diff}'
print(f'BatchNorm stats frozen across {len(bn_layers)} BN layers: OK')
"
```

- [ ] **Step 4: Commit**

```bash
cd /home/LXT/lxt/DINOSAM/DINOSAM && git add dinosam/model/resnet_encoder.py && git commit -m "feat: add ResNetEncoder with frozen backbone and trainable conv adapter

ResNet-50/101 layer3 features (1024, 64x64) projected to (256, 64x64).
Backbone frozen at init with eval() to prevent BatchNorm running stats
drift. Preprocessing matches DepthSam.preprocess semantics.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Task 2: Export ResNetEncoder from model package

**Files:**
- Modify: `dinosam/model/__init__.py`

**Interfaces:**
- Produces: `from dinosam.model import ResNetEncoder` usable by external code

- [ ] **Step 1: Add import after line 3**

Insert after `from .dinov3_encoder import DINOv3MonaEncoder`:

```python
from .resnet_encoder import ResNetEncoder
```

The file becomes:
```python
from .depth_decoder import DepthMaskDecoder
from .depth_sam import DepthSam
from .dinov3_encoder import DINOv3MonaEncoder
from .resnet_encoder import ResNetEncoder
from .mona import Mona, MonaOp
```

- [ ] **Step 2: Verify import works**

```bash
cd /home/LXT/lxt/DINOSAM/DINOSAM && python -c "from dinosam.model import ResNetEncoder; print('Import OK:', ResNetEncoder)"
```

- [ ] **Step 3: Commit**

```bash
cd /home/LXT/lxt/DINOSAM/DINOSAM && git add dinosam/model/__init__.py && git commit -m "feat: export ResNetEncoder from dinosam.model package

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Task 3: Integrate ResNet encoder into DepthSam

**Files:**
- Modify: `dinosam/model/depth_sam.py`

**Interfaces:**
- Consumes: `ResNetEncoder(model_name, img_size)` from Task 1
- Produces: `DepthSam(encoder_type="resnet50"|"resnet101")` builds and uses ResNetEncoder

- [ ] **Step 1: Add import on line 11 (after DINOv3MonaEncoder import)**

Current lines 10-11:
```python
from .dinov3_encoder import DINOv3MonaEncoder
```

Add after:
```python
from .resnet_encoder import ResNetEncoder
```

- [ ] **Step 2: Restructure __init__ encoder dispatch (lines 38-51)**

Replace:
```python
        if encoder_type == "dinov3":
            if dinov3_checkpoint is None:
                raise ValueError("dinov3_checkpoint is required when encoder_type='dinov3'")
            self.dinov3_encoder = DINOv3MonaEncoder(
                checkpoint_path=dinov3_checkpoint,
                img_size=image_size,
                embed_dim=768,
                out_dim=256,
                adapter_type=adapter_type,
                dpt_layers=dpt_layers,
                dpt_layers_depth=dpt_layers_depth,
            )
        else:
            self.dinov3_encoder = None
```

With:
```python
        self.dinov3_encoder = None
        if encoder_type == "dinov3":
            if dinov3_checkpoint is None:
                raise ValueError("dinov3_checkpoint is required when encoder_type='dinov3'")
            self.dinov3_encoder = DINOv3MonaEncoder(
                checkpoint_path=dinov3_checkpoint,
                img_size=image_size,
                embed_dim=768,
                out_dim=256,
                adapter_type=adapter_type,
                dpt_layers=dpt_layers,
                dpt_layers_depth=dpt_layers_depth,
            )
        elif encoder_type.startswith("resnet"):
            self.resnet_encoder = ResNetEncoder(
                model_name=encoder_type, img_size=image_size,
            )
```

- [ ] **Step 3: Add ResNet dispatch in encode_images (lines 204-209)**

Replace:
```python
        if self.encoder_type == "dinov3":
            return self.dinov3_encoder(images)
        else:
            input_images, input_size = self.preprocess(images)
            image_embeddings = self.sam.image_encoder(input_images)
            return image_embeddings, input_size
```

With:
```python
        if self.encoder_type == "dinov3":
            return self.dinov3_encoder(images)
        elif self.encoder_type.startswith("resnet"):
            return self.resnet_encoder(images)
        else:
            input_images, input_size = self.preprocess(images)
            image_embeddings = self.sam.image_encoder(input_images)
            return image_embeddings, input_size
```

- [ ] **Step 4: Verify DepthSam builds with all encoder types**

```bash
cd /home/LXT/lxt/DINOSAM/DINOSAM && python -c "
import torch, os
from segment_anything.build_sam import sam_model_registry

sam_ckpt = '/data3/LXT/data/checkpoint/sam_vit_b_01ec64.pth'
dinov3_ckpt = '/data3/LXT/data/checkpoint/dinov3_vitb16_pretrain.pth'

if not os.path.exists(sam_ckpt):
    print('SAM checkpoint not found, skipping')
    exit(0)

from dinosam.model.depth_sam import DepthSam

sam = sam_model_registry['vit_b'](checkpoint=sam_ckpt)

# Test resnet50
model = DepthSam(sam, encoder_type='resnet50')
print(f'resnet50: built OK, dual_dpt={model.dual_dpt}')
x = torch.randint(0, 255, (2, 3, 800, 600), dtype=torch.float32)
emb, input_size = model.encode_images(x)
print(f'  emb={tuple(emb.shape)}, input_size={input_size}')
assert emb.shape == (2, 256, 64, 64)
assert model.dual_dpt is False

# Test resnet101
model101 = DepthSam(sam, encoder_type='resnet101')
print(f'resnet101: built OK')
model101.freeze_image_encoder()
print('  freeze_image_encoder: OK')

# Test existing sam encoder still works
model_sam = DepthSam(sam, encoder_type='sam')
emb_sam, _ = model_sam.encode_images(x)
assert emb_sam.shape == (2, 256, 64, 64)
print('sam encoder: still OK')

# Test dinov3 encoder still works
if os.path.exists(dinov3_ckpt):
    model_dino = DepthSam(sam, encoder_type='dinov3', dinov3_checkpoint=dinov3_ckpt)
    emb_dino, _ = model_dino.encode_images(x)
    assert emb_dino.shape == (2, 256, 64, 64) or isinstance(emb_dino, dict)
    print('dinov3 encoder: still OK')

print('All integration checks passed')
"
```

- [ ] **Step 5: Commit**

```bash
cd /home/LXT/lxt/DINOSAM/DINOSAM && git add dinosam/model/depth_sam.py && git commit -m "feat: add ResNet encoder dispatch in DepthSam

Add dispatch branches in __init__ and encode_images. Restructure
if/elif to hoist dinov3_encoder=None, eliminating redundant else branch.
ResNetEncoder.forward handles its own preprocessing.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Task 4: Update trainer CLI for ResNet encoders

**Files:**
- Modify: `dinosam/train/trainer.py:651,658,672-678`

**Interfaces:**
- Consumes: `DepthSam(encoder_type="resnet50"|"resnet101")` from Task 3

- [ ] **Step 1: Extend --encoder choices (line 651)**

Replace:
```python
    parser.add_argument("--encoder", type=str, default="sam", choices=["sam", "dinov3"],
                        help="Image encoder type: 'sam' (default) or 'dinov3' (DINOv3+Mona)")
```

With:
```python
    parser.add_argument("--encoder", type=str, default="sam",
                        choices=["sam", "dinov3", "resnet50", "resnet101"],
                        help="Image encoder type: 'sam' (default), 'dinov3' (DINOv3+Mona), "
                             "'resnet50', or 'resnet101' (ImageNet pretrained, frozen)")
```

- [ ] **Step 2: Extend adapter_type warning (line 677-678)**

Replace:
```python
    if args.encoder == "sam" and args.adapter_type != "mona":
        print(f"Warning: --adapter_type '{args.adapter_type}' is ignored when --encoder sam")
```

With:
```python
    if args.encoder in ("sam", "resnet50", "resnet101") and args.adapter_type != "mona":
        print(f"Warning: --adapter_type '{args.adapter_type}' is ignored when --encoder {args.encoder}")
```

- [ ] **Step 3: Add guard for --dpt_layers_depth with ResNet (after existing guard, after line 675)**

Add after the existing `--dpt_layers_depth` validation:
```python
    if args.dpt_layers_depth is not None and args.encoder.startswith("resnet"):
        parser.error("--dpt_layers_depth is not supported with ResNet encoder")
```

- [ ] **Step 4: Verify CLI accepts new encoder types**

```bash
cd /home/LXT/lxt/DINOSAM/DINOSAM && python -c "
import sys; sys.argv = ['trainer', '--encoder', 'resnet50']
# Just check argparse doesn't reject resnet50 at parse time
try:
    from dinosam.train.trainer import main
    print('Trainer CLI: resnet50 accepted by argparse')
except SystemExit:
    pass
"

# Test that dpt_layers_depth is rejected with resnet
cd /home/LXT/lxt/DINOSAM/DINOSAM && python -m dinosam.train.trainer --encoder resnet50 --dpt_layers_depth 2,5,8,11 2>&1 | head -3
# Expected: "error: --dpt_layers_depth is not supported with ResNet encoder"
```

- [ ] **Step 5: Commit**

```bash
cd /home/LXT/lxt/DINOSAM/DINOSAM && git add dinosam/train/trainer.py && git commit -m "feat: add resnet50/resnet101 to trainer --encoder choices

Extend choices list, adapter_type ignored-warning, and add guard
rejecting --dpt_layers_depth with ResNet encoders.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Task 5: Update evaluate CLI for ResNet encoders

**Files:**
- Modify: `dinosam/evaluate.py:328,353-354`

**Interfaces:**
- Consumes: `DepthSam(encoder_type="resnet50"|"resnet101")` from Task 3

- [ ] **Step 1: Extend --encoder choices (line 328)**

Replace:
```python
    parser.add_argument("--encoder", type=str, default="sam", choices=["sam", "dinov3"],
                        help="Image encoder type: 'sam' (default) or 'dinov3' (DINOv3+Mona)")
```

With:
```python
    parser.add_argument("--encoder", type=str, default="sam",
                        choices=["sam", "dinov3", "resnet50", "resnet101"],
                        help="Image encoder type: 'sam' (default), 'dinov3' (DINOv3+Mona), "
                             "'resnet50', or 'resnet101'")
```

- [ ] **Step 2: Extend adapter_type ignored warning (line 353-354)**

Replace:
```python
    if args.encoder == "sam" and args.adapter_type != "mona":
        print(f"Warning: --adapter_type '{args.adapter_type}' is ignored when --encoder sam")
```

With:
```python
    if args.encoder in ("sam", "resnet50", "resnet101") and args.adapter_type != "mona":
        print(f"Warning: --adapter_type '{args.adapter_type}' is ignored when --encoder {args.encoder}")
```

- [ ] **Step 3: Verify CLI accepts new encoder types**

```bash
cd /home/LXT/lxt/DINOSAM/DINOSAM && python -c "
import sys; sys.argv = ['evaluate', '--encoder', 'resnet50', '--phase1_checkpoint', '/nonexistent', '--phase2_checkpoint', '/nonexistent']
try:
    from dinosam.evaluate import main
    print('Evaluate CLI: resnet50 accepted by argparse')
except SystemExit:
    pass
"
```

- [ ] **Step 4: Commit**

```bash
cd /home/LXT/lxt/DINOSAM/DINOSAM && git add dinosam/evaluate.py && git commit -m "feat: add resnet50/resnet101 to evaluate --encoder choices

Extend choices list and adapter_type ignored-warning.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Task 6: Update pca_viz CLI for ResNet encoders

**Files:**
- Modify: `dinosam/scripts/pca_viz.py:29`

- [ ] **Step 1: Extend --encoder choices (line 29)**

Replace:
```python
    parser.add_argument("--encoder", type=str, default="sam", choices=["sam", "dinov3"],
                        help="Image encoder type (default: sam)")
```

With:
```python
    parser.add_argument("--encoder", type=str, default="sam",
                        choices=["sam", "dinov3", "resnet50", "resnet101"],
                        help="Image encoder type (default: sam)")
```

Note: PCA visualization relies on ViT intermediate layers (`viz/pca.py` guards on `encoder_type == "dinov3"`). ResNet models will load correctly but PCA viz will be skipped. No functional change needed beyond the choices list.

- [ ] **Step 2: Commit**

```bash
cd /home/LXT/lxt/DINOSAM/DINOSAM && git add dinosam/scripts/pca_viz.py && git commit -m "feat: add resnet50/resnet101 to pca_viz --encoder choices

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Task 7: End-to-end smoke test

**Files:**
- No changes. Verification-only task.

- [ ] **Step 1: Verify all encoder types build, freeze, and pass shapes**

```bash
cd /home/LXT/lxt/DINOSAM/DINOSAM && python -c "
import torch, os
from segment_anything.build_sam import sam_model_registry
from dinosam.model.depth_sam import DepthSam

sam_ckpt = '/data3/LXT/data/checkpoint/sam_vit_b_01ec64.pth'
if not os.path.exists(sam_ckpt):
    print('SAM checkpoint not found, skipping')
    exit(0)

sam = sam_model_registry['vit_b'](checkpoint=sam_ckpt)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

configs = [
    {'encoder_type': 'sam'},
    {'encoder_type': 'dinov3', 'dinov3_checkpoint': '/data3/LXT/data/checkpoint/dinov3_vitb16_pretrain.pth'},
    {'encoder_type': 'resnet50'},
    {'encoder_type': 'resnet101'},
]

for cfg in configs:
    enc = cfg['encoder_type']
    dino_ckpt = cfg.get('dinov3_checkpoint')
    if enc == 'dinov3' and not os.path.exists(dino_ckpt):
        print(f'Skipping {enc} — no DINOv3 checkpoint')
        continue

    model = DepthSam(sam, **{k: v for k, v in cfg.items()}).to(device)
    model.freeze_image_encoder()
    model.init_depth_from_sam()

    # Check freeze correctness
    if hasattr(model, 'resnet_encoder'):
        assert all(not p.requires_grad for p in model.resnet_encoder.backbone.parameters()), \
            f'{enc}: backbone should be frozen'
        assert all(p.requires_grad for p in model.resnet_encoder.adapter.parameters()), \
            f'{enc}: adapter should be trainable'
    if hasattr(model, 'dinov3_encoder') and model.dinov3_encoder is not None:
        assert all(not p.requires_grad for p in model.dinov3_encoder.backbone.parameters()), \
            f'{enc}: backbone should be frozen'
    assert all(not p.requires_grad for p in model.sam.image_encoder.parameters()), \
        f'{enc}: SAM image_encoder should be frozen'

    # Forward pass shape check
    x = torch.randint(0, 255, (2, 3, 800, 600), dtype=torch.float32).to(device)
    emb, input_size = model.encode_images(x)
    if isinstance(emb, dict):
        assert emb['mask'].shape == (2, 256, 64, 64)
    else:
        assert emb.shape == (2, 256, 64, 64), f'{enc}: bad shape {emb.shape}'

    print(f'{enc}: OK')

print('All encoder types: PASS')
"
```

- [ ] **Step 2: Cross-encoder checkpoint compatibility check**

```bash
cd /home/LXT/lxt/DINOSAM/DINOSAM && python -c "
import torch, os
from segment_anything.build_sam import sam_model_registry
from dinosam.model.depth_sam import DepthSam

sam_ckpt = '/data3/LXT/data/checkpoint/sam_vit_b_01ec64.pth'
if not os.path.exists(sam_ckpt):
    print('SAM checkpoint not found, skipping')
    exit(0)

sam = sam_model_registry['vit_b'](checkpoint=sam_ckpt)

# Save fake resnet50 checkpoint, load into resnet50 → clean
model_r50 = DepthSam(sam, encoder_type='resnet50')
ckpt = {'model_state_dict': model_r50.state_dict()}
result = model_r50.load_state_dict(ckpt['model_state_dict'], strict=False)
resnet_missing = [k for k in result.missing_keys if 'resnet' in k]
resnet_unexpected = [k for k in result.unexpected_keys if 'resnet' in k]
assert len(resnet_missing) == 0, f'Unexpected missing: {resnet_missing}'
assert len(resnet_unexpected) == 0, f'Unexpected keys: {resnet_unexpected}'
print('resnet50→resnet50: clean')

# Load resnet50 ckpt into dinov3 → handles gracefully
dinov3_ckpt_path = '/data3/LXT/data/checkpoint/dinov3_vitb16_pretrain.pth'
if os.path.exists(dinov3_ckpt_path):
    model_dino = DepthSam(sam, encoder_type='dinov3', dinov3_checkpoint=dinov3_ckpt_path)
    result2 = model_dino.load_state_dict(ckpt['model_state_dict'], strict=False)
    assert any('resnet' in k for k in result2.unexpected_keys), 'Expected resnet keys as unexpected'
    print('resnet50→dinov3: unexpected keys handled gracefully')

print('Checkpoint compatibility: PASS')
"
```

- [ ] **Step 3: No commit (verification only)**

Verify all checks pass with zero errors.
