"""Swin-B Stage 3 encoder. Frozen backbone + trainable 1x1 channel projection.

Loads official Microsoft Swin-B ImageNet-22K checkpoint. Uses timm's Swin
Transformer implementation (which handles padding for arbitrary input sizes).
"""

import torch
from torch import nn
from torch.nn import functional as F

# Pre-downloaded official checkpoint path
_LOCAL_CHECKPOINT = "/data3/LXT/data/checkpoint/SwinTransformer/swin_base_patch4_window7_224_22k.pth"


def _official_to_timm(k: str) -> str:
    """Remap official Swin checkpoint keys to timm format.

    Official uses 'layers.i.downsample' (attached to source layer).
    timm uses 'layers_{i+1}.downsample' (attached to destination layer).

    Other than the downsample index shift, keys are identical modulo
    'layers.' → 'layers_' (dots vs underscores in the prefix).
    """
    if k.startswith("layers."):
        parts = k.split(".", 2)       # ["layers", "0", "blocks.0.attn..."]
        layer_idx = int(parts[1])
        suffix = parts[2] if len(parts) > 2 else ""
        if suffix.startswith("downsample"):
            return f"layers_{layer_idx + 1}.{suffix}"
        else:
            return f"layers_{layer_idx}.{suffix}"
    return k


class SwinBEncoder(nn.Module):
    """Swin-B (ImageNet-22K) → Stage 3 features → Conv1x1 projection.

    Backbone is frozen and locked in eval mode. Only the 512→256 Conv1x1
    projection is trainable.

    Output shape: (B, 256, 64, 64) — compatible with DepthSam.encode_images().
    """

    def __init__(self, img_size=1024, checkpoint_path=None):
        super().__init__()
        import timm

        self.img_size = img_size

        # Build model architecture (no pretrained weights yet)
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=False,
            features_only=True,
            out_indices=(2,),              # Stage 3 only: 64×64 at 1024²
            img_size=img_size,
        )

        # Load official 22K checkpoint with key remapping
        ckpt_path = checkpoint_path or _LOCAL_CHECKPOINT
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]

        converted = {}
        for k, v in state_dict.items():
            new_k = _official_to_timm(k)
            if new_k in self.backbone.state_dict():
                converted[new_k] = v

        missing, unexpected = self.backbone.load_state_dict(converted, strict=False)
        if missing:
            print(f"SwinBEncoder: {len(missing)} missing keys: {missing}")
        if unexpected:
            print(f"SwinBEncoder: {len(unexpected)} unexpected keys: {unexpected}")
        print(f"SwinBEncoder: loaded {len(converted)} weights from {ckpt_path}")

        # Freeze backbone + lock eval (prevents BatchNorm drift if any)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Trainable channel projection: 512 → 256
        self.proj = nn.Conv2d(512, 256, kernel_size=1)

        # ImageNet normalization (stored as buffers for .to(device))
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def train(self, mode=True):
        """Keep backbone in eval mode; only projection follows outer mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images):
        """Encode images to (B, 256, 64, 64) embeddings.

        Args:
            images: (B, 3, H, W) raw [0, 255] range

        Returns:
            image_embeddings: (B, 256, 64, 64)
            input_size: (1024, 1024)
        """
        # Bicubic resize to 1024×1024
        x = F.interpolate(
            images, size=(self.img_size, self.img_size),
            mode="bicubic", align_corners=False,
        )

        # ImageNet normalization
        x = (x / 255.0 - self.pixel_mean) / self.pixel_std

        # timm Swin-B Stage 3: features_only returns list[(B, H, W, C)]
        feats = self.backbone(x)[0]                     # (B, 64, 64, 512) NHWC
        feats = feats.permute(0, 3, 1, 2).contiguous()  # (B, 512, 64, 64) NCHW

        # 512 → 256 channel projection
        out = self.proj(feats)  # (B, 256, 64, 64)

        return out, (self.img_size, self.img_size)
