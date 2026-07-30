"""Swin-B Stage 3 encoder. Frozen backbone + trainable 1x1 channel projection.

Output: (B, 256, 64, 64) -- compatible with SAM mask decoder and depth decoder.
"""

import torch
from torch import nn
from torch.nn import functional as F


class SwinBEncoder(nn.Module):
    """Swin-B pretrained on ImageNet-1K (22K->1K finetuned), features_only Stage 3.

    Backbone is frozen and locked in eval mode. Only the 512->256 Conv1x1
    projection is trainable.

    Output shape matches DepthSam.encode_images() contract: (B, 256, 64, 64).
    """

    def __init__(self, img_size=1024):
        super().__init__()
        import timm

        self.img_size = img_size

        # Swin-B, ImageNet-22K pretrained -> 1K finetuned, features_only Stage 3
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=True,
            features_only=True,
            out_indices=(2,),      # Stage 3 only: 64x64 when input is 1024^2
            img_size=img_size,
        )

        # Freeze backbone + lock eval (prevents BatchNorm drift if any)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Trainable channel projection: 512 -> 256
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
        # Bicubic resize to 1024x1024
        x = F.interpolate(
            images, size=(self.img_size, self.img_size),
            mode="bicubic", align_corners=False,
        )

        # ImageNet normalization (mean/std x255 because input is [0, 255])
        x = (x / 255.0 - self.pixel_mean) / self.pixel_std

        # Swin-B Stage 3: timm features_only returns list[(B, H, W, C)]
        feats = self.backbone(x)[0]                     # (B, 64, 64, 512) NHWC
        feats = feats.permute(0, 3, 1, 2).contiguous()  # (B, 512, 64, 64) NCHW

        # 512 -> 256 channel projection
        out = self.proj(feats)  # (B, 256, 64, 64)

        return out, (self.img_size, self.img_size)
