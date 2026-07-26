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

        # Freeze backbone AND lock in eval mode (prevents BatchNorm running stats drift)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Store non-trainable bn1.running_mean/std from original ResNet on the
        # adapter state_dict — they won't appear in backbone.parameters() but
        # are buffers tracked by Sequential.  They're frozen because backbone
        # never leaves eval mode (see train() override below).

    def train(self, mode=True):
        """Override train() to keep backbone in eval mode.

        The adapter follows the outer model's training mode. The backbone
        stays in eval to freeze BatchNorm running stats.
        """
        super().train(mode)
        self.backbone.eval()
        return self

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
