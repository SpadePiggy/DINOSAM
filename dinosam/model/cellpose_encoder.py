"""Cellpose encoder: frozen cellpose eval() + trainable Conv downsampling head.

Takes raw [0,255] images, runs frozen CellposeModel.eval() to extract
dP (XY gradient flow) and cellprob (cell probability), concatenates into
a 3-channel feature map, then downsamples to (B, 256, 64, 64) for
compatibility with SAM mask decoder and depth decoder.
"""

import os
import numpy as np
import torch
from torch import nn


# Point cellpose to pre-downloaded models
_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "cellpose", "models"
)
os.environ.setdefault("CELLPOSE_LOCAL_MODELS_PATH", os.path.abspath(_MODEL_DIR))


class CellposeEncoder(nn.Module):
    """Frozen cellpose -> trainable 4-layer Conv downsample -> (B, 256, 64, 64).

    CellposeModel.eval() runs in eval mode with frozen weights. Its two raw
    output feature tensors — dP (XY gradient flow, 2ch) and cellprob (cell
    probability, 1ch) — are concatenated into a 3-channel feature map at the
    original image resolution, then downsampled by 4x stride-2 Conv layers to
    64x64 and projected to 256 channels.

    Output interface matches all other encoders (SamEncoder,
    DINOv3MonaEncoder, ResNetEncoder, SwinBEncoder).

    Args:
        model_type: cellpose model name (e.g. "cyto3", "livecell").
            Loaded from CELLPOSE_LOCAL_MODELS_PATH.
        device: torch device for cellpose model + downsample head.
    """

    def __init__(self, model_type="cyto3", device=None):
        super().__init__()

        from cellpose import models

        model_path = os.path.join(
            os.environ["CELLPOSE_LOCAL_MODELS_PATH"], model_type
        )
        if os.path.exists(model_path):
            self.cp_model = models.CellposeModel(
                gpu=(device.type == "cuda" if device else False),
                pretrained_model=model_path,
                device=device,
            )
        else:
            self.cp_model = models.CellposeModel(
                gpu=(device.type == "cuda" if device else False),
                model_type=model_type,
                device=device,
            )

        # Freeze cellpose
        for param in self.cp_model.net.parameters():
            param.requires_grad = False
        self.cp_model.net.eval()

        # Trainable downsample head: 1024 -> 64 (4x stride-2), 3 -> 256 channels
        # 3 input channels: dP (2ch) + cellprob (1ch)
        self.downsample = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),    # 1024 -> 512
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 512 -> 256
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 256 -> 128
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1), # 128 -> 64
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1),                       # 64 -> 64
        )

    def forward(self, images: torch.Tensor):
        """Encode images to (B, 256, 64, 64) embeddings.

        Args:
            images: (B, 3, H, W) raw [0, 255] range

        Returns:
            image_embeddings: (B, 256, 64, 64)
            input_size: (1024, 1024)
        """
        B = images.shape[0]
        device = images.device

        feat_batch = []
        for i in range(B):
            # Convert to numpy for cellpose eval()
            img_np = images[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            masks, flows, styles = self.cp_model.eval(
                img_np,
                channels=[0, 0],
                diameter=None,
            )

            # flows = [dx_to_circ(dP), dP, cellprob]
            #   flows[0]: (H, W, 3) uint8 RGB visualization — skip
            #   flows[1]: (2, H, W) float32 — raw XY gradient field
            #   flows[2]: (H, W) float32 — cell probability map
            dP = torch.as_tensor(flows[1], device=device)          # (2, H, W)
            cellprob = torch.as_tensor(flows[2], device=device)    # (H, W)

            # Resize to 1024x1024 if needed
            if dP.shape[-2:] != (1024, 1024):
                dP = nn.functional.interpolate(
                    dP.unsqueeze(0), size=(1024, 1024),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
            if cellprob.shape[-2:] != (1024, 1024):
                cellprob = nn.functional.interpolate(
                    cellprob.unsqueeze(0).unsqueeze(0), size=(1024, 1024),
                    mode="bilinear", align_corners=False,
                ).squeeze(0).squeeze(0)

            # Concatenate: dP(2) + cellprob(1) = 3 channels
            feat = torch.cat([dP, cellprob.unsqueeze(0)], dim=0)  # (3, 1024, 1024)
            feat = feat.unsqueeze(0)                                 # (1, 3, 1024, 1024)
            feat = self.downsample(feat)                             # (1, 256, 64, 64)
            feat_batch.append(feat)

        image_embeddings = torch.cat(feat_batch, dim=0)  # (B, 256, 64, 64)
        return image_embeddings, (1024, 1024)
