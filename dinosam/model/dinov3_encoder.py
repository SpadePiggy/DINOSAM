import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add DINOv3 to path if not already importable
_dinov3_path = os.path.join(os.path.expanduser("~"), "lxt", "dinov3")
if _dinov3_path not in sys.path:
    sys.path.insert(0, _dinov3_path)

from .adapters import get_adapter


class DINOv3MonaEncoder(nn.Module):
    """DINOv3 ViT-B/16 with Mona adapters, outputting 256-dim embeddings.

    DINOv3 backbone is frozen; only Mona adapters + projection are trainable.
    Output shape matches SAM image_encoder: (B, 256, 64, 64) for 1024×1024 input.
    """

    def __init__(self, checkpoint_path, img_size=1024, embed_dim=768, out_dim=256,
                 adapter_type="mona", dpt_layers=None):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.adapter_type = adapter_type

        # Validate dpt_layers usage
        if dpt_layers is not None and adapter_type != "dpt_simple":
            raise ValueError(
                f"dpt_layers parameter is only valid when adapter_type='dpt_simple'. "
                f"Got adapter_type='{adapter_type}', dpt_layers={dpt_layers}. "
                f"Either set adapter_type='dpt_simple' or remove dpt_layers."
            )

        from dinov3.models.vision_transformer import DinoVisionTransformer

        self.backbone = DinoVisionTransformer(
            img_size=img_size,
            patch_size=16,
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=768,
            depth=12,
            num_heads=12,
            ffn_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.0,
            layerscale_init=1e-5,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
        )

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.backbone.load_state_dict(state_dict, strict=True)
        print(f"Loaded DINOv3 ViT-B/16 from {checkpoint_path}")

        for param in self.backbone.parameters():
            param.requires_grad = False

        # DPT head + adapters
        if adapter_type == "dpt_simple":
            from .dpt_heads import DPTHead
            self.dpt_layers = dpt_layers or [2, 5, 8, 11]
            self.dpt_head = DPTHead(embed_dim=embed_dim, n_layers=len(self.dpt_layers))
            self.adapter1 = get_adapter("mona", embed_dim, factor=8)
            self.adapter2 = get_adapter("mona", embed_dim, factor=8)
        else:
            self.dpt_head = None
            self.adapter1 = get_adapter(adapter_type, embed_dim, factor=8)
            self.adapter2 = get_adapter(adapter_type, embed_dim, factor=8)

        self.projection = nn.Linear(embed_dim, out_dim)

        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255,
        )

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, images):
        """Encode images to (B, 256, 64, 64) embeddings.

        Args:
            images: (B, 3, H, W) raw [0, 255] range

        Returns:
            image_embeddings: (B, 256, 64, 64)
            input_size: (H, W) of resized image before padding
        """
        x = F.interpolate(
            images, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        )
        input_size = (self.img_size, self.img_size)

        x = (x - self.pixel_mean) / self.pixel_std

        # Critical: define all variables before if/else branch
        B = x.shape[0]
        patch_h = patch_w = self.img_size // 16
        H, W = patch_h, patch_w

        if self.dpt_head is not None:
            # DPT path: multi-level features, all at 64×64
            features = self.backbone.get_intermediate_layers(
                x, n=self.dpt_layers
            )
            x_2d = self.dpt_head(features, patch_h, patch_w)  # (B, 768, 64, 64)
            x = x_2d.flatten(2).transpose(1, 2)                # (B, N, 768)
        else:
            # Existing path: single-layer features
            features = self.backbone.forward_features(x)
            x = features["x_norm_patchtokens"]

        x = self.adapter1(x, (H, W))
        x = self.adapter2(x, (H, W))

        x = self.projection(x)  # (B, H*W, 256)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, 256, H, W)

        return x, input_size
