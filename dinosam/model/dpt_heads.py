import torch
import torch.nn as nn


class DPTSimpleHead(nn.Module):
    """DPT multi-level feature extraction head. No resize, no upsample.

    All 4 intermediate layers from DINOv3 are at the same spatial resolution
    (patch_h × patch_w = 64×64 for 1024×1024 input with patch_size=16).

    Source: adapted from segdino/dpt.py (simple concat).

    Data flow:
        Input: 4× (B, N, embed_dim) where N = patch_h × patch_w
        Output: (B, embed_dim, patch_h, patch_w)

    Architecture:
        1. Reshape each feature to (B, embed_dim, H, W)
        2. Project each to features channels via 1×1 Conv
        3. Refine each via 3×3 Conv
        4. Concatenate along channel dim → (B, features*4, H, W)
        5. Reduce to embed_dim via 1×1 Conv
    """

    def __init__(self, embed_dim=768, features=256):
        super().__init__()

        # 4× 1×1 projection: embed_dim → features (uniform, no out_channels)
        self.projects = nn.ModuleList([
            nn.Conv2d(embed_dim, features, kernel_size=1) for _ in range(4)
        ])

        # 4× 3×3 Conv: features → features (inline refine, no _make_scratch)
        self.refine_convs = nn.ModuleList([
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)
            for _ in range(4)
        ])

        # concat (4*features) → embed_dim
        self.output_conv = nn.Conv2d(features * 4, embed_dim, kernel_size=1)

    def forward(self, features, patch_h, patch_w):
        """
        Args:
            features: list of 4 tensors, each (B, N, embed_dim)
            patch_h, patch_w: spatial size of patch tokens (e.g., 64, 64)
        Returns:
            (B, embed_dim, patch_h, patch_w) — same spatial as input
        """
        out = []
        for i, x in enumerate(features):
            # (B, N, C) → (B, C, patch_h, patch_w)
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
            x = self.projects[i](x)         # (B, features, H, W)
            x = self.refine_convs[i](x)     # (B, features, H, W)
            out.append(x)

        fused = torch.cat(out, dim=1)       # (B, features*4, H, W)
        return self.output_conv(fused)      # (B, embed_dim, H, W)


class DPTHead(nn.Module):
    """Generalized DPT multi-level feature extraction head.

    Backward-compatible with DPTSimpleHead when n_layers=4: produces identical
    state_dict keys and forward output. Supports arbitrary n_layers (3, 4, 12, …).

    Data flow:
        Input: n_layers × (B, N, embed_dim) where N = patch_h × patch_w
        Output: (B, embed_dim, patch_h, patch_w)
    """

    def __init__(self, embed_dim=768, features=256, n_layers=4):
        super().__init__()
        self.n_layers = n_layers

        self.projects = nn.ModuleList([
            nn.Conv2d(embed_dim, features, kernel_size=1) for _ in range(n_layers)
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
            patch_h, patch_w: spatial size of patch tokens
            return_fused: if True, also return pre-output fused tensor
        Returns:
            (B, embed_dim, patch_h, patch_w), or tuple with fused tensor
        """
        out = []
        for i, x in enumerate(features):
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
            x = self.projects[i](x)
            x = self.refine_convs[i](x)
            out.append(x)

        fused = torch.cat(out, dim=1)
        result = self.output_conv(fused)

        if return_fused:
            return result, fused
        return result
