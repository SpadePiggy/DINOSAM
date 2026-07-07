import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedLayerFusion(nn.Module):
    """Learned global layer weights + top-k fusion.

    Training:  softmax(weights) → weighted sum of ALL layers (full gradient)
    Inference: select top-k layers by weight → weighted sum of k layers only
    """

    def __init__(self, embed_dim=768, num_layers=12, k=4):
        super().__init__()
        self.k = k
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        # Learnable layer weights (zero-init → softmax = uniform distribution)
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

        # Per-layer projection (Conv1x1 + GroupNorm + GELU).
        # GroupNorm ensures consistent output scale across layers so weight
        # learning isn't biased by random init scale differences.
        # GroupNorm over InstanceNorm: InstanceNorm produces NaN at B=1
        # (zero variance in single-sample batches).
        self.layer_projects = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 1),
                nn.GroupNorm(num_groups=min(32, embed_dim), num_channels=embed_dim),
                nn.GELU(),
            )
            for _ in range(num_layers)
        ])

    def forward(self, features, patch_h, patch_w):
        """Fuse multi-layer features into a single spatial map.

        Args:
            features: list of tensors, each (B, N, embed_dim).
                      Training: all num_layers tensors.
                      Inference: only top-k tensors (caller pre-selects).
            patch_h, patch_w: spatial dimensions to reshape to.

        Returns:
            fused: (B, embed_dim, patch_h, patch_w)
        """
        B = features[0].shape[0]
        C = features[0].shape[2]

        if self.training:
            assert len(features) == self.num_layers, \
                f"Training expects {self.num_layers} features, got {len(features)}"
            weights = F.softmax(self.layer_weights, dim=0)
            fused = torch.zeros(B, C, patch_h, patch_w, device=features[0].device)
            for l in range(self.num_layers):
                f = features[l].permute(0, 2, 1).reshape(B, C, patch_h, patch_w)
                f_proj = self.layer_projects[l](f)
                fused = fused + weights[l] * f_proj
            return fused
        else:
            assert len(features) == self.k, \
                f"Eval expects {self.k} features (top-k), got {len(features)}"
            topk_indices = self.layer_weights.topk(self.k).indices
            topk_weights = F.softmax(self.layer_weights[topk_indices], dim=0)
            fused = torch.zeros(B, C, patch_h, patch_w, device=features[0].device)
            for i, (feat, global_idx) in enumerate(zip(features, topk_indices)):
                f = feat.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)
                f_proj = self.layer_projects[global_idx](f)
                fused = fused + topk_weights[i] * f_proj
            return fused

    def get_topk_indices(self):
        """Return indices of the top-k layers by learned weight."""
        return self.layer_weights.topk(self.k).indices

    def get_weights(self):
        """Return softmax-normalized layer weights (detached, on CPU)."""
        return F.softmax(self.layer_weights, dim=0).detach().cpu()
