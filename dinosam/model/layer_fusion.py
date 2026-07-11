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

        # Learnable layer weights.
        # Truncated-normal init (std=0.1, clip ±0.2) gives ~4× more initial
        # spread while preventing extreme values from dominating softmax.
        self.layer_weights = nn.Parameter(torch.empty(num_layers))
        nn.init.trunc_normal_(self.layer_weights, mean=0.0, std=0.1, a=-0.2, b=0.2)

        # Per-layer projection: Conv1x1 + GELU.
        # No GroupNorm — it forces all 12 layers to same output scale, killing
        # the gradient signal needed for layer_weights to differentiate.
        #
        # Symmetric init: first Conv1x1 is kaiming-init'd with a fixed seed,
        # then its weights are copied to the remaining 11.  This guarantees all
        # layers start from the same projection, so early weight differentiation
        # reflects genuine feature quality, not random init lottery.
        g = torch.Generator()
        g.manual_seed(42)
        first_conv = nn.Conv2d(embed_dim, embed_dim, 1)
        nn.init.kaiming_normal_(first_conv.weight, generator=g)
        nn.init.zeros_(first_conv.bias)

        self.layer_projects = nn.ModuleList([
            nn.Sequential(first_conv, nn.GELU())
        ])
        for _ in range(1, num_layers):
            conv = nn.Conv2d(embed_dim, embed_dim, 1)
            conv.weight.data.copy_(first_conv.weight.data)
            conv.bias.data.copy_(first_conv.bias.data)
            self.layer_projects.append(
                nn.Sequential(conv, nn.GELU())
            )

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

    def variance_loss(self):
        """Variance of softmax weights — used as auxiliary loss to penalize
        uniform distributions.

        Returns -Var(w) so the caller writes  loss += λ · variance_loss
        (i.e. adds a penalty for low variance / uniform weights).

        Unlike entropy, variance has first-order gradient near uniformity:
        ∂Var/∂w_i ∝ (w_i − 1/n), so it immediately amplifies any small
        asymmetry the task loss creates.
        """
        w = F.softmax(self.layer_weights, dim=0)
        return -(w.var())
