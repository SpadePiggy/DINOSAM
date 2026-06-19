import torch
import torch.nn as nn
import torch.nn.functional as F


class FCAdapter(nn.Module):
    """Pure fully-connected bottleneck MLP adapter.

    Architecture: LayerNorm*gamma + identity*gammax → project1(dim→bottleneck) →
    GELU → dropout → project2(bottleneck→dim) → residual

    No spatial convolutions; hw_shapes is accepted for interface compatibility
    but not used since FC layers operate on the channel dimension only.
    """

    def __init__(self, in_dim, factor=8):
        super().__init__()
        bottleneck = in_dim // factor

        self.norm = nn.LayerNorm(in_dim)
        self.project1 = nn.Linear(in_dim, bottleneck)
        self.project2 = nn.Linear(bottleneck, in_dim)

        self.dropout = nn.Dropout(0.1)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):
        identity = x

        x = self.norm(x) * self.gamma + x * self.gammax

        x = self.project1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.project2(x)

        return identity + x
