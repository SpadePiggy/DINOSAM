import torch
import torch.nn as nn
import torch.nn.functional as F


class MonaOp(nn.Module):
    """Multi-scale depthwise convolution adapter.

    Applies 3×3, 5×5, 7×7 depthwise convs averaged + identity, then 1×1 projector.
    """

    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=3, groups=in_features)
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1)

    def forward(self, x):
        identity = x
        x = (self.conv1(x) + self.conv2(x) + self.conv3(x)) / 3.0 + identity
        return identity + self.projector(x)


class Mona(nn.Module):
    """Lightweight adapter with spatial conv (MonaOp) and bottleneck MLP.

    Architecture: LayerNorm*scale + identity*scale → project1(dim→bottleneck) →
    MonaOp(spatial) → GELU → dropout → project2(bottleneck→dim) → residual
    """

    def __init__(self, in_dim, factor=8):
        super().__init__()
        bottleneck = in_dim // factor

        self.project1 = nn.Linear(in_dim, bottleneck)
        self.adapter_conv = MonaOp(bottleneck)
        self.project2 = nn.Linear(bottleneck, in_dim)

        self.dropout = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):
        identity = x

        x = self.norm(x) * self.gamma + x * self.gammax

        x = self.project1(x)

        if hw_shapes is not None:
            b, n, c = x.shape
            h, w = hw_shapes
            x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
            x = self.adapter_conv(x)
            x = x.permute(0, 2, 3, 1).reshape(b, n, c)
        else:
            x = self.adapter_conv(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)

        x = F.gelu(x)
        x = self.dropout(x)
        x = self.project2(x)

        return identity + x
