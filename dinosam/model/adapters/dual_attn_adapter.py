import torch
import torch.nn as nn


class PAM_Module(nn.Module):
    """Position Attention Module — no internal residual.

    Based on DANet PAM_Module, but the internal residual (``+ x``) is removed.
    The global residual is added once in DualAttnAdapter.forward().
    """

    def __init__(self, in_dim, factor=8):
        super().__init__()
        reduction = in_dim // factor
        self.query_conv = nn.Conv2d(in_dim, reduction, 1)
        self.key_conv = nn.Conv2d(in_dim, reduction, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        query = self.query_conv(x).view(B, -1, N).permute(0, 2, 1)  # (B, N, C//factor)
        key = self.key_conv(x).view(B, -1, N)                         # (B, C//factor, N)
        attn = torch.softmax(torch.bmm(query, key), dim=-1)           # (B, N, N)
        value = self.value_conv(x).view(B, -1, N)                     # (B, C, N)
        out = torch.bmm(value, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out  # no residual here


class CAM_Module(nn.Module):
    """Channel Attention Module — no internal residual.

    Matches the official DANet CAM formula exactly:
    ``energy_new = max(energy, -1)[0].expand_as(energy) - energy``
    then softmax over the subtracted energy.
    The internal residual (``+ x`` from DANet) is removed.
    """

    def __init__(self, in_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        query = x.view(B, C, -1)                                       # (B, C, N)
        key = x.view(B, C, -1).permute(0, 2, 1)                        # (B, N, C)
        energy = torch.bmm(query, key)                                  # (B, C, C)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attn = torch.softmax(energy_new, dim=-1)
        out = torch.bmm(attn, query).view(B, C, H, W)
        return self.gamma * out  # no residual here


class DualAttnAdapter(nn.Module):
    """Dual Attention Adapter: PAM + CAM in parallel, summed and averaged.

    Takes (B, N, C) sequence features, internally reshapes to (B, C, H, W)
    using ``hw_shapes``, runs PAM and CAM, then adds a single global residual.
    """

    def __init__(self, in_dim, factor=8):
        super().__init__()
        self.pam = PAM_Module(in_dim, factor)
        self.cam = CAM_Module(in_dim)

    def forward(self, x, hw_shapes=None):
        if hw_shapes is None:
            raise ValueError(
                "DualAttnAdapter requires hw_shapes for 2D reshape. "
                "Got hw_shapes=None."
            )
        B, N, C = x.shape
        H, W = hw_shapes
        identity = x
        x_2d = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        attn_out = (self.pam(x_2d) + self.cam(x_2d)) / 2.0
        out = attn_out.permute(0, 2, 3, 1).reshape(B, N, C)
        return identity + out  # single global residual
