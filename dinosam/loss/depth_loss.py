import torch
from torch.nn import functional as F


def depth_mse_loss(depth_pred: torch.Tensor, depth_labels: torch.Tensor) -> torch.Tensor:
    """MSE loss for depth regression."""
    return F.mse_loss(depth_pred, depth_labels)
