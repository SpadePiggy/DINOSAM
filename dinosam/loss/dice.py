import torch


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dice loss for binary segmentation.

    Args:
        pred: (B, 1, H, W) predicted logits (before sigmoid)
        target: (B, 1, H, W) binary ground truth
    """
    pred_prob = torch.sigmoid(pred)
    intersection = (pred_prob * target).sum(dim=(2, 3))
    union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def compute_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred_mask = (torch.sigmoid(pred) > 0.5).float()
    overlap = (pred_mask * target).sum(dim=(2, 3))
    union = ((pred_mask + target) > 0).float().sum(dim=(2, 3))
    return overlap / (union + eps)
