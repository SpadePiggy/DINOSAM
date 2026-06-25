import torch
import torch.nn.functional as F


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


def boundary_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    boundary_width: int = 3,
    dt_weight_max: float = 5.0,
    dice_weight: float = 1.0,
    bce_weight: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Boundary-aware loss for reducing jagged edges in segmentation masks.

    Combines two terms:
    1. Boundary Dice Loss — Dice computed only on boundary pixels of the GT mask
    2. Distance-weighted BCE — higher penalty near mask boundaries

    Args:
        pred: (B, 1, H, W) predicted logits (before sigmoid)
        target: (B, 1, H, W) binary ground truth
        boundary_width: half-width of the morphological boundary band (pixels)
        dt_weight_max: max weight multiplier for pixels near the boundary in BCE term
        dice_weight: weight for the boundary Dice component
        bce_weight: weight for the distance-weighted BCE component
        eps: smoothing constant for Dice denominator
    """
    pred_prob = torch.sigmoid(pred)

    # Morphological erosion of GT: erode = min-pool = -max_pool(-target)
    # For binary masks this yields pixels whose full kxk neighbourhood is foreground
    k = 2 * boundary_width + 1
    eroded = -F.max_pool2d(-target, kernel_size=k, stride=1, padding=boundary_width)

    # Inner boundary band = mask minus its eroded core
    gt_boundary = target - eroded

    # --- Term 1: Boundary Dice Loss ---
    bd_pred = pred_prob * gt_boundary
    bd_target = target * gt_boundary
    intersection = (bd_pred * bd_target).sum(dim=(2, 3))
    union = bd_pred.sum(dim=(2, 3)) + bd_target.sum(dim=(2, 3))
    bd_dice = (2 * intersection + eps) / (union + eps)
    boundary_dice = 1 - bd_dice.mean()

    # --- Term 2: Distance-weighted BCE (high weight near boundary) ---
    B, _, H, W = target.shape
    from kornia.contrib import distance_transform
    # GPU batched distance transform — replaces scipy per-sample CPU loop
    dt_fg = distance_transform(target)             # distance to background (inside mask)
    dt_bg = distance_transform(1.0 - target)        # distance to foreground (outside mask)
    dt = dt_fg * target + dt_bg * (1.0 - target)   # combined signed distance

    # Normalize to [0, 1] per sample; weight = max_w * (1 - dt_norm) → high near boundary
    dt_flat = dt.view(B, -1)
    dt_max = dt_flat.max(dim=1, keepdim=True).values.clamp(min=1.0)
    dt_norm = (dt_flat / dt_max).view(B, 1, H, W)
    weights = 1.0 + (dt_weight_max - 1.0) * (1.0 - dt_norm)

    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    dt_bce = (bce * weights).mean()

    return dice_weight * boundary_dice + bce_weight * dt_bce


def compute_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred_mask = (torch.sigmoid(pred) > 0.5).float()
    overlap = (pred_mask * target).sum(dim=(2, 3))
    union = ((pred_mask + target) > 0).float().sum(dim=(2, 3))
    return overlap / (union + eps)
