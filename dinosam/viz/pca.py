import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def pca_visualize(
    features: torch.Tensor,
    mask: torch.Tensor = None,
    n_components: int = 3,
    sigmoid_scale: float = 2.0,
) -> np.ndarray:
    """PCA visualization of feature maps.

    Args:
        features: (B, C, H, W) feature map.
        mask: Optional (B, 1, H, W) binary mask. PCA is fit on foreground only;
              falls back to all patches when no foreground exists.
        n_components: Number of PCA components (must be 3 for RGB output).
        sigmoid_scale: Multiplier before sigmoid for vibrant colors.

    Returns:
        (B, H, W, 3) uint8 RGB visualization.
    """
    # ponytail: lazy import — sklearn is optional
    from sklearn.decomposition import PCA

    # Convert to numpy if torch tensors
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    B, C, H, W = features.shape
    results = np.zeros((B, H, W, 3), dtype=np.uint8)

    for b in range(B):
        # (H*W, C)
        flat = features[b].reshape(C, -1).T

        # Determine fitting data and whether masking applies
        has_foreground = False
        if mask is not None:
            fg = mask[b, 0].reshape(-1) > 0.5
            has_foreground = fg.any()
            fit_data = flat[fg] if has_foreground else flat
        else:
            fit_data = flat

        pca = PCA(n_components=n_components, whiten=True)
        pca.fit(fit_data)

        projected = pca.transform(flat).reshape(H, W, 3)

        # Sigmoid scaling for vibrant colors
        projected = 1.0 / (1.0 + np.exp(-projected * sigmoid_scale))

        # Scale to uint8
        img = (projected * 255).astype(np.uint8)

        # Zero out background only when mask had foreground
        if mask is not None and has_foreground:
            bg = (mask[b, 0] <= 0.5)
            img[bg] = 0

        results[b] = img

    return results


@torch.no_grad()
def visualize_model_features(
    model,
    images: torch.Tensor,
    device: torch.device,
    save_dir: str,
    prefix: str = "",
    mask: torch.Tensor = None,
):
    """Generate PCA visualizations for backbone layers + fused features.

    Only works with DINOv3 encoder path (encoder_type == "dinov3") and
    requires dpt_head (adapter_type == "dpt_simple").

    Args:
        model: DepthSam instance.
        images: (B, 3, H, W) raw [0, 255] range.
        device: Torch device.
        save_dir: Directory to save PNG files.
        prefix: Filename prefix for saved PNGs.
        mask: Optional (B, 1, H, W) binary mask passed to pca_visualize.
    """
    # Guards
    if model.encoder_type != "dinov3" or model.dinov3_encoder is None:
        print("visualize_model_features: only works with dinov3 encoder, skipping")
        return

    encoder = model.dinov3_encoder
    if encoder.dpt_head is None:
        print("visualize_model_features: dpt_head is None (non-dpt_simple adapter), skipping")
        return

    os.makedirs(save_dir, exist_ok=True)

    # Preprocessing — inline copy of DINOv3MonaEncoder.forward() preprocessing
    x = F.interpolate(
        images.to(device), size=(encoder.img_size, encoder.img_size),
        mode="bilinear", align_corners=False,
    )
    x = (x - encoder.pixel_mean) / encoder.pixel_std

    patch_h = patch_w = encoder.img_size // 16

    # Get intermediate features — same call as training path
    raw_features = encoder.backbone.get_intermediate_layers(x, n=encoder.dpt_layers)

    # Per-layer PCA
    for i, feat in enumerate(raw_features):
        # feat: (B, N, C) → (B, C, patch_h, patch_w)
        B, N, C = feat.shape
        feat_2d = feat.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)
        vis = pca_visualize(feat_2d, mask=mask)
        for b in range(B):
            fname = f"{prefix}layer{i}_b{b}.png" if prefix else f"layer{i}_b{b}.png"
            Image.fromarray(vis[b]).save(os.path.join(save_dir, fname))

    # Fused PCA
    _, fused = encoder.dpt_head(raw_features, patch_h, patch_w, return_fused=True)
    vis_fused = pca_visualize(fused, mask=mask)
    for b in range(B):
        fname = f"{prefix}fused_b{b}.png" if prefix else f"fused_b{b}.png"
        Image.fromarray(vis_fused[b]).save(os.path.join(save_dir, fname))
