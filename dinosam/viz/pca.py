import numpy as np


def pca_visualize(
    features: np.ndarray,
    mask: np.ndarray = None,
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
