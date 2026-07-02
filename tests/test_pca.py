import numpy as np
import pytest


def test_pca_visualize_no_mask():
    from dinosam.viz.pca import pca_visualize

    B, C, H, W = 2, 64, 8, 8
    features = np.random.randn(B, C, H, W).astype(np.float32)
    result = pca_visualize(features)

    assert result.shape == (B, H, W, 3)
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


def test_pca_visualize_with_mask():
    from dinosam.viz.pca import pca_visualize

    B, C, H, W = 1, 64, 8, 8
    features = np.random.randn(B, C, H, W).astype(np.float32)
    mask = np.zeros((B, 1, H, W), dtype=np.float32)
    # foreground is top half only
    mask[:, :, :4, :] = 1.0

    result = pca_visualize(features, mask=mask)

    assert result.shape == (B, H, W, 3)
    assert result.dtype == np.uint8
    # background (bottom half) must be black
    assert np.all(result[:, 4:, :, :] == 0)
    # foreground should have some non-zero pixels
    assert result[:, :4, :, :].sum() > 0


def test_pca_visualize_empty_mask():
    from dinosam.viz.pca import pca_visualize

    B, C, H, W = 1, 64, 8, 8
    features = np.random.randn(B, C, H, W).astype(np.float32)
    mask = np.zeros((B, 1, H, W), dtype=np.float32)  # all background

    result = pca_visualize(features, mask=mask)

    assert result.shape == (B, H, W, 3)
    assert result.dtype == np.uint8
    # fallback to full-image fit: output should not be all-black
    assert result.sum() > 0
