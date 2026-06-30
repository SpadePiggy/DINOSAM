import torch
import pytest
from dinosam.model.dpt_heads import DPTSimpleHead


def test_dpt_simple_head_shape():
    """DPTSimpleHead outputs (B, embed_dim, patch_h, patch_w) from 4 intermediate features."""
    head = DPTSimpleHead(embed_dim=768, features=256)

    # Mock 4 intermediate features: each (B, N, embed_dim)
    B, N, embed_dim = 2, 4096, 768
    patch_h = patch_w = 64
    features = [torch.randn(B, N, embed_dim) for _ in range(4)]

    output = head(features, patch_h, patch_w)

    assert output.shape == (B, embed_dim, patch_h, patch_w), \
        f"Expected {(B, embed_dim, patch_h, patch_w)}, got {output.shape}"


def test_dpt_simple_head_different_batch_sizes():
    """DPTSimpleHead works with batch_size=1 and batch_size=4."""
    head = DPTSimpleHead(embed_dim=768, features=256)

    for B in [1, 4]:
        features = [torch.randn(B, 4096, 768) for _ in range(4)]
        output = head(features, 64, 64)
        assert output.shape == (B, 768, 64, 64)


def test_dpt_simple_head_parameter_count():
    """DPTSimpleHead has ~3.9M parameters (projects + refine_convs + output_conv)."""
    head = DPTSimpleHead(embed_dim=768, features=256)
    total_params = sum(p.numel() for p in head.parameters())

    # projects: 4 × (768 × 256 × 1) = 786,432
    # refine_convs: 4 × (256 × 256 × 3 × 3) = 2,359,296
    # output_conv: 1024 × 768 × 1 = 786,432
    # Total: ~3.9M
    assert 3_500_000 < total_params < 4_500_000, \
        f"Expected ~3.9M params, got {total_params:,}"
