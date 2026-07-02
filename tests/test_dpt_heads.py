import torch
import pytest
from dinosam.model.dpt_heads import DPTSimpleHead, DPTHead


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


# --- DPTHead tests ---

def test_dpt_head_backward_compat_state_dict():
    """DPTHead(n_layers=4) state_dict keys match DPTSimpleHead exactly."""
    simple = DPTSimpleHead(embed_dim=768, features=256)
    dpt = DPTHead(embed_dim=768, features=256, n_layers=4)

    simple_keys = set(simple.state_dict().keys())
    dpt_keys = set(dpt.state_dict().keys())

    assert simple_keys == dpt_keys, \
        f"Key mismatch.\nOnly in simple: {simple_keys - dpt_keys}\nOnly in dpt: {dpt_keys - simple_keys}"


def test_dpt_head_backward_compat_forward():
    """DPTHead(n_layers=4) produces identical output to DPTSimpleHead with same weights."""
    embed_dim, features = 768, 256
    simple = DPTSimpleHead(embed_dim=embed_dim, features=features)
    dpt = DPTHead(embed_dim=embed_dim, features=features, n_layers=4)

    # Copy weights from simple → dpt (keys match per test above)
    dpt.load_state_dict(simple.state_dict())

    B, N = 2, 4096
    patch_h = patch_w = 64
    feats = [torch.randn(B, N, embed_dim) for _ in range(4)]

    with torch.no_grad():
        out_simple = simple(feats, patch_h, patch_w)
        out_dpt = dpt(feats, patch_h, patch_w)

    assert torch.allclose(out_simple, out_dpt, atol=1e-6), \
        f"Max diff: {(out_simple - out_dpt).abs().max().item()}"


def test_dpt_head_3_layers():
    """DPTHead works with n_layers=3, output shape uses features*3 concat."""
    embed_dim, features, n_layers = 768, 256, 3
    head = DPTHead(embed_dim=embed_dim, features=features, n_layers=n_layers)

    B, N = 1, 4096
    patch_h = patch_w = 64
    feats = [torch.randn(B, N, embed_dim) for _ in range(n_layers)]

    out = head(feats, patch_h, patch_w)
    assert out.shape == (B, embed_dim, patch_h, patch_w)

    # Verify parameter count reflects 3 layers
    # projects: 3 × (768×256 + 256 bias) = 590,592
    # refine_convs: 3 × (256×256×9, no bias) = 1,769,472
    # output_conv: (256×3)×768 + 768 bias = 590,592
    total = sum(p.numel() for p in head.parameters())
    proj = 3 * (768 * 256 + 256)       # weight + bias
    refine = 3 * (256 * 256 * 9)       # no bias
    out = (256 * 3) * 768 + 768        # weight + bias
    expected = proj + refine + out
    assert total == expected, f"Expected {expected}, got {total}"


def test_dpt_head_12_layers():
    """DPTHead works with n_layers=12."""
    embed_dim, features, n_layers = 768, 256, 12
    head = DPTHead(embed_dim=embed_dim, features=features, n_layers=n_layers)

    B, N = 1, 4096
    patch_h = patch_w = 64
    feats = [torch.randn(B, N, embed_dim) for _ in range(n_layers)]

    out = head(feats, patch_h, patch_w)
    assert out.shape == (B, embed_dim, patch_h, patch_w)


def test_dpt_head_return_fused():
    """DPTHead return_fused=True returns both output and pre-output fused tensor."""
    head = DPTHead(embed_dim=768, features=256, n_layers=4)

    B, N = 1, 4096
    patch_h = patch_w = 64
    feats = [torch.randn(B, N, 768) for _ in range(4)]

    out, fused = head(feats, patch_h, patch_w, return_fused=True)

    assert out.shape == (B, 768, patch_h, patch_w)
    assert fused.shape == (B, 256 * 4, patch_h, patch_w)

    # Verify fused is the pre-output_conv tensor
    with torch.no_grad():
        out_from_fused = head.output_conv(fused)
    assert torch.allclose(out, out_from_fused, atol=1e-6)


def test_dpt_head_wrong_feature_count_raises():
    """DPTHead raises AssertionError when len(features) != n_layers."""
    head = DPTHead(embed_dim=768, features=256, n_layers=4)

    B, N = 1, 4096
    patch_h = patch_w = 64
    # Pass 3 features instead of expected 4
    feats = [torch.randn(B, N, 768) for _ in range(3)]

    with pytest.raises(AssertionError, match="Expected 4 features, got 3"):
        head(feats, patch_h, patch_w)
