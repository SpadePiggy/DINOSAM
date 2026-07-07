import torch
import pytest
from dinosam.model.layer_fusion import LearnedLayerFusion


def test_output_shape_training():
    """Training mode: all 12 layers in, (B, C, H, W) out."""
    fusion = LearnedLayerFusion(embed_dim=768, num_layers=12, k=4)
    fusion.train()
    features = [torch.randn(2, 4096, 768) for _ in range(12)]
    fused = fusion(features, 64, 64)
    assert fused.shape == (2, 768, 64, 64)


def test_output_shape_eval():
    """Eval mode: only top-k features in, same shape out."""
    fusion = LearnedLayerFusion(embed_dim=768, num_layers=12, k=4)
    with torch.no_grad():
        fusion.layer_weights[0] = 10.0
        fusion.layer_weights[3] = 9.0
        fusion.layer_weights[7] = 8.0
        fusion.layer_weights[11] = 7.0
    fusion.eval()
    features = [torch.randn(2, 4096, 768) for _ in range(4)]
    fused = fusion(features, 64, 64)
    assert fused.shape == (2, 768, 64, 64)


def test_zero_init_uniform():
    """Zero-init weights → softmax = uniform 1/12 each."""
    fusion = LearnedLayerFusion(embed_dim=768, num_layers=12, k=4)
    weights = fusion.get_weights()
    expected = torch.ones(12) / 12
    assert torch.allclose(weights, expected, atol=1e-6)


def test_get_topk_indices():
    """get_topk_indices returns the k layers with highest weights."""
    fusion = LearnedLayerFusion(embed_dim=768, num_layers=12, k=4)
    with torch.no_grad():
        fusion.layer_weights[0] = 10.0
        fusion.layer_weights[3] = 9.0
        fusion.layer_weights[7] = 8.0
        fusion.layer_weights[11] = 7.0
    topk = fusion.get_topk_indices().tolist()
    assert set(topk) == {0, 3, 7, 11}


def test_gradient_flows_to_weights():
    """Task loss gradient reaches layer_weights parameter."""
    fusion = LearnedLayerFusion(embed_dim=768, num_layers=12, k=4)
    fusion.train()
    features = [torch.randn(2, 4096, 768) for _ in range(12)]
    fused = fusion(features, 64, 64)
    loss = fused.sum()
    loss.backward()
    assert fusion.layer_weights.grad is not None
    assert fusion.layer_weights.grad.abs().sum() > 0


def test_gradient_flows_to_all_projections():
    """Every layer projection receives gradient during training."""
    fusion = LearnedLayerFusion(embed_dim=768, num_layers=12, k=4)
    fusion.train()
    features = [torch.randn(2, 4096, 768) for _ in range(12)]
    fused = fusion(features, 64, 64)
    loss = fused.sum()
    loss.backward()
    for l, proj in enumerate(fusion.layer_projects):
        conv = proj[0]  # Conv2d is first in Sequential
        assert conv.weight.grad is not None, f"Layer {l} projection received no gradient"
        assert conv.weight.grad.abs().sum() > 0, f"Layer {l} projection gradient is zero"


def test_training_assert_wrong_feature_count():
    """Training raises AssertionError if feature count != num_layers."""
    fusion = LearnedLayerFusion(embed_dim=768, num_layers=12, k=4)
    fusion.train()
    features = [torch.randn(2, 4096, 768) for _ in range(6)]  # wrong count
    with pytest.raises(AssertionError, match="Training expects 12"):
        fusion(features, 64, 64)


def test_parameter_count():
    """Verify parameter count: 12 weights + 12 × (Conv1x1 + GroupNorm)."""
    fusion = LearnedLayerFusion(embed_dim=768, num_layers=12, k=4)
    # layer_weights: 12
    # per projection: Conv2d(768,768,1) = 768*768+768 = 590592
    #                 GroupNorm(32,768) = 2*768 = 1536
    #                 per layer = 592128, × 12 = 7105536
    # total = 7105548
    total = sum(p.numel() for p in fusion.parameters())
    assert total == 7105548


def test_batch_size_one():
    """GroupNorm must work with B=1 (InstanceNorm would NaN)."""
    fusion = LearnedLayerFusion(embed_dim=768, num_layers=12, k=4)
    fusion.train()
    features = [torch.randn(1, 4096, 768) for _ in range(12)]
    fused = fusion(features, 64, 64)
    assert fused.shape == (1, 768, 64, 64)
    assert not torch.isnan(fused).any()
