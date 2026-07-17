import pytest

import torch

from dinosam.model.dinov3_encoder import (
    split_union_features,
    has_depth_branch_keys,
    parse_layers_arg,
)


def _make_features(union):
    """每层一个可识别张量：全体元素值 = 层索引。"""
    return [torch.full((2, 16, 8), float(l)) for l in union]


def _layer_of(feat):
    return int(feat[0, 0, 0].item())


def test_split_overlapping_layers():
    """两组层有重叠：mask=[2,5,8,11], depth=[5,8,11]."""
    layers_a, layers_b = [2, 5, 8, 11], [5, 8, 11]
    union = sorted(set(layers_a) | set(layers_b))
    assert union == [2, 5, 8, 11]
    features = _make_features(union)

    fa, fb = split_union_features(features, union, layers_a, layers_b)

    assert [_layer_of(f) for f in fa] == layers_a
    assert [_layer_of(f) for f in fb] == layers_b


def test_split_disjoint_layers():
    """两组层无重叠：a=[0,3], b=[6,9]."""
    layers_a, layers_b = [0, 3], [6, 9]
    union = sorted(set(layers_a) | set(layers_b))
    assert union == [0, 3, 6, 9]
    features = _make_features(union)

    fa, fb = split_union_features(features, union, layers_a, layers_b)

    assert [_layer_of(f) for f in fa] == layers_a
    assert [_layer_of(f) for f in fb] == layers_b


def test_split_identical_layers():
    """两组层完全相同：分发结果一致且为同一批张量。"""
    layers = [2, 5, 8, 11]
    union = sorted(set(layers))
    features = _make_features(union)

    fa, fb = split_union_features(features, union, layers, layers)

    assert [_layer_of(f) for f in fa] == layers
    assert [_layer_of(f) for f in fb] == layers
    for a, b in zip(fa, fb):
        assert a is b  # 同一对象，不复制


def test_has_depth_branch_keys_hits_new_modules():
    """四个新模块的键全部命中。"""
    for mod in ("dpt_head_depth", "adapter1_depth", "adapter2_depth", "projection_depth"):
        assert has_depth_branch_keys([f"dinov3_encoder.{mod}.weight"])


def test_has_depth_branch_keys_no_false_positives():
    """既有含 'depth' 的键均不误伤。"""
    existing = [
        "simple_depth_head.2.weight",                     # 无 dinov3_encoder. 前缀
        "depth_decoder.depth_token.weight",               # 无前缀
        "depth_decoder.depth_prediction_head.0.weight",   # 无前缀
        "dinov3_encoder.dpt_head.projects.0.weight",      # 前缀命中但无 '_depth.'
        "dinov3_encoder.backbone.blocks.0.attn.qkv.weight",
        "dinov3_encoder.projection.weight",
    ]
    assert not has_depth_branch_keys(existing)
    assert not has_depth_branch_keys([])


def test_parse_layers_arg_normalizes():
    """乱序 + 重复 → 升序去重。"""
    assert parse_layers_arg("11, 8,5,2") == [2, 5, 8, 11]
    assert parse_layers_arg("5,5,8") == [5, 8]
    assert parse_layers_arg("7") == [7]
