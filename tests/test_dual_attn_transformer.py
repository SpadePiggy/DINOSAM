"""dual_attn_transformer 的 shape 验证测试。

使用 importlib.util 直接加载模块文件，完全绕过 dinosam/__init__.py。
"""

import sys
import os
import types
import importlib.util

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "segment-anything"))

import torch


def _load_module_from_file(module_name, file_path):
    """直接从文件加载模块，不触发包级 __init__.py。"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# 先注册必要的包命名空间（空壳），避免触发 __init__.py
for pkg in ["dinosam", "dinosam.model", "dinosam.model.adapters"]:
    if pkg not in sys.modules:
        sys.modules[pkg] = types.ModuleType(pkg)

# 按依赖顺序加载
_load_module_from_file(
    "dinosam.model.adapters.dual_attn_adapter",
    os.path.join(PROJECT_ROOT, "dinosam", "model", "adapters", "dual_attn_adapter.py"),
)
mod = _load_module_from_file(
    "dinosam.model.dual_attn_transformer",
    os.path.join(PROJECT_ROOT, "dinosam", "model", "dual_attn_transformer.py"),
)

DualAttnTransformerLayer = mod.DualAttnTransformerLayer
DualAttnTransformer = mod.DualAttnTransformer


def test_layer_shape():
    """验证 DualAttnTransformerLayer 输入输出 shape 一致。"""
    B, N, C = 2, 5, 256
    H, W = 16, 16
    HW = H * W

    layer = DualAttnTransformerLayer(
        embedding_dim=C, num_heads=8, mlp_dim=2048
    )

    queries = torch.randn(B, N, C)
    keys = torch.randn(B, HW, C)
    query_pe = torch.randn(B, N, C)
    key_pe = torch.randn(B, HW, C)

    out_q, out_k = layer(queries, keys, query_pe, key_pe, hw_shape=(H, W))

    assert out_q.shape == (B, N, C), f"queries shape 不匹配: {out_q.shape}"
    assert out_k.shape == (B, HW, C), f"keys shape 不匹配: {out_k.shape}"
    print("DualAttnTransformerLayer shape 测试通过")


def test_transformer_shape():
    """验证 DualAttnTransformer 输入输出 shape 一致。"""
    B, N, C = 2, 5, 256
    H, W = 16, 16
    HW = H * W

    model = DualAttnTransformer(
        embedding_dim=C, num_heads=8, mlp_dim=2048, depth=2
    )

    image_embedding = torch.randn(B, C, H, W)
    image_pe = torch.randn(B, C, H, W)
    point_embedding = torch.randn(B, N, C)

    out_q, out_k = model(image_embedding, image_pe, point_embedding)

    assert out_q.shape == (B, N, C), f"queries shape 不匹配: {out_q.shape}"
    assert out_k.shape == (B, HW, C), f"keys shape 不匹配: {out_k.shape}"
    print("DualAttnTransformer shape 测试通过")


if __name__ == "__main__":
    test_layer_shape()
    test_transformer_shape()
    print("所有测试通过")
