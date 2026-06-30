# DPT Multi-Level Fusion — End-to-End Smoke Test

**Date:** 2026-06-30
**Branch:** DINO (本地已实现 Task 2-6，需拉取最新代码)
**前置条件：** GPU 机器、DINOv3 checkpoint、SAM checkpoint、训练/测试数据

---

## 0. 准备工作

```bash
cd /home/LXT/lxt/DINOSAM   # 或你的项目路径
git pull origin DINO
```

确认新增/修改的文件已就位：

```bash
ls dinosam/model/dpt_heads.py          # 新增
git log --oneline -5                   # 确认最近 5 个 commit
```

---

## 1. 验证 DINOv3 `get_intermediate_layers` API（Task 1，之前跳过）

> **目标：** 确认 backbone 返回 `tuple[Tensor]`，每个 `(B, 4096, 768)`

```bash
python3 -c "
import sys
sys.path.insert(0, '/home/LXT/lxt/dinov3')
from dinov3.models.vision_transformer import DinoVisionTransformer
import torch

backbone = DinoVisionTransformer(
    img_size=1024, patch_size=16, in_chans=3, embed_dim=768, depth=12,
    num_heads=12, ffn_ratio=4, qkv_bias=True, drop_path_rate=0.0,
    layerscale_init=1e-5, norm_layer='layernormbf16', ffn_layer='mlp',
    ffn_bias=True, proj_bias=True, n_storage_tokens=4, mask_k_bias=True,
    pos_embed_rope_base=100, pos_embed_rope_normalize_coords='separate',
    pos_embed_rope_rescale_coords=2, pos_embed_rope_dtype='fp32',
)

x = torch.randn(1, 3, 1024, 1024)
features = backbone.get_intermediate_layers(x, n=[2, 5, 8, 11])

print(f'Type: {type(features)}')
print(f'Length: {len(features)}')
for i, f in enumerate(features):
    print(f'Layer {i}: shape={f.shape}')

# 关键断言
assert isinstance(features, tuple), f'Expected tuple, got {type(features)}'
assert len(features) == 4, f'Expected 4 features, got {len(features)}'
for i, f in enumerate(features):
    assert f.shape == (1, 4096, 768), f'Layer {i}: expected (1, 4096, 768), got {f.shape}'

print('SUCCESS: API verified')
"
```

### 预期输出

```
Type: <class 'tuple'>
Length: 4
Layer 0: shape=torch.Size([1, 4096, 768])
Layer 1: shape=torch.Size([1, 4096, 768])
Layer 2: shape=torch.Size([1, 4096, 768])
Layer 3: shape=torch.Size([1, 4096, 768])
SUCCESS: API verified
```

### 如果失败

| 错误 | 原因 | 修复方案 |
|------|------|----------|
| `AttributeError: no attribute 'get_intermediate_layers'` | 方法名不同 | `grep -rn "get_intermediate_layers" ~/lxt/dinov3/` 查找正确名称 |
| shape 为 `(1, 4100, 768)` | backbone 包含 4 个 storage tokens | 需要在 `dinov3_encoder.py` forward 中加切片：`features = [f[:, :4096, :] for f in features]`，然后修复代码 |
| shape 为 `(1, N, 768)` 但 N ≠ 4096 | patch 计算有误 | 检查 img_size 和 patch_size 配置 |

---

## 2. 单元测试

```bash
cd /home/LXT/lxt/DINOSAM
PYTHONPATH=. python -m pytest tests/test_dpt_heads.py -v
```

### 预期输出

```
tests/test_dpt_heads.py::test_dpt_simple_head_shape PASSED
tests/test_dpt_heads.py::test_dpt_simple_head_different_batch_sizes PASSED
tests/test_dpt_heads.py::test_dpt_simple_head_parameter_count PASSED
```

---

## 3. DPT 训练 — 1 epoch smoke test

> **目标：** 验证 `--adapter_type dpt_simple` 能正常训练，loss 下降

```bash
python -m dinosam.train.trainer \
    --sam_checkpoint /data3/LXT/data/checkpoint/sam_vit_b_01ec64.pth \
    --dinov3_checkpoint /data3/LXT/DINOv3-SAM/dinov3_vitb16.pth \
    --encoder dinov3 \
    --adapter_type dpt_simple \
    --dpt_layers "2,5,8,11" \
    --train_dir /data3/LXT/DINOv3-SAM/1724_train \
    --val_dir /data3/LXT/DINOv3-SAM/1724_test \
    --output_dir /tmp/dpt_smoke_test \
    --log_dir /tmp/dpt_smoke_test_logs \
    --train_phase 1 \
    --epochs 1 \
    --batch_size 2 \
    --log_interval 5
```

### 预期

- 训练启动无报错
- loss 在 epoch 内下降
- checkpoint 保存到 `/tmp/dpt_smoke_test/phase1_best.pt`
- 内存使用比 mona baseline 高约 2-3GB（batch_size=2 不应 OOM）

### 验证 checkpoint 内容

```bash
python3 -c "
import torch
ckpt = torch.load('/tmp/dpt_smoke_test/phase1_best.pt', map_location='cpu')
print(f'adapter_type: {ckpt.get(\"adapter_type\", \"MISSING\")}')

keys = list(ckpt['model_state_dict'].keys())
dpt_keys = [k for k in keys if 'dpt_head' in k]
print(f'DPT head keys ({len(dpt_keys)}):')
for k in dpt_keys[:5]:
    print(f'  {k}: {ckpt[\"model_state_dict\"][k].shape}')
print(f'  ...')

# 关键检查
assert ckpt.get('adapter_type') == 'dpt_simple', f'Expected dpt_simple, got {ckpt.get(\"adapter_type\")}'
assert len(dpt_keys) > 0, 'No dpt_head keys found in checkpoint'
print('CHECKPOINT VERIFIED')
"
```

预期：`adapter_type: dpt_simple`，dpt_head keys 存在且 shape 正确。

---

## 4. Mona 回归测试 — 确认默认行为不变

> **目标：** 验证 `--adapter_type mona`（默认）行为不受 DPT 改动影响

```bash
python -m dinosam.train.trainer \
    --sam_checkpoint /data3/LXT/data/checkpoint/sam_vit_b_01ec64.pth \
    --dinov3_checkpoint /data3/LXT/DINOv3-SAM/dinov3_vitb16.pth \
    --encoder dinov3 \
    --adapter_type mona \
    --train_dir /data3/LXT/DINOv3-SAM/1724_train \
    --val_dir /data3/LXT/DINOv3-SAM/1724_test \
    --output_dir /tmp/mona_regression_test \
    --log_dir /tmp/mona_regression_test_logs \
    --train_phase 1 \
    --epochs 1 \
    --batch_size 4 \
    --log_interval 5
```

### 预期

- 训练完成无报错
- 无新的 warning 或 error
- loss 曲线与 DPT 改动前相似

---

## 5. 跨 adapter_type checkpoint 加载

> **目标：** 验证 mona checkpoint 加载到 dpt_simple 模型时 `strict=False` + warning 正常工作

```bash
python -m dinosam.train.trainer \
    --sam_checkpoint /data3/LXT/data/checkpoint/sam_vit_b_01ec64.pth \
    --dinov3_checkpoint /data3/LXT/DINOv3-SAM/dinov3_vitb16.pth \
    --encoder dinov3 \
    --adapter_type dpt_simple \
    --train_dir /data3/LXT/DINOv3-SAM/1724_train \
    --val_dir /data3/LXT/DINOv3-SAM/1724_test \
    --output_dir /tmp/cross_adapter_test \
    --log_dir /tmp/cross_adapter_test_logs \
    --train_phase 2 \
    --epochs 1 \
    --batch_size 2 \
    --phase1_checkpoint /tmp/mona_regression_test/phase1_best.pt
```

### 预期

- 打印 warning：`Warning: Phase 1 checkpoint adapter_type='mona' differs from current 'dpt_simple'...`
- 打印 missing keys：`Missing keys: ['dinov3_encoder.dpt_head.projects.0.weight', ...]`
- **不 crash**，训练继续
- DPT head 权重随机初始化

---

## 6. 评估脚本验证

> **目标：** 验证 `evaluate.py` 支持 `--adapter_type dpt_simple`

使用 Step 3 产生的 DPT checkpoint 进行评估（Phase 1 only）：

```bash
python -m dinosam.evaluate \
    --sam_checkpoint /data3/LXT/data/checkpoint/sam_vit_b_01ec64.pth \
    --dinov3_checkpoint /data3/LXT/DINOv3-SAM/dinov3_vitb16.pth \
    --encoder dinov3 \
    --adapter_type dpt_simple \
    --test_dir /data3/LXT/DINOv3-SAM/1724_test \
    --phase1_checkpoint /tmp/dpt_smoke_test/phase1_best.pt \
    --phase2_checkpoint /tmp/dpt_smoke_test/phase2_best.pt \
    --batch_size 2
```

### 预期

- 评估完成无报错
- 打印指标（Dice, IoU, Depth MAE 等）
- 无 missing/unexpected key warnings（同 adapter_type）

---

## 7. 结果汇总

| # | 测试项 | 通过？ | 备注 |
|---|--------|--------|------|
| 1 | `get_intermediate_layers` API | ☐ | shape=(1, 4096, 768)? |
| 2 | DPTSimpleHead 单元测试 | ☐ | 3/3 PASSED? |
| 3 | DPT 训练 smoke test | ☐ | loss 下降？OOM？ |
| 4 | Mona 回归测试 | ☐ | 行为不变？ |
| 5 | 跨 adapter checkpoint 加载 | ☐ | warning 正确？不 crash？ |
| 6 | evaluate.py 评估 | ☐ | 指标输出？ |

---

## 快速执行（一键）

如果所有路径正确，可以一次性跑完：

```bash
cd /home/LXT/lxt/DINOSAM

# API 验证
python3 -c "
import sys; sys.path.insert(0, '/home/LXT/lxt/dinov3')
from dinov3.models.vision_transformer import DinoVisionTransformer
import torch
bb = DinoVisionTransformer(img_size=1024, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, ffn_ratio=4, qkv_bias=True, drop_path_rate=0.0, layerscale_init=1e-5, norm_layer='layernormbf16', ffn_layer='mlp', ffn_bias=True, proj_bias=True, n_storage_tokens=4, mask_k_bias=True, pos_embed_rope_base=100, pos_embed_rope_normalize_coords='separate', pos_embed_rope_rescale_coords=2, pos_embed_rope_dtype='fp32')
f = bb.get_intermediate_layers(torch.randn(1,3,1024,1024), n=[2,5,8,11])
for i,t in enumerate(f): print(f'Layer {i}: {t.shape}')
assert all(t.shape==(1,4096,768) for t in f); print('API OK')
" &&

# 单元测试
PYTHONPATH=. python -m pytest tests/test_dpt_heads.py -v &&

echo "=== Steps 1-2 passed ==="
```

Steps 3-6 需要分别执行（每个耗时较长）。
