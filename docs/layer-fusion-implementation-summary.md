# Layer Fusion 实现总结

> 日期：2026-07-08
> 分支：DINO
> 提交数：8

---

## 概述

为 DINOSAM 实现了 **Layer Fusion with Learned Global Weights** 特性，通过可学习的全局层权重 + top-k 推理，揭示 mask 和 depth 任务对 DINOv3 ViT-B/16 各层的重要性差异。

### 核心设计

- **12 个可学习标量权重**：每个 ViT 层一个权重，softmax 归一化后作为融合系数
- **训练时**：softmax over 全部 12 层，加权融合（全梯度流）
- **推理时**：只选择 top-k 层进行融合（稀疏高效）
- **双分支独立**：mask 和 depth 各一套 `LearnedLayerFusion`，Phase 1/2 分别训练
- **无辅助 loss**：权重通过任务 loss 直接反向传播学习

---

## 提交历史

| 提交 | 说明 |
|------|------|
| `3e0f7be` | feat: add LearnedLayerFusion module with unit tests |
| `1f19969` | fix: add eval-mode input validation to LearnedLayerFusion |
| `31177d4` | feat: integrate LearnedLayerFusion into DINOv3MonaEncoder |
| `4111994` | feat: integrate fusion branch into DepthSam |
| `71e6da8` | feat: add dpt_fusion CLI args and Phase 2 unfreeze in trainer |
| `b58b717` | feat: dual encode for fusion mode in training loops |
| `ee091e0` | feat: dual encode and fusion CLI args in evaluate.py |
| `8a3a1a9` | feat: add layer weight logging and analysis script |

---

## 修改文件清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `dinosam/model/layer_fusion.py` | `LearnedLayerFusion` 核心模块 |
| `tests/test_layer_fusion.py` | 9 个单元测试 |
| `scripts/analyze_layer_preferences.py` | 训练后层偏好分析脚本 |
| `docs/superpowers/specs/2026-07-07-layer-fusion-design.md` | 设计规格文档 (v2.3) |
| `docs/layer-fusion-related-papers.md` | 相关论文综述 |

### 修改文件

| 文件 | 修改内容 |
|------|---------|
| `dinosam/model/__init__.py` | 导出 `LearnedLayerFusion` |
| `dinosam/model/dinov3_encoder.py` | 新增 `dpt_fusion` 分支、`fusion_k` 参数、`branch` 参数 |
| `dinosam/model/depth_sam.py` | 新增 `fusion_k` 参数、`branch` 参数、freeze 方法更新、dual encode |
| `dinosam/train/trainer.py` | CLI 参数、模型构造、Phase 2 解冻、dual encode、层权重日志 |
| `dinosam/train/iterative.py` | `iterative_train_step` 和 `iterative_depth_step` 双编码 |
| `dinosam/evaluate.py` | CLI 参数、模型构造、dual encode |

---

## 架构概览

```
Image → DINOv3(frozen) → get_intermediate_layers([0..11]) → 12 层特征
                                                                ↓
                                                    LearnedLayerFusion
                                                    (mask_fusion 或 depth_fusion)
                                                    - softmax(layer_weights)
                                                    - Conv1x1 + GroupNorm + GELU per layer
                                                    - 加权融合
                                                                ↓
                                                    (B, 768, 64, 64) fused
                                                                ↓
                                                    adapter1 → adapter2 → projection
```

---

## 使用方法

### 训练

```bash
python dinosam/train/trainer.py \
    --encoder dinov3 \
    --dinov3_checkpoint /path/to/dinov3_vitb16.pth \
    --adapter_type dpt_fusion \
    --fusion_k 4 \
    --sam_checkpoint /path/to/sam_vit_b.pth \
    --train_dir /path/to/train \
    --val_dir /path/to/val \
    --output_dir outputs/fusion_k4
```

### 评估

```bash
python dinosam/evaluate.py \
    --encoder dinov3 \
    --dinov3_checkpoint /path/to/dinov3_vitb16.pth \
    --adapter_type dpt_fusion \
    --fusion_k 4 \
    --sam_checkpoint /path/to/sam_vit_b.pth \
    --phase1_checkpoint outputs/fusion_k4/phase1_best.pt \
    --phase2_checkpoint outputs/fusion_k4/phase2_best.pt \
    --test_dir /path/to/test
```

### 层偏好分析

```bash
python scripts/analyze_layer_preferences.py \
    --phase1_checkpoint outputs/fusion_k4/phase1_best.pt \
    --phase2_checkpoint outputs/fusion_k4/phase2_best.pt \
    --output_dir outputs/fusion_k4/analysis
```

输出：
- `layer_weights_comparison.png`：mask vs depth 权重柱状图
- `layer_weights_overlay.png`：叠加对比图
- `statistics.txt`：top-4 层、重叠度、权重数值

---

## 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 权重类型 | 全局 `nn.Parameter` | 直接可解释，12 个标量 |
| 训练策略 | softmax over all layers | 全梯度流，训练稳定 |
| 推理策略 | hard top-k | 只计算 k 层，推理高效 |
| 归一化 | GroupNorm（非 InstanceNorm） | 防止 B=1 时 NaN |
| 投影 | Conv1x1 + GroupNorm + GELU | 对齐不同层特征尺度 |
| 辅助 loss | 无 | 全局权重不存在坍缩问题 |
| Phase 2 adapter | 保持冻结 | 与现有 DPTHead 行为一致 |

---

## 实验变量

| 参数 | 默认值 | 可选范围 | 说明 |
|------|--------|---------|------|
| `--fusion_k` | 4 | [4, 6, 8] | 推理时 top-k 层数 |

---

## 测试结果

```
23 passed, 3 skipped in 10.15s
```

- 9 个新增 `test_layer_fusion.py` 测试全部通过
- 14 个现有测试全部通过，无回归
- 3 个 `test_dinov3_encoder.py` 测试跳过（需要 DINOv3 checkpoint）

---

## 已知风险

| 风险 | 缓解措施 |
|------|---------|
| 推理 top-k 性能下降 | 增大 k 值；或推理时仍用全部层 |
| Train/eval gap（12层 vs top-k） | GroupNorm 减轻分布偏移 |
| 12 层特征提取显存增加 | 可减小 batch size |
| Phase 2 adapter 冻结导致 depth 特征不匹配 | 如性能不佳可尝试解冻 adapter |
