# 训练速度优化与 Bug 修复总结

> Date: 2026-06-19
> Commit: `ca881f2`
> Branch: `DINO`
> Spec: `docs/superpowers/specs/2026-06-19-training-speed-optimization-design.md`

## 概述

本次修改在**不改变两阶段训练框架**的前提下，修复了 4 个 bug 并实施了 2 项性能优化。预期 Phase 2 训练速度提升 2-4x。

| 文件 | + | - |
|------|---|---|
| `dinosam/model/depth_sam.py` | +85 | -60 |
| `dinosam/train/trainer.py` | +38 | -27 |
| `dinosam/train/iterative.py` | +14 | -10 |

---

## 一、Bug 修复

### Bug 1: `preprocess` — SAM 的 `apply_image_torch` 索引错误

**根因:** SAM 官方的 `apply_image_torch` 在 batch 输入时使用 `image.shape[0]`（batch size B）和 `image.shape[1]`（channels C=3）作为高和宽计算 resize 目标尺寸，而不是正确的 `image.shape[2]`（H）和 `image.shape[3]`（W）。

| batch_size | `max(B, C=3)` | scale | target_size |
|------------|---------------|-------|-------------|
| 1 | 3 | 1024/3=341 | (341, 1024) |
| 4 | 4 | 1024/4=256 | (1024, 768) |
| 8 | 8 | 1024/8=128 | (1024, 384) |

**影响:** 不同 batch_size 产生完全不同的特征图空间维度，训练与推理 batch_size 不一致时分割结果错误。

**修复:** `depth_sam.py:134-146` — 绕过 `apply_image_torch`，直接用 `F.interpolate` + `ResizeLongestSide.get_preprocess_shape` 使用正确的 H、W 索引。

---

### Bug 2: `original_sizes[0]` — 坐标变换使用首张图尺寸作用于整个 batch

**根因:** `trainer.py` 和 `iterative.py` 中 4 处调用点使用 `original_sizes[0].tolist()` 取出 batch 中第一张图的尺寸，然后对整个 batch 做 `apply_coords_torch`。当 batch 内图像尺寸不同时，非首张图的坐标变换完全错误。

**修复:**
- 移除所有 4 处调用点的 `apply_coords_torch` 预变换（`trainer.py` L188, L264; `iterative.py` L90）
- `forward_depth` 内部添加逐样本 `apply_coords_torch`（`depth_sam.py` L269-278），与 `forward_mask` 约定对齐
- 坐标空间约定统一为「原始图像空间」

---

### Bug 3: `iterative_depth_step` — 256×256 坐标未缩放传入 `forward_depth`

**根因:** `iterative_depth_step` 的 prompt generator 产生 256×256 低分辨率坐标，原代码直接传入 `forward_depth`（期望原始图像空间坐标）。`iterative_mask_step` 已有对应缩放逻辑（L168-176），但 `iterative_depth_step` 缺少。

**修复:** `iterative.py:271-281` — 将 256×256 坐标按 `original_sizes` 缩放回原始图像空间。

---

### Bug 4: SAM `MaskDecoder` / `DepthMaskDecoder` 的 `repeat_interleave` 与 batch 化不兼容

**根因:** 两个 decoder 内部使用 `torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)`。当输入 batch 化时 `tokens.shape[0] = B`，产生 `(B², C, H, W)` 的非法形状张量，直接 crash。

**修复:**
- `prompt_encoder` 保持 batch 化（无 `repeat_interleave` 问题）
- `mask_decoder` / `depth_decoder` 保持逐样本调用（避免侵入式修改上游 SAM 代码）

---

## 二、性能优化

### 优化 1: `prompt_encoder` 批量处理

| 方法 | 原实现 | 优化后 |
|------|--------|--------|
| `forward_mask` | `prompt_encoder` 调用 B 次 | **1 次 batch 调用** |
| `forward_depth` | `outer_prompt_encoder` 调用 B 次 | **1 次 batch 调用** |
| `forward_mask` | `mask_decoder` 调用 B 次 | B 次（不变，避免 `repeat_interleave` B² 问题） |
| `forward_depth` | `depth_decoder` 调用 B 次 | B 次（同上） |

**加速来源:** 消除 B-1 次 GPU kernel launch，每次 batch 调用充分利用 GPU 并行计算能力。PromptEncoder 的计算量（坐标嵌入 + mask 降采样）远大于 decoder（2 层 transformer），因此 decoder 保持 per-sample 的损失很小。

**预期贡献:** 占总加速的 ~60-70%

---

### 优化 2: Phase 2 条件性 mask loss 计算

- `forward_mask` 新增 `return_fullres: bool = True` 参数。`False` 时跳过 `postprocess_masks`（低分辨率 mask → 原始分辨率的上采样），返回 `masks=None`
- Phase 2 训练中仅在 `log_interval` batch 时计算 mask loss（用于日志）
- `validate_phase2` 设置 `return_fullres=False`，完全移除 mask loss 计算
- 修正 mask loss 日志分母（使用实际计算 mask loss 的 batch 数而非总 batch 数）

**加速来源:** 跳过每个 batch 的 `postprocess_masks`（bilinear resize 到原图分辨率）和 `dice_loss` 计算。

**预期贡献:** 占总加速的 ~15-20%

---

## 三、预期加速效果

| 阶段 | 变化 | 说明 |
|------|------|------|
| Phase 1 | **不变** | 不调用 `forward_depth`，不涉及 mask loss 跳过 |
| Phase 2 | **2-4x 加速** | prompt_encoder 批量化（~60-70%）+ 条件 mask loss（~15-20%） |

---

## 四、验证清单

- [ ] Batch size 不变性：bsz=4 训练，bsz=1 和 bsz=4 评估，IoU/Dice 匹配（< 1e-4）
- [ ] Batched vs per-sample 等价：B=4 vs 4×B=1，`low_res_masks`/`iou_pred`/`depth_pred` 匹配（< 1e-6）
- [ ] `original_sizes[0]` fix：混合尺寸 batch 结果与逐样本评估一致
- [ ] Phase 1 回归：mask loss 曲线与修改前一致
- [ ] Phase 2 回归：depth loss 曲线与修改前一致
- [ ] DINO encoder：bsz=1/4 评估指标一致
- [ ] Phase 2 epoch 耗时显著减少（目标：30-50%）

---

## 五、坐标空间约定

修改后统一约定：

| 参数 | 空间 | 说明 |
|------|------|------|
| `point_coords` / `depth_point_coords` | **原始图像空间** | 与 `original_sizes` 对应 |
| `forward_mask` 内部 | 逐样本 `apply_coords_torch` 变换到 model-input 空间 | 已有逻辑 |
| `forward_depth` 内部 | 逐样本 `apply_coords_torch` 变换到 model-input 空间 | **新增**，对齐 `forward_mask` |

所有调用方传入原始图像空间坐标，不得预变换。
