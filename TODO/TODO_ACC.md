# 训练速度优化方案

## 当前训练速度瓶颈分析

从代码逐行分析来看，主要瓶颈按影响从大到小排列：

| 优先级 | 瓶颈 | 位置 | 影响估算 |
|--------|------|------|----------|
| 🔴 P0 | **BUG: `apply_image_torch` 用错 shape 索引** | `depth_sam.py:135` | 不仅限制 batch size, 还导致整体处理方式错误 |
| 🔴 P0 | **forward_mask/forward_depth 逐样本串行处理** | `depth_sam.py:188-222, 252-275` | ~4x 慢于批处理 |
| 🟠 P1 | **Phase 2 冗余计算 mask loss** | `trainer.py:203-204` | 额外前向+loss 计算，但不参与反向传播 |
| 🟡 P2 | **Phase 2 仍然跑完整 forward_mask** | `trainer.py:179-185` | mask branch 已冻结，不需要 full forward |
| 🟡 P2 | **mixed precision 未启用** | 全局 | 3090 支持 fp16，可提速 1.5-2x |
| 🟢 P3 | **每 batch 写 TensorBoard** | `trainer.py:85-87, 222-225` | 少量 I/O 开销 |
| 🟢 P3 | **同步 checkpoint 保存** | `trainer.py:328` | 每 epoch ~5-15s |

---

## 方案 A：安全修复（推荐 ✅）

只做低风险、高收益的改动：

1. **修复 `preprocess` bug** — 绕过 SAM 官方的 `apply_image_torch`（其在 batch 输入时用 `image.shape[0]`（B）和 `image.shape[1]`（C）计算 resize 尺寸，而不是 `image.shape[2]`（H）和 `image.shape[3]`（W）），直接用 `get_preprocess_shape` + `F.interpolate`
2. **批量化 `forward_mask`** — 不再 `for i in range(B)`，一次性处理整个 batch。SAM 的 PromptEncoder 和 MaskDecoder 均支持 batched 输入
3. **批量化 `forward_depth`** — 同上
4. **Phase 2 跳过冗余 mask loss 计算** — `inner_mask_loss` 和 `decoder_mask_loss` 不参与反向传播，只在需要 log 时才计算

**预计提速：2-4x**，改动集中在 `depth_sam.py`，对训练逻辑无影响。

---

## 方案 B：中等优化

在方案 A 基础上增加：

5. **启用 AMP (Automatic Mixed Precision)** — `torch.cuda.amp.autocast` + `GradScaler`，3090 的 fp16 tensor core 可显著加速
6. **增大 batch_size** — 修复后利用 24GB 显存，尝试 batch_size=8 或 16
7. **TensorBoard 写入降频** — 每 N 个 batch 写一次（如 `log_interval`）

**预计提速：3-6x**

---

## 方案 C：激进优化

在方案 B 基础上增加：

8. **预计算并缓存 image embeddings** — Phase 1 训完后把 embeddings 存盘，Phase 2 直接从磁盘读（因为 encoder 全程冻结），完全跳过 `encode_images`
9. **异步 checkpoint 保存** — 用独立线程异步写盘，不阻塞训练

**预计提速：5-8x**（Phase 2 可接近数据加载上限）

---

## 关键 Bug 详解：`apply_image_torch` 索引错误

SAM 官方 `transforms.py:62`:

```python
def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
    # Expects batched images with shape BxCxHxW
    target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
    #                                       ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^
    #                                       B (batch size!)  C (channels!)  ← BUG!
```

`get_preprocess_shape(oldh, oldw, long_side_length)`:
```python
scale = long_side_length * 1.0 / max(oldh, oldw)
newh, neww = oldh * scale, oldw * scale
```

不同 batch_size 的影响：

| batch_size | `max(B, C=3)` | scale | target_size |
|------------|---------------|-------|-------------|
| 1 | max(1,3)=3 | 1024/3=341 | **(341, 1024)** |
| 4 | max(4,3)=4 | 1024/4=256 | **(1024, 768)** |
| 8 | max(8,3)=8 | 1024/8=128 | **(1024, 384)** |

### 修复方式

```python
def preprocess(self, x):
    # 直接用正确的 H, W 索引 (2, 3) 而不是 (0, 1)
    oldh, oldw = x.shape[2], x.shape[3]
    newh, neww = ResizeLongestSide.get_preprocess_shape(oldh, oldw, self.img_size)
    x = F.interpolate(x, size=(newh, neww), mode="bilinear", align_corners=False)
    input_size = (newh, neww)

    x = (x - self.sam.pixel_mean.unsqueeze(0)) / self.sam.pixel_std.unsqueeze(0)
    padh = self.img_size - newh
    padw = self.img_size - neww
    x = F.pad(x, (0, padw, 0, padh))
    return x, input_size
```

---

## forward_mask 批量化设计

当前：逐样本 for 循环，PromptEncoder + MaskDecoder 各跑 B 次
改为：一次性传入 `(B, N, 2)` 的 points

关键验证：SAM PromptEncoder.forward 已支持 batched input：
- `_embed_points` 接收 `points: (B, N, 2)`, `labels: (B, N)`
- MaskDecoder 也通过 `_get_batch_size` 自动推断 batch size

注意：`apply_coords_torch` 在 batched 调用时需要 `original_size` shape 匹配，需确认其行为。
