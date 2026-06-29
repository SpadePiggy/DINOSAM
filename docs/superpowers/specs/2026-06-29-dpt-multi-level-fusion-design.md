# DPT Multi-Level Feature Fusion Adapter Design

**Date:** 2026-06-29  
**Branch:** DINO  
**Status:** APPROVED  
**Related:** `2026-06-19-dino-adapter-selectable-design.md` (adapter 框架), `2026-06-29-mask-loss-type-design.md` (损失选择)

---

## Overview

将 segdino 的 DPT 多层级特征提取迁移到 DINOSAM，作为 `--adapter_type dpt_simple` 新选项。与现有 `mona` / `fc` / `dual_attn` 互斥。

**动机：** 当前 `DINOv3MonaEncoder` 仅取 DINOv3 最后一层 `x_norm_patchtokens`，丢弃了浅层纹理/边缘信息。DPT 从 4 个中间层取特征并融合，能为 SAM 提供更丰富的特征表示。

**设计原则：** 4 个中间层全部在 64×64 patch 分辨率，无需任何 resize/上采样。只保留 `dpt_simple`（~45 行），不迁移 `dpt_full`（FeatureFusionBlock 在单分辨率下退化为 add+conv，无证据表明优于简单 concat，后续消融实验证明需要再加）。

---

## Architecture

### 数据流

```
现有 adapter (mona/fc/dual_attn):
  DINOv3.forward_features(x) → x_norm_patchtokens (B, N, 768)
    → adapter1 → adapter2 → projection (768→256) → (B, 256, 64, 64)

新增 dpt_simple adapter:
  DINOv3.get_intermediate_layers(x, n=[2,5,8,11])
    → 4× (B, N, 768) 中间层特征，全部在 64×64 patch 分辨率
    → DPTSimpleHead: project + refine + concat → (B, 768, 64, 64)
    → flatten → (B, N, 768)
    → adapter1 (Mona) → adapter2 (Mona) → projection → (B, 256, 64, 64)
```

### DPT 与 adapter 的关系

DPT 替换**特征提取方式**（单层 → 多层），adapter 是**特征精炼**步骤：

- `adapter_type in {mona, fc, dual_attn}`：单层特征 + 对应 adapter
- `adapter_type = "dpt_simple"`：多层 DPT 特征 + Mona adapter（encoder 内部决策，adapter registry 不感知）

**决策：** `dpt_simple` 固定搭配 Mona adapter。理由：
1. 减少组合爆炸
2. Mona 是现有默认且经过验证的 adapter
3. 如需实验其他 adapter 搭配 DPT，后续扩展 `--dpt_adapter_type`

---

## DPTSimpleHead 实现

从 `segdino/dpt.py` 的简单 concat 方式迁移。

**关键偏离 segdino：**
1. 输出通道数 = `embed_dim=768`（segdino 输出 `nclass` 通道用于分割）
2. 无 `resize_layers`（segdino 面向全分辨率分割，需要上采样）
3. 无 `_make_scratch` 辅助函数（内联为 `ModuleList`）
4. 无 `proj`（ConvTranspose 4× 上采样到 256×256，我们保持 64×64）
5. 无 `out_channels` 参数（统一使用 `features=256`，简化配置）

```python
import torch
import torch.nn as nn


class DPTSimpleHead(nn.Module):
    """DPT multi-level feature extraction head. No resize, no upsample.

    All 4 intermediate layers from DINOv3 are at the same spatial resolution
    (patch_h × patch_w = 64×64 for 1024×1024 input with patch_size=16).

    Source: adapted from segdino/dpt.py (simple concat).
    """

    def __init__(self, embed_dim=768, features=256):
        super().__init__()

        # 4× 1×1 projection: embed_dim → features (uniform, no out_channels)
        self.projects = nn.ModuleList([
            nn.Conv2d(embed_dim, features, kernel_size=1) for _ in range(4)
        ])

        # 4× 3×3 Conv: features → features (inline refine, no _make_scratch)
        self.refine_convs = nn.ModuleList([
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)
            for _ in range(4)
        ])

        # concat (4*features) → embed_dim
        self.output_conv = nn.Conv2d(features * 4, embed_dim, kernel_size=1)

    def forward(self, features, patch_h, patch_w):
        """
        Args:
            features: list of 4 tensors, each (B, N, embed_dim)
            patch_h, patch_w: spatial size of patch tokens (e.g., 64, 64)
        Returns:
            (B, embed_dim, patch_h, patch_w) — same spatial as input
        """
        out = []
        for i, x in enumerate(features):
            # (B, N, C) → (B, C, patch_h, patch_w)
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
            x = self.projects[i](x)         # (B, features, H, W)
            x = self.refine_convs[i](x)     # (B, features, H, W)
            out.append(x)

        fused = torch.cat(out, dim=1)       # (B, features*4, H, W)
        return self.output_conv(fused)      # (B, embed_dim, H, W)
```

**代码量：** ~45 行（含注释）。无需新文件 `dpt_blocks.py`。

---

## File Changes

### 新增文件

1. **`dinosam/model/dpt_heads.py`** (NEW) — ~45 行
   - `DPTSimpleHead` 类（上述代码）

### 修改文件

2. **`dinosam/model/adapters/__init__.py`** (MODIFY) — 无变化

   **不改。** adapter registry 不感知 DPT。encoder 内部直接调用 `get_adapter("mona", ...)` 当 `adapter_type == "dpt_simple"` 时。

3. **`dinosam/model/dinov3_encoder.py`** (MODIFY) — ~30 行改动

   - `__init__` 新增 `dpt_layers` 参数和条件 DPT head 创建
   - `forward` 分支：DPT 路径 vs 现有路径
   - **关键：** `patch_h`, `patch_w`, `B`, `H`, `W` 在 if/else **之前**定义

   ```python
   def __init__(self, checkpoint_path, img_size=1024, embed_dim=768,
                out_dim=256, adapter_type="mona", dpt_layers=None):
       # ... existing backbone setup ...
       self.adapter_type = adapter_type

       # DPT head (only for dpt_simple)
       if adapter_type == "dpt_simple":
           from .dpt_heads import DPTSimpleHead
           self.dpt_head = DPTSimpleHead(embed_dim=embed_dim)
       else:
           self.dpt_head = None

       # 中间层索引（CLI 可配置）
       if adapter_type == "dpt_simple":
           self.dpt_layers = dpt_layers or [2, 5, 8, 11]
       
       # adapter1/adapter2: 始终使用 Mona（dpt_simple 时 encoder 内部决策）
       if adapter_type == "dpt_simple":
           self.adapter1 = get_adapter("mona", embed_dim, factor=8)
           self.adapter2 = get_adapter("mona", embed_dim, factor=8)
       else:
           self.adapter1 = get_adapter(adapter_type, embed_dim, factor=8)
           self.adapter2 = get_adapter(adapter_type, embed_dim, factor=8)

   def forward(self, images):
       # ... preprocess (interpolate + normalize) ...
       
       # 关键：所有变量在分支前定义
       B = x.shape[0]
       patch_h = patch_w = self.img_size // 16
       H, W = patch_h, patch_w

       if self.dpt_head is not None:
           # DPT path: multi-level features, all at 64×64
           features = self.backbone.get_intermediate_layers(
               x, n=self.dpt_layers
           )
           x_2d = self.dpt_head(features, patch_h, patch_w)  # (B, 768, 64, 64)
           x = x_2d.flatten(2).transpose(1, 2)                # (B, N, 768)
       else:
           # Existing path: single-layer features
           features = self.backbone.forward_features(x)
           x = features["x_norm_patchtokens"]

       x = self.adapter1(x, (H, W))
       x = self.adapter2(x, (H, W))
       x = self.projection(x)  # (B, H*W, 256)
       x = x.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, 256, H, W)
       return x, input_size
   ```

4. **`dinosam/model/depth_sam.py`** (MODIFY) — +3 行

   `DepthSam.__init__` 新增 `dpt_layers` 参数，传递给 `DINOv3MonaEncoder`：

   ```python
   def __init__(self, sam, encoder_type="sam", dinov3_checkpoint=None,
                adapter_type="mona", depth_transformer_type="twoway",
                dpt_layers=None):
       # ...
       self.dinov3_encoder = DINOv3MonaEncoder(
           checkpoint_path=dinov3_checkpoint,
           img_size=image_size, embed_dim=768, out_dim=256,
           adapter_type=adapter_type,
           dpt_layers=dpt_layers,
       )
   ```

5. **`dinosam/train/trainer.py`** (MODIFY) — ~15 行改动

   **CLI 参数：**

   ```python
   parser.add_argument("--adapter_type", type=str, default="mona",
                       choices=["mona", "fc", "dual_attn", "dpt_simple"],
                       help="Adapter type: mona/fc/dual_attn (single-layer) or "
                            "dpt_simple (multi-level DPT fusion)")
   parser.add_argument("--dpt_layers", type=str, default="2,5,8,11",
                       help="Comma-separated intermediate layer indices for DPT "
                            "(e.g., '2,5,8,11'). Only used with --adapter_type dpt_simple")
   ```

   **模型构建（`_build_model_and_data`）：**

   ```python
   dpt_layers = (
       [int(x.strip()) for x in args.dpt_layers.split(",")]
       if args.adapter_type == "dpt_simple" else None
   )
   model = DepthSam(
       sam,
       encoder_type=args.encoder,
       dinov3_checkpoint=args.dinov3_checkpoint,
       adapter_type=args.adapter_type,
       depth_transformer_type=args.depth_transformer_type,
       dpt_layers=dpt_layers,
   )
   ```

   **Checkpoint 保存（`_save_checkpoint`）— 存储 adapter_type：**

   ```python
   torch.save({
       "model_state_dict": model.state_dict(),
       "optimizer_state_dict": optimizer.state_dict(),
       "scheduler_state_dict": scheduler.state_dict(),
       "epoch": epoch,
       "best_val_loss": best_val_loss,
       "adapter_type": args.adapter_type,  # NEW
   }, path)
   ```

   **Checkpoint 加载（`_load_checkpoint`）— strict=False + 日志：**

   ```python
   result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
   if result.missing_keys:
       print(f"Checkpoint missing keys (expected when switching adapter_type): {result.missing_keys}")
   if result.unexpected_keys:
       print(f"Checkpoint unexpected keys (ignored): {result.unexpected_keys}")
   ```

   **Phase 2 加载 Phase 1 checkpoint — strict=False + warning（不用 error）：**

   ```python
   ckpt = torch.load(phase1_checkpoint, map_location=device)
   ckpt_adapter_type = ckpt.get("adapter_type", "unknown")
   if ckpt_adapter_type != args.adapter_type:
       print(f"Warning: Phase 1 checkpoint adapter_type='{ckpt_adapter_type}' "
             f"differs from current '{args.adapter_type}'. "
             f"Some weights may not transfer.")
   result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
   if result.missing_keys:
       print(f"Missing keys: {result.missing_keys}")
   model.init_depth_from_sam()
   ```

6. **`dinosam/evaluate.py`** (MODIFY) — +12 行

   **CLI 参数：**

   ```python
   parser.add_argument("--adapter_type", type=str, default="mona",
                       choices=["mona", "fc", "dual_attn", "dpt_simple"],
                       help="Adapter type for DINOv3 encoder")
   parser.add_argument("--dpt_layers", type=str, default="2,5,8,11",
                       help="Comma-separated intermediate layer indices for DPT")
   ```

   **模型构建：**

   ```python
   dpt_layers = (
       [int(x.strip()) for x in args.dpt_layers.split(",")]
       if args.adapter_type == "dpt_simple" else None
   )
   model = DepthSam(
       sam,
       encoder_type=args.encoder,
       dinov3_checkpoint=args.dinov3_checkpoint,
       adapter_type=args.adapter_type,
       depth_transformer_type=args.depth_transformer_type,
       dpt_layers=dpt_layers,
   )
   ```

   **Checkpoint 加载（`_load_model_state_dict`）— strict=False：**

   ```python
   def _load_model_state_dict(model, checkpoint_path, device):
       ckpt = torch.load(checkpoint_path, map_location=device)
       if "model_state_dict" in ckpt:
           result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
       else:
           result = model.load_state_dict(ckpt, strict=False)
       if result.missing_keys:
           print(f"Missing keys: {result.missing_keys}")
       if result.unexpected_keys:
           print(f"Unexpected keys: {result.unexpected_keys}")
   ```

---

## Checkpoint 兼容性

### 同 adapter_type（round-trip）

✅ 正常工作。所有 keys 匹配。

### 跨 adapter_type

⚠️ 部分兼容。`strict=False` 加载：

- `mona` → `dpt_simple`：`adapter1/2` + `projection` 权重保留，`dpt_head.*` 随机初始化
- `dpt_simple` → `mona`：`adapter1/2` + `projection` 权重保留，`dpt_head.*` 被忽略

**建议：** 对新 adapter_type 从头训练以获得最佳效果。

### Phase 2 加载 Phase 1 checkpoint

- 使用 `strict=False` 加载
- adapter_type 不匹配时打印 warning（不抛异常，研究灵活性）
- adapter_type 存储在 checkpoint 中（不依赖 key 推断）

---

## Gradient Flow

| Phase | backbone | dpt_head | adapter1/2 | projection | depth branch |
|-------|----------|----------|------------|------------|-------------|
| Phase 1 | ❄️ frozen | ✅ trainable | ✅ trainable | ✅ trainable | ❄️ frozen |
| Phase 2 | ❄️ frozen | ❄️ frozen | ❄️ frozen | ❄️ frozen | ✅ trainable |

- `freeze_image_encoder()` 冻结 `backbone.parameters()`，DPT head 不受影响（Phase 1 可训练）
- Phase 2 冻结所有 encoder 参数（包括 DPT head），只训练 depth branch

---

## CLI Usage

### Phase 1: Mask training with DPT

```bash
python -m dinosam.train.trainer \
    --sam_checkpoint /path/to/sam_vit_b.pth \
    --encoder dinov3 \
    --dinov3_checkpoint /path/to/dinov3_vitb16.pth \
    --adapter_type dpt_simple \
    --dpt_layers "2,5,8,11" \
    --train_phase 1 \
    --batch_size 2  # 内存增加，建议减半
```

### 两阶段连续训练

```bash
python -m dinosam.train.trainer \
    --sam_checkpoint /path/to/sam_vit_b.pth \
    --encoder dinov3 \
    --dinov3_checkpoint /path/to/dinov3_vitb16.pth \
    --adapter_type dpt_simple \
    --dpt_layers "2,5,8,11" \
    --train_phase both
```

### Evaluation

```bash
python -m dinosam.evaluate \
    --checkpoint /path/to/best.pt \
    --encoder dinov3 \
    --dinov3_checkpoint /path/to/dinov3_vitb16.pth \
    --adapter_type dpt_simple
```

---

## Memory Considerations

DPT 存储 4 个中间特征图 `(B, 4096, 768)`：

- batch_size=4 时额外 ~2-3GB（含 backward 激活）
- 建议：DPT 模式下 `batch_size` 减半（4 → 2）

---

## Review 发现修复清单

| # | 严重度 | 问题 | 修复方案 |
|---|--------|------|----------|
| C1 | CRITICAL | DPT forward 路径变量未定义 | `B`, `patch_h`, `patch_w`, `H`, `W` 在 if/else **之前**定义 |
| M1 | MAJOR | `evaluate.py` checkpoint 加载未用 `strict=False` | 修改 `_load_model_state_dict` 使用 `strict=False` + 日志 |
| M2 | MAJOR | 源码归属错误 | 修正：DPTSimpleHead 来自 `segdino/dpt.py`（非 `fine_grained_decoder/dpt.py`） |
| M3 | MAJOR | `out_channels` 参数是死配置 | 删除 `out_channels`，统一使用 `features=256` |
| M4 | MAJOR | `get_adapter` registry 不应感知 DPT | encoder 内部直接调用 `get_adapter("mona", ...)` |
| m1 | MINOR | `dpt_layers` 解析未处理空格 | 加 `.strip()`：`[int(x.strip()) for x in args.dpt_layers.split(",")]` |
| m2 | MINOR | `evaluate.py` 模型构建缺代码 | 补充完整代码示例 |
| m3/m4 | MINOR | Verification checklist 缺断言 | 加 N=4096 断言 + `get_intermediate_layers` 返回格式确认 |

---

## Verification Checklist

实现完成后验证：

- [ ] 验证 `get_intermediate_layers` API 存在且返回 `tuple[Tensor]`，每个 `(B, N, 768)`
- [ ] **关键断言：** `assert features[0].shape[1] == 4096, f"Expected 4096 patch tokens, got {features[0].shape[1]} (storage tokens may be included)"`
- [ ] `--adapter_type dpt_simple` 训练正常，loss 下降
- [ ] `--adapter_type mona`（默认）行为不变（回归测试）
- [ ] DPT checkpoint round-trip 保存/加载正常
- [ ] 跨 adapter_type 加载：`strict=False` + 日志正确
- [ ] Phase 1 → Phase 2 adapter_type warning 正确触发
- [ ] `evaluate.py` 支持 `--adapter_type dpt_simple` 且 `strict=False`
- [ ] 内存使用：batch_size=2 时 OOM 不发生

---

## Out of Scope

1. **`dpt_full` / FeatureFusionBlock** — 单分辨率下退化为 add+conv，无消融实验证明价值。后续需要时再加
2. **DPT + 非 mona adapter 组合** — 后续扩展 `--dpt_adapter_type`
3. **Gradient checkpointing** — 内存优化可后续添加

---

## Implementation Order

1. 验证 `get_intermediate_layers` API
2. 新增 `dinosam/model/dpt_heads.py`（DPTSimpleHead，~45 行）
3. 修改 `dinosam/model/dinov3_encoder.py`（DPT 路径，~30 行）
4. 修改 `dinosam/model/depth_sam.py`（+3 行，传递 dpt_layers）
5. 修改 `dinosam/train/trainer.py`（CLI + checkpoint 兼容，~15 行）
6. 修改 `dinosam/evaluate.py`（CLI + checkpoint 兼容，~12 行）
7. 端到端测试

**预计总新增/修改：~105 行**
