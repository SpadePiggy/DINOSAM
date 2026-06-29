# DPT Multi-Level Feature Fusion Adapter Design

**Date:** 2026-06-29
**Branch:** DINO
**Status:** DRAFT
**Related:** `2026-06-19-dino-adapter-selectable-design.md` (adapter 框架), `2026-06-29-mask-loss-type-design.md` (损失选择)

---

## Overview

将 segdino 的 DPT 多层级特征融合机制迁移到 DINOSAM，作为 `--adapter_type` 的新选项。用户通过 `--adapter_type dpt_simple` 或 `--adapter_type dpt_full` 启用，与现有 `mona` / `fc` / `dual_attn` 互斥。

**动机：** 当前 `DINOv3MonaEncoder` 仅取 DINOv3 最后一层 `x_norm_patchtokens`，丢弃了浅层纹理/边缘信息。DPT 从 4 个中间层取特征并通过多尺度融合，理论上能为 SAM 提供更丰富的特征表示。

---

## Architecture

### 数据流对比

```
现有 adapter (mona/fc/dual_attn):
  DINOv3.forward_features(x) → x_norm_patchtokens (B, N, 768)
    → adapter1 → adapter2 → projection (768→256) → (B, 256, 64, 64)

新增 DPT adapter (dpt_simple / dpt_full):
  DINOv3.get_intermediate_layers(x, n=[2,5,8,11])
    → 4× (B, N, 768) 中间层特征
    → DPTHead: project + resize + fuse → (B, 768, 64, 64)
    → flatten → (B, N, 768)
    → adapter1 → adapter2 → projection → (B, 256, 64, 64)
```

### 关键设计：DPT 与 adapter 的组合关系

DPT 是**特征提取方式**的替换（从单层 → 多层），adapter 是**特征精炼**步骤。两者正交：

- `adapter_type in {mona, fc, dual_attn}`：单层特征 + 对应 adapter
- `adapter_type in {dpt_simple, dpt_full}`：多层 DPT 特征 + 固定使用 mona adapter（简单、经过验证）

**决策：** DPT 类型固定搭配 mona adapter，而非让用户选择。理由：
1. 减少组合爆炸（避免 2×4=8 种配置）
2. Mona 是现有默认且经过验证的 adapter
3. 如需实验其他 adapter 搭配 DPT，可在后续版本扩展为 `--dpt_adapter_type` 参数

---

## DPT Head 实现

### 共享组件

两种 DPT 头都需要：
- 4 个 1×1 Conv 投影层（768 → `out_channels[i]`）
- 统一的 resize 策略对齐到 patch 分辨率 (64×64)
- 输出通道数 = `embed_dim = 768`（**关键偏离 segdino**：segdino 输出 `nclass` 通道用于分割，我们输出 768 通道用于接 adapter）

### DPTSimpleHead（从 `segdino/dpt.py` 迁移）

```python
class DPTSimpleHead(nn.Module):
    """Simple DPT: project → resize → concat → 1×1 Conv."""
    
    def __init__(self, embed_dim=768, features=256, 
                 out_channels=(96, 192, 384, 768)):
        super().__init__()
        assert len(out_channels) == 4, "DPT requires exactly 4 intermediate layers"
        
        # 4× 1×1 projection: embed_dim → out_channels[i]
        self.projects = nn.ModuleList([
            nn.Conv2d(embed_dim, oc, kernel_size=1) for oc in out_channels
        ])
        
        # Layer 1 upsample 4× via ConvTranspose (96 → 256 spatial)
        # Layer 2 upsample 2× via ConvTranspose
        # Layer 3 identity
        # Layer 4 downsample 2× via stride-2 Conv
        # → 全部对齐到 layer_3 的空间尺寸 (patch_h × patch_w)
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, stride=4),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, stride=2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], 3, stride=2, padding=1),
        ])
        
        # 4× 3×3 Conv: out_channels[i] → features (统一通道数)
        self.scratch = _make_scratch(out_channels, features)
        
        # cat → 4*features → embed_dim (768)
        self.output_conv = nn.Conv2d(features * 4, embed_dim, kernel_size=1)
    
    def forward(self, features, patch_h, patch_w):
        """
        Args:
            features: list of 4 tensors, each (B, N, embed_dim)
            patch_h, patch_w: spatial size of patch tokens (e.g., 64, 64)
        Returns:
            (B, embed_dim, patch_h, patch_w)
        """
        out = []
        for i, x in enumerate(features):
            # (B, N, C) → (B, C, H, W)
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        
        # 对齐到 layer_3 的空间尺寸 (patch_h × patch_w)
        target_hw = out[2].shape[-2:]
        for i in [0, 1, 3]:
            if out[i].shape[-2:] != target_hw:
                out[i] = F.interpolate(out[i], size=target_hw, 
                                       mode="bilinear", align_corners=True)
        
        # 4× 3×3 Conv 精炼
        layers = [self.scratch.layer1_rn(out[0]), self.scratch.layer2_rn(out[1]),
                  self.scratch.layer3_rn(out[2]), self.scratch.layer4_rn(out[3])]
        
        # cat along channel: features*4
        fused = torch.cat(layers, dim=1)
        return self.output_conv(fused)  # (B, embed_dim, patch_h, patch_w)
```

### DPTFullHead（从 `segdino/fine_grained_decoder/dpt.py` 迁移）

在 SimpleHead 基础上，将 concat 替换为 4 个 `FeatureFusionBlock`（残差融合，逐层 2× 上采样）。代码约 150 行，含 `FeatureFusionBlock` 和 `ResidualConvUnit`。

```python
class DPTFullHead(nn.Module):
    """Full DPT: project → resize → FeatureFusionBlock chain → output_conv."""
    
    def forward(self, features, patch_h, patch_w):
        # ... project + resize (同 SimpleHead) ...
        
        # FeatureFusionBlock 残差融合（从深到浅）
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        return self.output_conv(path_1)  # (B, embed_dim, patch_h, patch_w)
```

### 辅助模块：从 segdino 迁移

`dinosam/model/dpt_blocks.py` (NEW)：

- `_make_scratch(in_shape, out_shape)`: 4 个 3×3 Conv 统一通道数
- `ResidualConvUnit`: 双 3×3 Conv + 残差
- `FeatureFusionBlock`: 残差融合 + 上采样

**关键修改（Review 发现）：** segdino 源码使用 `nn.quantized.FloatFunctional()` 做 skip connection。迁移时替换为普通 `+` 操作符，避免 mixed precision / torch.compile 兼容问题。

---

## Review 发现修复清单

| # | 严重度 | 问题 | 修复方案 |
|---|--------|------|----------|
| C1 | CRITICAL | DPT 输出空间尺寸错误：ConvTranspose 4× 后变成 256×256，adapter 期望 64×64 | DPT head 输出后加 `F.interpolate` 回到 `(patch_h, patch_w)` 再 flatten |
| C2 | CRITICAL | segdino `out_channels` 默认 3 元素，但代码假设 4 | 在 `DPTSimpleHead`/`DPTFullHead` 加 `assert len(out_channels) == 4`，且默认值固定为 `[96, 192, 384, 768]` |
| C3 | CRITICAL | DPT 头输出通道数是 `nclass`（2 或 21），adapter 期望 768 | 明确文档：迁移时将 `nclass` 替换为 `embed_dim=768` |
| M1 | MAJOR | `get_intermediate_layers` API 未验证 | 实现前必须运行验证脚本：`backbone.get_intermediate_layers(x, n=[2,5,8,11])` 返回格式 |
| M2 | MAJOR | DPT 头梯度流未说明 | DPT 头参数可训练（不冻结），`freeze_image_encoder` 只冻结 `backbone.parameters()` |
| M3 | MAJOR | Checkpoint 兼容性：切换 adapter_type 后 `strict=True` 加载失败 | `_load_checkpoint` 使用 `strict=False` + 日志记录 missing/unexpected keys |
| M4 | MAJOR | Phase 2 不验证 adapter_type 一致性 | 在 Phase 2 加载 Phase 1 checkpoint 时加断言：当前 `adapter_type` == checkpoint 中的 `adapter_type` |
| M5 | MAJOR | `evaluate.py` 缺少 `--adapter_type` 参数（DPT 场景） | 在 `evaluate.py` 添加 `--adapter_type` argparse，并在模型构建时传递 |
| m1 | MINOR | 4 个中间特征图内存增加 ~1.9GB | 文档说明，建议 batch_size 减半或使用 gradient checkpointing |
| m2 | MINOR | `n_storage_tokens=4` 影响 token count | `get_intermediate_layers` 应该排除 storage tokens（与 `x_norm_patchtokens` 行为一致），验证时确认 |
| m3 | MINOR | `nn.quantized.FloatFunctional` 兼容问题 | 迁移 `ResidualConvUnit`/`FeatureFusionBlock` 时替换为 `+` |

---

## File Changes

### 新增文件

1. **`dinosam/model/dpt_blocks.py`** (NEW)
   - `_make_scratch`, `ResidualConvUnit`, `FeatureFusionBlock`
   - 从 `segdino/fine_grained_decoder/blocks.py` 迁移，替换 `FloatFunctional` 为 `+`

2. **`dinosam/model/dpt_heads.py`** (NEW)
   - `DPTSimpleHead`, `DPTFullHead`
   - 输出通道 = `embed_dim=768`
   - 空间输出 = `(patch_h, patch_w)`

### 修改文件

3. **`dinosam/model/adapters/__init__.py`** (MODIFY)

   ```python
   def get_adapter(name: str, in_dim: int, factor: int = 8):
       if name == "mona":
           from ..mona import Mona
           return Mona(in_dim, factor)
       elif name == "fc":
           from .fc_adapter import FCAdapter
           return FCAdapter(in_dim, factor)
       elif name == "dual_attn":
           from .dual_attn_adapter import DualAttnAdapter
           return DualAttnAdapter(in_dim, factor)
       elif name in ("dpt_simple", "dpt_full"):
           # DPT adapters use mona for feature refinement
           from ..mona import Mona
           return Mona(in_dim, factor)
       else:
           raise ValueError(f"Unknown adapter type: {name}")
   ```

4. **`dinosam/model/dinov3_encoder.py`** (MODIFY)

   - `__init__` 新增：根据 `adapter_type` 决定是否创建 DPT head
   - `forward` 分支：DPT 路径 vs 现有路径

   ```python
   def __init__(self, ..., adapter_type="mona"):
       # ... existing ...
       self.adapter_type = adapter_type
       
       # DPT heads (only for dpt_simple / dpt_full)
       if adapter_type == "dpt_simple":
           from .dpt_heads import DPTSimpleHead
           self.dpt_head = DPTSimpleHead(embed_dim=embed_dim)
       elif adapter_type == "dpt_full":
           from .dpt_heads import DPTFullHead
           self.dpt_head = DPTFullHead(embed_dim=embed_dim)
       else:
           self.dpt_head = None
       
       # 中间层索引（CLI 可配置）
       self.intermediate_layer_idx = [2, 5, 8, 11]  # ViT-B/16 default
   
   def forward(self, images):
       # ... preprocess ...
       
       if self.dpt_head is not None:
           # DPT path: multi-level features
           features = self.backbone.get_intermediate_layers(
               x, n=self.intermediate_layer_idx
           )
           x_2d = self.dpt_head(features, patch_h, patch_w)  # (B, 768, 64, 64)
           x = x_2d.flatten(2).transpose(1, 2)  # (B, N, 768)
       else:
           # Existing path: single-layer features
           features = self.backbone.forward_features(x)
           x = features["x_norm_patchtokens"]
       
       x = self.adapter1(x, (H, W))
       x = self.adapter2(x, (H, W))
       x = self.projection(x)
       x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
       return x, input_size
   ```

5. **`dinosam/train/trainer.py`** (MODIFY)

   - `--adapter_type` choices 扩展为 `["mona", "fc", "dual_attn", "dpt_simple", "dpt_full"]`
   - 新增 `--dpt_layers` 参数（字符串格式，默认 `"2,5,8,11"`）
   - `_load_checkpoint` 使用 `strict=False` 并记录 missing/unexpected keys

   ```python
   parser.add_argument("--adapter_type", type=str, default="mona",
                       choices=["mona", "fc", "dual_attn", "dpt_simple", "dpt_full"],
                       help="Adapter type: mona/fc/dual_attn (single-layer) or "
                            "dpt_simple/dpt_full (multi-level DPT fusion)")
   parser.add_argument("--dpt_layers", type=str, default="2,5,8,11",
                       help="Comma-separated intermediate layer indices for DPT "
                            "(e.g., '2,5,8,11'). Only used when adapter_type is dpt_*")
   ```

6. **`dinosam/model/depth_sam.py`** (MODIFY)

   - `DepthSam.__init__` 已传递 `adapter_type`，无需改动
   - 确认 `DINOv3MonaEncoder` 接收 `adapter_type`

7. **`dinosam/evaluate.py`** (MODIFY)

   - 添加 `--adapter_type` 和 `--dpt_layers` 参数
   - 模型构建时传递这些参数

---

## Checkpoint 兼容性

### 同 adapter_type（round-trip）

✅ 正常工作。`adapter1.*`, `adapter2.*`, `projection.*` + `dpt_head.*`（如果存在）全部匹配。

### 跨 adapter_type

⚠️ 部分兼容。使用 `strict=False` 加载：

- `mona` → `dpt_simple`：`adapter1/2` 权重保留，`dpt_head.*` 随机初始化
- `dpt_simple` → `mona`：`adapter1/2` 权重保留，`dpt_head.*` 被忽略

**文档说明：** 切换 adapter_type 时，adapter 权重可能复用（如果结构兼容），但 DPT 头需要重新训练。建议对新 adapter_type 从头训练以获得最佳效果。

### Phase 2 加载 Phase 1 checkpoint

在 `run_phase2` 中加验证：

```python
ckpt = torch.load(phase1_checkpoint, map_location=device)
ckpt_adapter_type = _infer_adapter_type_from_checkpoint(ckpt["model_state_dict"])
if ckpt_adapter_type != args.adapter_type:
    raise ValueError(
        f"Phase 1 checkpoint was trained with adapter_type='{ckpt_adapter_type}', "
        f"but Phase 2 is using adapter_type='{args.adapter_type}'. "
        f"Please use matching adapter_type for both phases."
    )
```

---

## CLI Usage

### Phase 1: Mask training with DPT

```bash
# DPT simple fusion + mona adapter
python -m dinosam.train.trainer \
    --sam_checkpoint /path/to/sam_vit_b.pth \
    --encoder dinov3 \
    --dinov3_checkpoint /path/to/dinov3_vitb16.pth \
    --adapter_type dpt_simple \
    --dpt_layers "2,5,8,11" \
    --train_phase 1 \
    --batch_size 2  # Reduced due to memory increase
```

### Phase 2: Depth training

```bash
# adapter_type must match Phase 1
python -m dinosam.train.trainer \
    --sam_checkpoint /path/to/sam_vit_b.pth \
    --encoder dinov3 \
    --dinov3_checkpoint /path/to/dinov3_vitb16.pth \
    --adapter_type dpt_simple \
    --train_phase 2 \
    --phase1_checkpoint /path/to/phase1_best.pt
```

### 两阶段连续训练

```bash
python -m dinosam.train.trainer \
    --sam_checkpoint /path/to/sam_vit_b.pth \
    --encoder dinov3 \
    --dinov3_checkpoint /path/to/dinov3_vitb16.pth \
    --adapter_type dpt_full \
    --dpt_layers "2,5,8,11" \
    --train_phase both
```

### Evaluation

```bash
python -m dinosam.evaluate \
    --checkpoint /path/to/best.pt \
    --encoder dinov3 \
    --dinov3_checkpoint /path/to/dinov3_vitb16.pth \
    --adapter_type dpt_full
```

---

## Memory Considerations

DPT 存储 4 个中间特征图 `(B, 4096, 768)`：

- batch_size=4 时额外 ~1.9GB 激活内存
- 建议：DPT 模式下 `batch_size` 减半（4 → 2）
- 可选：启用 gradient checkpointing（future work）

---

## Verification Checklist

实现完成后验证：

- [ ] `python -c "from dinov3.models.vision_transformer import DinoVisionTransformer; ..."` 确认 `get_intermediate_layers` API 存在且返回 `list[Tensor]`
- [ ] `--adapter_type dpt_simple` 训练正常，loss 下降
- [ ] `--adapter_type dpt_full` 训练正常，loss 下降
- [ ] `--adapter_type mona`（默认）行为不变（回归测试）
- [ ] DPT checkpoint round-trip 保存/加载正常
- [ ] 跨 adapter_type 加载：`strict=False` + 日志正确
- [ ] Phase 1 → Phase 2 adapter_type 一致性验证生效
- [ ] `evaluate.py` 支持 `--adapter_type dpt_*`
- [ ] 内存使用符合预期（batch_size=2 时 OOM 不发生）

---

## Out of Scope

1. **DPT + 非 mona adapter 组合** — 如需实验，后续添加 `--dpt_adapter_type` 参数
2. **Gradient checkpointing** — 内存优化可后续添加
3. **消融实验** — `dpt_simple` vs `dpt_full` vs `mona` 的效果对比由用户实验决定
4. **Phase 2 depth decoder 改动** — DPT 仅影响 encoder，depth decoder 不变

---

## Implementation Order

1. 验证 `get_intermediate_layers` API
2. 实现 `dinosam/model/dpt_blocks.py`（从 segdino 迁移）
3. 实现 `dinosam/model/dpt_heads.py`
4. 修改 `dinosam/model/adapters/__init__.py` 支持 `dpt_simple` / `dpt_full`
5. 修改 `dinosam/model/dinov3_encoder.py` 添加 DPT 路径
6. 修改 `dinosam/train/trainer.py` 添加 CLI 参数 + checkpoint 兼容性
7. 修改 `dinosam/evaluate.py` 添加 CLI 参数
8. 端到端测试
