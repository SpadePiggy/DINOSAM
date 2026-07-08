# Layer Fusion with Learned Global Weights — 设计规格

> 日期：2026-07-07
> 项目：DINOSAM
> 状态：设计完成，待实现

---

## 1. 目标与动机

### 1.1 核心目标
理解 DINOv3 ViT-B/16 各层对密集预测任务（分割、深度估计）的重要性差异。通过可学习的全局层权重 + top-k 选择，揭示不同任务对层级特征的依赖模式。

### 1.2 设计目标
- **全局可学习权重**：每个层一个标量权重，通过 softmax 归一化为融合系数
- **Top-k 推理**：推理时只使用权重最高的 k 层，稀疏高效
- **任务差异化**：mask 和 depth 分支各一套独立权重，Phase 1/2 分别训练
- **直接可解释**：12 个权重值直接回答"哪些层对哪个任务最重要"

### 1.3 设计哲学
- 极简：核心代码约 20 行，无需评分网络、Gumbel 噪声、辅助 loss
- 稳定：全局权重通过 task loss 直接反向传播学习，无坍缩风险
- 兼容：完全兼容现有两阶段训练流程，不破坏任何现有接口

---

## 2. 架构设计

### 2.1 整体流程

```
训练时：
Image → DINOv3(frozen) → get_intermediate_layers([0..11]) → 12 层特征
                                                                ↓
                                                          softmax(weights)
                                                                ↓
                                                     加权融合全部 12 层
                                                                ↓
                                                     (B, 768, 64, 64) fused
                                                                ↓
                                                     adapter1 → adapter2 → projection

推理时：
Image → DINOv3(frozen) → get_intermediate_layers(top_k_indices) → k 层特征
                                                                ↓
                                                     加权融合 top-k 层
                                                                ↓
                                                     adapter1 → adapter2 → projection
```

### 2.2 关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 权重类型 | 全局 `nn.Parameter` | 直接可解释，12 个标量 = 12 层重要性 |
| 训练策略 | softmax over all layers | 全梯度流，训练稳定 |
| 推理策略 | hard top-k 选择 | 只计算 k 层，推理高效 |
| 融合方式 | 加权求和 | 保持固定输出维度，下游不变 |
| 双分支 | 两套独立权重 | mask/depth 各学各的层偏好 |
| 辅助 loss | 无 | 全局权重不存在坍缩问题 |

---

## 3. 核心模块

### 3.1 LearnedLayerFusion

**文件**: `dinosam/model/layer_fusion.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedLayerFusion(nn.Module):
    """Learned global layer weights + top-k fusion.

    Training:  softmax(weights) → weighted sum of ALL layers (full gradient)
    Inference: select top-k layers by weight → weighted sum of k layers only
    """

    def __init__(self, embed_dim=768, num_layers=12, k=4):
        super().__init__()
        self.k = k
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        # 可学习的层权重（零初始化 → softmax = 均匀分布）
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

        # 每层独立的投影（Conv1x1 + GroupNorm 归一化尺度 + GELU）
        # GroupNorm 确保不同层的投影输出尺度一致，
        # 避免权重学习被随机初始化的尺度差异偏置。
        # 选用 GroupNorm 而非 InstanceNorm，因为 InstanceNorm 在 B=1 时方差为 0 导致 NaN
        self.layer_projects = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 1),
                nn.GroupNorm(num_groups=min(32, embed_dim), num_channels=embed_dim),
                nn.GELU(),
            )
            for _ in range(num_layers)
        ])

    def forward(self, features, patch_h, patch_w):
        """
        Args:
            features: list/tuple of tensors, each (B, N, embed_dim)
                      Training: all num_layers tensors
                      Inference: only top-k tensors (caller pre-selects)
            patch_h, patch_w: spatial dimensions

        Returns:
            fused: (B, embed_dim, patch_h, patch_w)
        """
        B = features[0].shape[0]
        C = features[0].shape[2]

        if self.training:
            assert len(features) == self.num_layers, \
                f"Training expects {self.num_layers} features, got {len(features)}"
            # 训练：softmax over ALL layers，加权融合
            weights = F.softmax(self.layer_weights, dim=0)  # (num_layers,)
            fused = torch.zeros(B, C, patch_h, patch_w, device=features[0].device)
            for l in range(self.num_layers):
                f = features[l].permute(0, 2, 1).reshape(B, C, patch_h, patch_w)
                f_proj = self.layer_projects[l](f)
                fused += weights[l] * f_proj
            return fused
        else:
            # 推理：只使用 top-k 层（features 已经由调用方预筛选为 k 个）
            topk_indices = self.layer_weights.topk(self.k).indices  # (k,)
            topk_weights = F.softmax(self.layer_weights[topk_indices], dim=0)  # (k,)
            fused = torch.zeros(B, C, patch_h, patch_w, device=features[0].device)
            for i, (feat, global_idx) in enumerate(zip(features, topk_indices)):
                f = feat.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)
                f_proj = self.layer_projects[global_idx](f)
                fused += topk_weights[i] * f_proj
            return fused

    def get_topk_indices(self):
        """返回权重最高的 top-k 层的索引，用于推理时预筛选特征。"""
        return self.layer_weights.topk(self.k).indices

    def get_weights(self):
        """返回 softmax 归一化后的层权重，用于日志和分析。"""
        return F.softmax(self.layer_weights, dim=0).detach().cpu()
```

### 3.2 参数量估算

| 组件 | 参数量 |
|------|--------|
| `layer_weights` | 12 |
| `layer_projects` | 12 × (Conv1x1: 768×768+768 + GroupNorm: 2×768) = ~7.1M |
| **总计** | **~7.1M** |

对比 DPTHead（`projects` + `refine_convs` + `output_conv`）：约 ~1.2M。新模块的参数量主要来自 12 个独立的投影块。如果需要减少可以改为共享 projection + per-layer GroupNorm。

---

## 4. 集成设计

### 4.1 DINOv3MonaEncoder 修改

**文件**: `dinosam/model/dinov3_encoder.py`

```python
class DINOv3MonaEncoder(nn.Module):
    def __init__(self, checkpoint_path, img_size=1024, embed_dim=768, out_dim=256,
                 adapter_type="mona", dpt_layers=None, fusion_k=4):
        super().__init__()
        ...

        # 更新 dpt_layers 验证 guard，允许 dpt_fusion 模式
        if dpt_layers is not None and adapter_type not in ("dpt_simple", "dpt_fusion"):
            raise ValueError(
                f"dpt_layers parameter is only valid when adapter_type='dpt_simple' or 'dpt_fusion'. "
                f"Got adapter_type='{adapter_type}', dpt_layers={dpt_layers}."
            )

        # 新增: fusion 模式
        if adapter_type == "dpt_fusion":
            from .layer_fusion import LearnedLayerFusion
            self.dpt_layers = list(range(12))  # 全部 12 层
            self.mask_fusion = LearnedLayerFusion(
                embed_dim=embed_dim, num_layers=12, k=fusion_k,
            )
            self.depth_fusion = LearnedLayerFusion(
                embed_dim=embed_dim, num_layers=12, k=fusion_k,
            )
            self.dpt_head = None
            # 显式创建 adapter（与 dpt_simple 一致，使用 mona adapter）
            self.adapter1 = get_adapter("mona", embed_dim, factor=8)
            self.adapter2 = get_adapter("mona", embed_dim, factor=8)
        else:
            self.mask_fusion = None
            self.depth_fusion = None
            # 现有 dpt_head 逻辑不变
            ...

    def forward(self, images, branch="mask"):
        """Encode images to (B, 256, 64, 64) embeddings.

        Args:
            images: (B, 3, H, W) raw [0, 255] range
            branch: "mask" or "depth" — 决定使用哪个 fusion head

        Returns:
            image_embeddings: (B, 256, 64, 64)
            input_size: (H, W) of resized image before padding
        """
        x = F.interpolate(
            images, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        )
        input_size = (self.img_size, self.img_size)
        x = (x - self.pixel_mean) / self.pixel_std

        B = x.shape[0]
        patch_h = patch_w = self.img_size // 16
        H, W = patch_h, patch_w

        if self.mask_fusion is not None:
            # Fusion 模式
            fusion = self.mask_fusion if branch == "mask" else self.depth_fusion

            if self.training:
                # 训练：提取全部 12 层
                features = self.backbone.get_intermediate_layers(x, n=self.dpt_layers)
                x_2d = fusion(features, patch_h, patch_w)
            else:
                # 推理：只提取 top-k 层
                topk_indices = fusion.get_topk_indices().tolist()
                features = self.backbone.get_intermediate_layers(
                    x, n=topk_indices
                )
                x_2d = fusion(features, patch_h, patch_w)

            x = x_2d.flatten(2).transpose(1, 2)  # (B, N, C)

        elif self.dpt_head is not None:
            # 现有 DPT 路径不变
            features = self.backbone.get_intermediate_layers(x, n=self.dpt_layers)
            x_2d = self.dpt_head(features, patch_h, patch_w)
            x = x_2d.flatten(2).transpose(1, 2)

        else:
            # 现有单层路径不变
            features = self.backbone.forward_features(x)
            x = features["x_norm_patchtokens"]

        # Adapter + projection（不变，两个分支共享）
        x = self.adapter1(x, (H, W))
        x = self.adapter2(x, (H, W))
        x = self.projection(x)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)

        return x, input_size
```

### 4.2 返回值兼容性

**关键设计**：返回值保持 `(x, input_size)` 二元组，与现有代码完全一致。不需要返回 scores 或 dict——权重是全局的，随时可以通过 `fusion.get_weights()` 获取。

所有现有调用方无需任何修改：
- `depth_sam.py`: `return self.dinov3_encoder(images)` ✅
- `iterative.py`: `image_embeddings, input_size = model.encode_images(images)` ✅
- `evaluate.py`: 同上 ✅

### 4.3 DepthSam 调用链

**文件**: `dinosam/model/depth_sam.py`

```python
def encode_images(self, images: torch.Tensor, branch: str = "mask"):
    """Preprocess and encode images.

    Args:
        branch: "mask" or "depth" — 决定使用哪个 fusion head

    Returns:
        image_embeddings: (B, 256, 64, 64)
        input_size: (H, W)
    """
    if self.encoder_type == "dinov3":
        return self.dinov3_encoder(images, branch=branch)
    else:
        input_images, input_size = self.preprocess(images)
        image_embeddings = self.sam.image_encoder(input_images)
        return image_embeddings, input_size

def forward(self, images, ...):
    """Full forward: mask branch + depth branch."""
    # Mask 分支
    mask_embeddings, input_size = self.encode_images(images, branch="mask")
    low_res_masks, iou_pred, masks = self.forward_mask(
        image_embeddings=mask_embeddings, ...)

    # Depth 分支
    depth_embeddings, _ = self.encode_images(images, branch="depth")
    depth_pred, _ = self.forward_depth(
        image_embeddings=depth_embeddings, ...)

    return {
        "masks": masks,
        "low_res_masks": low_res_masks,
        "iou_predictions": iou_pred,
        "depth_pred": depth_pred,
    }
```

**注意**：`forward()` 中 mask 和 depth 各调用一次 `encode_images`，backbone 特征提取两次。由于权重是全局固定的，推理时 backbone 的 `get_intermediate_layers` 只提取 top-k 层（由 `get_topk_indices()` 决定），计算量可控。如果显存/速度成为瓶颈，后续可优化为 backbone 只跑一次。

### 4.4 DepthSam 构造函数传参

`DepthSam.__init__` 需要新增 `fusion_k` 参数并传递给 encoder：

```python
class DepthSam(nn.Module):
    def __init__(self, ..., fusion_k: int = 4):
        ...
        if encoder_type == "dinov3":
            self.dinov3_encoder = DINOv3MonaEncoder(
                checkpoint_path=dinov3_checkpoint,
                img_size=image_size,
                embed_dim=768,
                out_dim=256,
                adapter_type=adapter_type,
                dpt_layers=dpt_layers,
                fusion_k=fusion_k,  # 【新增】
            )
```

### 4.5 训练循环中的 encode_images 调用更新

**关键原则**：在 fusion 模式下，mask 分支和 depth 分支需要各自独立的融合特征。任何同时使用两个分支的代码路径都需要两次 `encode_images` 调用。

#### 需要双 encode 的调用点（mask + depth 都用到）

```python
# trainer.py — train_one_epoch_phase2 (line ~220)
# 修改前:
image_embeddings, input_size = model.encode_images(images)
low_res_masks, iou_pred, masks = model.forward_mask(image_embeddings, ...)
depth_pred, _ = model.forward_depth(image_embeddings, ...)

# 修改后:
mask_embeddings, input_size = model.encode_images(images, branch="mask")
depth_embeddings, _ = model.encode_images(images, branch="depth")
low_res_masks, iou_pred, masks = model.forward_mask(mask_embeddings, ...)
depth_pred, _ = model.forward_depth(depth_embeddings, ...)

# iterative.py — iterative_train_step (line ~31)
# 修改前:
image_embeddings, input_size = model.encode_images(images)
# mask 循环和 depth 调用共享 image_embeddings

# 修改后:
mask_embeddings, input_size = model.encode_images(images, branch="mask")
depth_embeddings, _ = model.encode_images(images, branch="depth")
# mask 循环用 mask_embeddings, depth 用 depth_embeddings

# iterative.py — iterative_depth_step (line ~226)
# 修改前:
image_embeddings, input_size = model.encode_images(images)
low_res_masks, ... = model.forward_mask(image_embeddings, ...)  # line 227
# ...
depth_pred, _ = model.forward_depth(image_embeddings, ...)      # line 246

# 修改后:
mask_embeddings, input_size = model.encode_images(images, branch="mask")
depth_embeddings, _ = model.encode_images(images, branch="depth")
low_res_masks, ... = model.forward_mask(mask_embeddings, ...)
# ...
depth_pred, _ = model.forward_depth(depth_embeddings, ...)

# trainer.py — validate_phase2 (line ~304)
# 同上：双 encode

# evaluate.py (line ~68)
# 修改前: 单次 encode_images 共享给两个分支
# 修改后:
mask_embeddings, input_size = model.encode_images(images, branch="mask")
# ... mask 预测 ...
depth_embeddings, _ = model.encode_images(images, branch="depth")
# ... depth 评估 ...
```

#### 不需要修改的调用点（仅用 mask 分支）

```python
# trainer.py:56 — train_one_epoch_phase1 (mask only)
# 不需要修改，默认 branch="mask" 即可

# trainer.py:133 — validate_phase1 (mask only)
# 不需要修改

# iterative.py:131 — iterative_mask_step (mask only)
# 不需要修改
```

#### 调用点完整性检查表

| 调用点 | 用到分支 | fusion 模式需要 |
|--------|---------|----------------|
| trainer.py:56 phase1 train | mask only | 无需修改 |
| trainer.py:133 validate_phase1 | mask only | 无需修改 |
| trainer.py:220 phase2 train | mask + depth | **双 encode** |
| trainer.py:304 validate_phase2 | mask + depth | **双 encode** |
| iterative.py:31 iterative_train_step | mask + depth | **双 encode** |
| iterative.py:131 iterative_mask_step | mask only | 无需修改 |
| iterative.py:226 iterative_depth_step | mask + depth | **双 encode** |
| evaluate.py:68 | mask + depth | **双 encode** |
| depth_sam.py:356 forward() | mask + depth | **双 encode**（§4.3 已处理）|

### 4.6 `_build_model_and_data` 传参更新

`trainer.py` 和 `evaluate.py` 中构建 `DepthSam` 的调用点需要传递 `fusion_k`：

```python
# trainer.py _build_model_and_data (line ~355-362)
# evaluate.py (line ~349-352, ~386-389)
model = DepthSam(
    sam, encoder_type=args.encoder, ...,
    dpt_layers=dpt_layers,
    fusion_k=getattr(args, 'fusion_k', 4),  # 【新增】
)
```

### 4.7 CLI argparse 更新

`trainer.py` 和 `iterative.py` 需要两处更新：

```python
# 1. adapter_type choices 新增 dpt_fusion
parser.add_argument("--adapter_type", choices=["mona", "fc", "dual_attn", "dpt_simple", "dpt_fusion"])

# 2. 新增 fusion_k 参数
parser.add_argument("--fusion_k", type=int, default=4, help="Top-k layers for fusion inference")
```

### 4.8 训练循环

**无需修改训练循环。** 全局权重通过 task loss 的反向传播自动学习——`layer_weights` 是 `nn.Parameter`，参与梯度计算。不需要辅助 loss、不需要额外的 loss 项。

Phase 1（mask-only）：
- `mask_fusion` 的 `layer_weights` 和 `layer_projects` 接收梯度
- `depth_fusion` 被冻结（随 depth branch 一起冻结）
- `adapter1/adapter2/projection` 接收梯度

Phase 2（depth-only）：
- **必须修改 `run_phase2()`**：在 blanket freeze 后显式解冻 `depth_fusion.parameters()`
- `depth_fusion` 的 `layer_weights` 和 `layer_projects` 接收梯度
- `mask_fusion` 被冻结（随 mask branch 一起冻结）
- `adapter1/adapter2/projection` 被冻结（与现有 Phase 2 行为一致）

```python
# trainer.py 中 run_phase2() 需要添加：
for param in model.parameters():
    param.requires_grad = False
# 解冻 depth decoder
for param in model.outer_prompt_encoder.parameters():
    param.requires_grad = True
for param in model.depth_decoder.parameters():
    param.requires_grad = True
# 【新增】解冻 depth fusion
if model.dinov3_encoder and model.dinov3_encoder.depth_fusion is not None:
    for param in model.dinov3_encoder.depth_fusion.parameters():
        param.requires_grad = True
```

---

## 5. 层偏好日志与分析

### 5.1 训练时日志

由于权重是全局的，只需在每个 epoch 结束时记录：

```python
def log_layer_weights(model, epoch, logger):
    """记录各分支的层权重分布和梯度状态。"""
    encoder = model.dinov3_encoder if hasattr(model, 'dinov3_encoder') else None
    if encoder and encoder.mask_fusion is not None:
        mask_w = encoder.mask_fusion.get_weights().numpy()
        depth_w = encoder.depth_fusion.get_weights().numpy()
        logger.info(f"Epoch {epoch} Mask weights: {mask_w.tolist()}")
        logger.info(f"Epoch {epoch} Depth weights: {depth_w.tolist()}")
        logger.info(f"Epoch {epoch} Mask top-{encoder.mask_fusion.k}: "
                     f"{encoder.mask_fusion.get_topk_indices().tolist()}")
        logger.info(f"Epoch {epoch} Depth top-{encoder.depth_fusion.k}: "
                     f"{encoder.depth_fusion.get_topk_indices().tolist()}")
        # 梯度监控：诊断权重是否在有效学习
        if encoder.mask_fusion.layer_weights.grad is not None:
            g_norm = encoder.mask_fusion.layer_weights.grad.norm().item()
            logger.info(f"Epoch {epoch} Mask weight grad norm: {g_norm:.6f}")
        if encoder.depth_fusion.layer_weights.grad is not None:
            g_norm = encoder.depth_fusion.layer_weights.grad.norm().item()
            logger.info(f"Epoch {epoch} Depth weight grad norm: {g_norm:.6f}")
        
        # 投影梯度监控：检测"权重活着但投影死了"的静默失败
        # 检查 top-k 和 bottom-k 层的投影梯度
        if encoder.mask_fusion.layer_weights.grad is not None:
            topk_indices = encoder.mask_fusion.get_topk_indices()
            all_indices = list(range(12))
            bottomk_indices = [i for i in all_indices if i not in topk_indices][-2:]  # bottom-2
            
            proj_grads_top = []
            for idx in topk_indices:
                proj = encoder.mask_fusion.layer_projects[idx]
                if proj[0].weight.grad is not None:
                    proj_grads_top.append(proj[0].weight.grad.norm().item())
            
            proj_grads_bottom = []
            for idx in bottomk_indices:
                proj = encoder.mask_fusion.layer_projects[idx]
                if proj[0].weight.grad is not None:
                    proj_grads_bottom.append(proj[0].weight.grad.norm().item())
            
            if proj_grads_top:
                logger.info(f"Epoch {epoch} Mask top-k proj grad norms: {proj_grads_top}")
            if proj_grads_bottom:
                logger.info(f"Epoch {epoch} Mask bottom-k proj grad norms: {proj_grads_bottom}")
```

### 5.2 训练后分析

```bash
python scripts/analyze_layer_preferences.py --checkpoint outputs/model.pth
```

**输出**:
- `layer_preference_comparison.png`: mask vs depth 的 12 层权重柱状图对比
- `layer_preference_topk.png`: 两个分支各自选中的 top-k 层标注

**分析维度**:
1. 全局层偏好：哪些层权重最高/最低
2. mask vs depth 差异：两组权重的相关性/差异性
3. 假设验证：低层（纹理）对 mask 更重要，高层（语义）对 depth 更重要

---

## 6. 超参数配置

### 6.1 默认配置

```python
FUSION_CONFIG = {
    "adapter_type": "dpt_fusion",
    "fusion_k": 4,              # top-k 推理时选择的层数
    "dpt_layers": list(range(12)),  # 全部 12 层作为候选
}
```

### 6.2 实验变量

| 参数 | 范围 | 说明 |
|------|------|------|
| `fusion_k` | [4, 6, 8] | 推理时 top-k 层数 |

注：无其他超参数需要调优。权重学习率与主网络共享，无需单独设置。

---

## 7. 兼容性

### 7.1 现有模式不变

| adapter_type | 行为 | 改动 |
|--------------|------|------|
| `"mona"` | 单层特征 + adapter | 无改动 |
| `"dpt_simple"` | 4 层 DPTHead | 无改动 |
| `"dpt_fusion"` (新增) | 12 层 LearnedLayerFusion | 本次实现 |

### 7.2 向后兼容

- `DINOv3MonaEncoder.forward()` 返回值保持 `(x, input_size)` 二元组
- `DepthSam.encode_images()` 返回值保持 `(image_embeddings, input_size)` 二元组
- 新增 `branch` 参数，默认值 `"mask"` 保持向后兼容
- 所有现有调用方无需任何修改

### 7.3 冻结兼容

- `freeze_image_encoder()`: 冻结 backbone（已有），fusion weights 和 projects 不受影响
- `freeze_depth_branch()`: 冻结 `depth_fusion`（需新增）+ outer_prompt_encoder + depth_decoder
- `freeze_mask_branch()`: 冻结 `mask_fusion`（需新增）+ SAM prompt_encoder + mask_decoder

需要在 `DepthSam` 中新增对 fusion head 的冻结逻辑：

```python
def freeze_depth_branch(self):
    ...
    if self.dinov3_encoder and self.dinov3_encoder.depth_fusion is not None:
        for param in self.dinov3_encoder.depth_fusion.parameters():
            param.requires_grad = False

def freeze_mask_branch(self):
    ...
    if self.dinov3_encoder and self.dinov3_encoder.mask_fusion is not None:
        for param in self.dinov3_encoder.mask_fusion.parameters():
            param.requires_grad = False
```

---

## 8. 测试计划

### 8.1 单元测试

**文件**: `tests/test_layer_fusion.py`

```python
import torch
import pytest
from dinosam.model.layer_fusion import LearnedLayerFusion


def test_output_shape():
    """验证输入输出 shape 正确。"""
    fusion = LearnedLayerFusion(embed_dim=768, num_layers=12, k=4)
    features = [torch.randn(2, 4096, 768) for _ in range(12)]
    fused = fusion(features, 64, 64)
    assert fused.shape == (2, 768, 64, 64)


def test_zero_init_uniform():
    """验证零初始化产生均匀权重。"""
    fusion = LearnedLayerFusion(embed_dim=768, num_layers=12, k=4)
    weights = fusion.get_weights()
    expected = torch.ones(12) / 12
    assert torch.allclose(weights, expected, atol=1e-6)


def test_training_uses_all_layers():
    """验证训练时所有层都参与了融合计算。"""
    fusion = LearnedLayerFusion(embed_dim=768, num_layers=12, k=4)
    fusion.train()
    features = [torch.randn(2, 4096, 768) for _ in range(12)]
    fused = fusion(features, 64, 64)
    loss = fused.sum()
    loss.backward()
    # 验证每个 projection 都接收了梯度（说明每个层都参与了融合）
    for l, proj in enumerate(fusion.layer_projects):
        conv = proj[0]  # Conv2d is first in Sequential
        assert conv.weight.grad is not None, f"Layer {l} projection received no gradient"
        assert conv.weight.grad.abs().sum() > 0, f"Layer {l} projection gradient is zero"


def test_eval_uses_topk():
    """验证推理时只使用 top-k 层。"""
    fusion = LearnedLayerFusion(embed_dim=768, num_layers=12, k=4)
    # 手动设置权重使 top-4 为 [0, 3, 7, 11]
    with torch.no_grad():
        fusion.layer_weights[0] = 10.0
        fusion.layer_weights[3] = 9.0
        fusion.layer_weights[7] = 8.0
        fusion.layer_weights[11] = 7.0
    topk = fusion.get_topk_indices().tolist()
    assert set(topk) == {0, 3, 7, 11}

    fusion.eval()
    # 只提供 top-k 层的特征
    features = [torch.randn(2, 4096, 768) for _ in range(4)]
    fused = fusion(features, 64, 64)
    assert fused.shape == (2, 768, 64, 64)


def test_gradient_flows_to_weights():
    """验证梯度能流向 layer_weights。"""
    fusion = LearnedLayerFusion(embed_dim=768, num_layers=12, k=4)
    fusion.train()
    features = [torch.randn(2, 4096, 768) for _ in range(12)]
    fused = fusion(features, 64, 64)
    loss = fused.sum()
    loss.backward()
    assert fusion.layer_weights.grad is not None
    assert fusion.layer_weights.grad.abs().sum() > 0
```

### 8.2 集成测试

```python
def test_encoder_fusion_mode_mask():
    """验证 encoder 在 fusion 模式下 mask 分支正确工作。"""
    encoder = DINOv3MonaEncoder(
        checkpoint_path="path/to/dinov3.pth",
        adapter_type="dpt_fusion",
        fusion_k=4,
    )
    encoder.train()
    images = torch.randn(2, 3, 1024, 1024)
    embeddings, input_size = encoder(images, branch="mask")
    assert embeddings.shape == (2, 256, 64, 64)


def test_encoder_fusion_mode_eval_topk():
    """验证 encoder 推理时只提取 top-k 层。"""
    encoder = DINOv3MonaEncoder(
        checkpoint_path="path/to/dinov3.pth",
        adapter_type="dpt_fusion",
        fusion_k=4,
    )
    encoder.eval()
    images = torch.randn(2, 3, 1024, 1024)
    embeddings, input_size = encoder(images, branch="depth")
    assert embeddings.shape == (2, 256, 64, 64)


def test_backward_compat_non_fusion():
    """验证非 fusion 模式返回值不变。"""
    encoder = DINOv3MonaEncoder(
        checkpoint_path="path/to/dinov3.pth",
        adapter_type="dpt_simple",
        dpt_layers=[2, 5, 8, 11],
    )
    images = torch.randn(2, 3, 1024, 1024)
    result = encoder(images)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0].shape == (2, 256, 64, 64)
```

---

## 9. 预期结果

### 9.1 层偏好预期

基于 ViT 文献的一般认知：
- **Mask 分支**：偏好中层（4-8），编码物体边界和纹理
- **Depth 分支**：偏好高层（8-11），编码语义和空间关系
- **低层（0-3）**：两个分支都可能权重较低，编码低级纹理

### 9.2 研究价值

训练后直接比较两组 12 维权重向量：
- Pearson 相关系数：mask 和 depth 的层偏好是否相关
- Top-k 重叠度：两个分支选中的层有多少重叠
- 权重分布形状：是集中在少数层还是均匀分布

---

## 10. 实现顺序

### Step 1: 核心模块 + 单元测试
1. 实现 `LearnedLayerFusion`（`dinosam/model/layer_fusion.py`）
2. 编写 `tests/test_layer_fusion.py`
3. 运行测试验证

### Step 2: Encoder 集成
1. 修改 `DINOv3MonaEncoder` 支持 `adapter_type="dpt_fusion"`
2. 添加 `branch` 参数（默认 `"mask"`）
3. 集成测试验证端到端流程

### Step 3: DepthSam 集成
1. 修改 `encode_images()` 添加 `branch` 参数
2. 修改 `forward()` 分别编码 mask 和 depth
3. 修改 `freeze_depth_branch()` / `freeze_mask_branch()` 处理 fusion head

### Step 4: 日志与分析
1. 添加 epoch 结束时层权重日志
2. 实现分析脚本 `scripts/analyze_layer_preferences.py`

---

## 11. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 权重收敛到均匀分布（无区分度） | 低 | 中 | 说明所有层同等重要，本身也是有价值的发现 |
| 推理 top-k 性能下降 | 中 | 中 | 调整 k 值；或推理时仍用全部层（全局权重推理也很快） |
| Train/eval gap（12层 vs top-k） | 中 | 中 | GroupNorm 减轻分布偏移；增大 k 值；或 eval 时也用全部层 |
| 12 层特征提取显存增加 | 中 | 低 | 训练时增加约 12×B×4096×768×4 ≈ 150MB，可减小 batch size |
| Phase 2 adapter 冻结导致 depth 特征不匹配 | 中 | 中 | 与现有 DPTHead 模式行为一致，不是新问题。如性能不佳，可尝试 Phase 2 解冻 adapter |
| 权重有效坍缩（单层权重 > 0.8） | 低 | 低 | 记录权重熵和最大值。如无辅助 loss 约束，坍缩可能反映真实层重要性差异 |

---

## 12. 与初始方案的对比

| | 初始方案（逐图像动态评分） | 当前方案（全局权重） |
|---|---|---|
| 核心代码 | ~200 行 + 辅助 loss | ~50 行 |
| 训练稳定性 | 需要防坍缩、温度退火 | 天然稳定 |
| 两阶段兼容 | 需要处理 adapter 共享矛盾 | 完全兼容 |
| 接口改动 | 返回值改为 dict，所有调用方更新 | 返回值不变 |
| 逐图像差异 | ✅ 能发现 | ❌ 不能（全局固定） |
| 稀疏推理 | ✅ top-k | ✅ top-k |
| 层可解释性 | 需统计平均 softmax | ✅ 直接读出 12 个权重 |

---

## 13. 参考文献

1. Shazeer et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (ICLR 2017)
2. Fedus et al. "Switch Transformers: Scaling to Trillion Parameter Models" (ICML 2022)

详细论文综述见 `docs/layer-fusion-related-papers.md`。

---

## 版本历史

- **v2.3** (2026-07-07): 修复 v2.2 review 发现的问题
  - M1: 补充 Phase 2 双 encode 说明（train/validate/iterative 全部需要 mask_embeddings + depth_embeddings）
  - M2: 补充投影梯度监控，检测"权重活着但投影死了"的静默失败
  - M3: 风险表中 InstanceNorm → GroupNorm
  - M4: 修复章节编号重复（两个 4.4 → 4.5/4.8）
  - m1: 修复 §3.1 注释缺字
  - 新增：§4.5 完整的调用点修改矩阵

- **v2.3** (2026-07-07): 最终审批
  - 补充 §4.1 dpt_fusion 分支显式创建 adapter1/adapter2（使用 mona adapter）
  - 补充 §4.1 dpt_fusion 分支显式设置 dpt_head = None
  - **状态**: ✅ 已审批，待实现

- **v2.2** (2026-07-07): 修复 v2.1 review 发现的问题
  - C1: Phase 2 run_phase2() blanket freeze 后显式解冻 depth_fusion
  - C2: forward() 中 mask 和 depth 各调用一次 encode_images
  - M1: dpt_layers guard 允许 dpt_fusion + argparse choices 新增 dpt_fusion
  - M2: DepthSam.__init__ 传递 fusion_k 到 encoder
  - M3: InstanceNorm2d → GroupNorm（避免 B=1 时 NaN）
  - M4: 添加权重梯度监控日志

- **v2.1** (2026-07-07): 修复初版 review 发现的问题
  - C1: forward() 改为双 encode（mask/depth 各一次）
  - M1: 训练循环所有 encode_images 调用点添加 branch 参数
  - M2: _build_model_and_data 传递 fusion_k
  - M3: 投影添加 GroupNorm 解决初始化尺度偏差
  - M4: 训练/推理 gap 记录为已知风险
  - m1: 训练路径 assert 输入长度
  - m2: 测试断言改为检查梯度流

- **v2.0** (2026-07-07): 初版设计
  - 从 v1.0 的逐图像动态评分简化为全局可学习权重 + top-k 推理
  - 核心思想：12 个可学习标量权重 + softmax → 加权融合（训练）/ top-k 选择（推理）
  - 两个独立 fusion head（mask_fusion / depth_fusion），Phase 1/2 分别训练
  - 无需辅助 loss，完全兼容现有两阶段训练

- **v1.0** (2026-07-07): 初始探索
  - 逐图像动态评分（输入相关）+ Gumbel-TopK + 稀疏融合
  - 借鉴 MoE 路由机制，使用 Switch Transformer Eq.4 负载均衡 loss
  - 被 review 否决：接口不兼容、训练循环不匹配、Gumbel-STE 梯度方差高
