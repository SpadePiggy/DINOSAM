# DualAttnTransformer 替换 TwoWayTransformer 设计文档

## 目标

为论文消融实验，在 DepthMaskDecoder 中创建 DualAttnTransformer 作为 TwoWayTransformer 的替代模块。通过 CLI 参数切换，其余训练/评估流程不变。

**消融变量**：唯一的架构差异是**每层的第一步**——TwoWayTransformer 用 token self-attention，DualAttnTransformer 用 PAM+CAM 对 image features 的增强。其余所有"脚手架"（LayerNorm、token PE、cross-attention、MLP、final_attn）保持一致，确保消融公平。

## 背景

### 当前架构

`DepthMaskDecoder` 使用 SAM 的 `TwoWayTransformer`（2 层）作为核心注意力模块。

**TwoWayAttentionBlock 每层结构**（`transformer.py:151-182`）：
1. Token self-attention（第一层 `skip_first_layer_pe=True`，不加 PE）+ norm1
2. Token→Image cross-attention：`Q=queries+query_pe, K=keys+key_pe, V=keys` + norm2
3. MLP on tokens + norm3
4. Image→Token cross-attention：`Q=keys+key_pe, K=queries+query_pe, V=queries` + norm4

**循环后**（`transformer.py:99-104`）：
5. `final_attn_token_to_image`：`Q=queries+point_embedding, K=keys+image_pe, V=keys` + `norm_final_attn`

调用接口（`depth_decoder.py:74`）：
```python
hs, src = self.transformer(src, pos_src, tokens)
# src: (B, C, H, W) → 内部 flatten 为 (B, HW, C)
# pos_src: (B, C, H, W) → 内部 flatten 为 (B, HW, C)
# tokens: (B, N_tokens, C)
# 返回 hs: (B, N_tokens, C), src: (B, HW, C)
```

后续使用（`depth_decoder.py:75-90`）：
- `hs[:, 0, :]` → depth_token_out → depth_prediction_head → 深度预测
- `hs[:, 1:1+num_mask_tokens, :]` → mask_tokens_out → hypernetwork → mask 预测
- `src.transpose(1,2).view(b,c,h,w)` → output_upscaling → 上采样特征

### 已有组件

`PAM_Module` 和 `CAM_Module`（`dinosam/model/adapters/dual_attn_adapter.py`）：
- PAM：位置注意力，输入 `(B,C,H,W)`，计算 `(B,N,N)` 空间注意力矩阵，可学习 `gamma`（初始 0）
- CAM：通道注意力，输入 `(B,C,H,W)`，计算 `(B,C,C)` 通道注意力矩阵，可学习 `gamma`（初始 0）
- 两者均无内部残差

SAM 的 `Attention` 类（`transformer.py:185-240`）：支持 `downsample_rate` 的多头注意力。
SAM 的 `MLPBlock`（`common.py`）：两层 FC + 激活函数。

## 设计

### 新模块：DualAttnTransformer

**文件**：`dinosam/model/dual_attn_transformer.py`

**构造参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `embedding_dim` | int | 256 | 特征维度，对齐 TwoWayTransformer |
| `num_heads` | int | 8 | attention 头数 |
| `mlp_dim` | int | 2048 | MLP 隐藏层维度 |
| `depth` | int | 2 | 堆叠层数，对齐 TwoWayTransformer |
| `activation` | Type[nn.Module] | nn.ReLU | MLP 激活函数，对齐 TwoWayTransformer |
| `attention_downsample_rate` | int | 2 | cross-attention 的 Q/K/V 缩减倍率，对齐 TwoWayTransformer |

注：PAM 的 `factor` 使用 PAM_Module 默认值 8，不额外暴露。

### 与 TwoWayTransformer 的逐步对比

| 步骤 | TwoWayAttentionBlock | DualAttnTransformerLayer | 差异说明 |
|------|---------------------|-------------------------|----------|
| 1 | Token self-attn + norm1 | PAM+CAM on src + norm1 | **唯一消融变量**：self-attn 作用于 tokens，PAM+CAM 作用于 image features |
| 2 | Token→Image cross-attn + norm2 | Token→Image cross-attn + norm2 | 完全一致，使用 SAM Attention 类 |
| 3 | MLP on tokens + norm3 | MLP on tokens + norm3 | 完全一致，使用 SAM MLPBlock |
| 4 | Image→Token cross-attn + norm4 | Image→Token cross-attn + norm4 | 完全一致 |
| final | final_attn_token_to_image + norm_final_attn | final_attn_token_to_image + norm_final_attn | 完全一致 |

### DualAttnTransformerLayer 每层结构

```
输入: queries (B, N, C), keys (B, HW, C), query_pe (B, N, C), key_pe (B, HW, C)
额外参数: hw_shape = (H, W)

1. PAM + CAM 增强 image features（替代 token self-attention）:
   keys_2d = keys.reshape(B, H, W, C).permute(0, 3, 1, 2)   → (B, C, H, W)
   enhanced = PAM(keys_2d) + CAM(keys_2d)                     （不除以2，gamma自己学）
   enhanced = enhanced.permute(0, 2, 3, 1).reshape(B, HW, C)
   keys = keys + enhanced                                      （残差连接）
   keys = norm1(keys)

2. Token→Image cross-attention（与 TwoWayAttentionBlock 完全一致）:
   q = queries + query_pe
   k = keys + key_pe
   attn_out = cross_attn_token_to_image(q=q, k=k, v=keys)
   queries = queries + attn_out
   queries = norm2(queries)

3. MLP on tokens（与 TwoWayAttentionBlock 完全一致）:
   mlp_out = mlp(queries)
   queries = queries + mlp_out
   queries = norm3(queries)

4. Image→Token cross-attention（与 TwoWayAttentionBlock 完全一致）:
   q = queries + query_pe
   k = keys + key_pe
   attn_out = cross_attn_image_to_token(q=k, k=q, v=queries)
   keys = keys + attn_out
   keys = norm4(keys)

输出: queries (B, N, C), keys (B, HW, C)
```

### DualAttnTransformer forward 签名

与 TwoWayTransformer 完全一致：

```python
def forward(
    self,
    image_embedding: Tensor,  # (B, C, H, W)
    image_pe: Tensor,          # (B, C, H, W)
    point_embedding: Tensor,   # (B, N, C)
) -> Tuple[Tensor, Tensor]:   # (queries: (B, N, C), keys: (B, HW, C))
```

内部流程：
1. 保存 `h, w = image_embedding.shape[2:]`，传给每层用于 PAM/CAM 的 2D reshape
2. `image_embedding` 和 `image_pe` 从 `(B,C,H,W)` flatten 为 `(B,HW,C)`
3. `queries = point_embedding`, `keys = image_embedding`
4. 逐层处理：`queries, keys = layer(queries, keys, query_pe=point_embedding, key_pe=image_pe, hw_shape=(h,w))`
5. **final_attn_token_to_image**：`Q=queries+point_embedding, K=keys+image_pe, V=keys` → 残差 + norm_final_attn
6. 返回 `(queries, keys)`

注意：`query_pe` 始终是原始的 `point_embedding`（与 TwoWayTransformer 一致），不随层更新。

### 关于 token self-attention 的去除

TwoWayAttentionBlock 的第一步是 token self-attention（tokens attend to tokens），其作用是让 depth_token、mask_token 和 point prompt tokens 之间建立关联。

DualAttnTransformer 将此替换为 PAM+CAM（image features attend to image features），这是两种不同的注意力策略：
- Token self-attn：少量 token 之间的交互（N_tokens 通常 < 10）
- PAM+CAM：4096 个空间位置之间的交互 + 256 个通道之间的交互

**这正是消融实验要验证的核心假设**：PAM+CAM 对 image features 的增强能否弥补（甚至超越）token self-attention 的缺失。

### 实现组件

- **Cross-attention**：复用 SAM 的 `Attention` 类（`segment_anything.modeling.transformer.Attention`），`downsample_rate=2`，对齐参数量
- **MLP**：复用 SAM 的 `MLPBlock`（`segment_anything.modeling.common.MLPBlock`）
- **LayerNorm**：标准 `nn.LayerNorm(embedding_dim)`，每层 4 个 + final 1 个
- **PAM/CAM**：直接使用 `PAM_Module` 和 `CAM_Module`，不包 DualAttnAdapter

### PAM 显存说明

PAM 在 64x64 特征图上生成 `(B, 4096, 4096)` 注意力矩阵，FP32 下约 64MB/sample。`factor=8` 缩减 Q/K 通道维度（256→32）但不影响空间维度的注意力矩阵大小。当前训练 batch_size 较小（通常 2-4），显存可接受。

### 参数量对比

| 组件 | TwoWayAttentionBlock | DualAttnTransformerLayer | 差异 |
|------|---------------------|-------------------------|------|
| Step 1 | self_attn (downsample=1): ~263K | PAM+CAM: ~82K | -181K |
| norm1 | 512 | 512 | 0 |
| cross_attn_token_to_image | ~132K | ~132K | 0 |
| norm2 | 512 | 512 | 0 |
| MLP | ~1,051K | ~1,051K | 0 |
| norm3 | 512 | 512 | 0 |
| cross_attn_image_to_token | ~132K | ~132K | 0 |
| norm4 | 512 | 512 | 0 |
| **每层合计** | **~1,580K** | **~1,399K** | **-181K** |

| | TwoWayTransformer | DualAttnTransformer |
|---|---|---|
| 2 层 | ~3,160K | ~2,798K |
| final_attn + norm | ~132K | ~132K |
| **总计** | **~3,292K (~3.3M)** | **~2,930K (~2.9M)** |

参数量差异约 11%（DualAttn 略少），主要因为 PAM+CAM（82K）比 token self-attn（263K）轻。对消融实验而言这个差距可接受，应在论文表格中报告。

## 集成改动

### depth_sam.py

1. `__init__` 新增 `depth_transformer_type: str = "twoway"` 参数，并保存属性：

```python
def __init__(self, sam: Sam, encoder_type: str = "sam",
             dinov3_checkpoint: str = None, adapter_type: str = "mona",
             depth_transformer_type: str = "twoway"):
    ...
    self.depth_transformer_type = depth_transformer_type

    if depth_transformer_type == "dual_attn":
        from .dual_attn_transformer import DualAttnTransformer
        transformer = DualAttnTransformer(
            embedding_dim=prompt_embed_dim, num_heads=8,
            mlp_dim=2048, depth=2,
        )
    else:
        transformer = TwoWayTransformer(
            depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048, num_heads=8,
        )
    self.depth_decoder = DepthMaskDecoder(
        transformer_dim=prompt_embed_dim, transformer=transformer,
    )
```

2. `init_depth_from_sam` 中显式跳过 transformer 权重迁移：

```python
# Transformer 权重
if self.depth_transformer_type != "dual_attn":
    for key in list(dd_sd.keys()):
        if key.startswith("transformer."):
            md_key = key
            if md_key in md_sd:
                dd_sd[key] = md_sd[md_key].clone()
else:
    print("DualAttn mode: skipping transformer weight migration (random init)")
```

### trainer.py

1. 新增 CLI 参数（与现有参数风格一致，使用下划线）：

```python
parser.add_argument(
    "--depth_transformer_type", type=str, default="twoway",
    choices=["twoway", "dual_attn"],
    help="Depth decoder transformer type. dual_attn requires training from scratch.",
)
```

2. `_build_model_and_data` 中透传：

```python
model = DepthSam(
    sam,
    encoder_type=args.encoder,
    dinov3_checkpoint=args.dinov3_checkpoint,
    adapter_type=args.adapter_type,
    depth_transformer_type=args.depth_transformer_type,
)
```

### evaluate.py

1. 同样新增 CLI 参数
2. **注意**：evaluate.py 中有两处模型构建（Phase 1 和 Phase 2 各一次），两处都需要透传 `depth_transformer_type`

## 不改动的文件

| 文件 | 原因 |
|------|------|
| `depth_decoder.py` | 通过 `transformer: nn.Module` 注入，接口兼容即零改动 |
| `iterative.py` | 通过 `model.forward_mask()` / `model.forward_depth()` 调用，不直接接触 transformer |
| `dual_attn_adapter.py` | 只 import PAM_Module 和 CAM_Module，不修改 |

## 初始化策略

- DualAttnTransformer（PAM、CAM、cross-attention、MLP、LayerNorm、final_attn）：随机初始化
- depth_token：仍从 SAM iou_token 迁移
- mask_tokens：仍从 SAM mask_tokens[0] 迁移
- output_upscaling：仍从 SAM 迁移
- output_hypernetworks_mlps：仍从 SAM 迁移
- depth_prediction_head：随机初始化（与 twoway 模式一致）

## Checkpoint 兼容性

`depth_transformer_type=dual_attn` 的模型与 `twoway` 的 checkpoint 不兼容（state_dict key 不同，`load_state_dict(strict=True)` 会自然报错）。消融实验应从头训练，不支持跨类型 `--resume`。

## 改动文件清单

| 文件 | 操作 | 改动量 |
|------|------|--------|
| `dinosam/model/dual_attn_transformer.py` | 新建 | ~120 行 |
| `dinosam/model/depth_sam.py` | 修改 | ~15 行 |
| `dinosam/train/trainer.py` | 修改 | ~5 行 |
| `dinosam/evaluate.py` | 修改 | ~8 行（含两处模型构建） |

总计：1 个新文件 + 3 个文件修改，约 148 行改动。
