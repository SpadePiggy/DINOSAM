# Dual DPT Layers Design — depth 头与 mask 头独立层配置

**Date:** 2026-07-16
**Status:** Approved

## 目标

允许 mask 分支（SAM mask_decoder）与 depth 分支（depth_decoder）使用不同的
DINOv3 中间层组合构建各自的 image embedding。当前两个分支共享同一个
`dpt_head` 输出；本设计在 `adapter_type="dpt_simple"` 下新增可选的双头模式。

**约束：**
- 完全向后兼容：不启用开关时，模型结构、state_dict 键、CLI 行为与现状逐字节一致
- backbone 只前向一次：`get_intermediate_layers` 取两组层的并集，再分发给两个 head
- mona adapter 与 projection 每分支独立一套

## CLI 接口

trainer.py 与 evaluate.py 各新增一个参数：

```
--dpt_layers        "2,5,8,11"   # 不变；双头模式下作为 mask 分支的层配置
--dpt_layers_depth  "5,8,11"     # 新增；提供即开启双头模式
```

- `--dpt_layers_depth` 仅在 `--adapter_type dpt_simple` 下合法，否则报
  ValueError（扩展 `dinov3_encoder.py` 现有的 dpt_layers 校验，不静默忽略）
- 不提供 `--dpt_layers_depth` → 完全走旧路径

## 架构

### DINOv3MonaEncoder

`__init__` 新增参数 `dpt_layers_depth: list[int] = None`。

| 模式 | 模块 |
|---|---|
| 单头（`dpt_layers_depth=None`，默认） | `dpt_head` / `adapter1` / `adapter2` / `projection` — 与现状一致 |
| 双头 | mask 分支保留原名；depth 分支新增 `dpt_head_depth` / `adapter1_depth` / `adapter2_depth` / `projection_depth` |

mask 分支保留原参数名是向后兼容的关键：
- 开关关闭时 state_dict 与旧版完全一致
- 双头模式下用 `strict=False` 加载旧单头 checkpoint 时，旧权重自动装入 mask 分支
- `viz/pca.py` 引用的 `encoder.dpt_head` / `encoder.dpt_layers` 属性继续有效
  （PCA 可视化展示 mask 分支，不改动）

### forward 数据流（双头模式）

```
union = sorted(set(dpt_layers) | set(dpt_layers_depth))
features = backbone.get_intermediate_layers(x, n=union)      # backbone 只跑一次
features_mask  = [features[union.index(l)] for l in dpt_layers]
features_depth = [features[union.index(l)] for l in dpt_layers_depth]
每分支独立: dpt_head → adapter1 → adapter2 → projection → (B, 256, 64, 64)
返回 ({"mask": emb_m, "depth": emb_d}, input_size)
```

单头模式返回值不变：`(emb, input_size)`。

union 计算 + 索引分发提取为模块级纯函数（便于单测）：

```python
def split_union_features(features, union, layers_a, layers_b):
    """按 union 中的索引把一次性提取的 features 分发给两组层配置。"""
```

### DepthSam

- `__init__` 新增 `dpt_layers_depth=None`，透传给 DINOv3MonaEncoder
- `forward_mask` / `forward_depth` 开头各加分支选择：

```python
if isinstance(image_embeddings, dict):
    image_embeddings = image_embeddings["mask"]   # forward_depth 取 "depth"
```

- `encode_images`、`forward` 及 trainer/iterative/evaluate 中全部 8 个
  `encode_images` 调用点零改动——调用点把 dict 原样传给两个分支函数，
  分支函数自取所需 embedding

**代价（已接受）：** phase1（纯 mask 训练）双头模式下会多计算一份 depth 分支的
head + adapter + projection，相对 12 层 ViT backbone 开销 <5%。

## checkpoint 兼容与 warm-start

| 场景 | 行为 |
|---|---|
| 开关关闭 | state_dict 键与旧版一致，旧 checkpoint 照常加载 |
| 双头模式加载旧单头 checkpoint | `strict=False` 把旧键装入 mask 分支；`*_depth` 键出现在 missing_keys → 触发 warm-start |
| 双头模式续训（resume 双头 checkpoint） | 所有键命中，不触发 warm-start |

**warm-start**（新方法 `DINOv3MonaEncoder.warm_start_depth_branch()`）：
- `adapter1_depth ← adapter1`、`adapter2_depth ← adapter2`、
  `projection_depth ← projection` 无条件复制
- `dpt_head_depth ← dpt_head` 仅当 `len(dpt_layers_depth) == len(dpt_layers)`
  时复制（结构相同），否则保持随机初始化并打印提示

**调用时机：** trainer（phase2 加载 phase1 checkpoint 后）与 evaluate
（加载 checkpoint 后），条件 = 双头模式 且 missing_keys 中含 `_depth` 键。

## 测试

1. `tests/test_dual_dpt.py`：纯张量测试 `split_union_features` —
   并集顺序、分发正确性，覆盖两组层重叠 / 无重叠 / 完全相同三种情况
2. `tests/test_dpt_heads.py` 已覆盖 DPTHead 任意层数，不动
3. 手动 smoke 验证（需真实 DINOv3 权重，写进实现计划）：
   - 双头模式 forward 输出 shape 为 (B, 256, 64, 64) × 2
   - 开关关闭时 state_dict 键集合与旧版一致（回归检查）

## 明确不做（YAGNI）

- depth 分支的 PCA 可视化
- 单头 → 双头 checkpoint 的离线迁移脚本（warm-start 在加载时完成）
- phase1 下跳过 depth 分支计算的按需编码
