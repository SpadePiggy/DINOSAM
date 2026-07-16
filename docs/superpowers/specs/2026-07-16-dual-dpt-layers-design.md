# Dual DPT Layers Design — depth 头与 mask 头独立层配置

**Date:** 2026-07-16
**Status:** Approved (rev 2 — 吸收三方 review 发现)

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

**校验（落在 CLI/build 层，不得静默忽略）：**
- `--dpt_layers_depth` 必须**无条件解析并透传**——不得沿用现有
  `... if args.adapter_type == "dpt_simple" else None` 的条件解析模式
  （那会让校验永不触发）
- 解析后立即校验组合：提供了 `--dpt_layers_depth` 但
  `encoder != "dinov3"` 或 `adapter_type != "dpt_simple"` → 直接报错退出
  （`--encoder sam` 时 DINOv3MonaEncoder 不会被构造，encoder 侧校验够不到，
  所以必须在 CLI 层拦截）
- encoder 侧保留防御性 ValueError（扩展 `dinov3_encoder.py` 现有 dpt_layers 校验）
- `dpt_layers` 与 `dpt_layers_depth` 各自必须严格升序且无重复，否则报错。
  背景：backbone `get_intermediate_layers` 恒按块索引升序返回特征，乱序/重复
  配置会造成新旧路径行为不一致。这是新增校验；以往的合法配置（升序）不受影响。

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
features_mask, features_depth = split_union_features(features, union,
                                                     dpt_layers, dpt_layers_depth)
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
- `forward_mask` / `forward_depth` 的**函数体第一条语句**做分支选择
  （必须先于 `B = image_embeddings.shape[0]` 和
  `depth_transformer_type == "simple"` 分支，否则 dict 会被直接当 Tensor 用）：

```python
if isinstance(image_embeddings, dict):
    image_embeddings = image_embeddings["mask"]   # forward_depth 取 "depth"
```

- `encode_images`、`forward` 及 trainer/iterative/evaluate 中全部 8 个
  `encode_images` 调用点零改动——调用点把 dict 原样传给两个分支函数，
  分支函数自取所需 embedding。核实：8 个调用点在 encode 与分支函数之间
  均不触碰 embeddings。
- 第 9 个调用点 `dinosam/scripts/pca_viz.py` 永远走单头路径，不受影响
  （加载双头 checkpoint 时会命中下述 `_depth` 键不匹配报错，行为明确）

**代价（已接受）：** phase1（纯 mask 训练）双头模式下会多计算一份 depth 分支的
head + adapter + projection 前向，相对 12 层 ViT backbone 开销 <5%。
`*_depth` 模块在 phase1 被冻结（见下节），不产生梯度与激活保存开销。

## 训练阶段的冻结/解冻语义（双头模式）

review 发现的关键缺口：不改冻结逻辑，`*_depth` 模块永远不会被训练。

| 阶段 | `*_depth` 编码模块（dpt_head_depth / adapter1_depth / adapter2_depth / projection_depth） |
|---|---|
| phase1（纯 mask） | **冻结**——扩展 `DepthSam.freeze_depth_branch()`：双头模式下额外冻结这四个模块 |
| phase2（depth 训练） | **解冻**——`run_phase2` 现有"全冻结 → 解冻 outer_prompt_encoder + depth_decoder"逻辑中，双头模式下把这四个模块加入解冻集合 |

phase2 的 mask 分支编码模块（`dpt_head` / `adapter1` / `adapter2` / `projection`）
维持现状（冻结），不受本设计影响。

## checkpoint 兼容与 warm-start

**checkpoint 元数据：** `_save_checkpoint` 新增保存 `dpt_layers` 与
`dpt_layers_depth`（沿用现有 `adapter_type` 字段的先例），便于加载时诊断。

| 场景 | 行为 |
|---|---|
| 开关关闭 + 旧单头 checkpoint | state_dict 键一致，照常加载，零影响 |
| 开关关闭 + 双头 checkpoint | **报错**：检测到 unexpected_keys 中含 `dinov3_encoder.` 前缀的 `*_depth` 模块键 → 提示补 `--dpt_layers_depth`（防止 depth_decoder 拿 mask embedding 安静地输出错误深度） |
| 双头模式 + phase2 加载 phase1 checkpoint（无论单头还是双头产物） | `strict=False` 加载后**无条件**执行 warm-start。理由：phase1 从不训练 `*_depth` 模块，即使双头 phase1 checkpoint 里存了这些键也只是随机初始值，覆盖拷贝恒正确。不依赖 missing_keys 判断（双头 phase1 产物 missing_keys 为空，条件触发会失效） |
| 双头模式 + resume 双头 `*_last.pt` | 正常续训，不 warm-start |
| 双头模式 + resume 旧单头 `*_last.pt` | **明确报错，不支持**：模型 state_dict 缺 `*_depth` 键且 optimizer param_groups 数量不匹配（`optimizer.load_state_dict` 会崩溃）。检测到该情形直接报错并提示：删除/改名 last.pt，从 phase1 checkpoint warm-start 重新开始 phase2 |
| 双头模式 + evaluate 加载单头 checkpoint | **报错**：missing_keys 含 `*_depth` 模块键 → 提示该 checkpoint 为单头产物，去掉 `--dpt_layers_depth` 评估，或改用双头 checkpoint。evaluate **不做** warm-start（eval 不训练，随机 depth 分支的深度输出无意义） |

`_depth` 键的检测按 `dinov3_encoder.` 前缀 + 四个模块名精确匹配，
不用子串包含（`simple_depth_head.*` 也含 `_depth`，子串匹配会误伤）。

**warm-start**（新方法 `DINOv3MonaEncoder.warm_start_depth_branch()`）：
- `adapter1_depth ← adapter1`、`adapter2_depth ← adapter2`、
  `projection_depth ← projection` 复制
- `dpt_head_depth` **恒保持随机初始化**，不复制。理由：本特性的动机场景
  就是两组层数不同，同层数复制分支在真实用法中几乎不触发（review 采纳的
  YAGNI 砍法）
- 调用时机：仅 trainer `run_phase2` 加载 phase1 checkpoint 之后（双头模式下
  无条件调用），evaluate 不调用

## 测试

1. `tests/test_dual_dpt.py`：纯张量测试 `split_union_features` —
   并集顺序、分发正确性，覆盖两组层重叠 / 无重叠 / 完全相同三种情况
2. `tests/test_dpt_heads.py` 已覆盖 DPTHead 任意层数，不动
3. 手动 smoke 验证（需真实 DINOv3 权重，写进实现计划）：
   - 双头模式 forward 输出 shape 为 (B, 256, 64, 64) × 2
   - 开关关闭时 state_dict 键集合与旧版一致——用脚本化断言
     （加载模型比对键集合），不靠肉眼
   - 双头 phase2 下可训练参数日志应包含 `*_depth` 参数量
     （同时验证解冻逻辑生效）

## 明确不做（YAGNI）

- depth 分支的 PCA 可视化
- 单头 → 双头 checkpoint 的离线迁移脚本（warm-start 在加载时完成）
- phase1 下跳过 depth 分支前向计算的按需编码（冻结已消除梯度开销）
- `dpt_head_depth` 的同层数 warm-start 复制
- evaluate 侧 warm-start（以双向 `_depth` 键不匹配报错代替）
