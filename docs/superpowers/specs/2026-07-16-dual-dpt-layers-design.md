# Dual DPT Layers Design — depth 头与 mask 头独立层配置

**Date:** 2026-07-16
**Status:** Approved (rev 3.1 — 三轮三方 review 收敛)

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

**解析与校验（落在 CLI/build 层，不得静默忽略）：**
- `--dpt_layers_depth` 必须**无条件解析并透传**——不得沿用现有
  `... if args.adapter_type == "dpt_simple" else None` 的条件解析模式
  （那会让校验永不触发）。evaluate.py 有两处模型构建点，共用一次解析+校验
- 解析后立即校验组合：提供了 `--dpt_layers_depth` 但
  `encoder != "dinov3"` 或 `adapter_type != "dpt_simple"` → 直接报错退出
  （`--encoder sam` 时 DINOv3MonaEncoder 不会被构造，encoder 侧校验够不到，
  所以必须在 CLI 层拦截）。encoder 侧现有 dpt_layers 校验原样不动，不扩展
- 两组层配置解析后各自做 `sorted(set(...))` 归一化（一行）。背景：backbone
  `get_intermediate_layers` 恒按块索引升序返回特征，归一化使配置顺序与实际
  特征顺序一致；对以往的合法（升序）配置无行为变化

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
- 第 9 个调用点 `dinosam/scripts/pca_viz.py` 永远走单头路径，不改动。
  它用裸 `load_state_dict(strict=False)` 加载：遇到双头 checkpoint 时
  `*_depth` 键被静默忽略，mask 分支权重正常装入——行为无害，接受现状

**代价（已接受）：** phase1（纯 mask 训练）双头模式下会多计算一份 depth 分支的
head + adapter + projection 前向，相对 12 层 ViT backbone 开销 <5%。
`*_depth` 模块在 phase1 被冻结（见下节），不产生梯度与激活保存开销。

## 训练阶段的冻结/解冻语义（双头模式）

review 发现的关键缺口：不改冻结逻辑，`*_depth` 模块永远不会被训练。

| 阶段 | `*_depth` 编码模块（dpt_head_depth / adapter1_depth / adapter2_depth / projection_depth） |
|---|---|
| phase1（纯 mask） | **冻结**——扩展 `DepthSam.freeze_depth_branch()`：双头模式下额外冻结这四个模块。注意扩展必须放在 simple 早退分支（代码字面条件为 `self.outer_prompt_encoder is None`）**之前**，两种 transformer 类型都要覆盖 |
| phase2（depth 训练） | **解冻**——`run_phase2` 现有"全冻结 → 解冻"逻辑之后，独立追加一段"双头则解冻四模块"（放在 simple/非 simple 的 if/else **之外**，与 phase1 冻结侧同构，避免两条分支重复写四个模块） |

phase2 的 mask 分支编码模块（`dpt_head` / `adapter1` / `adapter2` / `projection`）
维持现状（冻结），不受本设计影响。

## checkpoint 兼容与 warm-start

**统一检测 helper：** 加载 state_dict 后，用一个 helper 检查
missing_keys / unexpected_keys 中是否存在 `_depth` 分支键——判据为
`key.startswith("dinov3_encoder.") and "_depth." in key`（`simple_depth_head`
挂在 DepthSam 顶层，无 `dinov3_encoder.` 前缀，不会误伤）。三处报错场景
共用该 helper，动作因调用点而异（见下表）。

| 场景 | 行为 |
|---|---|
| 开关关闭 + 旧单头 checkpoint | state_dict 键一致，照常加载，零影响 |
| 开关关闭 + 双头 checkpoint（trainer resume / phase2 加载 phase1） | **报错**：unexpected_keys 含 `_depth` 分支键 → 静态错误消息同时列出 `--dpt_layers_depth` 与 `--encoder dinov3`（不做条件分支）。防止 depth_decoder 拿 mask embedding 安静地输出错误深度。phase1 resume 方向亦有意报错——机械上可续训，但报错提示配置漂移，属保守选择 |
| 双头模式 + phase2 加载 phase1 checkpoint（无论单头还是双头产物） | `strict=False` 加载后**无条件**执行 warm-start，不报错。理由：phase1 从不训练 `*_depth` 模块，即使双头 phase1 checkpoint 里存了这些键也只是随机初始值，覆盖拷贝恒正确。不依赖 missing_keys 判断（双头 phase1 产物 missing_keys 为空，条件触发会失效） |
| 双头模式 + resume 双头 `*_last.pt` | 正常续训，不 warm-start |
| 双头模式 + **phase2** resume 旧单头 `phase2_last.pt` | **明确报错，不支持**：missing_keys 含 `_depth` 分支键即报错（背后原因：phase2 optimizer 含 `*_depth` 参数，`optimizer.load_state_dict` 会因组内 param 个数不匹配抛错）。提示：删除/改名 last.pt，从 phase1 checkpoint warm-start 重新开始 phase2。**作用域仅 phase2**——`_load_checkpoint` 为两个 phase 共用，phase1 resume 不做此检查 |
| 双头模式 + **phase1** resume 旧单头 `phase1_last.pt` | **安全，正常续训**：`*_depth` 在 phase1 冻结、不进 optimizer，param_groups 一致；`*_depth` 保持随机初始化，后续 phase2 会 warm-start 覆盖 |
| evaluate 遇到任一方向不匹配（单头产物+传了 `--dpt_layers_depth`，或双头产物+没传） | **打印明确错误并跳过该 phase，继续评下一个 checkpoint，不硬退出**。理由：evaluate 一次运行依次评 phase1/phase2 且共用 CLI 参数，推荐工作流产物是"单头 phase1 + 双头 phase2"混合对，必有一个不匹配；跳过策略保证任何 checkpoint × 参数组合都有定义好的行为，旧模型（不传新参数）完全不受影响。evaluate **不做** warm-start（eval 不训练，随机 depth 分支的深度输出无意义）。**配套改动**：脚本尾部 Phase1/Phase2 对比与绘图逻辑的门槛须从"checkpoint 文件存在"改为"该 phase 实际产出 metrics"，否则被跳过的 phase 会让 `metrics_p1`/`metrics_p2` 未定义而 NameError |

**warm-start**（新方法 `DINOv3MonaEncoder.warm_start_depth_branch()`）：
- `adapter1_depth ← adapter1`、`adapter2_depth ← adapter2`、
  `projection_depth ← projection` 复制
- `dpt_head_depth` **恒保持随机初始化**，不复制。理由：本特性的动机场景
  就是两组层数不同，同层数复制分支在真实用法中几乎不触发
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
- evaluate 侧 warm-start（以 `_depth` 键不匹配"报错跳过"代替）
- checkpoint 元数据记录 `dpt_layers` / `dpt_layers_depth`（用户裁决：删掉。
  接受"同层数、不同索引"配置错配时键名/shape 全一致、无法检测、静默按错误
  层语义运行的风险——mask 分支现状亦如此，两边保持一致）
- encoder 侧防御性 ValueError 扩展（CLI/build 层已拦截全部到达路径）
