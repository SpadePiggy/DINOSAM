# SimpleDepthHead 设计文档

## 目标

新增 `--depth_transformer_type simple` 消融选项。用 GlobalAvgPool + MLP 瓶颈直接从 image_embeddings 预测深度，绕过 outer_prompt_encoder 和 DepthMaskDecoder，验证"mask 监督信号和 transformer 交互是否对深度预测必要"。

**内层 SAM（prompt_encoder + mask_decoder）完全不变**，Phase 1 mask 训练不受影响。

**约束**：Phase 1 和 Phase 2 必须使用相同的 `--depth_transformer_type`，跨类型 checkpoint 不兼容。

## 架构对比

| | twoway / dual_attn | simple |
|---|---|---|
| Phase 1 | mask 训练（不变） | mask 训练（不变） |
| Phase 2 输入 | image_emb + mask(from inner SAM) | image_emb only |
| Phase 2 模块 | outer_prompt_encoder → DepthMaskDecoder | nn.Sequential MLP |
| Phase 2 输出 | depth_pred + decoder_masks | depth_pred only |
| mask 监督 | decoder mask vs GT dice loss | 无 |
| 参数量 | ~4.7M (decoder) | ~0.53M (MLP) |

## 设计

不建新类，不建新文件，不建新函数。在 `DepthSam.__init__` 中用 `nn.Sequential` 内联，`forward_depth` 的 early return 处理一切，现有训练循环原样工作。

## 改动清单

### 1. depth_sam.py (~18 行)

#### `__init__` — 条件创建（包裹 L49-68）

```python
if depth_transformer_type == "simple":
    self.simple_depth_head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(256, 1024), nn.ReLU(),
        nn.Linear(1024, 256), nn.ReLU(),
        nn.Linear(256, 1),
    )
    self.outer_prompt_encoder = None
    self.depth_decoder = None
else:
    # 原有逻辑（outer_prompt_encoder + depth_decoder 创建，不动）
```

#### `forward_depth` early return — 不修改签名

```python
def forward_depth(self, image_embeddings, low_res_masks, input_size, original_sizes, ...):
    if self.depth_transformer_type == "simple":
        depth = self.simple_depth_head(image_embeddings)  # (B,1)
        return depth.squeeze(-1), None
    # ... 原有逻辑不动
```

`low_res_masks` 保持必填参数不变，调用方传什么都行，被 early return 忽略。

#### `init_depth_from_sam` early return

```python
if self.depth_transformer_type == "simple":
    print("Simple mode: no depth decoder to initialize from SAM")
    return
```

#### `freeze_depth_branch` 适配

```python
if self.depth_transformer_type == "simple":
    for param in self.simple_depth_head.parameters():
        param.requires_grad = False
else:
    # 原有逻辑
```

### 2. trainer.py (~8 行)

#### CLI choices

```python
choices=["twoway", "dual_attn", "simple"],
```

#### `train_one_epoch_phase2` 分发 — simple 优先于 n_sub 判断

```python
if args.depth_transformer_type == "simple":
    # 走非迭代路径，forward_depth 的 early return 处理一切
    # （复用现有 else 分支的 forward_mask → forward_depth → loss 逻辑）
elif args.n_sub_iterations > 1:
    loss_val, mask_val, depth_val = iterative_depth_step(...)
else:
    # 原有非迭代逻辑
```

simple 模式**永远不进入** `iterative_depth_step`，避免 decoder_masks=None 崩溃。

#### logging 分支守卫（L217）

```python
if decoder_masks is not None:
    decoder_mask_loss = dice_loss(decoder_masks, gt_masks)
else:
    decoder_mask_loss = torch.tensor(0.0)
```

#### `run_phase2` unfreeze 适配（L425-430）

```python
if args.depth_transformer_type == "simple":
    for param in model.simple_depth_head.parameters():
        param.requires_grad = True
else:
    for param in model.outer_prompt_encoder.parameters():
        param.requires_grad = True
    for param in model.depth_decoder.parameters():
        param.requires_grad = True
```

### 3. evaluate.py (~1 行)

choices 增加 `"simple"`。`forward_depth` 返回值第二个已用 `_` 丢弃，无需修改。`validate_phase2` 由 early return 覆盖，安全。

## 参数量

`256×1024 + 1024 + 1024×256 + 256 + 256×1 + 1 = 525,825 ≈ 0.53M`

与 DepthMaskDecoder (~4.7M) 差异约 9x。应在论文表格中报告。

## 改动量

| 文件 | 改动量 |
|------|--------|
| `depth_sam.py` | ~18 行 |
| `trainer.py` | ~8 行 |
| `evaluate.py` | ~1 行 |
| **总计** | **~27 行，无新文件、无新函数** |
