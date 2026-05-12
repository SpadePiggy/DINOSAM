# DINOSAM 训练策略

## 模型架构

DepthSam 在 SAM ViT-B 基础上新增 depth 分支，形成两阶段流水线：

```
Image ──→ Image Encoder (冻结) ──→ Image Embeddings
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
             Mask Branch                            Depth Branch
         (SAM PromptEncoder                        (Outer PromptEncoder
          + MaskDecoder)                            + DepthMaskDecoder)
                    │                                     │
                    ↓                                     ↓
              Segmentation Mask                    Depth Prediction
```

**训练参数量**：
- Image Encoder: 冻结 (85.7M)
- Mask 分支: prompt_encoder + mask_decoder = 4,064,560
- Depth 分支: outer_prompt_encoder + depth_decoder = 3,643,597

## 两阶段训练

### 阶段1：分割训练

- **目标**：训练 mask 分支的分割能力
- **冻结**：image_encoder + depth_decoder + outer_prompt_encoder
- **可训练**：SAM prompt_encoder + mask_decoder
- **Loss**：`dice_loss + iou_loss`
- **迭代训练**：每个 step 迭代 mask 分支 N 次，每轮根据预测误差生成新 prompt

### 阶段2：深度训练

- **目标**：训练 depth 分支的深度预测能力
- **冻结**：image_encoder + SAM prompt_encoder + mask_decoder
- **可训练**：outer_prompt_encoder + depth_decoder
- **Loss**：`dice_loss + depth_loss_weight * depth_mse_loss`
- **mask 分支冻结**：用 no_grad 运行 mask 分支获取 low_res_masks，再输入 depth 分支

## 迭代训练策略

参考 micro-sam 的迭代提示机制，仅在 mask 分支（阶段1）中使用：

```
初始: 数据集提供的 point prompts
  │
  ├── Sub-iter 0: forward_mask → mask_loss, iou_loss
  │     └── 比较预测 mask 与 GT，在误差区域采样新的正/负点
  │
  ├── Sub-iter 1: forward_mask (原始点 + 新增点) → mask_loss, iou_loss
  │     └── 继续采样误差点...
  │
  ├── ...
  │
  └── Sub-iter N-1: forward_mask → mask_loss, iou_loss

avg_loss = sum(losses) / N
```

**IterativePromptGenerator**：对比 GT 和预测 mask，在以下区域采点：
- 正点（label=1）：GT 有但预测漏掉的区域（漏检前景）
- 负点（label=0）：预测有但 GT 没有的区域（误报背景）
- 每轮新增 1 正 + 1 负点，追加到已有 prompts

**mask_prob**：每轮以概率 `mask_prob` 将上一轮的 low_res_masks 作为 mask 输入提示，帮助模型学习从粗到精的分割。

## 启动命令

```bash
# 两阶段顺序执行
python -m dinosam.train.trainer --train_phase both \
    --sam_checkpoint /data3/LXT/data/checkpoint/sam_vit_b_01ec64.pth \
    --n_sub_iterations 8 --mask_prob 0.5 --batch_size 4

# 只跑阶段1
python -m dinosam.train.trainer --train_phase 1 --n_sub_iterations 8 --mask_prob 0.5

# 只跑阶段2（需已有 phase1 权重）
python -m dinosam.train.trainer --train_phase 2 --n_sub_iterations 8 --mask_prob 0.5

# 断点恢复
python -m dinosam.train.trainer --train_phase 1 --resume

# 评估
python -m dinosam.evaluate --sam_checkpoint /data3/LXT/data/checkpoint/sam_vit_b_01ec64.pth
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 50 | 每阶段训练轮数 |
| `--n_sub_iterations` | 1 | 迭代子步数，1=无迭代，推荐 3-8 |
| `--mask_prob` | 0.5 | 迭代中使用 mask 输入的概率 |
| `--depth_loss_weight` | 1.0 | depth loss 权重 |
| `--lr` | 1e-5 | 学习率 |
| `--batch_size` | 2 | 批大小，可根据显存调大 |
| `--save_interval` | 10 | 每 N epoch 保存 checkpoint |

## 保存文件

| 文件 | 内容 |
|------|------|
| `phase1_best.pt` | 阶段1 最佳模型（含 optimizer/scheduler） |
| `phase1_last.pt` | 阶段1 最新模型（用于 resume） |
| `phase2_best.pt` | 阶段2 最佳模型 |
| `phase2_last.pt` | 阶段2 最新模型（用于 resume） |
