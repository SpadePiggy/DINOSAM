# DINOSAM

深度感知 SAM：通过两阶段迭代训练扩展 Segment Anything Model 的深度预测能力。

## 模型架构

DINOSAM 在 SAM (ViT-B) 基础上增加了两个深度预测模块：

```
图像 ──► 图像编码器（冻结）──► 图像嵌入
                                      │
                  ┌───────────────────┴───────────────────┐
                  │                                       │
            点提示                                   低分辨率掩码
                  │                                    （分离梯度）
                  ▼                                       │
        内层 PromptEncoder                        外层 PromptEncoder
                  │                                       │
                  ▼                                       ▼
           Mask Decoder ──► 掩码                    Depth Decoder ──► 深度
                  │
           IoU 预测
```

- **掩码分支**（SAM 内层）：`prompt_encoder` + `mask_decoder` → 分割掩码 + IoU 分数
- **深度分支**（外层）：`outer_prompt_encoder` + `DepthMaskDecoder` → 深度标量预测
- `DepthMaskDecoder` 将 SAM 的 `iou_token` 替换为 `depth_token`，`iou_prediction_head` 替换为 `depth_prediction_head`（MLP: 256→256→1）
- 深度分支权重从 SAM 内层 prompt_encoder 和 mask_decoder 初始化（`depth_prediction_head` 除外，随机初始化）

## 两阶段训练

### 阶段 1：掩码训练

冻结图像编码器 + 深度分支。训练 SAM 的 `prompt_encoder` + `mask_decoder`。

- 损失：`dice_loss + iou_loss`
- 支持基于误差提示更新的迭代掩码优化

### 阶段 2：深度训练

冻结图像编码器 + 掩码分支。训练 `outer_prompt_encoder` + `depth_decoder`。

- 损失：`dice_loss(decoder_masks, gt) + depth_loss_weight × depth_mse_loss`
- 掩码分支以 `no_grad` 方式运行，提供低分辨率掩码作为深度分支输入
- 支持在 depth decoder 掩码上使用点提示的迭代深度训练

### 迭代训练

每个训练步可执行多次子迭代：

1. 前向传播 → 得到预测掩码
2. 与 GT 对比 → 采样 1 个正点（漏检前景）+ 1 个负点（误检区域）
3. 将新点追加到下一次迭代的提示中
4. 可选地使用前一次的低分辨率掩码作为掩码输入（由 `--mask_prob` 控制）

## 项目结构

```
dinosam/
├── model/
│   ├── depth_sam.py       # DepthSam：SAM + 外层深度分支
│   └── depth_decoder.py   # DepthMaskDecoder：用于深度预测的 mask decoder 变体
├── dataset/
│   └── depth_dataset.py   # DepthDataset：加载深度分类的图像/掩码对
├── loss/
│   ├── dice.py            # dice_loss, compute_iou
│   └── depth_loss.py      # depth_mse_loss
├── prompt/
│   └── iterative.py       # DepthIterativePromptGenerator：基于误差的提示采样
├── train/
│   ├── trainer.py         # 两阶段训练入口
│   └── iterative.py       # iterative_mask_step, iterative_depth_step
└── evaluate.py            # 模型评估（Dice、IoU、Depth MAE/RMSE）
```

## 数据集格式

```
root/
  0/       (depth = 0)
  10/      (depth = 10)
  f10/     (depth = -10)
  ...
  每个文件夹：配对的 .jpg 图像和 .png 掩码，文件名一一对应
```

深度值从文件夹名解析：`f10` → -10，`10` → 10，`0` → 0。

## 环境依赖

- Python 3.10+
- PyTorch
- [segment-anything](https://github.com/facebookresearch/segment-anything)
- scipy, Pillow, numpy, tensorboard

## 使用方法

### 训练

**两阶段顺序训练（迭代）：**

```bash
python -m dinosam.train.trainer \
    --sam_checkpoint /path/to/sam_vit_b_01ec64.pth \
    --train_phase both \
    --n_sub_iterations 8 \
    --mask_prob 0.5 \
    --depth_loss_weight 0.01
```

**仅阶段 1（掩码训练）：**

```bash
python -m dinosam.train.trainer \
    --sam_checkpoint /path/to/sam_vit_b_01ec64.pth \
    --train_phase 1 \
    --n_sub_iterations 8 \
    --mask_prob 0.5
```

**仅阶段 2（深度训练），加载阶段 1 权重：**

```bash
python -m dinosam.train.trainer \
    --sam_checkpoint /path/to/sam_vit_b_01ec64.pth \
    --train_phase 2 \
    --phase1_checkpoint /path/to/phase1_best.pt \
    --n_sub_iterations 8 \
    --mask_prob 0.5 \
    --depth_loss_weight 0.01
```

**从断点恢复训练：**

```bash
python -m dinosam.train.trainer \
    --sam_checkpoint /path/to/sam_vit_b_01ec64.pth \
    --train_phase 2 \
    --resume
```

### 主要参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--train_phase` | `both` | `1` = 仅掩码，`2` = 仅深度，`both` = 顺序执行 |
| `--n_sub_iterations` | `1` | 每步子迭代次数（1 = 不迭代） |
| `--mask_prob` | `0.5` | 使用前一次预测掩码作为输入的概率 |
| `--depth_loss_weight` | `1.0` | 阶段 2 深度 MSE 损失权重 |
| `--epochs` | `50` | 每阶段训练轮数 |
| `--batch_size` | `2` | 批大小 |
| `--lr` | `1e-5` | 学习率 |
| `--resume` | — | 从最新断点恢复训练 |
| `--phase1_checkpoint` | — | 阶段 1 权重路径（单独运行阶段 2 时需要） |

### 评估

```bash
python -m dinosam.evaluate \
    --sam_checkpoint /path/to/sam_vit_b_01ec64.pth \
    --phase1_checkpoint /path/to/phase1_best.pt \
    --phase2_checkpoint /path/to/phase2_best.pt \
    --n_sub_iterations 8 \
    --save_masks_dir /path/to/output
```

输出指标：Dice、IoU、IoU 预测 MAE、Depth MAE/RMSE、按深度类别分解的详细指标。

### 模型保存

- `phase1_best.pt` / `phase2_best.pt` — 验证集最优模型
- `phase1_last.pt` / `phase2_last.pt` — 最新模型（用于恢复训练）
- `phase1_epoch_N.pt` / `phase2_epoch_N.pt` — 周期性保存（每 `--save_interval` 个 epoch）
