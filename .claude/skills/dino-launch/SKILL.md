---
name: dino-launch
description: Use when launching DINOSAM dual-DPT training or evaluation runs on this machine — creating output dirs, writing run_train.sh / config.txt, launching with nohup on specific GPUs, or running evaluate.py with result collection. Covers naming conventions, fixed checkpoint paths, GPU allocation, and the exact nohup + verification pattern.
---

# DINO Launch — 训练与评估启动

固化双头 DPT 训练的启动（train）和评估（eval）流程。所有路径硬编码于本机 `/data3` 挂载，不做参数化。

## 固定参数（所有训练共享）

```bash
SAM_CHECKPOINT=/data3/LXT/data/checkpoint/sam_vit_b_01ec64.pth
DINO_CHECKPOINT=/data3/LXT/data/checkpoint/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
TMPDIR=/data3/LXT/tmp      # 必须 export
```

## 数据集

| 简称 | train_dir | val_dir | 测试集 |
|------|-----------|---------|--------|
| `0sam` | `/data3/LXT/DINOv3-SAM/0_sam_train` | `/data3/LXT/DINOv3-SAM/0_sam_test` | 同 val_dir |
| `1724` | `/data3/LXT/DINOv3-SAM/1724_train` | `/data3/LXT/DINOv3-SAM/1724_test` | 同 val_dir |

## 输出目录命名

```
/data3/LXT/DINOv3-SAM/model_DINO/DPT_<dataset>_layers<mask>_depth<depth>_bs8_iter8_ep50
```

- `<dataset>`: `0sam` 或 `1724`
- `<mask>`: mask 分支层索引，逗号分隔去掉逗号，如 `0,3,6,9` → `0369`
- `<depth>`: depth 分支层索引，同上，如 `2,5,8,11` → `25811`

## 训练启动

### 流程

1. **检查重复**: 目标目录存在 `.pt` 文件 → 已训练过，跳过或询问
2. **创建目录**: `mkdir -p <output_dir>/logs`
3. **写 `config.txt`**: 记录全部参数（方便复现）。格式参考下文模板
4. **写 `run_train.sh`**: 必须 `export TMPDIR` + `CUDA_VISIBLE_DEVICES=<gpu>` + 完整命令行。格式参考下文模板
5. **启动**: **必须 `cd /home/LXT/lxt/DINOSAM/DINOSAM`**（项目根，否则 `dinosam` 模块找不到），然后 `nohup bash <output_dir>/run_train.sh > <output_dir>/log.txt 2>&1 &`；记录 PID
6. **验证**: 等 90s，`nvidia-smi` 确认 GPU 显存占用 ≥10GB + `tail -5 <output_dir>/log.txt` 确认 `Phase 1: Total params: 191,986,173  Trainable: 8,533,296`（双头模式预期值）

### `run_train.sh` 模板

```bash
#!/bin/bash
export TMPDIR=/data3/LXT/tmp
CUDA_VISIBLE_DEVICES=<GPU> python -m dinosam.train.trainer \
    --sam_checkpoint /data3/LXT/data/checkpoint/sam_vit_b_01ec64.pth \
    --dinov3_checkpoint /data3/LXT/data/checkpoint/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
    --encoder dinov3 \
    --adapter_type dpt_simple \
    --dpt_layers "<MASK_LAYERS>" \
    --dpt_layers_depth "<DEPTH_LAYERS>" \
    --depth_transformer_type twoway \
    --train_dir <TRAIN_DIR> \
    --val_dir <VAL_DIR> \
    --output_dir <OUTPUT_DIR> \
    --log_dir <OUTPUT_DIR>/logs \
    --train_phase both \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-5 \
    --n_sub_iterations 8 \
    --depth_loss_weight 1.0 \
    --max_points 3 \
    --log_interval 10 \
    --save_interval 10
```

### `config.txt` 模板

```text
# Dual DPT Layers Training Config
# Model: <MODEL_NAME>
# Mask layers: <MASK_LAYERS> | Depth layers: <DEPTH_LAYERS> (dual DPT heads)
# Date: YYYY-MM-DD

# Data
--train_dir         <TRAIN_DIR>
--val_dir           <VAL_DIR>

# Model
--sam_checkpoint    /data3/LXT/data/checkpoint/sam_vit_b_01ec64.pth
--dinov3_checkpoint /data3/LXT/data/checkpoint/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
--encoder           dinov3
--adapter_type      dpt_simple
--dpt_layers        <MASK_LAYERS>
--dpt_layers_depth  <DEPTH_LAYERS>
--depth_transformer_type  twoway

# Training
--train_phase       both
--epochs            50
--batch_size        8
--lr                1e-5
--n_sub_iterations  8
--depth_loss_weight 1.0
--max_points        3
--log_interval      10
--save_interval     10

# Output
--output_dir        <OUTPUT_DIR>
--log_dir           <OUTPUT_DIR>/logs
# GPU: CUDA_VISIBLE_DEVICES=<GPU>
```

## 评估启动

### 流程

1. **创建子目录**: `mkdir -p <model_dir>/eval_iter<N>`
2. **构建命令**: 参考下文模板，关键参数：`--dpt_layers` / `--dpt_layers_depth` 与训练一致，`--test_dir` 对应数据集的 test 路径，`--n_sub_iterations <N>`，`--save_masks_dir <model_dir>/eval_iter<N>`
3. **启动**: `nohup bash -c '...' > <model_dir>/eval_iter<N>/eval.log 2>&1 &`
4. **汇总结果**: 全部跑完后，提取各 `eval.log` 中的指标，参照 `/data3/LXT/DINOv3-SAM/model_DINO/eval_dual_dpt_summary.md` 格式写成 markdown 表格

### evaluate 命令模板

```bash
export TMPDIR=/data3/LXT/tmp
CUDA_VISIBLE_DEVICES=<GPU> python -m dinosam.evaluate \
    --sam_checkpoint /data3/LXT/data/checkpoint/sam_vit_b_01ec64.pth \
    --dinov3_checkpoint /data3/LXT/data/checkpoint/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
    --encoder dinov3 \
    --adapter_type dpt_simple \
    --dpt_layers "<MASK_LAYERS>" \
    --dpt_layers_depth "<DEPTH_LAYERS>" \
    --depth_transformer_type twoway \
    --test_dir <TEST_DIR> \
    --phase1_checkpoint <MODEL_DIR>/phase1_best.pt \
    --phase2_checkpoint <MODEL_DIR>/phase2_best.pt \
    --n_sub_iterations <N> \
    --save_masks_dir <MODEL_DIR>/eval_iter<N> \
    --batch_size 4 --max_points 3
```

## GPU 分配策略

GPU 0-3 可用时，按数据集/任务分配，避免混跑：

| GPU | 用途 |
|-----|------|
| 0,1 | eval 优先（轻量，显存 ~2.5GB） |
| 2,3 | train 优先（重量，显存 ~11GB） |

同卡跑多个 eval 可并行（显存充足）；同卡跑两个 train 不可行（会 OOM）。

## 训练产出文件

训练完成后，输出目录下预期文件：

```
phase1_best.pt   phase1_last.pt    # Phase 1 最优/最后
phase1_epoch_9.pt ... epoch_49.pt  # 每 save_interval=10 存一次
phase2_best.pt   phase2_last.pt    # Phase 2 最优/最后
phase2_epoch_9.pt ... epoch_49.pt
config.txt                          # 本次训练参数存档
run_train.sh                        # 启动脚本
log.txt                             # nohup 全量日志
logs/                               # TensorBoard 日志
```

## 常用命令

```bash
# 查看训练进度
tail -f <output_dir>/log.txt

# 查看 GPU 状态
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

# 查看训练进程
ps aux | grep "dinosam.train.trainer"
```

## 常见错误

| 症状 | 原因 | 修复 |
|------|------|------|
| `ImportError: No module named dinosam` | 未在项目根目录启动 | `cd /home/LXT/lxt/DINOSAM/DINOSAM && ...` |
| `CUDA out of memory` | GPU 已被占用或 bs 过大 | `nvidia-smi` 检查，换空闲 GPU 或降 bs |
| eval 跳过 phase1/phase2 | `--dpt_layers_depth` 与 checkpoint 不匹配 | 确认 checkpoint 是双头产物：`grep dpt_layers_depth config.txt` |
| warm-start 未触发 | phase2 加载了双头 phase1 产物（missing_keys 为空） | 正常——双头 phase1 产物不触发 warm-start 是设计行为；resume 双头 phase2_last.pt 即可 |

## 结果汇总模板

参考 `/data3/LXT/DINOv3-SAM/model_DINO/eval_dual_dpt_summary.md`：总体指标表 + Phase2 逐类 MAE 表 + 关键观察。
