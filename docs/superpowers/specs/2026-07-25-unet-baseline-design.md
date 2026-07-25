# U-Net Baseline Comparison Experiment Design

**Date:** 2026-07-25
**Status:** approved
**Goal:** 使用 Pytorch-UNet 在两个数据集上训练分割模型，作为 DINOSAM 方法的对比基线。

---

## 1. Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| epochs | 50 | Aligned with DINOSAM |
| batch_size | 8 | Aligned with DINOSAM |
| lr | 1e-5 | Aligned with DINOSAM |
| image resolution | 1024×1024 (scale=1.0) | Aligned with DINOSAM |
| num_classes | 2 (background + object) | Binary segmentation |
| AMP | off | Aligned with DINOSAM |
| optimizer | RMSprop (momentum=0.999, wd=1e-8) | U-Net original — give U-Net its best config |
| scheduler | ReduceLROnPlateau (patience=5, mode='max') | U-Net original |
| loss | Pure Dice (no BCE) | Aligned with DINOSAM mask loss |
| gradient clipping | 1.0 | U-Net original |
| DataLoader workers | 4, pin_memory=True | Aligned with DINOSAM |

### NOT applicable (DINOSAM-specific)
- SAM/DINOv3 checkpoints, encoder type, adapter_type, dpt_layers, depth_transformer_type
- train_phase, n_sub_iterations, max_points, depth_loss_weight
- U-Net has no depth branch, no prompt mechanism, no iterative refinement

---

## 2. Datasets

### Training
- Dataset A: `/data3/LXT/DINOv3-SAM/0_sam_train` → model `UNet_0sam_ep50_bs8`
- Dataset B: `/data3/LXT/DINOv3-SAM/1724_train` → model `UNet_1724_ep50_bs8`

### Validation (separate test sets, no train split)
- Dataset A: `/data3/LXT/DINOv3-SAM/0_sam_test`
- Dataset B: `/data3/LXT/DINOv3-SAM/1724_test`

### Data format
Each dataset has subdirectories (0/, 10/, 20/, ... f10/, f20/, ...), each containing:
- `*.jpg` — RGB image (1024×1024)
- `*.png` — ground-truth mask (1024×1024, RGB)

Images and masks are paired by filename stem within the same subdirectory.

### Data loading adaptation
U-Net's `BasicDataset` expects flat `imgs/` + `masks/` directories. We need a new or adapted Dataset class that:
1. Recursively walks subdirectories
2. Pairs .jpg (image) + .png (mask) by filename stem
3. Reuses `BasicDataset.preprocess()` for resizing and mask-value mapping

---

## 3. Implementation Plan

### Files to modify

| File | Changes |
|------|---------|
| `Pytorch-UNet/utils/data_loading.py` | Add `RecursiveDataset` class supporting subdirectory structure |
| `Pytorch-UNet/train.py` | Add `--train_dir`, `--val_dir`, `--output_dir` CLI args; replace internal train/val split with external val set; change loss from BCE+Dice to pure Dice; make wandb optional |

### Files NOT modified
- `unet/unet_model.py` — model unchanged
- `unet/unet_parts.py` — unchanged
- `evaluate.py` — unchanged (Dice evaluation reused)
- `utils/dice_score.py` — unchanged (dice_loss function reused)

### Output structure

```
/data3/LXT/DINOv3-SAM/model_UNet/
  UNet_0sam_ep50_bs8/
    config.txt
    run_train.sh
    checkpoints/
      checkpoint_epoch10.pth
      ...
  UNet_1724_ep50_bs8/
    config.txt
    run_train.sh
    checkpoints/
      ...
```

### Run convention
Each experiment launched via `run_train.sh` with `nohup`, matching the DINOSAM project's `dino-launch` convention. GPU allocation per experiment.

---

## 4. Validation & Evaluation

- Validation runs every `n_train // (5 * batch_size)` steps (U-Net default interval)
- Metric: Dice score (same as U-Net evaluate.py)
- Best checkpoint saved based on validation Dice
- Checkpoint saved every 10 epochs
- After training, evaluate on the corresponding test set using the best checkpoint

---

## 5. Scope Boundaries

### In scope
- Modify U-Net training script to work with the DINOSAM dataset format
- Parameterize data paths and output directories
- Train 2 models (one per dataset)

### Out of scope
- U-Net architecture changes
- New evaluation metrics (use existing Dice)
- Depth prediction (U-Net only does segmentation)
- Comparison analysis/report (separate task)
