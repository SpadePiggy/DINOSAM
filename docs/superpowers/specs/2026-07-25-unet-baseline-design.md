# U-Net Baseline Comparison Experiment Design

**Date:** 2026-07-25
**Status:** approved (reviewed — 3-agent fanout, 2026-07-25)
**Goal:** 使用 Pytorch-UNet 在两个数据集上训练分割模型，作为 DINOSAM 方法的对比基线。

**Comparison type:** Best-method comparison — U-Net uses its own optimal config (RMSprop, ReduceLROnPlateau, gradient clipping) while DINOSAM uses its own. This conflates architecture with optimizer/scheduler choices. Results must be interpreted accordingly: differences are not attributable to architecture alone. Additional known differences: U-Net normalizes inputs via `/255`, while DINOSAM uses SAM's `pixel_mean`/`pixel_std` normalization — another confound inherent to each method's standard pipeline.

---

## 1. Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| epochs | 50 | Aligned with DINOSAM |
| batch_size | 8 | Aligned with DINOSAM run config (OOM → fallback to use_checkpointing()) |
| lr | 1e-5 | Aligned with DINOSAM |
| image resolution | 1024×1024 (scale=1.0) | Aligned with DINOSAM |
| num_classes | 2 (background + object) | Binary segmentation |
| bilinear | False (transposed conv) | U-Net original |
| AMP | off | Aligned with DINOSAM |
| optimizer | RMSprop (momentum=0.999, wd=1e-8) | U-Net original — best-method config |
| scheduler | ReduceLROnPlateau (patience=5, mode='max') | U-Net original |
| loss | Pure Dice (no BCE/CE) | Aligned with DINOSAM mask loss |
| gradient clipping | 1.0 | U-Net original |
| DataLoader workers | 4, pin_memory=True | Aligned with DINOSAM |
| data augmentation | None | Consistent with DINOSAM (resize only) |

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
- `*.png` — ground-truth mask (1024×1024, RGB, 5 discrete values including anti-aliasing boundary pixels)

Images and masks are paired by filename stem within the same subdirectory. The `f` prefix in directory names has no semantic meaning for segmentation — all subdirectories are included.

### Mask preprocessing (critical design decision)
Source PNG masks contain 5 RGB values: (0,0,0), (64,64,64), (128,128,128), (191,191,191), (255,255,255). The intermediate values are anti-aliasing boundary pixels. To produce a binary mask compatible with `num_classes=2`:

1. Open mask image with PIL, convert to grayscale: `.convert("L")`
2. Threshold at > 0: `(mask_arr > 0).astype(np.uint8)` → yields {0, 1}
3. This matches DINOSAM's `DepthDataset` behavior exactly

Source: `dinosam/dataset/depth_dataset.py`, lines 95-105.

### Data loading adaptation
U-Net's `BasicDataset` expects flat `imgs/` + `masks/` directories. We add a new `RecursiveDataset` class in `data_loading.py` that:
1. Recursively walks subdirectories
2. Pairs .jpg (image) + .png (mask) by filename stem
3. Applies grayscale conversion + thresholding before mask preprocess
4. For the remaining resize/mask-value logic, delegates to `BasicDataset.preprocess()` (which with binary masks produces expected 2-class output)

### Dice evaluation metric compatibility
After grayscale+threshold preprocessing, U-Net's `multiclass_dice_coeff(mask_pred[:, 1:], ...)` computes Dice on the foreground class only — mathematically equivalent to DINOSAM's binary Dice over the whole mask. No evaluation code change needed.

---

## 3. Implementation Plan

### Files to modify

| File | Changes |
|------|---------|
| `Pytorch-UNet/utils/data_loading.py` | Add `RecursiveDataset` class: recursive subdirectory walk, .jpg/.png pairing, grayscale+threshold mask loading |
| `Pytorch-UNet/train.py` | Add `--train_dir`, `--val_dir`, `--output_dir` CLI args; replace internal train/val split with external val set; change loss from BCE+Dice to pure Dice; **remove wandb** (replace with CSV logging to `<output_dir>/metrics.csv`: epoch, step, train_loss, val_dice, lr); fix num_workers to 4 |

### Files NOT modified
- `unet/unet_model.py` — model unchanged
- `unet/unet_parts.py` — unchanged
- `evaluate.py` — unchanged (Dice evaluation reused; metric is compatible with binary masks)
- `utils/dice_score.py` — unchanged (dice_loss function reused)

### Output structure

```
/data3/LXT/DINOv3-SAM/model_UNet/
  UNet_0sam_ep50_bs8/
    config.txt
    run_train.sh
    checkpoints/
      checkpoint_epoch10.pth
      checkpoint_epoch20.pth
      ...
  UNet_1724_ep50_bs8/
    config.txt
    run_train.sh
    checkpoints/
      ...
```

### Run convention
Each experiment launched via `run_train.sh` with `nohup`, matching the DINOSAM project's `dino-launch` convention. GPU allocation per experiment.

### Checkpoint resume
Support `--resume` flag to load latest checkpoint and continue training. Resume restores model weights, optimizer state (RMSprop momentum buffers), scheduler state (ReduceLROnPlateau patience counter + best), and epoch counter from the checkpoint.

### Reproducibility
- DataLoader shuffle seed: `torch.Generator().manual_seed(0)` — consistent with existing U-Net code
- Model weight initialization: PyTorch defaults (no explicit seed set — matches U-Net original behavior)

---

## 4. Validation & Evaluation

- Validation runs every `n_train // (5 * batch_size)` steps (standard U-Net interval)
- Metric: Dice score (via existing `evaluate.py`, `multiclass_dice_coeff` foreground-only path)
- Best checkpoint saved based on highest validation Dice
- Checkpoint saved every 10 epochs (`--save_interval 10`)
- After training, evaluate on the corresponding test set using the best checkpoint

### Smoke test (DoD before full training)
1. Run 1 epoch on each dataset → no crash, loss decreases
2. Confirm checkpoint saved to output dir
3. Confirm validation Dice printed in logs
4. Confirm mask binarization produces only {0, 1} values

---

## 5. Scope Boundaries

### In scope
- Modify U-Net training script to work with the DINOSAM dataset format
- Parameterize data paths and output directories
- Train 2 models (one per dataset)
- Grayscale+threshold mask preprocessing

### Out of scope
- U-Net architecture changes
- New evaluation metrics beyond Dice (IoU/Boundary IoU deferred to evaluation phase)
- Depth prediction (U-Net only does segmentation)
- Comparison analysis/report (separate task)
