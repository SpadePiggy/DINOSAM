# DINOSAM

Depth-aware SAM: extending Segment Anything Model with depth prediction via iterative two-phase training.

## Architecture

DINOSAM builds on SAM (ViT-B) with two additional modules for depth prediction:

```
Image ──► Image Encoder (frozen) ──► Image Embeddings
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │                                           │
              Point Prompts                              Low-res Mask
                    │                                     (detached)
                    ▼                                           │
          Inner PromptEncoder                          Outer PromptEncoder
                    │                                           │
                    ▼                                           ▼
             Mask Decoder ──► Masks                  Depth Decoder ──► Depth
                    │
              IoU Prediction
```

- **Mask Branch** (SAM inner): `prompt_encoder` + `mask_decoder` → segmentation mask + IoU score
- **Depth Branch** (outer): `outer_prompt_encoder` + `DepthMaskDecoder` → depth scalar prediction
- `DepthMaskDecoder` replaces SAM's `iou_token` with `depth_token` and `iou_prediction_head` with `depth_prediction_head` (MLP: 256→256→1)
- Depth branch weights are initialized from SAM's inner prompt_encoder and mask_decoder (except `depth_prediction_head`, which is random)

## Two-Phase Training

### Phase 1: Mask-only Training

Freeze image encoder + depth branch. Train SAM's `prompt_encoder` + `mask_decoder`.

- Loss: `dice_loss + iou_loss`
- Supports iterative mask refinement with error-based prompt updates

### Phase 2: Depth Training

Freeze image encoder + mask branch. Train `outer_prompt_encoder` + `depth_decoder`.

- Loss: `dice_loss(decoder_masks, gt) + depth_loss_weight × depth_mse_loss`
- Mask branch runs in `no_grad` to provide low-res masks as depth branch input
- Supports iterative depth training with point prompts on depth decoder masks

### Iterative Training

Each training step can perform multiple sub-iterations:

1. Run forward pass → get predicted mask
2. Compare with GT → sample 1 positive point (missed foreground) + 1 negative point (false positive)
3. Append new points to prompts for next sub-iteration
4. Optionally use previous low-res mask as mask input (controlled by `--mask_prob`)

## Project Structure

```
dinosam/
├── model/
│   ├── depth_sam.py       # DepthSam: SAM + outer depth branch
│   └── depth_decoder.py   # DepthMaskDecoder: mask decoder variant for depth
├── dataset/
│   └── depth_dataset.py   # DepthDataset: loads depth-classified image/mask pairs
├── loss/
│   ├── dice.py            # dice_loss, compute_iou
│   └── depth_loss.py      # depth_mse_loss
├── prompt/
│   └── iterative.py       # DepthIterativePromptGenerator: error-based prompt sampling
├── train/
│   ├── trainer.py         # Two-phase training entry point
│   └── iterative.py       # iterative_mask_step, iterative_depth_step
└── evaluate.py            # Model evaluation (Dice, IoU, Depth MAE/RMSE)
```

## Dataset Format

```
root/
  0/       (depth = 0)
  10/      (depth = 10)
  f10/     (depth = -10)
  ...
  Each folder: paired .jpg images and .png masks with matching names
```

Depth values are parsed from folder names: `f10` → -10, `10` → 10, `0` → 0.

## Requirements

- Python 3.10+
- PyTorch
- [segment-anything](https://github.com/facebookresearch/segment-anything)
- scipy, Pillow, numpy, tensorboard

## Usage

### Training

**Two-phase sequential training (iterative):**

```bash
python -m dinosam.train.trainer \
    --sam_checkpoint /path/to/sam_vit_b_01ec64.pth \
    --train_phase both \
    --n_sub_iterations 8 \
    --mask_prob 0.5 \
    --depth_loss_weight 0.01
```

**Phase 1 only (mask):**

```bash
python -m dinosam.train.trainer \
    --sam_checkpoint /path/to/sam_vit_b_01ec64.pth \
    --train_phase 1 \
    --n_sub_iterations 8 \
    --mask_prob 0.5
```

**Phase 2 only (depth), loading Phase 1 weights:**

```bash
python -m dinosam.train.trainer \
    --sam_checkpoint /path/to/sam_vit_b_01ec64.pth \
    --train_phase 2 \
    --phase1_checkpoint /path/to/phase1_best.pt \
    --n_sub_iterations 8 \
    --mask_prob 0.5 \
    --depth_loss_weight 0.01
```

**Resume from checkpoint:**

```bash
python -m dinosam.train.trainer \
    --sam_checkpoint /path/to/sam_vit_b_01ec64.pth \
    --train_phase 2 \
    --resume
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--train_phase` | `both` | `1` = mask-only, `2` = depth-only, `both` = sequential |
| `--n_sub_iterations` | `1` | Sub-iterations per step (1 = no iteration) |
| `--mask_prob` | `0.5` | Probability of using mask input from previous prediction |
| `--depth_loss_weight` | `1.0` | Weight for depth MSE loss in Phase 2 |
| `--epochs` | `50` | Training epochs per phase |
| `--batch_size` | `2` | Batch size |
| `--lr` | `1e-5` | Learning rate |
| `--resume` | — | Resume from latest checkpoint |
| `--phase1_checkpoint` | — | Phase 1 checkpoint path (for standalone Phase 2) |

### Evaluation

```bash
python -m dinosam.evaluate \
    --sam_checkpoint /path/to/sam_vit_b_01ec64.pth \
    --phase1_checkpoint /path/to/phase1_best.pt \
    --phase2_checkpoint /path/to/phase2_best.pt \
    --n_sub_iterations 8 \
    --save_masks_dir /path/to/output
```

Outputs: Dice, IoU, IoU Prediction MAE, Depth MAE/RMSE, per-depth-class breakdown.

### Checkpoints

- `phase1_best.pt` / `phase2_best.pt` — best validation models
- `phase1_last.pt` / `phase2_last.pt` — latest models (for resume)
- `phase1_epoch_N.pt` / `phase2_epoch_N.pt` — periodic saves (every `--save_interval` epochs)
