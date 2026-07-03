# DINOSAM

**Depth-aware SAM**: Extending Segment Anything Model with depth prediction via iterative two-phase training.

DINOSAM builds on SAM (ViT-B) to add depth estimation capabilities while maintaining high-quality segmentation. The model uses a two-phase training strategy and supports multiple backbone encoders (SAM, DINOv3) with various adapter types.

## Architecture

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

**Components:**
- **Mask Branch** (SAM inner): `prompt_encoder` + `mask_decoder` → segmentation mask + IoU score
- **Depth Branch** (outer): `outer_prompt_encoder` + `DepthMaskDecoder` → depth scalar prediction
- **DepthMaskDecoder**: Replaces SAM's `iou_token` with `depth_token` and `iou_prediction_head` with `depth_prediction_head` (MLP: 256→256→1)
- Depth branch weights are initialized from SAM's inner `prompt_encoder` and `mask_decoder` (except `depth_prediction_head`, which is random)

## Two-Phase Training Strategy

### Phase 1: Mask-only Training
- **Freezes**: image encoder + depth branch
- **Trains**: SAM's `prompt_encoder` + `mask_decoder`
- **Loss**: `dice_loss + iou_loss`
- **Features**: Supports iterative mask refinement with error-based prompt updates

### Phase 2: Depth Training
- **Freezes**: image encoder + mask branch
- **Trains**: `outer_prompt_encoder` + `depth_decoder`
- **Loss**: `dice_loss(decoder_masks, gt) + depth_loss_weight × depth_mse_loss`
- **Features**: Mask branch runs in `no_grad` to provide low-res masks; supports iterative depth training

### Iterative Training
Each training step can perform multiple sub-iterations:
1. Run forward pass → get predicted mask
2. Compare with GT → sample 1 positive point (missed foreground) + 1 negative point (false positive)
3. Append new points to prompts for next sub-iteration
4. Optionally use previous low-res mask as mask input (controlled by `--mask_prob`)

## Installation

### Requirements
- Python 3.10+
- PyTorch 2.0+
- [segment-anything](https://github.com/facebookresearch/segment-anything)
- scipy, Pillow, numpy, tensorboard
- scikit-learn (optional, for PCA visualization)

### Setup
```bash
# Install segment-anything first
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .

# Download SAM checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Install DINOSAM dependencies
cd /path/to/DINOSAM
pip install scipy Pillow numpy tensorboard
pip install scikit-learn  # optional, for PCA visualization
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

## Usage Guide

DINOSAM provides three main command-line tools:

### 1. Training (`dinosam.train.trainer`)

Train the model using two-phase strategy.

```bash
python -m dinosam.train.trainer \
    --sam_checkpoint /path/to/sam_vit_b_01ec64.pth \
    --train_dir /path/to/train_data \
    --val_dir /path/to/val_data \
    --output_dir /path/to/output \
    --train_phase both \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-5
```

#### Training Examples

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

**Training with PCA visualization:**
```bash
python -m dinosam.train.trainer \
    --sam_checkpoint /path/to/sam_vit_b.pth \
    --dpt_layers "2,5,8,11" \
    --pca_viz \
    --pca_interval 5 \
    --pca_dir ./pca_viz
```

**Training with boundary loss:**
```bash
python -m dinosam.train.trainer \
    --sam_checkpoint /path/to/sam_vit_b_01ec64.pth \
    --train_phase 1 \
    --boundary_loss \
    --boundary_loss_weight 1.0 \
    --boundary_width 3
```

#### Training Parameters

##### Data & Output
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train_dir` | `/data3/LXT/DINOv3-SAM/1724_train` | Training data directory |
| `--val_dir` | `/data3/LXT/DINOv3-SAM/1724_test` | Validation data directory |
| `--output_dir` | `/data3/LXT/DINOv3-SAM/model_sam/5_1` | Output directory for checkpoints |
| `--log_dir` | `./logs` | TensorBoard log directory |

##### Training Control
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train_phase` | `both` | Training phase: `1` = mask-only, `2` = depth-only, `both` = sequential |
| `--epochs` | `50` | Number of training epochs per phase |
| `--batch_size` | `4` | Batch size |
| `--lr` | `1e-5` | Learning rate |
| `--depth_loss_weight` | `1.0` | Weight for depth MSE loss in Phase 2 |
| `--resume` | `false` | Resume training from latest checkpoint |
| `--phase1_checkpoint` | `None` | Path to Phase 1 checkpoint (required for standalone Phase 2) |

##### Iterative Training
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_sub_iterations` | `1` | Number of iterative mask refinement sub-iterations (1 = no iteration) |
| `--mask_prob` | `0.5` | Probability of using mask input from previous prediction in each sub-iteration |
| `--max_points` | `3` | Maximum number of point prompts per sample |

##### Boundary Loss (Optional)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--boundary_loss` | `false` | Enable boundary-aware loss (Dice + DT-weighted BCE) for smoother mask edges |
| `--boundary_loss_weight` | `1.0` | Weight of boundary loss added to total loss |
| `--boundary_width` | `3` | Half-width (px) of morphological boundary band used by boundary_loss |

##### PCA Visualization (Optional)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pca_viz` | `false` | Enable PCA feature visualization during validation |
| `--pca_interval` | `1` | Generate PCA visualizations every N epochs |
| `--pca_dir` | `./pca_viz` | Directory to save PCA visualization images |
| `--pca_samples` | `4` | Number of validation samples to visualize |

##### Model Architecture
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sam_checkpoint` | `None` | Path to SAM vit_b checkpoint (required) |
| `--encoder` | `sam` | Image encoder type: `sam` (default) or `dinov3` (DINOv3+Mona) |
| `--dinov3_checkpoint` | `None` | Path to DINOv3 ViT-B/16 checkpoint (required when `--encoder dinov3`) |
| `--adapter_type` | `mona` | Feature adapter type: `mona`, `fc`, `dual_attn` (single-layer) or `dpt_simple` (multi-level DPT fusion) |
| `--dpt_layers` | `2,5,8,11` | Comma-separated intermediate layer indices for DPT (only used with `--adapter_type dpt_simple`) |
| `--depth_transformer_type` | `twoway` | Depth decoder transformer type: `twoway`, `dual_attn`, `simple` (dual_attn requires training from scratch) |

##### Logging & Checkpoints
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--log_interval` | `10` | Log training metrics every N batches |
| `--save_interval` | `10` | Save checkpoint every N epochs |

#### Checkpoint Files
- `phase1_best.pt` / `phase2_best.pt` — best validation models
- `phase1_last.pt` / `phase2_last.pt` — latest models (for resume)
- `phase1_epoch_N.pt` / `phase2_epoch_N.pt` — periodic saves (every `--save_interval` epochs)

### 2. Evaluation (`dinosam.evaluate`)

Evaluate trained models on test data.

```bash
python -m dinosam.evaluate \
    --sam_checkpoint /path/to/sam_vit_b_01ec64.pth \
    --phase1_checkpoint /path/to/phase1_best.pt \
    --phase2_checkpoint /path/to/phase2_best.pt \
    --test_dir /path/to/test_data \
    --n_sub_iterations 8 \
    --save_masks_dir /path/to/output
```

#### Evaluation Examples

**Basic evaluation:**
```bash
python -m dinosam.evaluate \
    --sam_checkpoint /path/to/sam_vit_b_01ec64.pth \
    --phase1_checkpoint /path/to/phase1_best.pt \
    --phase2_checkpoint /path/to/phase2_best.pt
```

**Evaluation with iterative refinement:**
```bash
python -m dinosam.evaluate \
    --sam_checkpoint /path/to/sam_vit_b_01ec64.pth \
    --phase1_checkpoint /path/to/phase1_best.pt \
    --phase2_checkpoint /path/to/phase2_best.pt \
    --n_sub_iterations 8 \
    --mask_prob 0.5
```

**Evaluation with mask saving:**
```bash
python -m dinosam.evaluate \
    --sam_checkpoint /path/to/sam_vit_b_01ec64.pth \
    --phase1_checkpoint /path/to/phase1_best.pt \
    --phase2_checkpoint /path/to/phase2_best.pt \
    --save_masks_dir ./eval_output \
    --batch_size 4
```

#### Evaluation Parameters

##### Data
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--test_dir` | `/data3/LXT/DINOv3-SAM/1724_test` | Test data directory |
| `--sam_checkpoint` | `/data3/LXT/data/checkpoint/sam_vit_b_01ec64.pth` | Path to SAM vit_b pretrained checkpoint |
| `--phase1_checkpoint` | `/data3/LXT/DINOv3-SAM/model_sam/5_1/phase1_best.pt` | Path to Phase 1 checkpoint |
| `--phase2_checkpoint` | `/data3/LXT/DINOv3-SAM/model_sam/5_1/phase2_best.pt` | Path to Phase 2 checkpoint |

##### Evaluation Control
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | `4` | Batch size for evaluation |
| `--max_points` | `3` | Maximum number of point prompts per sample |
| `--n_sub_iterations` | `1` | Number of iterative sub-iterations for evaluation |
| `--mask_prob` | `0.0` | Mask input probability for iterative evaluation |
| `--save_masks_dir` | `None` | Directory to save per-depth-class first sample pred/gt masks and scatter plots |

##### Model Architecture
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--encoder` | `sam` | Image encoder type: `sam` (default) or `dinov3` (DINOv3+Mona) |
| `--adapter_type` | `mona` | Feature adapter type: `mona`, `fc`, `dual_attn`, `dpt_simple` |
| `--dpt_layers` | `2,5,8,11` | Comma-separated intermediate layer indices for DPT (only used with `--adapter_type dpt_simple`) |
| `--depth_transformer_type` | `twoway` | Depth decoder transformer type: `twoway`, `dual_attn`, `simple` |
| `--dinov3_checkpoint` | `None` | Path to DINOv3 ViT-B/16 checkpoint (required when `--encoder dinov3`) |

#### Evaluation Metrics

**Segmentation Metrics:**
- **Dice**: F1 score for binary segmentation
- **IoU**: Intersection over Union
- **IoU Prediction**: Model's predicted IoU score
- **IoU Pred MAE**: Mean absolute error between predicted and actual IoU
- **Boundary IoU**: IoU computed only on boundary regions
- **Precision**: Positive predictive value
- **Recall**: Sensitivity / true positive rate

**Depth Metrics:**
- **Depth MSE**: Mean squared error
- **Depth MAE**: Mean absolute error
- **Depth RMSE**: Root mean squared error
- **Depth R²**: Coefficient of determination
- **Depth Pred/GT Mean**: Mean of predicted and ground truth depth values

**Per-Depth-Class Breakdown:**
- MAE and RMSE for each depth class
- Predicted mean depth per class
- Sample count per class

**Output Files (when `--save_masks_dir` is set):**
- `phase1/` and `phase2/` subdirectories
- Per-depth-class pred/gt mask images
- `depth_pred_vs_gt.png` — scatter plot of predicted vs ground truth depth
- `depth_residual_vs_gt.png` — scatter plot of residuals vs ground truth depth

### 3. PCA Visualization (`dinosam.scripts.pca_viz`)

Generate PCA visualizations of backbone features for inference images.

```bash
python -m dinosam.scripts.pca_viz \
    --checkpoint /path/to/phase1_best.pt \
    --sam_checkpoint /path/to/sam_vit_b.pth \
    --image /path/to/image1.jpg /path/to/image2.jpg \
    --output_dir ./pca_output \
    --dpt_layers "2,5,8,11" \
    --mask
```

#### PCA Visualization Examples

**Basic visualization:**
```bash
python -m dinosam.scripts.pca_viz \
    --checkpoint /path/to/phase1_best.pt \
    --sam_checkpoint /path/to/sam_vit_b.pth \
    --image /path/to/image.jpg \
    --output_dir ./pca_output
```

**Multiple images with mask:**
```bash
python -m dinosam.scripts.pca_viz \
    --checkpoint /path/to/phase1_best.pt \
    --sam_checkpoint /path/to/sam_vit_b.pth \
    --image /path/to/image1.jpg /path/to/image2.jpg \
    --mask
```

**Custom DPT layers:**
```bash
python -m dinosam.scripts.pca_viz \
    --checkpoint /path/to/phase1_best.pt \
    --sam_checkpoint /path/to/sam_vit_b.pth \
    --image /path/to/image.jpg \
    --dpt_layers "3,6,9,12" \
    --output_dir ./custom_pca
```

#### PCA Visualization Parameters

##### Required
| Parameter | Description |
|-----------|-------------|
| `--checkpoint` | DINOSAM model checkpoint path |
| `--sam_checkpoint` | SAM ViT-B checkpoint path |
| `--image` | Input image path(s), supports multiple images |

##### Optional
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output_dir` | `./pca_output` | Output directory |
| `--dpt_layers` | `2,5,8,11` | Comma-separated DPT layer indices |
| `--mask` | `false` | Use model-predicted mask for foreground-only PCA fitting |
| `--encoder` | `sam` | Image encoder type: `sam` or `dinov3` |
| `--dinov3_checkpoint` | `None` | DINOv3 ViT-B/16 checkpoint (required when `--encoder dinov3`) |
| `--adapter_type` | `mona` | Feature adapter type |
| `--img_size` | `1024` | Image size for preprocessing |

#### PCA Output Files

For each input image, creates a subdirectory with:
- `original.png` — original input image
- `mask.png` — predicted mask (if `--mask` is used)
- `layer_N_pca.png` — PCA visualization for each DPT layer N
- `fused_pca.png` — PCA visualization of fused features

## Advanced Features

### DPT Multi-Level Fusion

When using `--adapter_type dpt_simple`, the model uses DPT (Dense Prediction Transformer) architecture to fuse features from multiple DINOv3 transformer layers.

**Ablation Configurations:**
- `--dpt_layers "2,5,8"` — Three-layer early fusion
- `--dpt_layers "3,6,9"` — Three-layer uniform fusion
- `--dpt_layers "2,5,8,11"` — Four-layer SegDINO-style fusion (default)
- `--dpt_layers "3,6,9,12"` — Four-layer uniform fusion
- `--dpt_layers "8,9,10,11"` — High-level fusion
- `--dpt_layers "0,1,2,3,4,5,6,7,8,9,10,11"` — All-layer fusion

Layer indices are 0-indexed transformer block indices. For ViT-B/16 (12 blocks), valid range is [0, 11].

### Boundary Loss

Boundary-aware loss combines:
1. **Boundary Dice Loss**: Dice computed only on boundary pixels of GT mask
2. **Distance-weighted BCE**: Higher penalty near mask boundaries

Enable with `--boundary_loss` and adjust with `--boundary_loss_weight` and `--boundary_width`.

### Iterative Training

Iterative training improves mask quality through error-based prompt updates:

1. Model predicts mask
2. Compare prediction with ground truth
3. Sample error points: 1 positive (missed foreground) + 1 negative (false positive)
4. Append points to prompts for next iteration
5. Optionally feed back low-res mask as input (controlled by `--mask_prob`)

Use `--n_sub_iterations` to control number of iterations (default: 1 = no iteration).

### PCA Feature Visualization

PCA visualization projects high-dimensional backbone features to 3D RGB space:
- Fits PCA on feature vectors (optionally only foreground patches)
- Projects all patches through PCA
- Maps 3 components to RGB channels
- Applies sigmoid scaling for vibrant colors
- Masks background to black (if mask provided)

Useful for:
- Visualizing what each transformer layer learns
- Understanding feature fusion effectiveness
- Debugging model behavior

## Project Structure

```
DINOSAM/
├── dinosam/
│   ├── __init__.py
│   ├── model/
│   │   ├── depth_sam.py          # DepthSam: main model class
│   │   ├── depth_decoder.py      # DepthMaskDecoder: depth prediction head
│   │   ├── dinov3_encoder.py     # DINOv3 encoder with adapters
│   │   ├── dpt_heads.py          # DPTSimpleHead and DPTHead for multi-level fusion
│   │   ├── dual_attn_transformer.py  # Dual attention transformer
│   │   ├── mona.py               # Mona adapter implementation
│   │   └── adapters/             # Adapter implementations (fc, dual_attn)
│   ├── dataset/
│   │   └── depth_dataset.py      # DepthDataset: loads depth-classified image/mask pairs
│   ├── loss/
│   │   ├── dice.py               # dice_loss, compute_iou
│   │   └── depth_loss.py         # depth_mse_loss
│   ├── prompt/
│   │   └── iterative.py          # DepthIterativePromptGenerator: error-based prompt sampling
│   ├── train/
│   │   ├── trainer.py            # Two-phase training entry point
│   │   └── iterative.py          # iterative_mask_step, iterative_depth_step
│   ├── evaluate.py               # Model evaluation
│   ├── viz/
│   │   └── pca.py                # PCA visualization utilities
│   └── scripts/
│       └── pca_viz.py            # PCA visualization CLI tool
├── tests/
│   ├── test_dpt_heads.py         # DPTHead backward compatibility tests
│   ├── test_pca.py               # PCA visualization tests
│   ├── test_dinov3_encoder.py    # Encoder tests
│   └── test_dual_attn_transformer.py  # Transformer tests
├── docs/
│   └── superpowers/
│       ├── specs/                # Design specifications
│       └── plans/                # Implementation plans
└── README.md
```

## Troubleshooting

### Common Issues

**Out of memory during training:**
- Reduce `--batch_size`
- Use gradient accumulation
- Reduce `--n_sub_iterations`

**Phase 2 training fails to load Phase 1 checkpoint:**
- Ensure `--phase1_checkpoint` path is correct
- Check that Phase 1 training completed successfully

**PCA visualization not working:**
- Install scikit-learn: `pip install scikit-learn`
- Ensure checkpoint paths are correct
- Check that image paths exist

**Boundary loss causes training instability:**
- Reduce `--boundary_loss_weight` (try 0.5 or 0.1)
- Reduce `--boundary_width` (try 2 or 1)

**Iterative training is slow:**
- Reduce `--n_sub_iterations`
- Reduce `--mask_prob` to 0 (disables mask feedback)

### Performance Tips

**For faster training:**
- Use larger `--batch_size` if GPU memory allows
- Increase `--num_workers` for data loading
- Use `--log_interval` to reduce logging overhead

**For better segmentation:**
- Increase `--n_sub_iterations` (try 4-8)
- Enable `--boundary_loss`
- Use `--mask_prob 0.5` for iterative training

**For better depth prediction:**
- Adjust `--depth_loss_weight` (try 0.01-0.1)
- Use iterative depth training with `--n_sub_iterations`

## License

This project is for research purposes. Please ensure you comply with the licenses of all dependencies, especially [segment-anything](https://github.com/facebookresearch/segment-anything).

## Citation

If you use DINOSAM in your research, please cite:

```bibtex
@software{dinosam2024,
  title = {DINOSAM: Depth-aware Segment Anything Model},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/DINOSAM}
}
```
