# U-Net Baseline Comparison Experiment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify Pytorch-UNet to train on DINOSAM-format datasets with parameterized paths, pure Dice loss, CSV logging, and checkpoint resume — producing 2 trained models as DINOSAM baselines.

**Architecture:** Two-file change. `data_loading.py` gains a `RecursiveDataset` class that walks subdirectories and pairs .jpg/.png, applying grayscale+threshold mask binarization. `train.py` is rewritten to accept CLI args for data/output paths, use external validation sets, pure Dice loss, CSV metrics logging, and checkpoint resume.

**Tech Stack:** Python 3, PyTorch, torchvision, PIL, tqdm, csv (stdlib). No wandb.

## Global Constraints

- num_classes=2, bilinear=False, AMP off, gradient_clipping=1.0
- Optimizer: RMSprop (momentum=0.999, weight_decay=1e-8)
- Scheduler: ReduceLROnPlateau (mode='max', patience=5)
- Loss: pure multiclass Dice (no BCE/CE term)
- Image scale: 1.0 (1024×1024), no data augmentation
- DataLoader: num_workers=4, pin_memory=True
- Mask preprocessing: PIL `.convert("L")` → numpy `(arr > 0).astype(np.uint8)`
- Checkpoint: best_checkpoint.pth (on Dice improvement) + checkpoint_epochN.pth (every save_interval) + last.pt (resume)
- Logging: CSV to `<output_dir>/metrics.csv` (epoch, step, train_loss, val_dice, lr)
- No wandb dependency

---

### Task 1: Add RecursiveDataset to data_loading.py

**Files:**
- Modify: `Pytorch-UNet/utils/data_loading.py` (append at end of file)

**Interfaces:**
- Produces: `RecursiveDataset(root_dir, scale=1.0)` — torch.utils.data.Dataset
  - `__len__() -> int`
  - `__getitem__(idx) -> dict` with keys `'image'` (FloatTensor [3,H,W]) and `'mask'` (LongTensor [H,W])
  - `self.mask_values` = [0, 1] (list of int, for checkpoint compatibility)
  - `self.pairs` = list of (Path, Path) tuples (image_path, mask_path)

- [ ] **Step 1: Append RecursiveDataset class**

Add at the end of `Pytorch-UNet/utils/data_loading.py`:

```python
class RecursiveDataset(Dataset):
    """Dataset that recursively walks subdirectories, pairing .jpg images
    with .png masks by filename stem. Masks are binarized via grayscale
    conversion + threshold (>0), matching DINOSAM's DepthDataset behavior."""

    def __init__(self, root_dir: str, scale: float = 1.0):
        self.root_dir = Path(root_dir)
        self.scale = scale
        self.mask_values = [0, 1]  # binary after threshold

        # Collect all (image, mask) pairs across subdirectories
        self.pairs = []
        for subdir in sorted(self.root_dir.iterdir()):
            if not subdir.is_dir():
                continue
            jpgs = {p.stem: p for p in subdir.glob("*.jpg")}
            pngs = {p.stem: p for p in subdir.glob("*.png")}
            for stem in jpgs:
                if stem in pngs:
                    self.pairs.append((jpgs[stem], pngs[stem]))

        if not self.pairs:
            raise RuntimeError(
                f"No .jpg/.png pairs found in {self.root_dir}"
            )
        logging.info(f"RecursiveDataset: {len(self.pairs)} pairs from {self.root_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # Load image as RGB
        img = Image.open(img_path)

        # Load mask: grayscale + threshold → binary {0, 1}
        mask = Image.open(mask_path).convert("L")
        mask_arr = np.array(mask)
        mask_binary = (mask_arr > 0).astype(np.uint8)
        mask = Image.fromarray(mask_binary)

        # Reuse BasicDataset.preprocess for resize + normalization
        img = BasicDataset.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = BasicDataset.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            "image": torch.as_tensor(img.copy()).float().contiguous(),
            "mask": torch.as_tensor(mask.copy()).long().contiguous(),
        }
```

- [ ] **Step 2: Verify imports are sufficient**

`PIL.Image`, `pathlib.Path`, `numpy`, `torch`, `logging`, and `Dataset` from `torch.utils.data` are already imported at the top of `data_loading.py`. No new imports needed.

Run quick syntax check:
```bash
cd /home/LXT/lxt/DINOSAM/DINOSAM/Pytorch-UNet && python -c "from utils.data_loading import RecursiveDataset; print('Import OK')"
```
Expected: `Import OK` (no errors)

- [ ] **Step 3: Smoke test RecursiveDataset on real data**

```bash
cd /home/LXT/lxt/DINOSAM/DINOSAM/Pytorch-UNet && python -c "
from utils.data_loading import RecursiveDataset
ds = RecursiveDataset('/data3/LXT/DINOv3-SAM/0_sam_train', scale=1.0)
print(f'Dataset size: {len(ds)}')
sample = ds[0]
print(f'Image shape: {sample[\"image\"].shape}')  # expected: torch.Size([3, 1024, 1024])
print(f'Mask shape: {sample[\"mask\"].shape}')    # expected: torch.Size([1024, 1024])
print(f'Mask unique values: {sample[\"mask\"].unique()}')  # expected: tensor([0, 1])
print(f'Image dtype: {sample[\"image\"].dtype}')  # expected: torch.float32
print(f'Mask dtype: {sample[\"mask\"].dtype}')    # expected: torch.int64
print(f'Mask values: {ds.mask_values}')           # expected: [0, 1]
"
```
Expected output: dataset size > 0, shapes (3,1024,1024) and (1024,1024), unique mask values [0,1], mask_values [0,1].

- [ ] **Step 4: Commit**

```bash
git add Pytorch-UNet/utils/data_loading.py
git commit -m "feat: add RecursiveDataset for subdirectory-structured mask datasets

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Rewrite train.py

**Files:**
- Modify: `Pytorch-UNet/train.py` (full rewrite of main block + train_model function)

**Interfaces:**
- Consumes: `RecursiveDataset` from `utils.data_loading` (Task 1), `UNet` from `unet`, `evaluate` from `evaluate`, `dice_loss` from `utils.dice_score`
- Produces: CLI entry point with `--train_dir`, `--val_dir`, `--output_dir`, `--epochs`, `--batch-size`, `--learning-rate`, `--scale`, `--bilinear`, `--classes`, `--save_interval`, `--resume`

- [ ] **Step 1: Replace entire train.py content**

```python
import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import evaluate
from unet import UNet
from utils.data_loading import RecursiveDataset
from utils.dice_score import dice_loss


def train_model(
    model,
    device,
    train_dir: str,
    val_dir: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    img_scale: float = 1.0,
    save_interval: int = 10,
    resume: bool = False,
):
    # 1. Create datasets
    train_dataset = RecursiveDataset(train_dir, scale=img_scale)
    val_dataset = RecursiveDataset(val_dir, scale=img_scale)
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # 2. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)

    # 3. Setup CSV logging
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.csv")
    csv_file = open(metrics_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "step", "train_loss", "val_dice", "lr"])

    logging.info(
        f"Training: {n_train} samples | Validation: {n_val} samples\n"
        f"  Epochs: {epochs}  Batch size: {batch_size}  LR: {learning_rate}\n"
        f"  Scale: {img_scale}  Output: {output_dir}"
    )

    # 4. Optimizer, scheduler, loss
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-8,
        momentum=0.999,
        foreach=True,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=5
    )
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)

    # 5. Resume from last checkpoint
    start_epoch = 0
    best_val_dice = 0.0
    last_path = os.path.join(checkpoint_dir, "last.pt")
    if resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_dice = ckpt.get("best_val_dice", 0.0)
        logging.info(
            f"Resumed from epoch {ckpt['epoch']}, best val Dice: {best_val_dice:.4f}"
        )

    global_step = 0

    # 6. Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        division_step = max(n_train // (5 * batch_size), 1)

        with tqdm(total=n_train, desc=f"Epoch {epoch+1}/{epochs}", unit="img") as pbar:
            for batch in train_loader:
                images = batch["image"]
                true_masks = batch["mask"]

                images = images.to(
                    device=device,
                    dtype=torch.float32,
                    memory_format=torch.channels_last,
                )
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type, enabled=False):
                    masks_pred = model(images)

                    # Pure Dice loss (multiclass, background + foreground)
                    masks_pred_prob = F.softmax(masks_pred, dim=1)
                    true_masks_onehot = (
                        F.one_hot(true_masks, model.n_classes)
                        .permute(0, 3, 1, 2)
                        .float()
                    )
                    loss = dice_loss(
                        masks_pred_prob, true_masks_onehot, multiclass=True
                    )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{"loss": f"{loss.item():.4f}"})

                # Validation
                if global_step % division_step == 0:
                    val_score = evaluate(model, val_loader, device, amp=False)
                    scheduler.step(val_score)

                    current_lr = optimizer.param_groups[0]["lr"]
                    csv_writer.writerow(
                        [
                            epoch + 1,
                            global_step,
                            f"{loss.item():.6f}",
                            f"{val_score_f:.6f}",
                            f"{current_lr:.2e}",
                        ]
                    )
                    csv_file.flush()

                    val_score_f = float(val_score)
                    if val_score_f > best_val_dice:
                        best_val_dice = val_score_f
                        ckpt = model.state_dict()
                        ckpt["mask_values"] = train_dataset.mask_values
                        torch.save(
                            ckpt,
                            os.path.join(checkpoint_dir, "best_checkpoint.pth"),
                        )
                        logging.info(
                            f"  New best val Dice: {val_score_f:.4f}"
                        )

        # End of epoch
        avg_loss = epoch_loss / max(len(train_loader), 1)
        logging.info(
            f"Epoch {epoch+1}/{epochs}  avg_loss={avg_loss:.4f}  "
            f"best_val_dice={best_val_dice:.4f}"
        )

        # Periodic checkpoint
        if (epoch + 1) % save_interval == 0:
            ckpt = model.state_dict()
            ckpt["mask_values"] = train_dataset.mask_values
            torch.save(
                ckpt,
                os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch+1}.pth"),
            )

        # Always save last for resume
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_dice": best_val_dice,
            },
            last_path,
        )

    csv_file.close()
    logging.info(f"Training complete. Best val Dice: {best_val_dice:.4f}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Train UNet on images and target masks"
    )
    parser.add_argument(
        "--train_dir", type=str, required=True, help="Training data directory"
    )
    parser.add_argument(
        "--val_dir", type=str, required=True, help="Validation data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=50, help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=8,
        dest="batch_size",
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        "-l",
        type=float,
        default=1e-5,
        dest="lr",
        help="Learning rate",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=1.0,
        help="Downscaling factor of the images",
    )
    parser.add_argument(
        "--bilinear",
        action="store_true",
        default=False,
        help="Use bilinear upsampling",
    )
    parser.add_argument(
        "--classes",
        "-c",
        type=int,
        default=2,
        help="Number of classes",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from last checkpoint",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)
    model.to(device=device)

    logging.info(
        f"Network:\n"
        f"\t{model.n_channels} input channels\n"
        f"\t{model.n_classes} output channels (classes)\n"
        f"\t{'Bilinear' if model.bilinear else 'Transposed conv'} upscaling"
    )

    try:
        train_model(
            model=model,
            device=device,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            img_scale=args.scale,
            save_interval=args.save_interval,
            resume=args.resume,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error(
            "Detected OutOfMemoryError! "
            "Enabling checkpointing to reduce memory usage, but this slows down training."
        )
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            device=device,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            img_scale=args.scale,
            save_interval=args.save_interval,
            resume=args.resume,
        )
```

- [ ] **Step 2: Verify syntax**

```bash
cd /home/LXT/lxt/DINOSAM/DINOSAM/Pytorch-UNet && python -c "import py_compile; py_compile.compile('train.py', doraise=True); print('Syntax OK')"
```
Expected: `Syntax OK`

- [ ] **Step 3: Smoke test — run 1 epoch on 0_sam dataset**

```bash
cd /home/LXT/lxt/DINOSAM/DINOSAM/Pytorch-UNet && python train.py \
  --train_dir /data3/LXT/DINOv3-SAM/0_sam_train \
  --val_dir /data3/LXT/DINOv3-SAM/0_sam_test \
  --output_dir /tmp/unet_smoke_0sam \
  --epochs 1 \
  --batch-size 8 \
  --classes 2
```

Expected: no crash, loss decreases, `best_checkpoint.pth` saved to `/tmp/unet_smoke_0sam/checkpoints/`, `last.pt` saved, `metrics.csv` populated.

Verify outputs:
```bash
ls /tmp/unet_smoke_0sam/checkpoints/
# Expected: best_checkpoint.pth  last.pt
head /tmp/unet_smoke_0sam/metrics.csv
# Expected: epoch,step,train_loss,val_dice,lr header + data rows
```

- [ ] **Step 4: Verify mask binarization in smoke test**

```bash
cd /home/LXT/lxt/DINOSAM/DINOSAM/Pytorch-UNet && python -c "
from utils.data_loading import RecursiveDataset
ds = RecursiveDataset('/data3/LXT/DINOv3-SAM/0_sam_train', scale=1.0)
s = ds[0]
vals = s['mask'].unique()
assert vals.tolist() == [0, 1], f'Expected [0, 1], got {vals.tolist()}'
print(f'Mask binarization OK: unique values = {vals.tolist()}')
"
```
Expected: `Mask binarization OK: unique values = [0, 1]`

- [ ] **Step 5: Verify resume checkpoint format**

```bash
cd /home/LXT/lxt/DINOSAM/DINOSAM/Pytorch-UNet && python -c "
import torch
ckpt = torch.load('/tmp/unet_smoke_0sam/checkpoints/last.pt', map_location='cpu')
print(f'Keys: {list(ckpt.keys())}')
print(f'Epoch: {ckpt[\"epoch\"]}')
print(f'Best val Dice: {ckpt[\"best_val_dice\"]}')
# Expected: model_state_dict, optimizer_state_dict, scheduler_state_dict, epoch, best_val_dice
"
```
Expected: 5 keys present, epoch=0, best_val_dice is a float.

- [ ] **Step 6: Clean up smoke test artifacts**

```bash
rm -rf /tmp/unet_smoke_0sam
```

- [ ] **Step 7: Commit**

```bash
git add Pytorch-UNet/train.py
git commit -m "feat: rewrite train.py for DINOSAM baseline comparison

- Add --train_dir, --val_dir, --output_dir CLI args (parameterized paths)
- Replace internal train/val split with external RecursiveDataset
- Pure multiclass Dice loss (no BCE/CE term)
- Remove wandb, replace with CSV logging to output_dir/metrics.csv
- Add --resume flag restoring model, optimizer, scheduler, epoch state
- Save best_checkpoint.pth (Dice improvement) + periodic checkpoint_epochN.pth + last.pt

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Create launch scripts per DINOSAM convention

**Files:**
- Create: (generated at train time — not committed to repo, but template documented here)

**Note:** This task documents the launch command pattern. Actual `run_train.sh` and `config.txt` are created at launch time (see Step 2).

- [ ] **Step 1: Document launch convention**

Launch scripts follow this template. For each experiment:

**config.txt** (plain-text key=value):
```
train_dir=/data3/LXT/DINOv3-SAM/0_sam_train
val_dir=/data3/LXT/DINOv3-SAM/0_sam_test
output_dir=/data3/LXT/DINOv3-SAM/model_UNet/UNet_0sam_ep50_bs8
epochs=50
batch_size=8
lr=1e-5
scale=1.0
classes=2
bilinear=False
save_interval=10
```

**run_train.sh**:
```bash
#!/bin/bash
cd /home/LXT/lxt/DINOSAM/DINOSAM/Pytorch-UNet
CUDA_VISIBLE_DEVICES=1 nohup python -u train.py \
  --train_dir /data3/LXT/DINOv3-SAM/0_sam_train \
  --val_dir /data3/LXT/DINOv3-SAM/0_sam_test \
  --output_dir /data3/LXT/DINOv3-SAM/model_UNet/UNet_0sam_ep50_bs8 \
  --epochs 50 \
  --batch-size 8 \
  --classes 2 \
  > /data3/LXT/DINOv3-SAM/model_UNet/UNet_0sam_ep50_bs8/train.log 2>&1 &
echo $! > /data3/LXT/DINOv3-SAM/model_UNet/UNet_0sam_ep50_bs8/pid.txt
```

**Second experiment** (1724 dataset):
```bash
#!/bin/bash
cd /home/LXT/lxt/DINOSAM/DINOSAM/Pytorch-UNet
CUDA_VISIBLE_DEVICES=2 nohup python -u train.py \
  --train_dir /data3/LXT/DINOv3-SAM/1724_train \
  --val_dir /data3/LXT/DINOv3-SAM/1724_test \
  --output_dir /data3/LXT/DINOv3-SAM/model_UNet/UNet_1724_ep50_bs8 \
  --epochs 50 \
  --batch-size 8 \
  --classes 2 \
  > /data3/LXT/DINOv3-SAM/model_UNet/UNet_1724_ep50_bs8/train.log 2>&1 &
echo $! > /data3/LXT/DINOv3-SAM/model_UNet/UNet_1724_ep50_bs8/pid.txt
```

**GPU allocation:** Use available GPUs (adjust `CUDA_VISIBLE_DEVICES`). If both experiments run concurrently, use different GPUs.

- [ ] **Step 2: Create output directories and launch**

```bash
# Experiment 1: 0_sam dataset
mkdir -p /data3/LXT/DINOv3-SAM/model_UNet/UNet_0sam_ep50_bs8/checkpoints
# Write config.txt and run_train.sh (from template above)
# Launch with nohup

# Experiment 2: 1724 dataset
mkdir -p /data3/LXT/DINOv3-SAM/model_UNet/UNet_1724_ep50_bs8/checkpoints
# Write config.txt and run_train.sh (from template above)
# Launch with nohup
```

- [ ] **Step 3: Verify training started**

```bash
# Check processes
ps aux | grep "train.py" | grep -v grep

# Check logs (wait 30s for first output)
sleep 30 && head -20 /data3/LXT/DINOv3-SAM/model_UNet/UNet_0sam_ep50_bs8/train.log
```

Expected: log shows device info, dataset size, "Epoch 1/50" progress bar.
