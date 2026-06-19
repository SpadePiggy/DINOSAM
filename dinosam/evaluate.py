import os
import argparse
import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from PIL import Image
from segment_anything.build_sam import sam_model_registry

from dinosam.model import DepthSam
from dinosam.dataset import DepthDataset, collate_fn
from dinosam.loss.dice import dice_loss, compute_iou
from dinosam.loss.depth_loss import depth_mse_loss
from dinosam.prompt import DepthIterativePromptGenerator


DEPTH_SCALE = 100.0  # dataset normalizes depth by dividing by this


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=4, n_sub_iterations=1, mask_prob=0.0, save_masks_dir=None):
    """Evaluate model: iterative mask refinement + depth prediction.

    Args:
        save_masks_dir: if set, save per-depth-class first sample's pred/gt mask as images
    """
    model.eval()
    prompt_generator = DepthIterativePromptGenerator() if n_sub_iterations > 1 else None

    if save_masks_dir:
        os.makedirs(save_masks_dir, exist_ok=True)

    all_dice = []
    all_iou = []
    all_iou_pred = []
    all_depth_mae = []
    all_depth_pred = []
    all_depth_gt = []

    # Process one sample at a time for mask saving simplicity
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )

    sample_idx = 0
    for batch in dataloader:
        images = batch["image"].to(device)
        point_coords = batch["point_coords"].to(device)
        point_labels = batch["point_labels"].to(device)
        depth_labels = batch["depth"].to(device)
        gt_masks = batch["gt_mask"].to(device)
        original_sizes = batch["original_size"].to(device)

        image_embeddings, input_size = model.encode_images(images)
        B = images.shape[0]

        # ---- Mask branch (iterative) ----
        if n_sub_iterations > 1:
            cur_point_coords = point_coords.clone()
            cur_point_labels = point_labels.clone()
            cur_mask_input = None
            for sub_iter in range(n_sub_iterations):
                low_res_masks, iou_pred, masks_fullres = model.forward_mask(
                    image_embeddings=image_embeddings,
                    point_coords=cur_point_coords,
                    point_labels=cur_point_labels,
                    original_sizes=original_sizes,
                    input_size=input_size,
                    mask_input=cur_mask_input,
                )
                if sub_iter < n_sub_iterations - 1:
                    pred_binary = (low_res_masks > 0).float()
                    gt_lowres = F.interpolate(
                        gt_masks.float(), size=low_res_masks.shape[-2:], mode='nearest',
                    )
                    new_coords, new_labels = prompt_generator(gt_lowres, pred_binary)
                    lr_h, lr_w = low_res_masks.shape[-2:]
                    scale_x = original_sizes[:, 1].float().view(B, 1, 1) / lr_w
                    scale_y = original_sizes[:, 0].float().view(B, 1, 1) / lr_h
                    new_coords_scaled = new_coords.clone()
                    new_coords_scaled[:, :, 0] = new_coords[:, :, 0] * scale_x.squeeze(-1)
                    new_coords_scaled[:, :, 1] = new_coords[:, :, 1] * scale_y.squeeze(-1)
                    cur_point_coords = torch.cat([cur_point_coords, new_coords_scaled], dim=1)
                    cur_point_labels = torch.cat([cur_point_labels, new_labels], dim=1)
                    if mask_prob > 0 and random.random() < mask_prob:
                        cur_mask_input = low_res_masks.detach()
                    else:
                        cur_mask_input = None
        else:
            low_res_masks, iou_pred, masks_fullres = model.forward_mask(
                image_embeddings=image_embeddings,
                point_coords=point_coords,
                point_labels=point_labels,
                original_sizes=original_sizes,
                input_size=input_size,
            )

        # ---- Depth branch ----
        depth_pred, _ = model.forward_depth(
            image_embeddings=image_embeddings,
            low_res_masks=low_res_masks.detach(),
            input_size=input_size,
            original_sizes=original_sizes,
        )

        # ---- Per-sample metrics ----
        pred_binary = (torch.sigmoid(masks_fullres) > 0.5).float()
        for i in range(B):
            pred_i = pred_binary[i, 0]
            gt_i = gt_masks[i, 0]

            intersection = (pred_i * gt_i).sum()
            union = pred_i.sum() + gt_i.sum()
            dice = (2 * intersection / (union + 1e-7)).item()
            all_dice.append(dice)

            overlap = (pred_i * gt_i).sum()
            union_mask = ((pred_i + gt_i) > 0).float().sum()
            iou = (overlap / (union_mask + 1e-7)).item()
            all_iou.append(iou)

            all_iou_pred.append(iou_pred[i, 0].item())

            # Store in original scale [-100, 100]
            pred_orig = depth_pred[i].item() * DEPTH_SCALE
            gt_orig = depth_labels[i].item() * DEPTH_SCALE
            depth_mae = abs(pred_orig - gt_orig)
            all_depth_mae.append(depth_mae)
            all_depth_pred.append(pred_orig)
            all_depth_gt.append(gt_orig)

            # Save masks per sample in subfolder matching original depth class folder
            if save_masks_dir:
                sample_info = dataset.samples[sample_idx]
                # e.g. "/data3/.../f10/seq140516.jpg" -> folder_name="f10", img_name="seq140516"
                folder_name = os.path.basename(os.path.dirname(sample_info["image"]))
                img_name = os.path.splitext(os.path.basename(sample_info["image"]))[0]
                sub_dir = os.path.join(save_masks_dir, folder_name)
                os.makedirs(sub_dir, exist_ok=True)

                pred_arr = (pred_i.cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(pred_arr).save(
                    os.path.join(sub_dir, f"{img_name}_pred.png")
                )
                gt_arr = (gt_i.cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(gt_arr).save(
                    os.path.join(sub_dir, f"{img_name}_gt.png")
                )

            sample_idx += 1

    return {
        "dice": np.array(all_dice),
        "iou": np.array(all_iou),
        "iou_pred": np.array(all_iou_pred),
        "depth_mae": np.array(all_depth_mae),
        "depth_pred": np.array(all_depth_pred),
        "depth_gt": np.array(all_depth_gt),
    }


def print_metrics(name, metrics):
    depth_pred = metrics["depth_pred"]
    depth_gt = metrics["depth_gt"]
    depth_mae = metrics["depth_mae"]

    mse = np.mean((depth_pred - depth_gt) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((depth_gt - depth_pred) ** 2)
    ss_tot = np.sum((depth_gt - np.mean(depth_gt)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Segmentation Dice:  {metrics['dice'].mean():.4f}  (median: {np.median(metrics['dice']):.4f})")
    print(f"  Segmentation IoU:   {metrics['iou'].mean():.4f}  (median: {np.median(metrics['iou']):.4f})")
    print(f"  IoU Prediction:     {metrics['iou_pred'].mean():.4f}")
    print(f"  IoU Pred MAE:       {abs(metrics['iou_pred'] - metrics['iou']).mean():.4f}")
    print(f"  Depth MSE:          {mse:.4f}")
    print(f"  Depth MAE:          {depth_mae.mean():.4f}")
    print(f"  Depth RMSE:         {rmse:.4f}")
    print(f"  Depth R²:           {r2:.4f}")
    print(f"  Depth Pred:         mean={depth_pred.mean():.2f}  std={depth_pred.std():.2f}")
    print(f"  Depth GT:           mean={depth_gt.mean():.2f}  std={depth_gt.std():.2f}")
    print(f"  Samples:            {len(metrics['dice'])}")


def print_depth_by_class(metrics, dataset):
    depth_values = sorted(set(s["depth"] for s in dataset.samples))
    depth_gts = np.array([s["depth"] for s in dataset.samples])
    print(f"\n  Depth MAE by class (original scale):")
    print(f"    {'Depth':>8s}  {'MAE':>8s}  {'RMSE':>8s}  {'Pred Mean':>10s}  {'N':>4s}")
    print(f"    {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*4}")
    for d in depth_values:
        mask = depth_gts == d
        mae = metrics['depth_mae'][mask].mean()
        rmse = np.sqrt((metrics['depth_mae'][mask] ** 2).mean())
        pred_mean = metrics['depth_pred'][mask].mean()
        count = mask.sum()
        print(f"    {d:>+8.0f}  {mae:>8.2f}  {rmse:>8.2f}  {pred_mean:>10.2f}  {count:>4d}")


def plot_depth_scatter(metrics, save_dir, prefix=""):
    """Save scatter plots: pred vs gt, and residual vs gt."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depth_gt = metrics["depth_gt"]
    depth_pred = metrics["depth_pred"]
    residual = depth_pred - depth_gt

    os.makedirs(save_dir, exist_ok=True)

    # Plot 1: GT vs Pred
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(depth_gt, depth_pred, alpha=0.5, s=20, edgecolors="none")
    lim_min = min(depth_gt.min(), depth_pred.min()) - 5
    lim_max = max(depth_gt.max(), depth_pred.max()) + 5
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1, label="y = x")
    ax.set_xlabel("Depth GT", fontsize=14)
    ax.set_ylabel("Depth Pred", fontsize=14)
    ax.set_title(f"{prefix}Depth: Pred vs GT", fontsize=16)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path1 = os.path.join(save_dir, f"{prefix}depth_pred_vs_gt.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path1}")

    # Plot 2: GT vs Residual
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(depth_gt, residual, alpha=0.5, s=20, edgecolors="none", c="tab:orange")
    ax.axhline(y=0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Depth GT", fontsize=14)
    ax.set_ylabel("Residual (Pred - GT)", fontsize=14)
    ax.set_title(f"{prefix}Depth: Residual vs GT", fontsize=16)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path2 = os.path.join(save_dir, f"{prefix}depth_residual_vs_gt.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path2}")


def _load_model_state_dict(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    from dinosam.model.adapters import _remap_legacy_state_dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt["model_state_dict"] = _remap_legacy_state_dict(ckpt["model_state_dict"])
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        ckpt = _remap_legacy_state_dict(ckpt)
        model.load_state_dict(ckpt)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DepthSam models")
    parser.add_argument("--test_dir", type=str, default="/data3/LXT/DINOv3-SAM/1724_test")
    parser.add_argument("--sam_checkpoint", type=str, default="/data3/LXT/data/checkpoint/sam_vit_b_01ec64.pth",
                        help="Path to SAM vit_b pretrained checkpoint")
    parser.add_argument("--phase1_checkpoint", type=str,
                        default="/data3/LXT/DINOv3-SAM/model_sam/5_1/phase1_best.pt")
    parser.add_argument("--phase2_checkpoint", type=str,
                        default="/data3/LXT/DINOv3-SAM/model_sam/5_1/phase2_best.pt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_points", type=int, default=3)
    parser.add_argument("--n_sub_iterations", type=int, default=1,
                        help="Number of iterative sub-iterations for evaluation")
    parser.add_argument("--mask_prob", type=float, default=0.0,
                        help="Mask input probability for iterative evaluation")
    parser.add_argument("--save_masks_dir", type=str, default=None,
                        help="Directory to save per-depth-class first sample pred/gt masks and scatter plots")
    parser.add_argument("--encoder", type=str, default="sam", choices=["sam", "dinov3"],
                        help="Image encoder type: 'sam' (default) or 'dinov3' (DINOv3+Mona)")
    parser.add_argument("--adapter_type", type=str, default="mona",
                        choices=["mona", "fc", "dual_attn"],
                        help="Feature adapter type for DINOv3 encoder (default: mona)")
    parser.add_argument("--dinov3_checkpoint", type=str, default=None,
                        help="Path to DINOv3 ViT-B/16 checkpoint (required when --encoder dinov3)")
    args = parser.parse_args()

    if args.encoder == "sam" and args.adapter_type != "mona":
        print(f"Warning: --adapter_type '{args.adapter_type}' is ignored when --encoder sam")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DepthDataset(args.test_dir, max_points=args.max_points)
    print(f"Test dataset: {args.test_dir}")
    print(f"Test samples: {len(dataset)}")
    print(f"Iterative eval: n_sub={args.n_sub_iterations}, mask_prob={args.mask_prob}")

    # ---- Phase 1 ----
    if os.path.exists(args.phase1_checkpoint):
        print(f"\nLoading Phase 1 model: {args.phase1_checkpoint}")
        sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint)
        model = DepthSam(sam, encoder_type=args.encoder, dinov3_checkpoint=args.dinov3_checkpoint,
                         adapter_type=args.adapter_type)
        _load_model_state_dict(model, args.phase1_checkpoint, device)
        model.freeze_image_encoder()
        model.to(device)

        mask_dir_p1 = os.path.join(args.save_masks_dir, "phase1") if args.save_masks_dir else None
        metrics_p1 = evaluate(model, dataset, device,
                              batch_size=args.batch_size,
                              n_sub_iterations=args.n_sub_iterations,
                              mask_prob=args.mask_prob,
                              save_masks_dir=mask_dir_p1)
        print_metrics("Phase 1 Model (mask-only trained)", metrics_p1)
        print_depth_by_class(metrics_p1, dataset)
        if mask_dir_p1:
            plot_depth_scatter(metrics_p1, mask_dir_p1, prefix="phase1_")
            print(f"\n  Masks + plots saved to: {mask_dir_p1}")
    else:
        print(f"\nPhase 1 checkpoint not found: {args.phase1_checkpoint}")

    # ---- Phase 2 ----
    if os.path.exists(args.phase2_checkpoint):
        print(f"\nLoading Phase 2 model: {args.phase2_checkpoint}")
        sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint)
        model = DepthSam(sam, encoder_type=args.encoder, dinov3_checkpoint=args.dinov3_checkpoint,
                         adapter_type=args.adapter_type)
        _load_model_state_dict(model, args.phase2_checkpoint, device)
        model.freeze_image_encoder()
        model.to(device)

        mask_dir_p2 = os.path.join(args.save_masks_dir, "phase2") if args.save_masks_dir else None
        metrics_p2 = evaluate(model, dataset, device,
                              batch_size=args.batch_size,
                              n_sub_iterations=args.n_sub_iterations,
                              mask_prob=args.mask_prob,
                              save_masks_dir=mask_dir_p2)
        print_metrics("Phase 2 Model (depth trained)", metrics_p2)
        print_depth_by_class(metrics_p2, dataset)
        if mask_dir_p2:
            plot_depth_scatter(metrics_p2, mask_dir_p2, prefix="phase2_")
            print(f"\n  Masks + plots saved to: {mask_dir_p2}")

        # Comparison
        if os.path.exists(args.phase1_checkpoint):
            print(f"\n{'=' * 60}")
            print(f"  Phase 1 vs Phase 2 Comparison")
            print(f"{'=' * 60}")
            d_dice = metrics_p2['dice'].mean() - metrics_p1['dice'].mean()
            d_iou = metrics_p2['iou'].mean() - metrics_p1['iou'].mean()
            d_depth = metrics_p2['depth_mae'].mean() - metrics_p1['depth_mae'].mean()
            print(f"  Dice:       {metrics_p1['dice'].mean():.4f} -> {metrics_p2['dice'].mean():.4f}  ({d_dice:+.4f})")
            print(f"  IoU:        {metrics_p1['iou'].mean():.4f} -> {metrics_p2['iou'].mean():.4f}  ({d_iou:+.4f})")
            print(f"  Depth MAE:  {metrics_p1['depth_mae'].mean():.4f} -> {metrics_p2['depth_mae'].mean():.4f}  ({d_depth:+.4f})")
    else:
        print(f"\nPhase 2 checkpoint not found: {args.phase2_checkpoint} (skip)")


if __name__ == "__main__":
    main()
