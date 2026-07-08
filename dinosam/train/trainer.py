import os
import argparse
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from segment_anything.build_sam import sam_model_registry

from dinosam.model import DepthSam
from dinosam.dataset import DepthDataset, collate_fn
from dinosam.loss.dice import dice_loss, boundary_loss, compute_iou
from dinosam.loss.depth_loss import depth_mse_loss
from dinosam.prompt import DepthIterativePromptGenerator
from dinosam.train.iterative import iterative_train_step, iterative_mask_step, iterative_depth_step


# ---------------------------------------------------------------------------
# Phase 1: Mask-only training (dice + iou loss, iterative)
# ---------------------------------------------------------------------------

def train_one_epoch_phase1(model, dataloader, optimizer, device, epoch, writer, args):
    model.train()
    total_loss = 0.0
    total_mask_loss = 0.0
    total_iou_loss = 0.0
    total_boundary_loss = 0.0
    n_batches = 0

    use_bl = getattr(args, "boundary_loss", False)
    prompt_generator = DepthIterativePromptGenerator() if args.n_sub_iterations > 1 else None

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        point_coords = batch["point_coords"].to(device)
        point_labels = batch["point_labels"].to(device)
        gt_masks = batch["gt_mask"].to(device)
        original_sizes = batch["original_size"].to(device)

        if args.n_sub_iterations > 1:
            loss_val, mask_val, iou_val, bd_val = iterative_mask_step(
                model=model,
                images=images,
                point_coords=point_coords,
                point_labels=point_labels,
                gt_masks=gt_masks,
                original_sizes=original_sizes,
                prompt_generator=prompt_generator,
                optimizer=optimizer,
                device=device,
                args=args,
            )
        else:
            image_embeddings, input_size = model.encode_images(images)
            low_res_masks, iou_pred, masks_fullres = model.forward_mask(
                image_embeddings=image_embeddings,
                point_coords=point_coords,
                point_labels=point_labels,
                original_sizes=original_sizes,
                input_size=input_size,
            )
            mask_loss = dice_loss(masks_fullres, gt_masks)
            with torch.no_grad():
                true_iou = compute_iou(masks_fullres, gt_masks)
            iou_loss = nn.functional.mse_loss(
                iou_pred.squeeze(1), true_iou.squeeze(1)
            )

            if use_bl:
                bl = boundary_loss(masks_fullres, gt_masks, boundary_width=args.boundary_width)
                mask_loss = mask_loss + args.boundary_loss_weight * bl
                bd_val = bl.item()
            else:
                bd_val = 0.0

            loss = mask_loss + iou_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            mask_val = mask_loss.item()
            iou_val = iou_loss.item()

        total_loss += loss_val
        total_mask_loss += mask_val
        total_iou_loss += iou_val
        total_boundary_loss += bd_val
        n_batches += 1

        global_step = epoch * len(dataloader) + batch_idx
        if writer is not None:
            writer.add_scalar("train/loss", total_loss / n_batches, global_step)
            writer.add_scalar("train/mask_loss", total_mask_loss / n_batches, global_step)
            writer.add_scalar("train/iou_loss", total_iou_loss / n_batches, global_step)
            if use_bl:
                writer.add_scalar("train/boundary_loss", total_boundary_loss / n_batches, global_step)

        if (batch_idx + 1) % args.log_interval == 0:
            bd_str = f" bd={total_boundary_loss/n_batches:.4f}" if use_bl else ""
            print(
                f"  Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                f"loss={total_loss/n_batches:.4f} mask={total_mask_loss/n_batches:.4f} "
                f"iou={total_iou_loss/n_batches:.4f}{bd_str}"
            )

    return {
        "loss": total_loss / n_batches,
        "mask_loss": total_mask_loss / n_batches,
        "iou_loss": total_iou_loss / n_batches,
        "boundary_loss": total_boundary_loss / n_batches,
    }


@torch.no_grad()
def validate_phase1(model, dataloader, device, epoch, writer, args):
    model.eval()
    total_loss = 0.0
    total_boundary_loss = 0.0
    n_batches = 0
    use_bl = getattr(args, "boundary_loss", False)

    for batch in dataloader:
        images = batch["image"].to(device)
        point_coords = batch["point_coords"].to(device)
        point_labels = batch["point_labels"].to(device)
        gt_masks = batch["gt_mask"].to(device)
        original_sizes = batch["original_size"].to(device)

        image_embeddings, input_size = model.encode_images(images)
        low_res_masks, iou_pred, masks_fullres = model.forward_mask(
            image_embeddings=image_embeddings,
            point_coords=point_coords,
            point_labels=point_labels,
            original_sizes=original_sizes,
            input_size=input_size,
        )

        mask_loss = dice_loss(masks_fullres, gt_masks)
        true_iou = compute_iou(masks_fullres, gt_masks)
        iou_loss = nn.functional.mse_loss(
            iou_pred.squeeze(1), true_iou.squeeze(1)
        )

        if use_bl:
            bl = boundary_loss(masks_fullres, gt_masks, boundary_width=args.boundary_width)
            mask_loss = mask_loss + args.boundary_loss_weight * bl
            total_boundary_loss += bl.item()

        total_loss += (mask_loss + iou_loss).item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    bd_str = f" boundary={total_boundary_loss/n_batches:.4f}" if use_bl else ""
    print(f"  Val epoch {epoch}: loss={avg_loss:.4f}{bd_str}")
    if writer is not None:
        writer.add_scalar("val/loss", avg_loss, epoch)
        if use_bl:
            writer.add_scalar("val/boundary_loss", total_boundary_loss / n_batches, epoch)

    # ponytail: lazy import keeps viz optional; next(iter()) is deterministic on unshuffled val_loader
    if args.pca_viz and epoch % args.pca_interval == 0:
        from dinosam.viz.pca import visualize_model_features
        sample_batch = next(iter(dataloader))
        save_dir = os.path.join(args.pca_dir, f"epoch_{epoch:03d}")
        visualize_model_features(
            model, sample_batch["image"][:args.pca_samples],
            device, save_dir, prefix=f"ep{epoch:03d}_",
        )

    return avg_loss


# ---------------------------------------------------------------------------
# Phase 2: Depth training (dice + depth loss, mask branch frozen)
# ---------------------------------------------------------------------------

def train_one_epoch_phase2(model, dataloader, optimizer, device, epoch, writer, args):
    model.train()
    total_loss = 0.0
    total_mask_loss = 0.0
    total_depth_loss = 0.0
    n_batches = 0

    prompt_generator = DepthIterativePromptGenerator() if args.n_sub_iterations > 1 else None

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        point_coords = batch["point_coords"].to(device)
        point_labels = batch["point_labels"].to(device)
        depth_labels = batch["depth"].to(device)
        gt_masks = batch["gt_mask"].to(device)
        original_sizes = batch["original_size"].to(device)
        is_log_batch = (batch_idx + 1) % args.log_interval == 0

        if args.depth_transformer_type != "simple" and args.n_sub_iterations > 1:
            # Iterative depth training
            loss_val, mask_val, depth_val = iterative_depth_step(
                model=model,
                images=images,
                point_coords=point_coords,
                point_labels=point_labels,
                depth_labels=depth_labels,
                gt_masks=gt_masks,
                original_sizes=original_sizes,
                prompt_generator=prompt_generator,
                optimizer=optimizer,
                device=device,
                args=args,
            )
            total_loss += loss_val
            total_mask_loss += mask_val
            total_depth_loss += depth_val
            n_batches += 1
        else:
            # Standard (non-iterative) depth training
            image_embeddings, input_size = model.encode_images(images)

            # Skip full-res postprocessing on non-log batches (mask branch frozen)
            low_res_masks, iou_pred, masks_fullres = model.forward_mask(
                image_embeddings=image_embeddings,
                point_coords=point_coords,
                point_labels=point_labels,
                original_sizes=original_sizes,
                input_size=input_size,
                return_fullres=is_log_batch,
            )

            # Pass point coords in original image space (forward_depth does
            # per-sample apply_coords_torch internally — fixes original_sizes[0] bug)
            depth_pred, decoder_masks = model.forward_depth(
                image_embeddings=image_embeddings,
                low_res_masks=low_res_masks,
                input_size=input_size,
                original_sizes=original_sizes,
                depth_point_coords=point_coords,
                depth_point_labels=point_labels,
            )

            depth_loss = depth_mse_loss(depth_pred, depth_labels)
            loss = args.depth_loss_weight * depth_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            depth_val = depth_loss.item()
            if is_log_batch:
                inner_mask_loss = dice_loss(masks_fullres, gt_masks)
                if decoder_masks is not None:
                    decoder_mask_loss = dice_loss(decoder_masks, gt_masks)
                else:
                    decoder_mask_loss = 0.0
                mask_val = inner_mask_loss.item()
            else:
                mask_val = 0.0

            total_loss += loss_val
            total_depth_loss += depth_val
            n_batches += 1
            if is_log_batch:
                total_mask_loss += mask_val

        global_step = epoch * len(dataloader) + batch_idx
        if writer is not None:
            writer.add_scalar("train/loss", total_loss / n_batches, global_step)
            writer.add_scalar("train/mask_loss",
                              total_mask_loss / max(1, (batch_idx + 1) // args.log_interval),
                              global_step)
            writer.add_scalar("train/depth_loss", total_depth_loss / n_batches, global_step)

        if is_log_batch:
            print(
                f"  Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                f"loss={total_loss/n_batches:.4f} mask={total_mask_loss / max(1, (batch_idx + 1) // args.log_interval):.4f} "
                f"depth={total_depth_loss/n_batches:.4f}"
            )

    return {
        "loss": total_loss / n_batches,
        "mask_loss": total_mask_loss / max(1, n_batches // args.log_interval),
        "depth_loss": total_depth_loss / n_batches,
    }


@torch.no_grad()
def validate_phase2(model, dataloader, device, epoch, writer, args):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        point_coords = batch["point_coords"].to(device)
        point_labels = batch["point_labels"].to(device)
        depth_labels = batch["depth"].to(device)
        gt_masks = batch["gt_mask"].to(device)
        original_sizes = batch["original_size"].to(device)

        image_embeddings, input_size = model.encode_images(images)
        low_res_masks, iou_pred, _ = model.forward_mask(
            image_embeddings=image_embeddings,
            point_coords=point_coords,
            point_labels=point_labels,
            original_sizes=original_sizes,
            input_size=input_size,
            return_fullres=False,
        )

        # Pass point coords in original image space (forward_depth does
        # per-sample apply_coords_torch internally — fixes original_sizes[0] bug)
        depth_pred, _ = model.forward_depth(
            image_embeddings=image_embeddings,
            low_res_masks=low_res_masks.detach(),
            input_size=input_size,
            original_sizes=original_sizes,
            depth_point_coords=point_coords,
            depth_point_labels=point_labels,
        )

        depth_loss = depth_mse_loss(depth_pred, depth_labels)
        total_loss += (args.depth_loss_weight * depth_loss).item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    print(f"  Val epoch {epoch}: loss={avg_loss:.4f}")
    if writer is not None:
        writer.add_scalar("val/loss", avg_loss, epoch)
    return avg_loss


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def _log_trainable_params(model, label=""):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{label}Total params: {total:,}  Trainable: {trainable:,}")


def _build_model_and_data(args, device):
    sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint)

    # Parse dpt_layers (only for dpt_simple)
    dpt_layers = (
        [int(x.strip()) for x in args.dpt_layers.split(",")]
        if hasattr(args, 'dpt_layers') and args.adapter_type == "dpt_simple" else None
    )

    model = DepthSam(
        sam,
        encoder_type=args.encoder,
        dinov3_checkpoint=args.dinov3_checkpoint if args.encoder == "dinov3" else None,
        adapter_type=getattr(args, 'adapter_type', 'mona'),
        depth_transformer_type=getattr(args, 'depth_transformer_type', 'twoway'),
        dpt_layers=dpt_layers,
        fusion_k=getattr(args, 'fusion_k', 4),
    )
    model.freeze_image_encoder()
    model.init_depth_from_sam()
    model.to(device)

    train_dataset = DepthDataset(args.train_dir, max_points=args.max_points)
    val_dataset = DepthDataset(args.val_dir, max_points=args.max_points)
    print(f"Train samples: {len(train_dataset)}  Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )
    return model, train_loader, val_loader


def _save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss, adapter_type):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "adapter_type": adapter_type,
    }, path)


def _load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    # Remap legacy mona1/mona2 -> adapter1/adapter2 keys
    from dinosam.model.adapters import _remap_legacy_state_dict
    ckpt["model_state_dict"] = _remap_legacy_state_dict(ckpt["model_state_dict"])

    # Use strict=False to allow cross-adapter_type loading
    result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if result.missing_keys:
        print(f"Checkpoint missing keys (expected when switching adapter_type): {result.missing_keys}")
    if result.unexpected_keys:
        print(f"Checkpoint unexpected keys (ignored): {result.unexpected_keys}")

    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_val_loss = ckpt["best_val_loss"]
    print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    return start_epoch, best_val_loss


def run_phase1(model, train_loader, val_loader, device, args):
    """Phase 1: mask-only iterative training (dice + iou)."""
    print("\n" + "=" * 60)
    print("PHASE 1: Mask-only training (dice + iou loss)")
    print("=" * 60)

    model.freeze_image_encoder()
    model.freeze_depth_branch()
    _log_trainable_params(model, "Phase 1: ")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    start_epoch = 0
    best_val_loss = float("inf")

    # Resume from checkpoint
    resume_path = os.path.join(args.output_dir, "phase1_last.pt")
    if args.resume and os.path.exists(resume_path):
        start_epoch, best_val_loss = _load_checkpoint(
            resume_path, model, optimizer, scheduler, device,
        )

    log_dir = os.path.join(args.log_dir, "phase1")
    writer = SummaryWriter(log_dir)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Phase1 Epoch {epoch}/{args.epochs-1} ===")
        t0 = time.time()
        metrics = train_one_epoch_phase1(model, train_loader, optimizer, device, epoch, writer, args)
        scheduler.step()
        elapsed = time.time() - t0
        use_bl = getattr(args, "boundary_loss", False)
        bd_str = f" bd={metrics['boundary_loss']:.4f}" if use_bl else ""
        print(f"  Train: loss={metrics['loss']:.4f} mask={metrics['mask_loss']:.4f} "
              f"iou={metrics['iou_loss']:.4f}{bd_str} time={elapsed:.1f}s")

        val_loss = validate_phase1(model, val_loader, device, epoch, writer, args)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = os.path.join(args.output_dir, "phase1_best.pt")
            _save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss, args.adapter_type)
            print(f"  Saved best phase1 model (val_loss={val_loss:.4f})")

        if (epoch + 1) % args.save_interval == 0:
            path = os.path.join(args.output_dir, f"phase1_epoch_{epoch}.pt")
            _save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss, args.adapter_type)

        # Always save latest for resume
        _save_checkpoint(resume_path, model, optimizer, scheduler, epoch, best_val_loss, args.adapter_type)

    writer.close()
    print(f"\nPhase 1 complete. Best val loss: {best_val_loss:.4f}")
    return os.path.join(args.output_dir, "phase1_best.pt")


def run_phase2(model, train_loader, val_loader, device, args, phase1_checkpoint=None):
    """Phase 2: depth training (depth loss only), only depth branch trainable."""
    print("\n" + "=" * 60)
    print("PHASE 2: Depth training (depth loss, only depth branch trainable)")
    print("=" * 60)

    start_epoch = 0
    best_val_loss = float("inf")

    # Freeze everything, then only unfreeze depth branch
    for param in model.parameters():
        param.requires_grad = False
    if args.depth_transformer_type == "simple":
        # simple 模式：解冻 simple_depth_head
        for param in model.simple_depth_head.parameters():
            param.requires_grad = True
    else:
        for param in model.outer_prompt_encoder.parameters():
            param.requires_grad = True
        for param in model.depth_decoder.parameters():
            param.requires_grad = True

    # 【fusion mode】解冻 depth fusion head
    if hasattr(model, 'dinov3_encoder') and model.dinov3_encoder is not None \
            and hasattr(model.dinov3_encoder, 'depth_fusion') \
            and model.dinov3_encoder.depth_fusion is not None:
        for param in model.dinov3_encoder.depth_fusion.parameters():
            param.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Resume from checkpoint
    resume_path = os.path.join(args.output_dir, "phase2_last.pt")
    if args.resume and os.path.exists(resume_path):
        start_epoch, best_val_loss = _load_checkpoint(
            resume_path, model, optimizer, scheduler, device,
        )
    elif phase1_checkpoint and os.path.exists(phase1_checkpoint):
        print(f"Loading phase 1 checkpoint: {phase1_checkpoint}")
        ckpt = torch.load(phase1_checkpoint, map_location=device)

        # Check adapter_type compatibility
        ckpt_adapter_type = ckpt.get("adapter_type", "unknown")
        if ckpt_adapter_type != args.adapter_type:
            print(f"Warning: Phase 1 checkpoint adapter_type='{ckpt_adapter_type}' "
                  f"differs from current '{args.adapter_type}'. "
                  f"Some weights may not transfer.")

        # Remap legacy keys
        from dinosam.model.adapters import _remap_legacy_state_dict
        ckpt["model_state_dict"] = _remap_legacy_state_dict(ckpt["model_state_dict"])

        # Use strict=False for cross-adapter compatibility
        result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if result.missing_keys:
            print(f"Missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"Unexpected keys: {result.unexpected_keys}")

        # Re-init depth branch from SAM (phase1 checkpoint has random depth weights)
        model.init_depth_from_sam()

    _log_trainable_params(model, "Phase 2: ")

    log_dir = os.path.join(args.log_dir, "phase2")
    writer = SummaryWriter(log_dir)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Phase2 Epoch {epoch}/{args.epochs-1} ===")
        t0 = time.time()
        metrics = train_one_epoch_phase2(model, train_loader, optimizer, device, epoch, writer, args)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"  Train: loss={metrics['loss']:.4f} mask={metrics['mask_loss']:.4f} "
              f"depth={metrics['depth_loss']:.4f} time={elapsed:.1f}s")

        val_loss = validate_phase2(model, val_loader, device, epoch, writer, args)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = os.path.join(args.output_dir, "phase2_best.pt")
            _save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss, args.adapter_type)
            print(f"  Saved best phase2 model (val_loss={val_loss:.4f})")

        if (epoch + 1) % args.save_interval == 0:
            path = os.path.join(args.output_dir, f"phase2_epoch_{epoch}.pt")
            _save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss, args.adapter_type)

        # Always save latest for resume
        _save_checkpoint(resume_path, model, optimizer, scheduler, epoch, best_val_loss, args.adapter_type)

    writer.close()
    print(f"\nPhase 2 complete. Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train DepthSam (two-phase)")
    parser.add_argument("--train_dir", type=str, default="/data3/LXT/DINOv3-SAM/1724_train")
    parser.add_argument("--val_dir", type=str, default="/data3/LXT/DINOv3-SAM/1724_test")
    parser.add_argument("--sam_checkpoint", type=str, default=None, help="Path to SAM vit_b checkpoint")
    parser.add_argument("--output_dir", type=str, default="/data3/LXT/DINOv3-SAM/model_sam/5_1")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--depth_loss_weight", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--max_points", type=int, default=3)
    parser.add_argument("--n_sub_iterations", type=int, default=1,
                        help="Number of iterative mask refinement sub-iterations. 1 = no iteration")
    parser.add_argument("--mask_prob", type=float, default=0.5,
                        help="Probability of using mask input from previous prediction in each sub-iteration")
    parser.add_argument("--boundary_loss", action="store_true",
                        help="Enable boundary-aware loss (Dice + DT-weighted BCE) for smoother mask edges")
    parser.add_argument("--boundary_loss_weight", type=float, default=1.0,
                        help="Weight of boundary loss added to total loss (default: 1.0)")
    parser.add_argument("--boundary_width", type=int, default=3,
                        help="Half-width (px) of the morphological boundary band used by boundary_loss")
    parser.add_argument("--pca_viz", action="store_true",
                        help="Enable PCA visualization during validation")
    parser.add_argument("--pca_interval", type=int, default=1,
                        help="Generate PCA visualizations every N epochs")
    parser.add_argument("--pca_dir", type=str, default="./pca_viz",
                        help="Directory to save PCA visualization images")
    parser.add_argument("--pca_samples", type=int, default=4,
                        help="Number of validation samples to visualize")
    parser.add_argument("--train_phase", type=str, default="both",
                        choices=["1", "2", "both"],
                        help="Training phase: '1' = mask-only, '2' = depth-only, 'both' = sequential")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest checkpoint")
    parser.add_argument("--phase1_checkpoint", type=str, default=None,
                        help="Path to phase1 checkpoint (required when --train_phase 2 without --resume)")
    parser.add_argument("--encoder", type=str, default="sam", choices=["sam", "dinov3"],
                        help="Image encoder type: 'sam' (default) or 'dinov3' (DINOv3+Mona)")
    parser.add_argument("--dinov3_checkpoint", type=str, default=None,
                        help="Path to DINOv3 ViT-B/16 checkpoint (required when --encoder dinov3)")
    parser.add_argument("--adapter_type", type=str, default="mona",
                        choices=["mona", "fc", "dual_attn", "dpt_simple", "dpt_fusion"],
                        help="Adapter type: mona/fc/dual_attn (single-layer) or "
                             "dpt_simple (multi-level DPT fusion) or dpt_fusion (learned layer fusion)")
    parser.add_argument("--dpt_layers", type=str, default="2,5,8,11",
                        help="Comma-separated intermediate layer indices for DPT "
                             "(e.g., '2,5,8,11'). Only used with --adapter_type dpt_simple")
    parser.add_argument("--depth_transformer_type", type=str, default="twoway",
                        choices=["twoway", "dual_attn", "simple"],
                        help="Depth decoder transformer type. dual_attn requires training from scratch.")
    parser.add_argument("--fusion_k", type=int, default=4,
                        help="Top-k layers for fusion inference (only used with dpt_fusion)")
    args = parser.parse_args()

    if args.encoder == "sam" and args.adapter_type != "mona":
        print(f"Warning: --adapter_type '{args.adapter_type}' is ignored when --encoder sam")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    model, train_loader, val_loader = _build_model_and_data(args, device)

    phase1_checkpoint = None

    if args.train_phase in ("1", "both"):
        phase1_checkpoint = run_phase1(model, train_loader, val_loader, device, args)

    if args.train_phase in ("2", "both"):
        p1_ckpt = phase1_checkpoint if args.train_phase == "both" else args.phase1_checkpoint
        run_phase2(model, train_loader, val_loader, device, args,
                   phase1_checkpoint=p1_ckpt)


if __name__ == "__main__":
    main()
