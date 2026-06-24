import random

import torch
from torch.nn import functional as F

from dinosam.loss.dice import dice_loss, boundary_loss, compute_iou
from dinosam.loss.depth_loss import depth_mse_loss


def iterative_train_step(
    model,
    images,
    point_coords,
    point_labels,
    depth_labels,
    gt_masks,
    original_sizes,
    prompt_generator,
    optimizer,
    device,
    args,
):
    """One training step with iterative mask refinement.

    Mask branch iterates n_sub_iterations times with error-based prompt updates.
    Depth branch runs once on the final refined mask (detached).
    """
    model.train()
    B = images.shape[0]

    image_embeddings, input_size = model.encode_images(images)

    n_sub = args.n_sub_iterations
    total_mask_loss = 0.0
    total_iou_loss = 0.0

    cur_point_coords = point_coords.clone()
    cur_point_labels = point_labels.clone()
    cur_mask_input = None

    final_low_res_masks = None

    for sub_iter in range(n_sub):
        low_res_masks, iou_pred, masks_fullres = model.forward_mask(
            image_embeddings=image_embeddings,
            point_coords=cur_point_coords,
            point_labels=cur_point_labels,
            original_sizes=original_sizes,
            input_size=input_size,
            mask_input=cur_mask_input,
        )

        mask_loss = dice_loss(masks_fullres, gt_masks)
        with torch.no_grad():
            true_iou = compute_iou(masks_fullres, gt_masks)
        iou_loss = F.mse_loss(iou_pred.squeeze(1), true_iou.squeeze(1))

        total_mask_loss += mask_loss
        total_iou_loss += iou_loss

        if sub_iter < n_sub - 1:
            with torch.no_grad():
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

                if args.mask_prob > 0 and random.random() < args.mask_prob:
                    cur_mask_input = low_res_masks.detach()
                else:
                    cur_mask_input = None

        final_low_res_masks = low_res_masks

    avg_mask_loss = total_mask_loss / n_sub
    avg_iou_loss = total_iou_loss / n_sub

    # cur_point_coords are already in original image space (scaled from low-res
    # by iterative_mask_step logic at the top of the loop). forward_depth does
    # per-sample apply_coords_torch internally — fixes original_sizes[0] bug.
    depth_pred, decoder_masks_fullres = model.forward_depth(
        image_embeddings=image_embeddings,
        low_res_masks=final_low_res_masks.detach(),
        input_size=input_size,
        original_sizes=original_sizes,
        depth_point_coords=cur_point_coords,
        depth_point_labels=cur_point_labels,
    )
    depth_loss = depth_mse_loss(depth_pred, depth_labels)

    loss = avg_mask_loss + avg_iou_loss + args.depth_loss_weight * depth_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), avg_mask_loss.item(), avg_iou_loss.item(), depth_loss.item()


def iterative_mask_step(
    model,
    images,
    point_coords,
    point_labels,
    gt_masks,
    original_sizes,
    prompt_generator,
    optimizer,
    device,
    args,
):
    """Phase 1: iterative mask refinement step with dice + iou loss.

    Optionally adds boundary loss (when args.boundary_loss is True) to
    reduce jagged mask edges. No depth branch is executed.
    """
    model.train()
    B = images.shape[0]

    image_embeddings, input_size = model.encode_images(images)

    n_sub = args.n_sub_iterations
    total_mask_loss = 0.0
    total_iou_loss = 0.0
    total_boundary_loss = 0.0

    cur_point_coords = point_coords.clone()
    cur_point_labels = point_labels.clone()
    cur_mask_input = None

    for sub_iter in range(n_sub):
        low_res_masks, iou_pred, masks_fullres = model.forward_mask(
            image_embeddings=image_embeddings,
            point_coords=cur_point_coords,
            point_labels=cur_point_labels,
            original_sizes=original_sizes,
            input_size=input_size,
            mask_input=cur_mask_input,
        )

        mask_loss = dice_loss(masks_fullres, gt_masks)
        with torch.no_grad():
            true_iou = compute_iou(masks_fullres, gt_masks)
        iou_loss = F.mse_loss(iou_pred.squeeze(1), true_iou.squeeze(1))

        if getattr(args, "boundary_loss", False):
            bl = boundary_loss(masks_fullres, gt_masks, boundary_width=args.boundary_width)
            mask_loss = mask_loss + args.boundary_loss_weight * bl
            total_boundary_loss += bl.item()

        total_mask_loss += mask_loss
        total_iou_loss += iou_loss

        if sub_iter < n_sub - 1:
            with torch.no_grad():
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

                if args.mask_prob > 0 and random.random() < args.mask_prob:
                    cur_mask_input = low_res_masks.detach()
                else:
                    cur_mask_input = None

    avg_mask_loss = total_mask_loss / n_sub
    avg_iou_loss = total_iou_loss / n_sub
    avg_boundary_loss = total_boundary_loss / n_sub

    loss = avg_mask_loss + avg_iou_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), avg_mask_loss.item(), avg_iou_loss.item(), avg_boundary_loss


def iterative_depth_step(
    model,
    images,
    point_coords,
    point_labels,
    depth_labels,
    gt_masks,
    original_sizes,
    prompt_generator,
    optimizer,
    device,
    args,
):
    """Phase 2: iterative depth training step.

    Mask branch runs once to get low_res_masks.
    Depth branch iterates n_sub_iterations times with point prompt updates.
    Each sub-iteration: depth_decoder outputs mask + depth, error-based points
    are sampled from the decoder mask vs GT mask.
    Loss = avg_dice_loss + depth_loss_weight * avg_depth_loss.
    """
    model.train()
    B = images.shape[0]

    # Mask branch: get low_res_masks
    image_embeddings, input_size = model.encode_images(images)
    low_res_masks, _, masks_fullres = model.forward_mask(
        image_embeddings=image_embeddings,
        point_coords=point_coords,
        point_labels=point_labels,
        original_sizes=original_sizes,
        input_size=input_size,
    )
    inner_mask_loss = dice_loss(masks_fullres, gt_masks)
    mask_input = low_res_masks

    n_sub = args.n_sub_iterations
    total_depth_loss = 0.0
    total_mask_loss = 0.0

    # Depth branch point prompts: start empty, build up iteratively
    cur_depth_coords = None
    cur_depth_labels = None

    for sub_iter in range(n_sub):
        depth_pred, decoder_masks_fullres = model.forward_depth(
            image_embeddings=image_embeddings,
            low_res_masks=mask_input,
            input_size=input_size,
            original_sizes=original_sizes,
            depth_point_coords=cur_depth_coords,
            depth_point_labels=cur_depth_labels,
        )

        # Losses
        d_loss = depth_mse_loss(depth_pred, depth_labels)
        # Decoder mask loss against GT at original resolution
        m_loss = dice_loss(decoder_masks_fullres, gt_masks)

        total_depth_loss += d_loss
        total_mask_loss += m_loss

        # Update depth point prompts for next sub-iteration
        if sub_iter < n_sub - 1:
            with torch.no_grad():
                # Downsample GT and pred to 256x256 for prompt generation
                decoder_masks_lowres = F.interpolate(
                    decoder_masks_fullres, size=(256, 256), mode='bilinear', align_corners=False,
                )
                gt_lowres = F.interpolate(
                    gt_masks.float(), size=(256, 256), mode='nearest',
                )
                pred_binary = (decoder_masks_lowres > 0).float()
                new_coords, new_labels = prompt_generator(gt_lowres, pred_binary)

                # Scale 256x256 coords to original image space (forward_depth
                # expects original-image-space coords and does per-sample
                # apply_coords_torch internally)
                lr_h, lr_w = 256, 256
                scale_x = original_sizes[:, 1].float().view(B, 1, 1) / lr_w
                scale_y = original_sizes[:, 0].float().view(B, 1, 1) / lr_h
                new_coords_scaled = new_coords.clone()
                new_coords_scaled[:, :, 0] = new_coords[:, :, 0] * scale_x.squeeze(-1)
                new_coords_scaled[:, :, 1] = new_coords[:, :, 1] * scale_y.squeeze(-1)

                if cur_depth_coords is None:
                    cur_depth_coords = new_coords_scaled
                    cur_depth_labels = new_labels
                else:
                    cur_depth_coords = torch.cat([cur_depth_coords, new_coords_scaled], dim=1)
                    cur_depth_labels = torch.cat([cur_depth_labels, new_labels], dim=1)

    avg_depth_loss = total_depth_loss / n_sub
    avg_decoder_mask_loss = total_mask_loss / n_sub

    loss = args.depth_loss_weight * avg_depth_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), avg_decoder_mask_loss.item(), avg_depth_loss.item()
