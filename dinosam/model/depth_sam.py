import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Dict, Any

from segment_anything.modeling import Sam, PromptEncoder, TwoWayTransformer
from segment_anything.utils.transforms import ResizeLongestSide

from .depth_decoder import DepthMaskDecoder
from .dinov3_encoder import DINOv3MonaEncoder


class DepthSam(nn.Module):
    """SAM with an outer depth-prediction head.

    Pipeline:
      image → image_encoder → image_embeddings
      image_embeddings + point_prompts → inner prompt_encoder + mask_decoder → masks
      masks → outer prompt_encoder (mask input) → dense embeddings
      image_embeddings + outer embeddings → depth_decoder → depth prediction
    """

    def __init__(self, sam: Sam, encoder_type: str = "sam", dinov3_checkpoint: str = None):
        super().__init__()
        self.sam = sam
        self.encoder_type = encoder_type
        self.img_size = sam.image_encoder.img_size
        self.transform = ResizeLongestSide(self.img_size)

        prompt_embed_dim = 256
        image_size = self.img_size
        image_embedding_size = image_size // 16

        if encoder_type == "dinov3":
            if dinov3_checkpoint is None:
                raise ValueError("dinov3_checkpoint is required when encoder_type='dinov3'")
            self.dinov3_encoder = DINOv3MonaEncoder(
                checkpoint_path=dinov3_checkpoint,
                img_size=image_size,
                embed_dim=768,
                out_dim=256,
            )
        else:
            self.dinov3_encoder = None

        self.outer_prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )
        self.depth_decoder = DepthMaskDecoder(
            transformer_dim=prompt_embed_dim,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048, num_heads=8,
            ),
        )

    def init_depth_from_sam(self):
        """Initialize depth branch weights from SAM's inner prompt_encoder + mask_decoder.

        outer_prompt_encoder ← SAM prompt_encoder (identical structure)
        depth_decoder:
          - transformer ← mask_decoder.transformer (identical)
          - depth_token ← iou_token (same shape)
          - mask_tokens[0] ← mask_decoder.mask_tokens[0] (take first of 4)
          - output_upscaling ← mask_decoder.output_upscaling (identical)
          - output_hypernetworks_mlps[0] ← mask_decoder.output_hypernetworks_mlps[0] (take first)
          - depth_prediction_head: randomly initialized (no SAM equivalent)
        """
        sam_pe = self.sam.prompt_encoder
        sam_md = self.sam.mask_decoder

        # outer_prompt_encoder ← SAM prompt_encoder
        self.outer_prompt_encoder.load_state_dict(sam_pe.state_dict())

        # depth_decoder ← partial from mask_decoder
        dd_sd = self.depth_decoder.state_dict()
        md_sd = sam_md.state_dict()

        # Transformer (identical structure)
        for key in list(dd_sd.keys()):
            if key.startswith("transformer."):
                md_key = key  # same key name in mask_decoder
                if md_key in md_sd:
                    dd_sd[key] = md_sd[md_key].clone()

        # depth_token ← iou_token
        dd_sd["depth_token.weight"] = md_sd["iou_token.weight"].clone()

        # mask_tokens[0] ← mask_decoder.mask_tokens[0]
        dd_sd["mask_tokens.weight"] = md_sd["mask_tokens.weight"][0:1].clone()

        # output_upscaling (identical structure)
        for key in list(dd_sd.keys()):
            if key.startswith("output_upscaling."):
                if key in md_sd:
                    dd_sd[key] = md_sd[key].clone()

        # output_hypernetworks_mlps[0] ← mask_decoder.output_hypernetworks_mlps[0]
        for key in list(dd_sd.keys()):
            if key.startswith("output_hypernetworks_mlps.0."):
                md_key = key  # same key
                if md_key in md_sd:
                    dd_sd[key] = md_sd[md_key].clone()

        # depth_prediction_head: keep random init (no equivalent in SAM)

        self.depth_decoder.load_state_dict(dd_sd)
        print("Initialized depth branch from SAM inner weights (depth_prediction_head random)")

    def freeze_image_encoder(self):
        if self.encoder_type == "dinov3":
            self.dinov3_encoder.freeze_backbone()
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False

    def freeze_depth_branch(self):
        """Freeze outer_prompt_encoder + depth_decoder (for phase 1: mask-only training)."""
        for param in self.outer_prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.depth_decoder.parameters():
            param.requires_grad = False

    def freeze_mask_branch(self):
        """Freeze SAM prompt_encoder + mask_decoder (for phase 2: depth-only training)."""
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = False

    def preprocess(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.transform.apply_image_torch(x)
        input_size = x.shape[-2:]
        x = (x - self.sam.pixel_mean.unsqueeze(0)) / self.sam.pixel_std.unsqueeze(0)
        h, w = x.shape[-2:]
        padh = self.sam.image_encoder.img_size - h
        padw = self.sam.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x, input_size

    def encode_images(self, images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Preprocess and encode images. Called once per training step.

        Returns:
            image_embeddings: (B, 256, 64, 64)
            input_size: (H, W) of preprocessed images before padding
        """
        if self.encoder_type == "dinov3":
            return self.dinov3_encoder(images)
        else:
            input_images, input_size = self.preprocess(images)
            image_embeddings = self.sam.image_encoder(input_images)
            return image_embeddings, input_size

    def forward_mask(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        original_sizes: torch.Tensor,
        input_size: Tuple[int, int],
        mask_input: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run mask branch (inner SAM prompt_encoder + mask_decoder) per image.

        Args:
            image_embeddings: (B, 256, 64, 64)
            point_coords: (B, N, 2) in original image space
            point_labels: (B, N)
            original_sizes: (B, 2)
            mask_input: (B, 1, 256, 256) optional low-res mask from previous iteration

        Returns:
            low_res_masks: (B, 1, 256, 256)
            iou_pred: (B, 1)
            masks: (B, 1, H, W) postprocessed to original resolution
        """
        B = image_embeddings.shape[0]
        inner_pe = self.sam.prompt_encoder.get_dense_pe()

        all_masks = []
        all_low_res = []
        all_iou = []

        for i in range(B):
            curr_embedding = image_embeddings[i:i+1]
            orig_h, orig_w = original_sizes[i].tolist()

            pc = self.transform.apply_coords_torch(
                point_coords[i:i+1], (orig_h, orig_w)
            )
            points = (pc, point_labels[i:i+1])

            masks_input_i = mask_input[i:i+1] if mask_input is not None else None

            sparse_emb, dense_emb = self.sam.prompt_encoder(
                points=points, boxes=None, masks=masks_input_i,
            )
            low_res_masks, iou_pred = self.sam.mask_decoder(
                image_embeddings=curr_embedding,
                image_pe=inner_pe,
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )

            masks = self.sam.postprocess_masks(
                low_res_masks, input_size=input_size, original_size=(orig_h, orig_w),
            )

            all_masks.append(masks)
            all_low_res.append(low_res_masks)
            all_iou.append(iou_pred)

        return (
            torch.cat(all_low_res, dim=0),
            torch.cat(all_iou, dim=0),
            torch.cat(all_masks, dim=0),
        )

    def forward_depth(
        self,
        image_embeddings: torch.Tensor,
        low_res_masks: torch.Tensor,
        input_size: Tuple[int, int],
        original_sizes: torch.Tensor,
        depth_point_coords: torch.Tensor = None,
        depth_point_labels: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run depth branch (outer prompt_encoder + depth_decoder) per image.

        Args:
            image_embeddings: (B, 256, 64, 64)
            low_res_masks: (B, 1, 256, 256) typically detached from mask branch
            input_size: (H, W) of preprocessed images before padding
            original_sizes: (B, 2) original image sizes
            depth_point_coords: (B, N, 2) optional point prompts in 256x256 space
            depth_point_labels: (B, N) optional point labels

        Returns:
            depth_pred: (B,)
            decoder_masks: (B, 1, H, W) postprocessed to original resolution
        """
        B = image_embeddings.shape[0]
        outer_pe = self.outer_prompt_encoder.get_dense_pe()
        all_depth = []
        all_decoder_masks = []

        for i in range(B):
            curr_embedding = image_embeddings[i:i+1]
            orig_h, orig_w = original_sizes[i].tolist()

            points_i = None
            if depth_point_coords is not None:
                pc = depth_point_coords[i:i+1]
                pl = depth_point_labels[i:i+1]
                points_i = (pc, pl)

            outer_sparse, outer_dense = self.outer_prompt_encoder(
                points=points_i, boxes=None, masks=low_res_masks[i:i+1],
            )
            low_res_decoder_masks, depth_pred = self.depth_decoder(
                image_embeddings=curr_embedding,
                image_pe=outer_pe,
                sparse_prompt_embeddings=outer_sparse,
                dense_prompt_embeddings=outer_dense,
            )
            decoder_masks = self.sam.postprocess_masks(
                low_res_decoder_masks, input_size=input_size, original_size=(orig_h, orig_w),
            )
            all_depth.append(depth_pred.squeeze(-1))
            all_decoder_masks.append(decoder_masks)

        return torch.cat(all_depth, dim=0), torch.cat(all_decoder_masks, dim=0)

    def forward(
        self,
        images: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        original_sizes: torch.Tensor,
        depth_labels: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> Dict[str, Any]:
        """Full forward: mask branch + depth branch (non-iterative path)."""
        image_embeddings, input_size = self.encode_images(images)

        low_res_masks, iou_pred, masks = self.forward_mask(
            image_embeddings=image_embeddings,
            point_coords=point_coords,
            point_labels=point_labels,
            original_sizes=original_sizes,
            input_size=input_size,
        )

        depth_pred, _ = self.forward_depth(
            image_embeddings=image_embeddings,
            low_res_masks=low_res_masks.detach(),
            input_size=input_size,
            original_sizes=original_sizes,
        )

        return {
            "masks": masks,
            "low_res_masks": low_res_masks,
            "iou_predictions": iou_pred,
            "depth_pred": depth_pred,
        }
