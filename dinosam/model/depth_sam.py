import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Dict, Any

from segment_anything.modeling import Sam, PromptEncoder, TwoWayTransformer
from segment_anything.utils.transforms import ResizeLongestSide

from .depth_decoder import DepthMaskDecoder
from .dinov3_encoder import DINOv3MonaEncoder
from .resnet_encoder import ResNetEncoder
from .swin_encoder import SwinBEncoder


class DepthSam(nn.Module):
    """SAM with an outer depth-prediction head.

    Pipeline:
      image → image_encoder → image_embeddings
      image_embeddings + point_prompts → inner prompt_encoder + mask_decoder → masks
      masks → outer prompt_encoder (mask input) → dense embeddings
      image_embeddings + outer embeddings → depth_decoder → depth prediction
    """

    def __init__(self, sam: Sam, encoder_type: str = "sam", dinov3_checkpoint: str = None,
                 adapter_type: str = "mona", depth_transformer_type: str = "twoway",
                 dpt_layers: list[int] = None, dpt_layers_depth: list[int] = None,
                 cellpose_model: str = "cyto3"):
        super().__init__()
        self.sam = sam
        self.encoder_type = encoder_type
        self.adapter_type = adapter_type
        self.depth_transformer_type = depth_transformer_type
        self.img_size = sam.image_encoder.img_size
        self.transform = ResizeLongestSide(self.img_size)

        prompt_embed_dim = 256
        image_size = self.img_size
        image_embedding_size = image_size // 16

        self.dinov3_encoder = None
        self.swin_encoder = None
        self.cellpose_encoder = None
        if encoder_type == "dinov3":
            if dinov3_checkpoint is None:
                raise ValueError("dinov3_checkpoint is required when encoder_type='dinov3'")
            self.dinov3_encoder = DINOv3MonaEncoder(
                checkpoint_path=dinov3_checkpoint,
                img_size=image_size,
                embed_dim=768,
                out_dim=256,
                adapter_type=adapter_type,
                dpt_layers=dpt_layers,
                dpt_layers_depth=dpt_layers_depth,
            )
        elif encoder_type.startswith("resnet"):
            self.resnet_encoder = ResNetEncoder(
                model_name=encoder_type, img_size=image_size,
            )
        elif encoder_type == "swin_b":
            self.swin_encoder = SwinBEncoder(img_size=image_size)
        elif encoder_type == "cellpose":
            from .cellpose_encoder import CellposeEncoder
            device = next(sam.image_encoder.parameters()).device
            self.cellpose_encoder = CellposeEncoder(
                model_type=cellpose_model,
                device=device,
            )

        # simple 模式：仅用全局池化 + MLP 预测深度，不需要 decoder
        if depth_transformer_type == "simple":
            self.simple_depth_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(256, 1024), nn.ReLU(),
                nn.Linear(1024, 256), nn.ReLU(),
                nn.Linear(256, 1),
            )
            self.outer_prompt_encoder = None
            self.depth_decoder = None
        else:
            self.outer_prompt_encoder = PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            )
            # 根据 depth_transformer_type 选择 transformer
            if depth_transformer_type == "dual_attn":
                from .dual_attn_transformer import DualAttnTransformer
                transformer = DualAttnTransformer(
                    embedding_dim=prompt_embed_dim, num_heads=8,
                    mlp_dim=2048, depth=2,
                )
            else:
                transformer = TwoWayTransformer(
                    depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048, num_heads=8,
                )
            self.depth_decoder = DepthMaskDecoder(
                transformer_dim=prompt_embed_dim, transformer=transformer,
            )

    @property
    def dual_dpt(self):
        """双头 DPT 模式是否启用。"""
        return (self.dinov3_encoder is not None
                and self.dinov3_encoder.dpt_layers_depth is not None)

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
        if self.depth_transformer_type == "simple":
            print("Simple mode: no depth decoder to initialize from SAM")
            return

        sam_pe = self.sam.prompt_encoder
        sam_md = self.sam.mask_decoder

        # outer_prompt_encoder ← SAM prompt_encoder
        self.outer_prompt_encoder.load_state_dict(sam_pe.state_dict())

        # depth_decoder ← partial from mask_decoder
        dd_sd = self.depth_decoder.state_dict()
        md_sd = sam_md.state_dict()

        # Transformer 权重迁移（dual_attn 模式跳过，使用随机初始化）
        if self.depth_transformer_type != "dual_attn":
            for key in list(dd_sd.keys()):
                if key.startswith("transformer."):
                    md_key = key  # same key name in mask_decoder
                    if md_key in md_sd:
                        dd_sd[key] = md_sd[md_key].clone()
        else:
            print("DualAttn mode: skipping transformer weight migration (random init)")

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
        # 双头模式：phase1 冻结 depth 编码分支（放在 simple 早退之前）
        if self.dual_dpt:
            enc = self.dinov3_encoder
            for module in (enc.dpt_head_depth, enc.adapter1_depth,
                           enc.adapter2_depth, enc.projection_depth):
                for param in module.parameters():
                    param.requires_grad = False
        if self.outer_prompt_encoder is None:
            # simple 模式：冻结 simple_depth_head
            for param in self.simple_depth_head.parameters():
                param.requires_grad = False
            return
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
        # Bypass SAM's apply_image_torch which uses shape[0],[1] (B,C) instead of
        # shape[2],[3] (H,W) to compute the resize target (upstream SAM bug).
        oldh, oldw = x.shape[2], x.shape[3]
        newh, neww = ResizeLongestSide.get_preprocess_shape(oldh, oldw, self.img_size)
        x = F.interpolate(x, size=(newh, neww), mode="bilinear", align_corners=False,
                          antialias=True)
        input_size = (newh, neww)
        x = (x - self.sam.pixel_mean.unsqueeze(0)) / self.sam.pixel_std.unsqueeze(0)
        padh = self.img_size - newh
        padw = self.img_size - neww
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
        elif self.encoder_type.startswith("resnet"):
            return self.resnet_encoder(images)
        elif self.encoder_type == "swin_b":
            return self.swin_encoder(images)
        elif self.encoder_type == "cellpose":
            return self.cellpose_encoder(images)
        else:
            input_images, input_size = self.preprocess(images)
            image_embeddings = self.sam.image_encoder(input_images)
            return image_embeddings, input_size

    def forward_mask(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor = None,
        point_labels: torch.Tensor = None,
        original_sizes: torch.Tensor = None,
        input_size: Tuple[int, int] = None,
        mask_input: torch.Tensor = None,
        return_fullres: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run mask branch (inner SAM prompt_encoder + mask_decoder).

        Args:
            image_embeddings: (B, 256, 64, 64)
            point_coords: (B, N, 2) in original image space; None for no prompt
            point_labels: (B, N); None for no prompt
            original_sizes: (B, 2)
            mask_input: (B, 1, 256, 256) optional low-res mask from previous iteration
            return_fullres: if False, skip postprocess_masks and return None for masks

        Returns:
            low_res_masks: (B, 1, 256, 256)
            iou_pred: (B, 1)
            masks: (B, 1, H, W) or None if return_fullres=False
        """
        if isinstance(image_embeddings, dict):
            image_embeddings = image_embeddings["mask"]
        B = image_embeddings.shape[0]
        inner_pe = self.sam.prompt_encoder.get_dense_pe()

        # Encode point prompts (if any) with per-sample coord transform
        if point_coords is not None:
            pc_list = []
            for i in range(B):
                orig_h, orig_w = original_sizes[i].tolist()
                pc = self.transform.apply_coords_torch(
                    point_coords[i:i+1], (orig_h, orig_w)
                )
                pc_list.append(pc)
            pc_batched = torch.cat(pc_list, dim=0)  # (B, N, 2)
            points = (pc_batched, point_labels)
        else:
            points = None

        # When no points and no mask, bypass prompt_encoder to avoid _get_batch_size
        # returning 1 regardless of actual batch size (upstream SAM issue)
        if points is None and mask_input is None:
            no_mask = self.sam.prompt_encoder.no_mask_embed.weight
            sparse_emb = torch.empty(B, 0, 256, device=image_embeddings.device)
            dense_emb = no_mask.reshape(1, -1, 1, 1).expand(B, -1, 64, 64)
        else:
            sparse_emb, dense_emb = self.sam.prompt_encoder(
                points=points, boxes=None, masks=mask_input,
            )

        # Per-sample mask_decoder (MaskDecoder.predict_masks uses repeat_interleave
        # which produces B²-shaped tensors when inputs are batched)
        all_low_res = []
        all_iou = []
        for i in range(B):
            masks_input_i = mask_input[i:i+1] if mask_input is not None else None
            low_res_i, iou_i = self.sam.mask_decoder(
                image_embeddings=image_embeddings[i:i+1],
                image_pe=inner_pe,
                sparse_prompt_embeddings=sparse_emb[i:i+1],
                dense_prompt_embeddings=dense_emb[i:i+1],
                multimask_output=False,
            )
            all_low_res.append(low_res_i)
            all_iou.append(iou_i)

        low_res_masks = torch.cat(all_low_res, dim=0)
        iou_pred = torch.cat(all_iou, dim=0)

        # Postprocessing stays per-sample (original_size varies per image)
        if return_fullres:
            masks_list = []
            for i in range(B):
                orig_h, orig_w = original_sizes[i].tolist()
                mask_i = self.sam.postprocess_masks(
                    low_res_masks[i:i+1], input_size=input_size,
                    original_size=(orig_h, orig_w),
                )
                masks_list.append(mask_i)
            masks = torch.cat(masks_list, dim=0)
        else:
            masks = None

        return low_res_masks, iou_pred, masks

    def forward_depth(
        self,
        image_embeddings: torch.Tensor,
        low_res_masks: torch.Tensor,
        input_size: Tuple[int, int],
        original_sizes: torch.Tensor,
        depth_point_coords: torch.Tensor = None,
        depth_point_labels: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run depth branch (outer prompt_encoder + depth_decoder).

        Args:
            image_embeddings: (B, 256, 64, 64)
            low_res_masks: (B, 1, 256, 256) typically detached from mask branch
            input_size: (H, W) of preprocessed images before padding
            original_sizes: (B, 2) original image sizes
            depth_point_coords: (B, N, 2) optional point prompts in original image space
            depth_point_labels: (B, N) optional point labels

        Returns:
            depth_pred: (B,)
            decoder_masks: (B, 1, H, W) postprocessed to original resolution
        """
        if isinstance(image_embeddings, dict):
            image_embeddings = image_embeddings["depth"]
        # simple 模式：全局池化 + MLP，跳过 decoder
        if self.depth_transformer_type == "simple":
            depth = self.simple_depth_head(image_embeddings)
            return depth.squeeze(-1), None

        B = image_embeddings.shape[0]
        outer_pe = self.outer_prompt_encoder.get_dense_pe()

        # Per-sample coord transform, then batch (matches forward_mask convention)
        points_batched = None
        if depth_point_coords is not None:
            pc_list = []
            for i in range(B):
                orig_h, orig_w = original_sizes[i].tolist()
                pc = self.transform.apply_coords_torch(
                    depth_point_coords[i:i+1], (orig_h, orig_w)
                )
                pc_list.append(pc)
            pc_batched = torch.cat(pc_list, dim=0)  # (B, N, 2)
            points_batched = (pc_batched, depth_point_labels)

        # Batched outer_prompt_encoder (PromptEncoder has no repeat_interleave issue)
        outer_sparse, outer_dense = self.outer_prompt_encoder(
            points=points_batched, boxes=None, masks=low_res_masks,
        )

        # Per-sample depth_decoder (DepthMaskDecoder uses repeat_interleave which
        # produces B²-shaped tensors when inputs are batched)
        all_decoder_masks = []
        all_depth = []
        for i in range(B):
            low_res_decoder_i, depth_i = self.depth_decoder(
                image_embeddings=image_embeddings[i:i+1],
                image_pe=outer_pe,
                sparse_prompt_embeddings=outer_sparse[i:i+1],
                dense_prompt_embeddings=outer_dense[i:i+1],
            )
            orig_h, orig_w = original_sizes[i].tolist()
            decoder_masks_i = self.sam.postprocess_masks(
                low_res_decoder_i, input_size=input_size,
                original_size=(orig_h, orig_w),
            )
            all_decoder_masks.append(decoder_masks_i)
            all_depth.append(depth_i.squeeze(-1))

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
