import torch
from torch import nn
from typing import Tuple, Type

from segment_anything.modeling import TwoWayTransformer
from segment_anything.modeling.mask_decoder import MLP
from segment_anything.modeling.common import LayerNorm2d


class DepthMaskDecoder(nn.Module):
    """Like MaskDecoder but replaces the IoU head with a depth regression head.

    Always outputs a single mask + a single depth value (no multimask logic).
    """

    def __init__(
        self,
        *,
        transformer_dim: int = 256,
        transformer: nn.Module = None,
        activation: Type[nn.Module] = nn.GELU,
        depth_head_depth: int = 3,
        depth_head_hidden_dim: int = 256,
    ):
        super().__init__()
        self.transformer_dim = transformer_dim
        if transformer is None:
            transformer = TwoWayTransformer(
                depth=2, embedding_dim=transformer_dim, mlp_dim=2048, num_heads=8,
            )
        self.transformer = transformer

        self.depth_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
             for _ in range(self.num_mask_tokens)]
        )

        self.depth_prediction_head = MLP(
            transformer_dim, depth_head_hidden_dim, 1, depth_head_depth,
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate output tokens
        output_tokens = torch.cat([self.depth_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        depth_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : 1 + self.num_mask_tokens, :]

        # Upscale and predict masks
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list = [
            self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            for i in range(self.num_mask_tokens)
        ]
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Predict depth
        depth_pred = self.depth_prediction_head(depth_token_out)  # (B, 1)

        return masks, depth_pred
