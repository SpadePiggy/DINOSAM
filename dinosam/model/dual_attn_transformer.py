"""双注意力 Transformer：用 PAM + CAM 替换 token self-attention。

结构上对标 SAM 的 TwoWayTransformer / TwoWayAttentionBlock，
将第一步的 token self-attention 替换为对 image features 的
Position Attention (PAM) + Channel Attention (CAM) 增强。
"""

import torch
from torch import Tensor, nn
from typing import Tuple, Type

from segment_anything.modeling.transformer import Attention
from segment_anything.modeling.common import MLPBlock

from dinosam.model.adapters.dual_attn_adapter import PAM_Module, CAM_Module


class DualAttnTransformerLayer(nn.Module):
    """单层双注意力 Transformer Block。

    与 SAM TwoWayAttentionBlock 对应，但第一步用 PAM+CAM 增强
    image features，而非 token self-attention。
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()

        # 步骤 1：PAM + CAM 增强 image features
        self.pam = PAM_Module(embedding_dim)
        self.cam = CAM_Module(embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)

        # 步骤 2：Token→Image cross-attention
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        # 步骤 3：MLP on tokens
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        # 步骤 4：Image→Token cross-attention
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm4 = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_pe: Tensor,
        key_pe: Tensor,
        hw_shape: Tuple[int, int],
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            queries: (B, N, C) — token embeddings
            keys: (B, HW, C) — image embeddings
            query_pe: (B, N, C) — token 位置编码
            key_pe: (B, HW, C) — image 位置编码
            hw_shape: (H, W) — 空间尺寸，用于 PAM/CAM 的 2D reshape

        Returns:
            queries: (B, N, C)
            keys: (B, HW, C)
        """
        B, HW, C = keys.shape
        H, W = hw_shape

        # 步骤 1：PAM + CAM 增强 image features（替代 token self-attention）
        keys_2d = keys.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        enhanced = self.pam(keys_2d) + self.cam(keys_2d)  # gamma 自学，不除以 2
        enhanced = enhanced.permute(0, 2, 3, 1).reshape(B, HW, C)
        keys = keys + enhanced  # 残差连接
        keys = self.norm1(keys)

        # 步骤 2：Token→Image cross-attention
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # 步骤 3：MLP on tokens
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # 步骤 4：Image→Token cross-attention
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class DualAttnTransformer(nn.Module):
    """双注意力 Transformer，接口与 SAM TwoWayTransformer 一致。"""

    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        depth: int = 2,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim

        self.layers = nn.ModuleList([
            DualAttnTransformerLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                activation=activation,
                attention_downsample_rate=attention_downsample_rate,
            )
            for _ in range(depth)
        ])

        # 最终 Token→Image attention + LayerNorm
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """与 SAM TwoWayTransformer.forward 签名一致。

        Args:
            image_embedding: (B, C, H, W)
            image_pe: (B, C, H, W)
            point_embedding: (B, N, C)

        Returns:
            queries: (B, N, C) — 处理后的 point embedding
            keys: (B, HW, C) — 处理后的 image embedding
        """
        # 记录空间尺寸，然后 flatten
        h, w = image_embedding.shape[2:]
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        queries = point_embedding
        keys = image_embedding

        # 逐层处理
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
                hw_shape=(h, w),
            )

        # 最终 Token→Image attention
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys
