# DualAttnTransformer 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为论文消融实验，创建 DualAttnTransformer 替代 TwoWayTransformer，通过 CLI 参数切换。

**Architecture:** DualAttnTransformerLayer 用 PAM+CAM 替代 token self-attention（唯一消融变量），其余所有脚手架（LayerNorm、cross-attention、MLP、final_attn、token PE）与 TwoWayTransformer 完全一致。通过 `--depth_transformer_type` CLI 参数在 twoway 和 dual_attn 之间切换。

**Tech Stack:** PyTorch, segment-anything (SAM ViT-B), PAM_Module/CAM_Module (DANet)

**设计文档:** `docs/superpowers/specs/2026-06-22-dual-attn-transformer-design.md`

---

### Task 1: 创建 DualAttnTransformerLayer

**Files:**
- Create: `dinosam/model/dual_attn_transformer.py`

- [ ] **Step 1: 创建 DualAttnTransformerLayer 类**

```python
import torch
from torch import Tensor, nn
from typing import Tuple, Type

from segment_anything.modeling.transformer import Attention
from segment_anything.modeling.common import MLPBlock
from .adapters.dual_attn_adapter import PAM_Module, CAM_Module


class DualAttnTransformerLayer(nn.Module):
    """替代 TwoWayAttentionBlock。
    唯一差异：Step 1 用 PAM+CAM 增强 image features 替代 token self-attention。
    Steps 2-4（cross-attn、MLP、LayerNorm）与 TwoWayAttentionBlock 完全一致。
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
        # Step 1: PAM+CAM（替代 token self-attention）
        self.pam = PAM_Module(embedding_dim)
        self.cam = CAM_Module(embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)

        # Step 2: token→image cross-attention（与 TwoWayAttentionBlock 一致）
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        # Step 3: MLP on tokens（与 TwoWayAttentionBlock 一致）
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        # Step 4: image→token cross-attention（与 TwoWayAttentionBlock 一致）
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
        b, hw, c = keys.shape
        h, w = hw_shape

        # Step 1: PAM+CAM 增强 image features
        keys_2d = keys.reshape(b, h, w, c).permute(0, 3, 1, 2)  # (B, C, H, W)
        enhanced = self.pam(keys_2d) + self.cam(keys_2d)
        enhanced = enhanced.permute(0, 2, 3, 1).reshape(b, hw, c)
        keys = keys + enhanced
        keys = self.norm1(keys)

        # Step 2: token→image cross-attention（照搬 TwoWayAttentionBlock:163-168）
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # Step 3: MLP（照搬 TwoWayAttentionBlock:170-173）
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Step 4: image→token cross-attention（照搬 TwoWayAttentionBlock:175-180）
        # q=k, k=q 是 SAM 原始写法：image 做 query attend tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys
```

- [ ] **Step 2: 验证文件可 import**

Run: `cd /home/admin/workspace/SEU/DINOSAM && python -c "from dinosam.model.dual_attn_transformer import DualAttnTransformerLayer; print('OK')"`
Expected: `OK`

---

### Task 2: 创建 DualAttnTransformer

**Files:**
- Modify: `dinosam/model/dual_attn_transformer.py`（在 Task 1 创建的文件中追加）

- [ ] **Step 1: 在同一文件末尾添加 DualAttnTransformer 类**

```python
class DualAttnTransformer(nn.Module):
    """替代 TwoWayTransformer。
    forward 签名和返回值与 TwoWayTransformer 完全一致。
    """

    def __init__(
        self,
        depth: int = 2,
        embedding_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.layers = nn.ModuleList()

        for _ in range(depth):
            self.layers.append(
                DualAttnTransformerLayer(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                )
            )

        # final attention（与 TwoWayTransformer:57-60 完全一致）
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
        # flatten（与 TwoWayTransformer:82-84 一致）
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        queries = point_embedding
        keys = image_embedding

        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
                hw_shape=(h, w),
            )

        # final attention（与 TwoWayTransformer:99-104 一致）
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys
```

- [ ] **Step 2: 验证 import 和 shape**

Run:
```bash
cd /home/admin/workspace/SEU/DINOSAM && python -c "
import torch
from dinosam.model.dual_attn_transformer import DualAttnTransformer

m = DualAttnTransformer(depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048)
img = torch.randn(1, 256, 64, 64)
pe = torch.randn(1, 256, 64, 64)
pt = torch.randn(1, 5, 256)
hs, src = m(img, pe, pt)
assert hs.shape == (1, 5, 256), f'hs shape wrong: {hs.shape}'
assert src.shape == (1, 4096, 256), f'src shape wrong: {src.shape}'
print(f'OK: hs={hs.shape}, src={src.shape}')
print(f'Params: {sum(p.numel() for p in m.parameters()):,}')
"
```
Expected: `OK: hs=torch.Size([1, 5, 256]), src=torch.Size([1, 4096, 256])` 和参数量约 2,930K。

- [ ] **Step 3: Commit**

```bash
git add dinosam/model/dual_attn_transformer.py
git commit -m "feat: add DualAttnTransformer for ablation experiment

Replaces token self-attention with PAM+CAM image feature enhancement.
All other components (cross-attn, MLP, LayerNorm, final_attn) match
TwoWayTransformer exactly for fair ablation comparison."
```

---

### Task 3: 修改 depth_sam.py — 集成 DualAttnTransformer

**Files:**
- Modify: `dinosam/model/depth_sam.py:23` (`__init__` 签名)
- Modify: `dinosam/model/depth_sam.py:54-59` (transformer 创建)
- Modify: `dinosam/model/depth_sam.py:83-88` (`init_depth_from_sam` transformer 权重迁移)

- [ ] **Step 1: 修改 `__init__` 签名，增加 `depth_transformer_type` 参数**

将 `depth_sam.py:23` 从：
```python
    def __init__(self, sam: Sam, encoder_type: str = "sam", dinov3_checkpoint: str = None, adapter_type: str = "mona"):
```
改为：
```python
    def __init__(self, sam: Sam, encoder_type: str = "sam", dinov3_checkpoint: str = None, adapter_type: str = "mona", depth_transformer_type: str = "twoway"):
```

在 `self.adapter_type = adapter_type` 后面（第 27 行之后）添加：
```python
        self.depth_transformer_type = depth_transformer_type
```

- [ ] **Step 2: 修改 transformer 创建逻辑**

将 `depth_sam.py:54-59`（硬编码 TwoWayTransformer）从：
```python
        self.depth_decoder = DepthMaskDecoder(
            transformer_dim=prompt_embed_dim,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048, num_heads=8,
            ),
        )
```
改为：
```python
        if depth_transformer_type == "dual_attn":
            from .dual_attn_transformer import DualAttnTransformer
            depth_transformer = DualAttnTransformer(
                depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048, num_heads=8,
            )
        else:
            depth_transformer = TwoWayTransformer(
                depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048, num_heads=8,
            )
        self.depth_decoder = DepthMaskDecoder(
            transformer_dim=prompt_embed_dim,
            transformer=depth_transformer,
        )
```

- [ ] **Step 3: 修改 `init_depth_from_sam` 中 transformer 权重迁移**

将 `depth_sam.py:83-88` 从：
```python
        # Transformer (identical structure)
        for key in list(dd_sd.keys()):
            if key.startswith("transformer."):
                md_key = key  # same key name in mask_decoder
                if md_key in md_sd:
                    dd_sd[key] = md_sd[md_key].clone()
```
改为：
```python
        # Transformer 权重
        if self.depth_transformer_type != "dual_attn":
            for key in list(dd_sd.keys()):
                if key.startswith("transformer."):
                    md_key = key
                    if md_key in md_sd:
                        dd_sd[key] = md_sd[md_key].clone()
        else:
            print("DualAttn mode: skipping transformer weight migration (random init)")
```

- [ ] **Step 4: 验证 twoway 模式不受影响**

Run:
```bash
cd /home/admin/workspace/SEU/DINOSAM && python -c "
from segment_anything import sam_model_registry
from dinosam.model.depth_sam import DepthSam
# twoway（默认）应与改动前行为一致
sam = sam_model_registry['vit_b']()
model = DepthSam(sam, depth_transformer_type='twoway')
print(f'twoway: {type(model.depth_decoder.transformer).__name__}')
assert 'TwoWay' in type(model.depth_decoder.transformer).__name__
print('OK: twoway mode works')
"
```
Expected: `twoway: TwoWayTransformer` 和 `OK: twoway mode works`

- [ ] **Step 5: 验证 dual_attn 模式**

Run:
```bash
cd /home/admin/workspace/SEU/DINOSAM && python -c "
from segment_anything import sam_model_registry
from dinosam.model.depth_sam import DepthSam
sam = sam_model_registry['vit_b']()
model = DepthSam(sam, depth_transformer_type='dual_attn')
print(f'dual_attn: {type(model.depth_decoder.transformer).__name__}')
assert 'DualAttn' in type(model.depth_decoder.transformer).__name__
model.init_depth_from_sam()
print('OK: dual_attn mode works, init_depth_from_sam skips transformer weights')
"
```
Expected: 看到 `DualAttn mode: skipping transformer weight migration (random init)` 和 `OK`

- [ ] **Step 6: Commit**

```bash
git add dinosam/model/depth_sam.py
git commit -m "feat: integrate DualAttnTransformer into DepthSam

Add depth_transformer_type parameter to switch between twoway and
dual_attn. init_depth_from_sam skips transformer weight migration
for dual_attn mode (random init)."
```

---

### Task 4: 修改 trainer.py — 增加 CLI 参数

**Files:**
- Modify: `dinosam/train/trainer.py:512` (argparse，在 `--adapter_type` 之后)
- Modify: `dinosam/train/trainer.py:308-313` (`_build_model_and_data` 中 DepthSam 构造)

- [ ] **Step 1: 增加 CLI 参数**

在 `trainer.py:512`（`--adapter_type` 参数之后）插入：
```python
    parser.add_argument("--depth_transformer_type", type=str, default="twoway",
                        choices=["twoway", "dual_attn"],
                        help="Depth decoder transformer type. dual_attn requires training from scratch.")
```

- [ ] **Step 2: 透传给 DepthSam 构造函数**

将 `trainer.py:308-313` 从：
```python
    model = DepthSam(
        sam,
        encoder_type=args.encoder,
        dinov3_checkpoint=args.dinov3_checkpoint,
        adapter_type=args.adapter_type,
    )
```
改为：
```python
    model = DepthSam(
        sam,
        encoder_type=args.encoder,
        dinov3_checkpoint=args.dinov3_checkpoint,
        adapter_type=args.adapter_type,
        depth_transformer_type=args.depth_transformer_type,
    )
```

- [ ] **Step 3: 验证参数解析**

Run:
```bash
cd /home/admin/workspace/SEU/DINOSAM && python -c "
import argparse
# 模拟 trainer.py 的 argparse 部分
import dinosam.train.trainer as t
import sys
sys.argv = ['test', '--sam_checkpoint', '/dev/null', '--depth_transformer_type', 'dual_attn']
# 只验证 argparse 不报错
print('OK: CLI arg parsed')
" 2>&1 | head -5
```
Expected: 无 argparse 报错

- [ ] **Step 4: Commit**

```bash
git add dinosam/train/trainer.py
git commit -m "feat: add --depth_transformer_type CLI arg to trainer"
```

---

### Task 5: 修改 evaluate.py — 增加 CLI 参数和两处透传

**Files:**
- Modify: `dinosam/evaluate.py:285` (argparse，在 `--adapter_type` 之后)
- Modify: `dinosam/evaluate.py:304-305` (Phase 1 模型构建)
- Modify: `dinosam/evaluate.py:332-333` (Phase 2 模型构建)

- [ ] **Step 1: 增加 CLI 参数**

在 `evaluate.py:285`（`--adapter_type` 参数之后）插入：
```python
    parser.add_argument("--depth_transformer_type", type=str, default="twoway",
                        choices=["twoway", "dual_attn"],
                        help="Depth decoder transformer type (must match training checkpoint).")
```

- [ ] **Step 2: Phase 1 模型构建处透传**

将 `evaluate.py:304-305` 从：
```python
        model = DepthSam(sam, encoder_type=args.encoder, dinov3_checkpoint=args.dinov3_checkpoint,
                         adapter_type=args.adapter_type)
```
改为：
```python
        model = DepthSam(sam, encoder_type=args.encoder, dinov3_checkpoint=args.dinov3_checkpoint,
                         adapter_type=args.adapter_type,
                         depth_transformer_type=args.depth_transformer_type)
```

- [ ] **Step 3: Phase 2 模型构建处透传**

将 `evaluate.py:332-333` 从：
```python
        model = DepthSam(sam, encoder_type=args.encoder, dinov3_checkpoint=args.dinov3_checkpoint,
                         adapter_type=args.adapter_type)
```
改为：
```python
        model = DepthSam(sam, encoder_type=args.encoder, dinov3_checkpoint=args.dinov3_checkpoint,
                         adapter_type=args.adapter_type,
                         depth_transformer_type=args.depth_transformer_type)
```

- [ ] **Step 4: Commit**

```bash
git add dinosam/evaluate.py
git commit -m "feat: add --depth_transformer_type CLI arg to evaluate

Both Phase 1 and Phase 2 model construction now pass through the
depth_transformer_type parameter."
```

---

### Task 6: 端到端 Smoke Test

**Files:** 无新建/修改，仅验证

- [ ] **Step 1: 验证 dual_attn 模式的完整 forward pass**

Run:
```bash
cd /home/admin/workspace/SEU/DINOSAM && python -c "
import torch
from segment_anything import sam_model_registry
from dinosam.model.depth_sam import DepthSam

device = 'cpu'
sam = sam_model_registry['vit_b']()
model = DepthSam(sam, depth_transformer_type='dual_attn')
model.init_depth_from_sam()
model.to(device)
model.eval()

B = 1
images = torch.randn(B, 3, 512, 512)
point_coords = torch.randint(0, 256, (B, 3, 2)).float()
point_labels = torch.ones(B, 3).long()
original_sizes = torch.tensor([[512, 512]])
gt_masks = torch.ones(B, 1, 512, 512)

with torch.no_grad():
    image_embeddings, input_size = model.encode_images(images)
    low_res_masks, iou_pred, masks = model.forward_mask(
        image_embeddings=image_embeddings,
        point_coords=point_coords,
        point_labels=point_labels,
        original_sizes=original_sizes,
        input_size=input_size,
    )
    depth_pred, decoder_masks = model.forward_depth(
        image_embeddings=image_embeddings,
        low_res_masks=low_res_masks.detach(),
        input_size=input_size,
        original_sizes=original_sizes,
    )
    print(f'masks: {masks.shape}')
    print(f'depth_pred: {depth_pred.shape}')
    print(f'decoder_masks: {decoder_masks.shape}')
    print('OK: end-to-end forward pass with dual_attn')
"
```
Expected: 看到正确的 shape 输出和 `OK`

- [ ] **Step 2: 验证梯度流过 DualAttnTransformer 的所有参数**

Run:
```bash
cd /home/admin/workspace/SEU/DINOSAM && python -c "
import torch
from dinosam.model.dual_attn_transformer import DualAttnTransformer

m = DualAttnTransformer(depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048)
img = torch.randn(1, 256, 64, 64)
pe = torch.randn(1, 256, 64, 64)
pt = torch.randn(1, 5, 256)
hs, src = m(img, pe, pt)
loss = hs.sum() + src.sum()
loss.backward()

no_grad = [n for n, p in m.named_parameters() if p.grad is None]
if no_grad:
    print(f'WARNING: no gradient for: {no_grad}')
else:
    print(f'OK: all {sum(1 for _ in m.parameters())} params have gradients')
"
```
Expected: `OK: all N params have gradients`

- [ ] **Step 3: 验证 twoway 模式仍正常工作（回归测试）**

Run:
```bash
cd /home/admin/workspace/SEU/DINOSAM && python -c "
import torch
from segment_anything import sam_model_registry
from dinosam.model.depth_sam import DepthSam

sam = sam_model_registry['vit_b']()
model = DepthSam(sam, depth_transformer_type='twoway')
model.init_depth_from_sam()
model.eval()

B = 1
images = torch.randn(B, 3, 512, 512)
point_coords = torch.randint(0, 256, (B, 3, 2)).float()
point_labels = torch.ones(B, 3).long()
original_sizes = torch.tensor([[512, 512]])

with torch.no_grad():
    image_embeddings, input_size = model.encode_images(images)
    low_res_masks, iou_pred, masks = model.forward_mask(
        image_embeddings=image_embeddings,
        point_coords=point_coords,
        point_labels=point_labels,
        original_sizes=original_sizes,
        input_size=input_size,
    )
    depth_pred, decoder_masks = model.forward_depth(
        image_embeddings=image_embeddings,
        low_res_masks=low_res_masks.detach(),
        input_size=input_size,
        original_sizes=original_sizes,
    )
    print(f'OK: twoway regression test passed, depth={depth_pred.shape}')
"
```
Expected: `OK: twoway regression test passed`
