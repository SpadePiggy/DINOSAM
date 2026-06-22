# Training Strategy: DINOv3 ViT-B/16 Backbone with Mona Adapters

## Draft

### 3.Y Training with DINOv3 ViT-B/16 Image Encoder

#### 3.Y.1 Overview

When configured with the DINOv3 ViT-B/16 image encoder, DINOSAM replaces the SAM image encoder with a frozen DINOv3 backbone augmented by lightweight Mona adapters and a linear projection layer. The downstream architecture — mask branch, depth branch, loss functions, and iterative refinement — remains identical to the SAM-backbone configuration. The key distinction is that Phase 1 jointly optimizes the Mona adapters alongside the mask branch, enabling the encoder output to co-adapt with the downstream decoders. The overall pipeline is:

$$\text{Image} \xrightarrow{\text{frozen DINOv3}} \mathbf{Z} \xrightarrow{\text{Mona}_1 \to \text{Mona}_2 \to \text{Proj}} \mathbf{F} \xrightarrow[\text{Phase 1}]{\text{mask branch}} \hat{M} \xrightarrow[\text{Phase 2}]{\text{depth branch}} \hat{d}$$

#### 3.Y.2 Image Encoding with Mona Adaptation

**Preprocessing.** Unlike the SAM-backbone configuration which uses longest-side resizing with zero-padding, the DINOv3 configuration directly resizes input images to $1024 \times 1024$ via bilinear interpolation and normalizes with ImageNet statistics ($\boldsymbol{\mu}_{\text{IN}} \times 255$, $\boldsymbol{\sigma}_{\text{IN}} \times 255$).

**Frozen backbone.** The DINOv3 ViT-B/16 backbone (12 transformer layers, embedding dimension 768, 12 attention heads) with RoPE positional encoding extracts patch tokens:

$$\mathbf{Z} = f_{\text{DINOv3}}(\text{norm}(\text{resize}(\mathbf{I}))) \in \mathbb{R}^{B \times 4096 \times 768}$$

where $4096 = 64 \times 64$ corresponds to the number of $16 \times 16$ patches. The backbone weights are loaded from a pre-trained checkpoint and frozen at initialization.

**Mona adaptation.** Two sequential Mona adapters transform the patch tokens while preserving their dimensionality. Each Mona adapter applies a dual-branch normalization with learnable scales, a bottleneck MLP, and a multi-scale spatial convolution module:

$$\mathbf{X}' = \text{LN}(\mathbf{X}) \odot \boldsymbol{\gamma} + \mathbf{X} \odot \boldsymbol{\gamma}_x$$

$$\hat{\mathbf{X}} = \mathbf{X} + W_2\!\left(\text{dropout}\!\left(\text{GELU}\!\left(\text{MonaOp}\!\left(W_1 \mathbf{X}'\right)\right)\right)\right)$$

where $W_1 \in \mathbb{R}^{768 \times 96}$ and $W_2 \in \mathbb{R}^{96 \times 768}$ form a bottleneck with reduction factor $r = 8$, and dropout rate is $p = 0.1$.

**MonaOp: multi-scale spatial convolution.** The MonaOp module applies three depthwise separable convolutions at kernel sizes $3 \times 3$, $5 \times 5$, and $7 \times 7$, averaged with an identity shortcut, followed by a $1 \times 1$ pointwise convolution:

$$\text{MonaOp}(\mathbf{x}) = \mathbf{x} + \text{Conv}_{1 \times 1}\!\left(\frac{\text{DWConv}_3(\mathbf{x}) + \text{DWConv}_5(\mathbf{x}) + \text{DWConv}_7(\mathbf{x})}{3} + \mathbf{x}\right)$$

This multi-scale design captures spatial patterns at multiple receptive fields — local texture ($3 \times 3$), medium-range context ($5 \times 5$), and broader structural cues ($7 \times 7$) — while the depthwise-separable structure and bottleneck keep the parameter overhead minimal.

**Projection.** After the two Mona adapters, a linear projection maps the token dimension from 768 to 256, and the sequence is reshaped to a spatial feature map:

$$\mathbf{F} = \text{Reshape}\!\left(\text{Proj}(\hat{\mathbf{Z}})\right) \in \mathbb{R}^{B \times 256 \times 64 \times 64}$$

where $\text{Proj}: \mathbb{R}^{768} \to \mathbb{R}^{256}$. The output $\mathbf{F}$ matches the SAM encoder output shape, making the downstream mask and depth branches agnostic to the encoder choice.

#### 3.Y.3 Depth Branch Initialization

Identical to the SAM-backbone configuration. The outer prompt encoder is initialized from the SAM prompt encoder, and the DepthMaskDecoder is partially initialized from the SAM mask decoder (transformer, tokens, upscaling, and hypernetwork MLP are transferred; the depth prediction head is randomly initialized). See Section 3.X.3 for the full weight transfer table.

#### 3.Y.4 Phase 1: Joint Mask and Adapter Training

Phase 1 freezes the DINOv3 backbone and the depth branch, and trains the mask branch together with the Mona adapters and projection layer. The trainable parameter set is:

$$\Theta_{\text{Phase1}}^{\text{DINOv3}} = \{\theta_{\text{prompt\_encoder}},\; \theta_{\text{mask\_decoder}},\; \theta_{\text{Mona}_1},\; \theta_{\text{Mona}_2},\; \theta_{\text{Proj}}\}$$

This is the central difference from the SAM-backbone configuration, where $\Theta_{\text{Phase1}}^{\text{SAM}} = \{\theta_{\text{prompt\_encoder}},\; \theta_{\text{mask\_decoder}}\}$ only. The joint training enables the image-level adaptation (Mona + Proj) to co-optimize with the downstream mask prediction, allowing the DINOv3 features to align with the prompt-mask decoding pipeline.

**Motivation.** The `freeze_image_encoder()` method freezes only the DINOv3 backbone parameters (`self.backbone`), leaving the Mona adapters and projection layer as trainable modules. This design reflects the principle that the pre-trained DINOv3 representations are strong but domain-generic; the lightweight adapters bridge the gap to the specific segmentation task without modifying the backbone itself.

**Loss function.** Identical to the SAM-backbone Phase 1:

$$\mathcal{L}_{\text{Phase1}} = \mathcal{L}_{\text{dice}}(\hat{M}, M^*) + \text{MSE}\!\left(\hat{q},\; \text{IoU}(\hat{M}, M^*)\right)$$

**Iterative refinement.** Identical to the SAM-backbone Phase 1. The error-driven prompt generator operates on the mask decoder output and produces cumulative corrective point prompts:

$$P_{k+1} = P_k \cup \{\mathbf{p}_k^+,\; \mathbf{p}_k^-\}$$

with optional low-resolution mask feedback (probability $p_{\text{mask}}$). Loss averaged over $K$ sub-iterations.

#### 3.Y.5 Phase 2: Depth Branch Training (Adapters Frozen)

Phase 2 freezes **all** model parameters — including the Mona adapters and projection layer that were trainable in Phase 1 — and then selectively unfreezes only the depth branch:

$$\Theta_{\text{Phase2}}^{\text{DINOv3}} = \{\theta_{\text{outer\_prompt\_encoder}},\; \theta_{\text{depth\_decoder}}\}$$

This is identical to the SAM-backbone Phase 2. The Mona adapters, having been optimized during Phase 1, now produce fixed, task-adapted image embeddings that serve as a stable input for depth branch learning.

**Phase transition.** The best Phase 1 checkpoint is loaded (which includes the trained Mona adapter weights), then `init_depth_from_sam()` re-initializes the depth branch from the fine-tuned SAM mask decoder. The depth prediction head is randomly initialized.

**Loss function and iterative refinement.** Identical to the SAM-backbone Phase 2:

$$\mathcal{L}_{\text{Phase2}}^{\text{iter}} = \lambda_d \cdot \frac{1}{K} \sum_{k=1}^{K} \text{MSE}(\hat{d}_k, d^*)$$

The depth branch iterates $K$ times from empty prompts ($P_0^{\text{depth}} = \emptyset$), generating corrective points from the depth decoder's mask output at each sub-iteration.

#### 3.Y.6 Optimization

Both phases use AdamW with a StepLR scheduler (decay factor $0.5$ every $10$ epochs). The optimizer in Phase 1 collects gradients from both the mask branch and the encoder adaptation layers (Mona + Proj), ensuring end-to-end gradient flow from the segmentation loss through the adapters to the DINOv3 patch tokens.

**Trainable parameter summary (DINOv3 backbone):**

| Component | Phase 1 | Phase 2 |
|-----------|---------|---------|
| DINOv3 ViT-B/16 backbone | Frozen | Frozen |
| Mona$_1$ adapter | **Trainable** | Frozen |
| Mona$_2$ adapter | **Trainable** | Frozen |
| Linear projection | **Trainable** | Frozen |
| SAM prompt\_encoder | **Trainable** | Frozen |
| SAM mask\_decoder | **Trainable** | Frozen |
| Outer prompt\_encoder | Frozen | **Trainable** |
| DepthMaskDecoder | Frozen | **Trainable** |

#### 3.Y.7 Comparison with SAM-Backbone Training

| Aspect | SAM backbone | DINOv3 backbone |
|--------|-------------|----------------|
| Image encoder | SAM ViT-B (frozen) | DINOv3 ViT-B/16 (frozen) + Mona + Proj |
| Preprocessing | Longest-side resize + SAM normalization + zero-pad | Bilinear interpolation + ImageNet normalization |
| Phase 1 trainable | prompt\_encoder, mask\_decoder | prompt\_encoder, mask\_decoder, **Mona$_1$, Mona$_2$, Proj** |
| Phase 2 trainable | outer\_prompt\_encoder, depth\_decoder | outer\_prompt\_encoder, depth\_decoder |
| Encoder adaptation | None | Multi-scale spatial adapters (Mona) |
| Decoder & losses | Identical | Identical |
| Iterative refinement | Identical | Identical |

---

## Assumptions or missing inputs

- [Evidence needed: parameter counts for Mona adapters and projection layer to quantify the adaptation overhead]
- [Evidence needed: ablation comparing DINOv3+Mona vs SAM ViT-B on segmentation and depth metrics]
- [Evidence needed: whether Mona adapters are inserted between DINOv3 transformer layers (interleaved) or applied post-hoc — current code applies them post-hoc on the final patch tokens]
- [Evidence needed: hyperparameter values — $K$, $p_{\text{mask}}$, $\lambda_d$, learning rate, epochs]

## Claim-evidence map

| Claim | Evidence | Status |
|-------|----------|--------|
| DINOv3 backbone frozen, only Mona + Proj trainable at encoder level | `freeze_backbone()` freezes `self.backbone` only (`dinov3_encoder.py:75-77`); Mona/Proj not included | Supported |
| Phase 1 jointly trains Mona with mask branch | `freeze_image_encoder()` → `freeze_backbone()` only; optimizer collects all `requires_grad=True` (`trainer.py:358-359`) | Supported |
| Phase 2 freezes Mona adapters | `run_phase2`: all params set to `requires_grad=False` before unfreezing depth branch (`trainer.py:414-420`) | Supported |
| Multi-scale convolution captures patterns at 3 receptive fields | DWConv at 3×3, 5×5, 7×7 in `MonaOp.__init__` (`mona.py:14-16`) | Supported |
| Output shape matches SAM encoder | `forward` returns `(B, 256, H, W)` after Proj + reshape (`dinov3_encoder.py:107-108`) | Supported |
| Mona applies post-hoc, not interleaved | `forward`: `backbone.forward_features()` → `mona1()` → `mona2()` sequentially (`dinov3_encoder.py:97-105`) | Supported |

## 中文结构说明

DINOv3 主干的训练流程与 SAM 主干共享相同的两阶段框架和迭代精炼机制，核心差异集中在：

1. **编码器层面**：DINOv3 ViT-B/16 冻结骨干 + 2 层 Mona 适配器（可训练）+ 线性投影（可训练）。Mona 的多尺度空间卷积（3×3/5×5/7×7）是该变体的独特贡献，通过瓶颈结构控制参数量。
2. **Phase 1 可训练范围扩大**：不仅训练分割分支，还同时训练 Mona 适配器和投影层，实现编码器适配与分割解码的联合优化。
3. **Phase 2 一致性**：全局冻结后仅解冻深度分支，Mona 适配器在此阶段被冻结，产出固定的任务适配特征。
4. **预处理差异**：DINOv3 使用双线性插值至 1024×1024 + ImageNet 归一化，而非 SAM 的最长边缩放 + zero-pad。

将这些差异独立成文，可以让读者清晰对比两种主干配置的设计选择和训练行为。
