# Method: Training Strategy (SAM Backbone & DINOv3 Backbone)

## Draft

### 3.X Training Strategy

#### 3.X.1 Overview

DINOSAM employs a two-phase training strategy with iterative prompt refinement that is agnostic to the choice of image encoder. The framework supports two backbone configurations: (i) the original SAM ViT-B image encoder, and (ii) a DINOv3 ViT-B/16 encoder augmented with lightweight Mona adapters. In both configurations, the backbone weights are frozen throughout training, and the segmentation and depth prediction tasks are learned sequentially — Phase 1 trains only the mask branch, and Phase 2 trains only the depth branch. The two configurations share identical decoder architectures, loss functions, and iterative refinement mechanisms; they differ only in how image embeddings are produced and which parameters are trainable at the encoder level. We describe each component below.

#### 3.X.2 Image Encoder Configurations

**SAM ViT-B encoder.** In the default configuration (`encoder_type = "sam"`), input images are resized using longest-side resizing to $1024 \times 1024$, normalized with the SAM-specific pixel statistics, and zero-padded to a square. The frozen SAM ViT-B encoder produces image embeddings $\mathbf{F} \in \mathbb{R}^{B \times 256 \times 64 \times 64}$. No additional adaptation layers are applied; the encoder output is passed directly to both the mask branch and the depth branch.

**DINOv3 ViT-B/16 encoder with Mona adapters.** In the alternative configuration (`encoder_type = "dinov3"`), the SAM ViT-B encoder is replaced by a DINOv3 ViT-B/16 backbone. Input images are bilinearly interpolated to $1024 \times 1024$ and normalized with ImageNet statistics. The frozen DINOv3 backbone extracts patch tokens $\mathbf{Z} \in \mathbb{R}^{B \times N \times 768}$, where $N = 64 \times 64 = 4096$. These tokens are then processed by two sequential Mona adapters and a linear projection layer to produce embeddings in the same space as the SAM encoder:

$$\mathbf{F} = \text{Reshape}\!\left(\text{Proj}\!\left(\text{Mona}_2\!\left(\text{Mona}_1(\mathbf{Z})\right)\right)\right) \in \mathbb{R}^{B \times 256 \times 64 \times 64}$$

where $\text{Proj}: \mathbb{R}^{768} \rightarrow \mathbb{R}^{256}$ is a learned linear projection. The Mona adapters and projection layer are the only trainable components at the encoder level; the DINOv3 backbone itself remains frozen. This design enables task-specific adaptation of the pre-trained DINOv3 features without modifying the backbone weights.

**Mona adapter architecture.** Each Mona adapter consists of a layer normalization branch with a learnable scale $\boldsymbol{\gamma}$ and an identity branch with a learnable scale $\boldsymbol{\gamma}_x$, followed by a bottleneck MLP with a multi-scale spatial convolution module (MonaOp):

$$\mathbf{X}' = \text{LN}(\mathbf{X}) \odot \boldsymbol{\gamma} + \mathbf{X} \odot \boldsymbol{\gamma}_x$$

$$\hat{\mathbf{X}} = \mathbf{X} + W_2\!\left(\text{GELU}\!\left(\text{MonaOp}\!\left(W_1 \mathbf{X}'\right)\right)\right)$$

where $W_1 \in \mathbb{R}^{768 \times 96}$ and $W_2 \in \mathbb{R}^{96 \times 768}$ form the bottleneck (reduction factor $r = 8$), and MonaOp applies three depthwise separable convolutions at kernel sizes $3 \times 3$, $5 \times 5$, and $7 \times 7$, averaged with an identity connection, followed by a $1 \times 1$ pointwise convolution:

$$\text{MonaOp}(\mathbf{x}) = \mathbf{x} + \text{Conv}_{1 \times 1}\!\left(\frac{\text{DWConv}_{3}(\mathbf{x}) + \text{DWConv}_{5}(\mathbf{x}) + \text{DWConv}_{7}(\mathbf{x})}{3} + \mathbf{x}\right)$$

The multi-scale design captures spatial patterns at multiple receptive fields while the bottleneck structure keeps the parameter count low. Dropout ($p = 0.1$) is applied after the GELU activation for regularization.

#### 3.X.3 Depth Branch Initialization

Because the depth branch shares the same architectural template as the mask branch — both consist of a prompt encoder followed by a transformer-based decoder — we initialize the depth branch from the pre-trained SAM weights to provide a warm start. The outer prompt encoder $\mathcal{E}_{\text{outer}}$ is initialized by copying the full state dictionary of the SAM prompt encoder $\mathcal{E}_{\text{inner}}$. For the DepthMaskDecoder $\mathcal{D}_{\text{depth}}$, we transfer the following components from the SAM mask decoder $\mathcal{D}_{\text{mask}}$:

| Depth branch component | Initialized from | Notes |
|---|---|---|
| Two-way transformer | $\mathcal{D}_{\text{mask}}$.transformer | Identical structure, all weights copied |
| Depth token $\mathbf{e}_{\text{depth}}$ | IoU token $\mathbf{e}_{\text{iou}}$ | Same shape: $(1, 256)$ |
| Mask token $\mathbf{e}_{\text{mask}}^{(1)}$ | $\mathcal{D}_{\text{mask}}$.mask\_tokens$[0]$ | First of SAM's four mask tokens |
| Output upscaling layers | $\mathcal{D}_{\text{mask}}$.output\_upscaling | Identical ConvTranspose2d stack |
| Hypernetwork MLP$[0]$ | $\mathcal{D}_{\text{mask}}$.output\_hypernetworks\_mlps$[0]$ | First of four hypernetwork MLPs |
| Depth prediction head | Random initialization | MLP: $256 \rightarrow 256 \rightarrow 1$ (3 layers) |

This initialization applies identically to both the SAM-backbone and DINOv3-backbone configurations, as the depth branch operates on image embeddings $\mathbf{F}$ that share the same dimensionality regardless of the encoder choice.

#### 3.X.4 Phase 1: Mask Branch Training

In Phase 1, the mask branch (SAM prompt encoder + mask decoder) is trained while the image encoder and depth branch are frozen. The training objective combines the Dice loss for spatial overlap with an IoU prediction loss:

$$\mathcal{L}_{\text{Phase1}} = \mathcal{L}_{\text{dice}}(\hat{M}, M^*) + \text{MSE}\!\left(\hat{q},\; \text{IoU}(\hat{M}, M^*)\right)$$

where $\hat{M}$ is the predicted mask, $M^*$ is the ground-truth mask, $\hat{q}$ is the predicted IoU score, and

$$\mathcal{L}_{\text{dice}} = 1 - \frac{2 \sum \sigma(\hat{M}) \cdot M^* + \epsilon}{\sum \sigma(\hat{M}) + \sum M^* + \epsilon}$$

**Trainable parameter difference.** In the SAM-backbone configuration, only the prompt encoder and mask decoder parameters are updated. In the DINOv3-backbone configuration, the Mona adapters and projection layer are additionally trainable, because `freeze_image_encoder()` freezes only the DINOv3 backbone weights while leaving the adaptation modules learnable. This means Phase 1 jointly optimizes the image-level adaptation and mask prediction for the DINOv3 variant, enabling the encoder output to co-adapt with the downstream decoders.

#### 3.X.5 Phase 2: Depth Branch Training

Phase 2 freezes all model parameters and then selectively unfreezes only the depth branch (outer prompt encoder + DepthMaskDecoder). This applies uniformly to both backbone configurations — in particular, for the DINOv3 variant, the Mona adapters trained in Phase 1 are frozen in Phase 2. The mask branch executes a single forward pass to produce a low-resolution mask $\hat{M}^{\text{low}} \in \mathbb{R}^{1 \times 256 \times 256}$, which is detached from the computational graph and passed to the depth branch as a dense prompt.

The training objective is:

$$\mathcal{L}_{\text{Phase2}} = \lambda_d \cdot \text{MSE}(\hat{d}, d^*)$$

where $\hat{d}$ is the predicted depth, $d^*$ is the ground-truth depth, and $\lambda_d$ is a weighting coefficient.

**Phase transition.** At the start of Phase 2, the model loads the best Phase 1 checkpoint, then re-initializes the depth branch from the (now fine-tuned) SAM mask decoder via `init_depth_from_sam()`. This ensures the depth branch benefits from the improved mask decoder representations learned during Phase 1, while the depth prediction head starts from a clean random initialization.

#### 3.X.6 Iterative Prompt Refinement

Both phases support an iterative refinement mechanism with $K$ sub-iterations per training step. This mechanism is backbone-agnostic and operates identically for both encoder configurations.

**Phase 1 iterative refinement.** After each sub-iteration $k < K$, an error-driven prompt generator computes the difference map $\Delta_k = \hat{M}_k - M^*$ and produces one corrective positive point (sampled from missed foreground, $\Delta_k = -1$) and one corrective negative point (sampled from false positives, $\Delta_k = +1$; with a fallback to background near the object boundary). The new points are appended cumulatively:

$$P_{k+1} = P_k \cup \{\mathbf{p}_k^+, \mathbf{p}_k^-\}$$

With probability $p_{\text{mask}}$, the low-resolution mask $\hat{M}_k^{\text{low}}$ from the current sub-iteration is additionally fed as a dense mask prompt for the next iteration. The loss is averaged:

$$\mathcal{L}_{\text{Phase1}}^{\text{iter}} = \frac{1}{K} \sum_{k=1}^{K} \left[\mathcal{L}_{\text{dice}}(\hat{M}_k, M^*) + \text{MSE}(\hat{q}_k, \text{IoU}_k)\right]$$

**Phase 2 iterative refinement.** The mask branch runs once to produce $\hat{M}^{\text{low}}$. The depth branch then iterates $K$ times, starting from no point prompts ($P_0^{\text{depth}} = \emptyset$). After each sub-iteration, the depth decoder's mask output $\hat{M}_{\text{dec},k}$ is downsampled to $256 \times 256$ and compared with $M^*$ to generate new corrective point prompts via the same error-driven generator. The loss is:

$$\mathcal{L}_{\text{Phase2}}^{\text{iter}} = \lambda_d \cdot \frac{1}{K} \sum_{k=1}^{K} \text{MSE}(\hat{d}_k, d^*)$$

#### 3.X.7 Implementation Details

Both backbone configurations use the AdamW optimizer with a StepLR scheduler (decay factor 0.5 every 10 epochs). Data loading uses 4 workers with pinned memory. Checkpoints are saved at regular intervals, at the best validation loss, and at every epoch for training resumption. The two backbone configurations are selected at model construction time via the `encoder_type` argument and share all other hyperparameters.

**Summary of trainable parameters by phase and backbone:**

| Phase | SAM backbone | DINOv3 backbone |
|-------|-------------|----------------|
| Phase 1 | prompt\_encoder, mask\_decoder | prompt\_encoder, mask\_decoder, **Mona$_1$, Mona$_2$, Proj** |
| Phase 2 | outer\_prompt\_encoder, depth\_decoder | outer\_prompt\_encoder, depth\_decoder |

---

## Section outline

1. Overview — unified two-phase framework, backbone-agnostic design
2. Image encoder configurations — SAM ViT-B vs DINOv3 + Mona, preprocessing differences
3. Mona adapter architecture — multi-scale depthwise conv + bottleneck MLP
4. Depth branch initialization — weight transfer table from SAM
5. Phase 1 — mask training, loss, trainable parameter difference between backbones
6. Phase 2 — depth training, loss, phase transition mechanism
7. Iterative refinement — error-driven prompt generation, Phase 1 vs Phase 2 differences
8. Implementation details — optimizer, scheduler, trainable parameter summary table

## Assumptions or missing inputs

- [Evidence needed: specific values for $K$ (sub-iterations), $p_{\text{mask}}$ (mask feedback probability), $\lambda_d$ (depth loss weight), learning rate $\eta$, batch size, and total epochs]
- [Evidence needed: dataset details — training/validation split, data augmentation strategy, number of point prompts per image (`max_points`)]
- [Evidence needed: ablation results comparing SAM-backbone vs DINOv3-backbone performance to support the claim that Mona adaptation improves encoder quality]
- [Evidence needed: parameter count comparison between the two configurations]

## Claim-evidence map

| Claim | Evidence | Status |
|-------|----------|--------|
| DINOv3 backbone is frozen, only Mona + projection are trainable | `freeze_backbone()` freezes `self.backbone`, Mona/Proj not included (`dinov3_encoder.py:58-59,75-77`) | Supported |
| Phase 1 DINOv3 variant trains Mona adapters alongside mask branch | `freeze_image_encoder()` calls `freeze_backbone()` only, not Mona params (`depth_sam.py:112-116`); Phase 1 uses `filter(requires_grad)` (`trainer.py:358-359`) | Supported |
| Phase 2 freezes Mona adapters for DINOv3 variant | `run_phase2` sets all params to `requires_grad=False` then only unfreezes depth branch (`trainer.py:414-420`) | Supported |
| Mona captures multi-scale spatial patterns | Three depthwise convolutions at 3×3, 5×5, 7×7 (`mona.py:14-16`) | Supported |
| Depth branch re-initialized from fine-tuned SAM at Phase 2 start | `model.init_depth_from_sam()` called after loading Phase 1 checkpoint (`trainer.py:436-437`) | Supported |
| Iterative refinement is backbone-agnostic | Same `DepthIterativePromptGenerator` and `iterative_mask_step`/`iterative_depth_step` used regardless of encoder type | Supported |

## Why this structure

- **Encoder configurations placed before training phases**: readers need to understand what $\mathbf{F}$ is and which parameters are trainable at the encoder level before they can follow the Phase 1/Phase 2 freezing logic.
- **Mona described inline with the DINOv3 encoder**: the adapter is architecturally part of the encoder pipeline, not a standalone module — placing it here avoids a forward reference.
- **Phase 1 highlights the trainable parameter difference explicitly**: this is the central structural difference between the two backbone configurations; burying it would mislead readers into thinking the training is identical.
- **Iterative refinement in a shared subsection**: since the mechanism is backbone-agnostic, a single description avoids redundancy. Phase 1 vs Phase 2 differences are contrasted within the subsection.

---

## 中文结构说明

1. **概述段**：强调框架的主干无关性——两种编码器共享同一套解码器、损失和迭代精炼机制。
2. **图像编码器配置**：分别描述 SAM 和 DINOv3+Mona 两条路径的预处理、前向过程和输出形状。关键点：两者输出都是 `(B, 256, 64, 64)`，对下游透明。
3. **Mona 适配器**：详细写出公式（LayerNorm 双分支缩放 → 瓶颈 → MonaOp 多尺度卷积 → 残差），因为这是 DINOv3 变体的核心可训练组件。
4. **深度分支初始化**：用表格清晰列出每个组件的来源，比纯文字描述更易查阅。
5. **Phase 1**：损失公式后紧跟"可训练参数差异"——这是两种主干的**唯一结构性区别**，必须显式指出。SAM 变体只训练 prompt_encoder + mask_decoder；DINOv3 变体额外训练 Mona 适配器和投影层。
6. **Phase 2**：强调全局冻结后仅解冻深度分支，两种主干在此阶段完全一致（DINOv3 的 Mona 也被冻结）。还描述了 Phase 1 → Phase 2 的过渡：加载最佳权重后重新 `init_depth_from_sam()`。
7. **迭代精炼**：合并为一节避免重复。Phase 1 和 Phase 2 的差异通过对比描述：Phase 1 迭代分割分支 + 支持 mask_input 反馈；Phase 2 迭代深度分支 + 从空提示开始。
8. **实现细节**：优化器、调度器等放最后，并以表格总结两种主干在两个阶段的可训练参数差异。
