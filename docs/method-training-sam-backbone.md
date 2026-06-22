# Training Strategy: SAM ViT-B Backbone

## Draft

### 3.X Training with SAM ViT-B Image Encoder

#### 3.X.1 Overview

When configured with the SAM ViT-B image encoder, DINOSAM adopts a two-phase decoupled training strategy in which the segmentation mask branch and the depth prediction branch are trained sequentially. The image encoder remains frozen throughout both phases, preserving its pre-trained visual representations. An error-driven iterative refinement mechanism operates within each phase to progressively improve predictions by generating corrective point prompts from prediction errors. The overall pipeline is:

$$\text{Image} \xrightarrow{\text{frozen SAM ViT-B}} \mathbf{F} \xrightarrow[\text{Phase 1}]{\text{mask branch}} \hat{M} \xrightarrow[\text{Phase 2}]{\text{depth branch}} \hat{d}$$

#### 3.X.2 Image Encoding

Input images are preprocessed using longest-side resizing to fit within $1024 \times 1024$ pixels, normalized with SAM-specific pixel statistics ($\boldsymbol{\mu}_{\text{SAM}}$, $\boldsymbol{\sigma}_{\text{SAM}}$), and zero-padded to a square. The frozen SAM ViT-B encoder then produces image embeddings:

$$\mathbf{F} = f_{\text{SAM}}(\text{pad}(\text{norm}(\text{resize}(\mathbf{I})))) \in \mathbb{R}^{B \times 256 \times 64 \times 64}$$

No additional adaptation layers are applied. The encoder output $\mathbf{F}$ is shared by both the mask branch and the depth branch, and is computed once per training step.

#### 3.X.3 Depth Branch Initialization

The depth branch (outer prompt encoder $\mathcal{E}_{\text{outer}}$ + DepthMaskDecoder $\mathcal{D}_{\text{depth}}$) shares the same architectural template as the mask branch. To provide a warm start, we initialize its weights from the pre-trained SAM components:

$$\theta_{\mathcal{E}_{\text{outer}}} \leftarrow \theta_{\mathcal{E}_{\text{inner}}}$$

For $\mathcal{D}_{\text{depth}}$, the weight transfer follows:

| Depth component | Source | Shape |
|---|---|---|
| Two-way transformer | $\mathcal{D}_{\text{mask}}$.transformer | all weights |
| Depth token $\mathbf{e}_{\text{depth}}$ | IoU token $\mathbf{e}_{\text{iou}}$ | $(1, 256)$ |
| Mask token | $\mathcal{D}_{\text{mask}}$.mask\_tokens$[0]$ | $(1, 256)$ |
| Output upscaling | $\mathcal{D}_{\text{mask}}$.output\_upscaling | ConvTranspose2d stack |
| Hypernetwork MLP$[0]$ | $\mathcal{D}_{\text{mask}}$.output\_hypernetworks\_mlps$[0]$ | 3-layer MLP |
| Depth prediction head | **Random init** | MLP: $256 \to 256 \to 1$ |

#### 3.X.4 Phase 1: Mask-Only Training

Phase 1 trains the mask branch while freezing the image encoder and depth branch. The trainable parameters are:

$$\Theta_{\text{Phase1}} = \{\theta_{\text{prompt\_encoder}},\; \theta_{\text{mask\_decoder}}\}$$

**Loss function.** The objective combines a Dice loss for spatial overlap with an IoU prediction loss:

$$\mathcal{L}_{\text{Phase1}} = \mathcal{L}_{\text{dice}}(\hat{M}, M^*) + \text{MSE}\!\left(\hat{q},\; \text{IoU}(\hat{M}, M^*)\right)$$

where

$$\mathcal{L}_{\text{dice}} = 1 - \frac{2 \sum \sigma(\hat{M}) \cdot M^* + \epsilon}{\sum \sigma(\hat{M}) + \sum M^* + \epsilon}$$

$\sigma(\cdot)$ denotes the sigmoid function, and the true IoU is computed in a no-gradient context.

**Iterative refinement.** When $K > 1$ sub-iterations are used, each training step performs $K$ consecutive forward passes through the mask decoder. After sub-iteration $k$ ($k < K$), the error-driven prompt generator analyzes the difference map $\Delta_k = \hat{M}_k - M^*$:

- A **positive point** $\mathbf{p}^+$ is sampled from the missed foreground region ($\Delta_k = -1$). Fallback: overlap region, then any GT foreground.
- A **negative point** $\mathbf{p}^-$ is sampled from the false positive region ($\Delta_k = +1$). Fallback: background near the object boundary, then any background.

The corrective prompts are appended cumulatively to the existing prompt set:

$$P_{k+1} = P_k \cup \{\mathbf{p}_k^+,\; \mathbf{p}_k^-\}$$

Additionally, with probability $p_{\text{mask}}$, the low-resolution mask output $\hat{M}_k^{\text{low}} \in \mathbb{R}^{1 \times 256 \times 256}$ is fed as a dense mask prompt to the next sub-iteration, providing the decoder with a spatial prior. The final loss is averaged over all sub-iterations:

$$\mathcal{L}_{\text{Phase1}}^{\text{iter}} = \frac{1}{K} \sum_{k=1}^{K} \left[\mathcal{L}_{\text{dice}}(\hat{M}_k, M^*) + \text{MSE}(\hat{q}_k, \text{IoU}_k)\right]$$

#### 3.X.5 Phase 2: Depth Branch Training

Phase 2 freezes all model parameters and selectively unfreezes only the depth branch:

$$\Theta_{\text{Phase2}} = \{\theta_{\text{outer\_prompt\_encoder}},\; \theta_{\text{depth\_decoder}}\}$$

**Phase transition.** The best Phase 1 checkpoint is loaded, then `init_depth_from_sam()` is called again. Because the SAM mask decoder has now been fine-tuned during Phase 1, the depth branch inherits improved decoder representations. The depth prediction head remains randomly initialized.

**Forward pass.** The mask branch executes a single forward pass to produce $\hat{M}^{\text{low}}$, which is detached from the computational graph and passed through the outer prompt encoder as a dense prompt. The DepthMaskDecoder then predicts both a depth value $\hat{d} \in \mathbb{R}$ and a decoder mask $\hat{M}_{\text{dec}}$.

**Loss function.**

$$\mathcal{L}_{\text{Phase2}} = \lambda_d \cdot \text{MSE}(\hat{d}, d^*)$$

**Iterative refinement.** The depth branch iterates $K$ times, starting from no point prompts ($P_0^{\text{depth}} = \emptyset$). After each sub-iteration $k$, the decoder mask $\hat{M}_{\text{dec},k}$ is downsampled to $256 \times 256$ and compared with $M^*$ via the same error-driven prompt generator to produce corrective points:

$$P_{k+1}^{\text{depth}} = P_k^{\text{depth}} \cup \{\mathbf{p}_k^+,\; \mathbf{p}_k^-\}$$

The final loss is:

$$\mathcal{L}_{\text{Phase2}}^{\text{iter}} = \lambda_d \cdot \frac{1}{K} \sum_{k=1}^{K} \text{MSE}(\hat{d}_k, d^*)$$

#### 3.X.6 Optimization

Both phases use AdamW with a StepLR scheduler (decay factor $0.5$ every $10$ epochs). The best checkpoint is selected by validation loss. Training supports resumption from the latest checkpoint.

**Trainable parameter summary (SAM backbone):**

| Component | Phase 1 | Phase 2 |
|-----------|---------|---------|
| SAM ViT-B image encoder | Frozen | Frozen |
| SAM prompt\_encoder | **Trainable** | Frozen |
| SAM mask\_decoder | **Trainable** | Frozen |
| Outer prompt\_encoder | Frozen | **Trainable** |
| DepthMaskDecoder | Frozen | **Trainable** |

---

## Assumptions or missing inputs

- [Evidence needed: hyperparameter values — $K$, $p_{\text{mask}}$, $\lambda_d$, learning rate, batch size, epochs]
- [Evidence needed: dataset details and augmentation strategy]
- [Evidence needed: total trainable parameter count for SAM-backbone configuration]

## Claim-evidence map

| Claim | Evidence | Status |
|-------|----------|--------|
| Image encoder is fully frozen throughout | `freeze_image_encoder()` freezes `sam.image_encoder` (`depth_sam.py:112-116`) | Supported |
| Only mask branch params trained in Phase 1 | `freeze_depth_branch()` + `freeze_image_encoder()` (`trainer.py:354-355`) | Supported |
| Phase 2 re-initializes depth from fine-tuned SAM | `init_depth_from_sam()` after loading Phase 1 checkpoint (`trainer.py:436-437`) | Supported |
| Iterative prompts are cumulative | `torch.cat([cur_point_coords, new_coords_scaled])` (`iterative.py:76,175,274`) | Supported |

## 中文结构说明

SAM 主干的训练流程相对简洁：编码器全程冻结、无任何适配层。Phase 1 只训练 prompt_encoder + mask_decoder（分割分支），Phase 2 只训练 outer_prompt_encoder + depth_decoder（深度分支）。两个阶段通过检查点传递，Phase 2 开始时重新从 Phase 1 微调后的 SAM 权重初始化深度分支。迭代精炼机制在两个阶段中复用同一套误差驱动提示生成器，但 Phase 1 迭代分割分支、Phase 2 迭代深度分支。
