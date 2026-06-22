# Training Strategy

## Draft

### 3.X Training Strategy

#### 3.X.1 Overview

DINOSAM adopts a two-phase training strategy with iterative prompt refinement to decouple the learning of segmentation and depth prediction. The image encoder remains frozen throughout both phases to preserve its pre-trained representations. In Phase 1, only the mask branch (the SAM prompt encoder and mask decoder) is trained, while the depth branch is frozen. In Phase 2, the mask branch is frozen and only the depth branch (the outer prompt encoder and DepthMaskDecoder) is updated. Within each phase, an error-driven iterative refinement mechanism progressively improves predictions by generating corrective point prompts from the discrepancy between the current prediction and the ground truth. We describe each component below.

#### 3.X.2 Weight Initialization of the Depth Branch

Because the depth branch shares the same architectural template as the mask branch — both consist of a prompt encoder followed by a transformer-based decoder — we initialize the depth branch weights from the pre-trained SAM parameters rather than training from scratch. Specifically, the outer prompt encoder $\mathcal{E}_{\text{outer}}$ is initialized from the SAM prompt encoder $\mathcal{E}_{\text{inner}}$:

$$\theta_{\mathcal{E}_{\text{outer}}} \leftarrow \theta_{\mathcal{E}_{\text{inner}}}$$

For the DepthMaskDecoder $\mathcal{D}_{\text{depth}}$, we transfer the following components from the SAM mask decoder $\mathcal{D}_{\text{mask}}$: the two-way transformer weights, the output upscaling layers, and the first hypernetwork MLP. The depth token embedding $\mathbf{e}_{\text{depth}}$ is initialized from the IoU token embedding $\mathbf{e}_{\text{iou}}$, and the single mask token is initialized from the first of SAM's four mask tokens:

$$\mathbf{e}_{\text{depth}} \leftarrow \mathbf{e}_{\text{iou}}, \quad \mathbf{e}_{\text{mask}}^{(1)} \leftarrow \mathbf{e}_{\text{mask}}^{\text{SAM},(1)}$$

The depth prediction head $f_{\text{depth}}: \mathbb{R}^{256} \rightarrow \mathbb{R}^{1}$, a three-layer MLP with hidden dimension 256, has no counterpart in SAM and is randomly initialized. This initialization strategy provides a warm start that accelerates convergence while allowing the depth-specific components to adapt freely.

#### 3.X.3 Phase 1: Mask Branch Training with Iterative Refinement

In Phase 1, the image encoder and depth branch are frozen, and only the mask branch parameters are updated. The training objective combines a Dice loss for spatial overlap and an IoU prediction loss:

$$\mathcal{L}_{\text{Phase1}} = \mathcal{L}_{\text{dice}}(\hat{M}, M^*) + \mathcal{L}_{\text{IoU}}$$

where $\hat{M}$ denotes the predicted mask, $M^*$ the ground-truth mask, and the IoU loss is defined as the mean squared error between the predicted IoU score $\hat{q}$ and the true IoU computed from the predicted and ground-truth masks:

$$\mathcal{L}_{\text{dice}} = 1 - \frac{2 \sum \sigma(\hat{M}) \cdot M^* + \epsilon}{\sum \sigma(\hat{M}) + \sum M^* + \epsilon}$$

$$\mathcal{L}_{\text{IoU}} = \text{MSE}\!\left(\hat{q},\; \text{IoU}(\hat{M}, M^*)\right)$$

where $\sigma(\cdot)$ denotes the sigmoid function.

**Iterative refinement.** When the number of sub-iterations $K > 1$, each training step performs $K$ consecutive forward passes through the mask decoder. After each sub-iteration $k < K$, an error-driven prompt generator analyzes the prediction error to produce corrective point prompts. The generator computes the difference map $\Delta = \hat{M}_k - M^*$ and identifies two error regions:

- **Missed foreground** ($\Delta = -1$): regions present in $M^*$ but absent from $\hat{M}_k$. A positive point $\mathbf{p}^+ \in \mathbb{R}^2$ is sampled uniformly from this region.
- **False positive** ($\Delta = +1$): regions present in $\hat{M}_k$ but absent from $M^*$. A negative point $\mathbf{p}^-$ is sampled from this region, with a fallback to the background near the object boundary when no false positives exist.

The new prompts are appended to the existing prompt set in a cumulative fashion:

$$P_{k+1} = P_k \cup \{\mathbf{p}^+, \mathbf{p}^-\}$$

Additionally, with probability $p_{\text{mask}}$, the low-resolution mask output $\hat{M}_k^{\text{low}} \in \mathbb{R}^{1 \times 256 \times 256}$ from the current sub-iteration is fed as a dense mask prompt to the next sub-iteration, providing the decoder with a spatial prior. The final loss is averaged over all sub-iterations:

$$\mathcal{L}_{\text{Phase1}}^{\text{iter}} = \frac{1}{K} \sum_{k=1}^{K} \left[ \mathcal{L}_{\text{dice}}(\hat{M}_k, M^*) + \mathcal{L}_{\text{IoU},k} \right]$$

This iterative scheme enables the model to learn self-correction: the mask decoder is trained not only on initial prompts but also on progressively refined prompts that highlight its own errors, improving robustness to imprecise initial point placements.

#### 3.X.4 Phase 2: Depth Branch Training with Iterative Refinement

In Phase 2, the image encoder and mask branch are frozen, and only the depth branch parameters (outer prompt encoder and DepthMaskDecoder) are updated. The mask branch executes a single forward pass to produce a low-resolution mask $\hat{M}^{\text{low}}$, which is detached from the computational graph and passed to the depth branch as a dense prompt.

The depth branch receives $\hat{M}^{\text{low}}$ through the outer prompt encoder and produces both a depth prediction $\hat{d} \in \mathbb{R}$ and a decoder mask $\hat{M}_{\text{dec}}$. The training objective is:

$$\mathcal{L}_{\text{Phase2}} = \lambda_d \cdot \text{MSE}(\hat{d}, d^*)$$

where $d^*$ is the ground-truth depth label and $\lambda_d$ is a weighting coefficient.

**Iterative refinement.** Unlike Phase 1 where the mask branch iterates, Phase 2 iterates the depth branch while the mask branch runs only once. The depth decoder begins with no point prompts ($P_0^{\text{depth}} = \emptyset$). After each sub-iteration $k$, the decoder mask $\hat{M}_{\text{dec},k}$ is compared against $M^*$ using the same error-driven prompt generator:

$$P_{k+1}^{\text{depth}} = P_k^{\text{depth}} \cup \{\mathbf{p}^+, \mathbf{p}^-\}$$

where the error regions are computed by downsampling both $\hat{M}_{\text{dec},k}$ and $M^*$ to a common resolution of $256 \times 256$. The cumulative point prompts guide the depth decoder toward more accurate spatial attention over successive iterations. The final loss is:

$$\mathcal{L}_{\text{Phase2}}^{\text{iter}} = \lambda_d \cdot \frac{1}{K} \sum_{k=1}^{K} \text{MSE}(\hat{d}_k, d^*)$$

This design reflects the insight that depth prediction quality depends on accurate spatial localization: by iteratively correcting the decoder's internal mask representation, the depth branch receives increasingly precise spatial cues, which in turn improve the depth regression.

#### 3.X.5 Training Schedule

The two phases are executed sequentially. Phase 1 trains until convergence using AdamW with an initial learning rate $\eta$ and a step-decay schedule (decay factor 0.5 every 10 epochs). The best Phase 1 checkpoint, selected by validation loss, initializes Phase 2. At the start of Phase 2, the depth branch weights are re-initialized from the trained SAM mask decoder (which now reflects Phase 1 fine-tuning), except for the depth prediction head which is randomly initialized. Phase 2 then trains with the same optimizer and schedule configuration.

---

## Section outline

1. Overview paragraph — two-phase decoupled training with iterative refinement
2. Depth branch initialization — weight transfer from SAM
3. Phase 1 — mask-only training, loss function, iterative prompt refinement mechanism
4. Phase 2 — depth training, loss function, depth-branch iterative refinement
5. Training schedule — optimizer, learning rate, checkpoint selection

## Assumptions or missing inputs

- **Specific hyperparameter values**: The exact number of sub-iterations $K$, mask probability $p_{\text{mask}}$, depth loss weight $\lambda_d$, learning rate $\eta$, and number of training epochs are referenced symbolically. [Evidence needed: final hyperparameter table from experiments]
- **Dataset details**: The training and validation data splits, data augmentation, and batch size are not covered here — they belong in a separate "Datasets and Implementation Details" subsection.
- **Phase interaction**: Whether multiple rounds of Phase 1 → Phase 2 alternation are used (e.g., re-running Phase 1 after Phase 2) is not described in the current codebase — the two phases appear to run once each in sequence.

## Claim-evidence map

| Claim | Evidence | Status |
|-------|----------|--------|
| Two-phase training decouples segmentation and depth learning | `freeze_depth_branch()` in Phase 1, manual param freezing in Phase 2 (`trainer.py:415-420`) | Supported |
| Depth branch initialized from SAM weights accelerates convergence | `init_depth_from_sam()` method (`depth_sam.py:59-110`) | Supported (mechanism described; convergence claim needs ablation) |
| Error-driven prompts improve robustness to imprecise initial placements | `DepthIterativePromptGenerator` (`prompt/iterative.py:6-125`) | Supported (mechanism described; needs experimental validation) |
| Phase 2 iterates the depth branch, not the mask branch | `iterative_depth_step` (`train/iterative.py:195-286`): mask branch runs once, depth branch loops | Supported |

## Why this structure

- **Overview first**: Establishes the two-phase + iterative architecture before diving into details, so readers have a mental map.
- **Weight initialization before training phases**: The initialization is a prerequisite for understanding why Phase 2 begins from SAM weights — placing it first avoids a forward reference.
- **Phase 1 before Phase 2**: Sequential ordering mirrors the actual training pipeline and allows Phase 2 to reference concepts (error-driven prompts, cumulative prompt sets) already introduced in Phase 1.
- **Training schedule last**: Implementation details are separated from algorithmic contributions, following the method fragment's guidance.

---

## 中文结构说明

1. **概述段**：先给出两阶段 + 迭代精炼的全局图景，让读者建立心理模型。
2. **权重初始化**：单独成节是因为这是理解 Phase 2 设计的前提——深度分支的 transformer、upscaling 等组件来自 SAM，只有 depth_prediction_head 随机初始化。
3. **Phase 1 详述**：先写损失函数（Dice + IoU），再写迭代精炼机制。迭代部分的关键是"基于误差的提示生成"——从漏检前景采样正点、从误检区域采样负点，累积追加到提示集合。
4. **Phase 2 详述**：与 Phase 1 的关键区别是：(a) 分割分支只跑一次；(b) 深度分支从无提示开始迭代；(c) 损失只有深度 MSE。迭代提示生成复用同一个 DepthIterativePromptGenerator。
5. **训练时间表**：优化器、学习率衰减、检查点选择等实现细节放在最后，与算法贡献分开。
