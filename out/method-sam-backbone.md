# Methods

## Overview

We present DepthSam, a framework that extends the Segment Anything Model (SAM) to jointly predict instance-level depth from a single image, without modifying or fine-tuning SAM's image encoder. The core idea is to augment SAM's segmentation pipeline with an external depth branch that receives the segmentation mask as an intermediate representation and regresses a scalar depth value per instance. The architecture comprises three components: a frozen SAM image encoder (Section 3.1), a mask branch for interactive segmentation (Section 3.2), and a depth branch for mask-conditioned depth regression (Section 3.3). An iterative prompt refinement mechanism (Section 3.4) improves both branches through error-guided point sampling. The model is trained in two phases: Phase 1 trains the mask branch while freezing all other components; Phase 2 freezes the mask branch and trains the depth branch from SAM-initialized weights (Section 3.5).

## 3.1 Image Encoder

SAM's image encoder is a Vision Transformer Base (ViT-B/16) pre-trained on the SA-1B dataset. We keep this encoder entirely frozen throughout both training phases, treating it as a fixed feature extractor. Given an input image, we first resize it so that the longest edge equals 1,024 pixels while preserving the aspect ratio. The image is then normalized using SAM's pre-trained pixel mean and standard deviation, and zero-padded to 1,024 x 1,024 pixels. The encoder produces a dense feature map of shape (B, 256, 64, 64), where B is the batch size. This feature map is shared by both the mask branch and the depth branch, ensuring that the two prediction tasks operate on a common visual representation without additional encoding cost.

Freezing the image encoder serves two purposes. First, it preserves the strong spatial priors learned from large-scale segmentation pre-training, which are difficult to recover from small-scale depth datasets. Second, it enables a clean ablation of the decoder-side modifications introduced in this work, isolating their contribution from encoder-level effects.

## 3.2 Mask Branch

The mask branch follows SAM's original prompt-to-mask pipeline and produces a segmentation mask for each prompted region.

**Motivation.** Interactive segmentation through point prompts is the standard interface of SAM. Rather than replacing this interface, we retain it so that the depth branch can leverage the segmentation mask as a structured spatial prior, converting the ill-posed monocular depth problem into a mask-conditioned regression that focuses on a single object instance at a time.

**Design.** The mask branch consists of SAM's internal prompt encoder and mask decoder. The prompt encoder maps point prompts (coordinates and binary foreground/background labels) into sparse embeddings and dense embeddings. The mask decoder, built on a two-way transformer, cross-attends between the prompt embeddings and the image features to produce a single segmentation mask (multimask output is disabled). We process each sample independently to avoid the quadratic memory cost of batch-level repeat-and-interleave operations within the mask decoder. The branch outputs a low-resolution mask of shape (B, 1, 256, 256), an IoU prediction score of shape (B, 1), and a full-resolution binary mask of shape (B, 1, H, W) obtained by bilinear upsampling and thresholding.

**Role in the pipeline.** The low-resolution mask serves as the input to the depth branch (after gradient detachment; see Section 3.3). The IoU prediction provides a self-assessment of mask quality that can be used for downstream filtering. In Phase 1 training, the mask branch is the sole trainable component.

## 3.3 Depth Branch

The depth branch receives the segmentation mask produced by the mask branch and regresses a scalar depth value for the segmented instance.

**Motivation.** Monocular depth estimation from a single image is inherently ambiguous. By conditioning depth prediction on the segmentation mask, we reduce the problem from dense per-pixel regression to a per-instance scalar prediction, where the mask delineates the spatial extent of the object and allows the decoder to focus its capacity on depth-discriminative features within that region. This formulation also decouples segmentation errors from depth errors: the mask branch can be trained and evaluated independently before the depth branch is introduced.

**Design.** The depth branch mirrors the structure of SAM's mask decoder but replaces the prediction heads. It consists of an outer prompt encoder and a DepthMaskDecoder. The outer prompt encoder is architecturally identical to SAM's prompt encoder and receives the low-resolution mask (detached from the mask branch's computation graph) through its mask encoding channel (mask_in_chans = 16). The DepthMaskDecoder modifies SAM's mask decoder in three ways:

1. **Depth token.** A learned depth token embedding (dimension 256) replaces SAM's IoU token. This token attends to the image features through the two-way transformer and is then projected by a depth prediction head -- a three-layer MLP (256 -> 256 -> 1) -- to produce a scalar depth value.
2. **Reduced mask tokens.** Only one mask token is used (SAM uses four for multi-mask output), and correspondingly only one hypernetwork MLP is retained. This token produces an auxiliary mask output that is used during Phase 2 iterative training for error-guided point sampling.
3. **Preserved transformer and upsampling.** The two-way transformer (depth = 2, embedding dimension = 256, MLP dimension = 2,048, 8 attention heads) and the output upsampling path (two transposed convolutions: 256 -> 64 -> 32) are kept identical to SAM's mask decoder.

The depth branch also receives the original point prompt coordinates, providing explicit spatial anchoring for the depth regression.

**Weight initialization.** To provide a strong starting point and accelerate convergence, the depth branch is initialized from SAM's trained mask decoder weights through a structured copy:

- Outer prompt encoder weights are copied from SAM's prompt encoder.
- The two-way transformer weights are copied from SAM's mask decoder transformer.
- The depth token is initialized from SAM's IoU token (same dimensionality).
- The single mask token is initialized from the first of SAM's four mask tokens.
- The output upsampling layers and the single hypernetwork MLP are copied from the corresponding SAM components.
- The depth prediction head is the only randomly initialized component, as SAM has no corresponding structure.

This initialization strategy ensures that the depth branch begins training with a decoder that already understands how to attend to SAM's image features and prompt embeddings, requiring only the depth prediction head to learn a new mapping.

**Output.** The depth branch produces a scalar depth prediction per instance (B,) and an auxiliary mask of shape (B, 1, H, W).

## 3.4 Iterative Prompt Refinement

**Motivation.** A single set of point prompts may be insufficient to capture the target object, particularly when the initial prompts fall near object boundaries or in ambiguous regions. Iterative refinement progressively corrects segmentation errors by sampling new prompts from the error regions of the current prediction, following the error-guided prompting paradigm introduced in interactive segmentation.

**Design.** The iterative prompt generator compares the predicted mask against the ground-truth mask at each iteration and samples two correction points:

- One positive point (foreground label) is sampled from the false-negative region (pixels present in the ground truth but absent from the prediction). If the false-negative region is empty, the point falls back to the overlap region, then to any ground-truth foreground pixel, and finally to the image center.
- One negative point (background label) is sampled from the false-positive region (pixels present in the prediction but absent from the ground truth). If this region is empty, the point falls back to background pixels near the object boundary (obtained by dilating the bounding box by 3 pixels), then to any background pixel, and finally to the coordinate origin.

The newly sampled points are appended to the existing prompt set, so the prompt grows by two points per iteration. The low-resolution mask from the previous iteration is optionally fed back to the mask branch as a mask input with probability mask_prob (default 0.5), providing the decoder with a spatial prior from its own previous output.

In Phase 2 iterative training, the mask branch runs only once (frozen), and the iterative loop operates on the depth branch: the auxiliary mask output of the DepthMaskDecoder is compared against the ground truth to generate correction points for subsequent depth refinement rounds.

## 3.5 Training Strategy

Training proceeds in two phases with complementary freezing schedules.

**Phase 1: Mask training.** The image encoder, outer prompt encoder, and depth decoder are frozen. Only SAM's internal prompt encoder and mask decoder are trained. The loss function is:

$$\mathcal{L}_{\text{mask}} = \mathcal{L}_{\text{dice}}(M_{\text{pred}}, M_{\text{gt}}) + \mathcal{L}_{\text{IoU}}(\hat{q}, q)$$

where $\mathcal{L}_{\text{dice}}$ is the soft dice loss computed as $1 - 2 |M_{\text{pred}} \cap M_{\text{gt}}| / (|M_{\text{pred}}| + |M_{\text{gt}}| + \epsilon)$ after sigmoid activation, and $\mathcal{L}_{\text{IoU}}$ is the mean squared error between the predicted IoU score $\hat{q}$ and the true IoU $q$ computed from the binarized masks. When iterative refinement is enabled (n_sub_iterations > 1), multiple forward passes are performed with progressively augmented prompts, and the final loss is the mean across all iterations.

We optimize with AdamW at a learning rate of $1 \times 10^{-5}$, with a step learning rate schedule (step size 10, decay factor 0.5).

**Phase 2: Depth training.** The image encoder, SAM's prompt encoder, and mask decoder are frozen. The outer prompt encoder and depth decoder are trained. At the start of Phase 2, the model is loaded from the Phase 1 checkpoint and the depth branch weights are re-initialized from SAM via the structured copy described in Section 3.3. The mask branch runs under no-gradient mode, and its low-resolution mask output is passed (detached) to the depth branch.

The loss function is:

$$\mathcal{L}_{\text{depth}} = \lambda \cdot \text{MSE}(d_{\text{pred}}, d_{\text{gt}})$$

where $d_{\text{pred}}$ is the predicted scalar depth, $d_{\text{gt}}$ is the ground-truth depth (normalized by dividing by 100), and $\lambda$ is the depth loss weight (default 1.0).

The two-phase design ensures that the mask branch converges to reliable segmentation before the depth branch is introduced. Because the depth branch is initialized from SAM's decoder weights, it inherits a meaningful attention structure over image features, and only the depth prediction head must learn from scratch.

## 3.6 Dataset and Point Prompt Generation

Training and evaluation data are organized by depth category, where each category folder encodes the ground-truth depth value in its name (e.g., "f10" denotes a depth of -10, "10" denotes +10). Each folder contains paired RGB images (.jpg) and binary segmentation masks (.png). Depth values are normalized to the range [-1, 1] by dividing by 100.

Point prompts are generated automatically from ground-truth masks. Connected-component analysis is applied to each mask, components are sorted by area in descending order, and the centroids of the largest components (up to max_points, default 3) are used as foreground point prompts. A custom collate function handles variable-length prompt sets across the batch by padding with a label of -1.

## 3.7 Evaluation

Models from Phase 1 and Phase 2 are evaluated separately.

**Segmentation metrics.** We report Dice coefficient, intersection-over-union (IoU), IoU prediction accuracy, and IoU prediction mean absolute error (MAE). These metrics assess both mask quality and the model's self-calibration of mask confidence.

**Depth metrics.** We report MAE, mean squared error (MSE), root mean squared error (RMSE), and the coefficient of determination ($R^2$). At evaluation time, predicted depth values are rescaled to the original range (multiplied by 100) before metric computation. Per-category depth MAE is also reported to identify systematic biases across depth ranges.

**Iterative evaluation.** Evaluation supports the same iterative prompt refinement used during training, enabling a direct comparison between single-pass and multi-iteration inference.

---

**Assumptions or missing inputs:**

- [Evidence needed: exact SA-1B pre-training details or citation for the SAM ViT-B/16 weights used]
- [Evidence needed: dataset name, size, and depth range distribution for quantitative claims]
- [Evidence needed: hardware specification (GPU type, memory) and training time for reproducibility]
- [Evidence needed: overview/pipeline figure reference (e.g., "Fig. 1") to anchor the architecture description]
- [Assumption: "depth" refers to metric depth or ordinal depth -- the text assumes metric depth normalized by 100; confirm with authors]

**Claim-evidence map:**

| Claim | Evidence | Status |
|---|---|---|
| Freezing the image encoder preserves spatial priors from large-scale pre-training | Ablation comparing frozen vs. fine-tuned encoder | Needs evidence |
| Mask-conditioned depth regression reduces ambiguity vs. dense prediction | Comparison with dense depth baselines | Needs evidence |
| SAM weight initialization accelerates depth branch convergence | Training curves with vs. without initialization | Needs evidence |
| Iterative prompt refinement improves mask and depth quality | Ablation on n_sub_iterations (1 vs. 8) | Needs evidence |
| Two-phase training outperforms joint training | Comparison with single-phase training | Needs evidence |

**Why this structure:**

- The Method section follows the pipeline order (encoder -> mask branch -> depth branch -> refinement -> training), matching the data flow through the model. This lets the reader build understanding incrementally.
- Each module is presented with the three-element pattern (motivation, design, role/advantage) so that no component reads as an unjustified black box.
- Training is placed after architecture to separate "what the model is" from "how it is trained," following Nature Methods conventions.
- Evaluation is included as the final subsection because the depth normalization and per-category reporting are integral to understanding the method's scope, not just its results.
