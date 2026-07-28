# 3. Method

We present **DINOSAM**, a promptable framework that jointly performs interactive object segmentation and object-level signed depth regression from a single monocular image. The framework couples a frozen self-supervised DINOv3 backbone with the Segment Anything Model (SAM) decoding machinery through a parameter-efficient multi-level fusion adapter, and extends SAM with a dedicated depth branch that reuses SAM's prompt-conditioned decoding paradigm for scalar depth estimation. Training simulates an interactive user via error-driven prompt refinement and is decoupled into two sequential phases. This section exposes every design decision of the system: data construction (Sec. 3.2), the encoder and its adapter family (Sec. 3.3), the two decoding branches (Secs. 3.4–3.5), the iterative refinement loops (Sec. 3.6), the complete loss inventory (Sec. 3.7), the two-phase optimization protocol (Sec. 3.8), the decoupled dual-DPT extension (Sec. 3.9), and the inference/evaluation protocol (Sec. 3.10).

## 3.1 Problem Formulation and Overview

Given an RGB image $I \in \mathbb{R}^{H \times W \times 3}$ and a set of point prompts $\mathcal{P} = \{(p_i, l_i)\}_{i=1}^{N}$, where $p_i \in \mathbb{R}^2$ is a coordinate in the original image space and $l_i \in \{0, 1\}$ a background/foreground label, the model predicts

$$
(\hat{M}, \hat{s}, \hat{d}) = f_\theta(I, \mathcal{P}),
$$

where $\hat{M} \in \mathbb{R}^{H \times W}$ is the segmentation logit map of the prompted object, $\hat{s} \in [0,1]$ a self-assessed mask-quality (IoU) estimate, and $\hat{d} \in \mathbb{R}$ a signed scalar depth associated with the object. Depth labels are signed offsets relative to a reference plane — negative values denote positions behind it — and are normalized by a constant scale $s_d = 100$ before regression, so the network regresses $d^{\text{gt}}/s_d$ and predictions are rescaled by $s_d$ at evaluation time.

The architecture comprises four components:

1. **Image encoder** (Sec. 3.3) — a frozen DINOv3 ViT-B/16 whose multi-level intermediate features are fused by a DPT-style head and adapted to the SAM embedding space via lightweight adapters, producing a SAM-compatible embedding $E \in \mathbb{R}^{256 \times 64 \times 64}$;
2. **Mask branch** (Sec. 3.4) — SAM's prompt encoder and mask decoder operating on $E$;
3. **Depth branch** (Sec. 3.5) — a structurally parallel *outer* prompt encoder and a *depth-mask decoder* in which SAM's IoU token is repurposed as a depth token; it is conditioned on the mask branch's low-resolution prediction and, during training, on its own accumulated point prompts;
4. **Iterative prompt refinement** (Sec. 3.6) — a training-time interaction simulator that injects error-corrective clicks over $T$ sub-iterations per optimization step, reusing a single image encoding.

## 3.2 Data Construction and Prompt Synthesis

**Depth-encoded dataset layout.** Each training sample is an (image, mask, depth) triplet. Samples are organized in directories whose names encode the signed depth class: a directory named `10` holds objects at depth $+10$, and an `f`-prefixed name (`f10`) holds depth $-10$ (behind the reference plane). Every directory contains paired JPEG images and same-named PNG masks; unpaired images are skipped at scan time. The depth scalar is normalized as $d = d_{\text{raw}} / s_d$ with $s_d = 100$. Masks are binarized with a positive-value threshold and kept at original resolution for full-resolution loss computation. Our dataset provides 1,345 training and 343 validation triplets over a discrete set of signed depth classes.

**Automatic prompt synthesis from annotations.** To emulate realistic user clicks without manual interaction, initial positive prompts are derived from the ground-truth mask by connected-component analysis: components are labeled, ranked by pixel area, and the centroids $(\bar{x}, \bar{y})$ of the $\min(k_{\max}, \#\text{components})$ largest components become positive points ($k_{\max} = 3$). If the mask is empty, the image center is used as a fallback prompt. This yields between 1 and $k_{\max}$ prompts per sample.

**Batch collation with prompt padding.** Because the number of prompts varies per sample, batches are collated by padding every prompt list to the batch maximum: padded coordinates are zeros and padded labels are $-1$, SAM's "not a point" sentinel, which the prompt encoder maps to a dedicated padding embedding rather than a foreground/background embedding. Images within a batch share the same resolution in our dataset, but original sizes are carried per sample and all coordinate transforms and mask post-processing are performed per sample, so the pipeline supports heterogeneous batches.

## 3.3 Image Encoding

### 3.3.1 Frozen DINOv3 Backbone

The backbone is a DINOv3 ViT-B/16 — 12 transformer blocks, embedding dimension $C = 768$, 12 heads, patch size 16, rotary position embeddings (RoPE) with coordinate normalization, 4 storage tokens, LayerScale init $10^{-5}$ — initialized from LVD-1689M pre-trained weights and **kept entirely frozen throughout both training phases**: it contributes zero gradients and zero optimizer state. Inputs are bilinearly resized to a fixed $1024 \times 1024$ square (no aspect-preserving padding on this path) and normalized with ImageNet statistics scaled to the raw $[0, 255]$ range, yielding $N_p = 64 \times 64 = 4096$ patch tokens.

Two rationales drive the frozen-backbone choice: (i) with 1.3K training images, fine-tuning an 86M-parameter ViT invites catastrophic overfitting; (ii) DINOv3's self-supervised features already encode the geometric and semantic structure that both segmentation and monocular depth reasoning require — the adaptation problem is one of *re-projection*, not *re-learning*.

### 3.3.2 Multi-Level DPT-Style Fusion

A single last-layer feature discards the representational spectrum formed across transformer depth: early blocks retain high-frequency spatial detail, deep blocks encode global semantics. We therefore extract features from a configurable set of intermediate blocks $\mathcal{L} = \{\ell_1, \dots, \ell_K\}$ in a single backbone pass,

$$
\{F^{(\ell)}\}_{\ell \in \mathcal{L}} = \mathrm{ViT}_{\text{frozen}}(I), \qquad F^{(\ell)} \in \mathbb{R}^{B \times N_p \times C},
$$

and fuse them with a resolution-preserving simplification of the DPT reassemble-and-fuse pipeline. Since a plain ViT keeps all blocks at the same $64 \times 64$ token resolution, no cross-scale resampling is needed; each stream is reshaped, projected, spatially refined, and the $K$ streams are concatenated and re-projected:

$$
\begin{aligned}
X^{(\ell)} &= \mathrm{reshape}\big(F^{(\ell)}\big) &&\in \mathbb{R}^{B \times C \times 64 \times 64}, \\
Z^{(\ell)} &= \mathrm{Conv}_{3\times3}^{(\ell)}\big(\mathrm{Conv}_{1\times1}^{(\ell)}(X^{(\ell)})\big) &&\in \mathbb{R}^{B \times 256 \times 64 \times 64}, \\
Z &= \mathrm{Conv}_{1\times1}\big([\,Z^{(\ell_1)}; \cdots; Z^{(\ell_K)}\,]\big) &&\in \mathbb{R}^{B \times C \times 64 \times 64},
\end{aligned}
$$

where $[\cdot\,;\cdot]$ is channel concatenation, each stream owns its private $1{\times}1$ projection (to 256 channels) and bias-free $3{\times}3$ refinement convolution (injecting the local spatial mixing that raw token features lack), and the output $1{\times}1$ convolution restores the backbone dimension $C$ so downstream modules are agnostic to $K$. The head supports arbitrary $K$ (3, 4, 12, …), which both layer-selection ablations and the dual-DPT extension (Sec. 3.9) exploit; the pre-projection concatenated tensor can also be exposed for feature-space analysis (Sec. 3.10).

**Layer selection.** Our main configuration uses $\mathcal{L} = \{0, 3, 6, 9\}$ — an *early-start, uniformly spaced* selection with stride 3 — in contrast to the deep-biased default $\{2, 5, 8, 11\}$. Including block 0 preserves near-input spatial detail that benefits boundary quality, on which both the mask decoder and the mask-conditioned depth branch depend.

### 3.3.3 Adapter Family

The fused feature is flattened back to a token sequence and passed through **two cascaded adapters**. The adapter is a pluggable interface — all variants consume $(B, N_p, C)$ sequences plus the spatial shape and are drop-in exchangeable — instantiated in our main configuration as **Mona** (multi-cognitive visual adapter):

$$
\begin{aligned}
u &= \gamma \odot \mathrm{LN}(x) + \gamma_x \odot x, \\
v &= W_{\downarrow} u \in \mathbb{R}^{B \times N_p \times C/8}, \\
w &= v' + \mathrm{Conv}_{1\times1}\!\Big(\tfrac{1}{3}\big(\mathrm{DWC}_{3}(v') + \mathrm{DWC}_{5}(v') + \mathrm{DWC}_{7}(v')\big) + v'\Big), \\
y &= x + W_{\uparrow}\,\mathrm{Drop}_{0.1}\big(\mathrm{GELU}(w)\big),
\end{aligned}
$$

where $v'$ is $v$ reshaped to its $64 \times 64$ spatial layout, $\mathrm{DWC}_k$ is a $k \times k$ depthwise convolution, the bottleneck factor is 8 ($768 \to 96$), and the gating vector $\gamma$ is initialized to $10^{-6}$ while the identity gate $\gamma_x$ is initialized to $1$ — each adapter therefore starts as a near-identity mapping and learns a gradual departure from the frozen features. The three parallel receptive fields (3/5/7) aggregate multi-scale local context, compensating for the frozen backbone's inability to adapt its own spatial mixing to the target domain.

Two alternative adapters instantiate the same interface for controlled comparison: (i) an **FC adapter** — the identical gated bottleneck MLP with the multi-scale convolution removed, isolating the contribution of spatial mixing; and (ii) a **dual-attention adapter** that replaces the bottleneck with parallel position attention (PAM) and channel attention (CAM) modules operating on the 2-D layout, each output gated by a zero-initialized scalar and merged by averaging under a single global residual. The adapter choice composes with the fusion choice as follows: the multi-level DPT configuration is paired with Mona adapters by construction, while in the non-DPT (single-level) configurations the DPT head is bypassed, the adapters consume the backbone's final normalized patch tokens directly, and any of the three adapter variants may be selected.

### 3.3.4 SAM-Space Projection

A final linear projection maps adapted tokens to the SAM prompt-embedding dimension, and the sequence is reshaped to the SAM-native layout:

$$
E = \mathrm{reshape}\big(W_{\text{proj}}\, y\big) \in \mathbb{R}^{B \times 256 \times 64 \times 64}.
$$

This matches SAM's image-encoder output specification exactly, so the entire SAM decoding stack — dense positional encodings included — operates unmodified downstream, and the DINOv3 path is a drop-in replacement for SAM's ViT encoder (which remains available as a baseline path with SAM's own longest-side-resize, normalize, and pad preprocessing). The trainable encoder-side parameters are confined to the DPT head, the two adapters, and the projection.

## 3.4 Mask Branch: Prompt-Conditioned Segmentation

The mask branch reuses SAM ViT-B's prompt encoder and mask decoder, initialized from the official SAM checkpoint. Its forward pass involves four steps, with a deliberate batched/per-sample split:

1. **Coordinate transform (per sample).** Point prompts given in original image coordinates are mapped into the $1024 \times 1024$ model space by the longest-side-resize transform, applied per sample since original sizes may differ.
2. **Prompt encoding (batched).** The prompt encoder embeds the transformed points — foreground, background, and padding labels each receiving distinct learned embeddings — together with an optional low-resolution mask prompt $M^{\text{prev}} \in \mathbb{R}^{B \times 1 \times 256 \times 256}$ carried over from a previous refinement iteration, into sparse and dense prompt embeddings.
3. **Mask decoding (per sample).** The mask decoder performs two-way cross-attention between prompt tokens and the image embedding and outputs, in single-mask mode, a low-resolution logit map $M^{\text{low}} \in \mathbb{R}^{B \times 1 \times 256 \times 256}$ and an IoU estimate $\hat{s} \in \mathbb{R}^{B \times 1}$. Decoding runs per sample because SAM's decoder duplicates image embeddings per prompt set via `repeat_interleave`; feeding batched inputs would produce $B^2$-shaped intermediate tensors. Prompt encoding has no such coupling and stays batched.
4. **Post-processing (per sample, optional).** Full-resolution logits are recovered by SAM's two-stage interpolation — upsample to $1024^2$, crop padding, resize to each sample's original size. When full-resolution masks are not needed (e.g., non-logging steps in phase 2 where the mask branch is frozen), this step is skipped to save compute.

## 3.5 Depth Branch: Prompt- and Mask-Conditioned Depth Regression

### 3.5.1 Design

The key observation motivating the depth branch is that SAM's decoder is a general query-based readout mechanism: learnable output tokens attend to a prompt-conditioned image embedding and are decoded into task predictions. We instantiate a second, structurally parallel copy of this mechanism for depth, consisting of:

- an **outer prompt encoder**, identical in structure to SAM's prompt encoder (its own point embeddings, mask downscaling network, and dense positional encoding), that consumes (i) the depth branch's own point prompts and (ii) the mask branch's low-resolution prediction $M^{\text{low}}$ as a dense mask prompt;
- a **depth-mask decoder** carrying **two output tokens** — one *mask token* and one *depth token* that structurally replaces SAM's IoU token.

The mask conditioning is the central design element: the dense mask prompt tells the depth decoder *which* object the scalar regression refers to, converting an ill-posed global regression into an object-grounded one. $M^{\text{low}}$ enters the depth branch as conditioning only — in the joint forward and at evaluation it is explicitly detached, and during phase-2 training every module upstream of it is frozen — so no depth gradient ever reshapes the segmentation.

### 3.5.2 Depth-Mask Decoder

The decoder mirrors SAM's mask decoder computation. The depth and mask tokens are concatenated with the sparse prompt embeddings; the image embedding, summed with the dense prompt embedding, undergoes two-way attention against the token set:

$$
(\mathbf{t}^{\text{out}}, E') = \mathrm{Transformer}\big(E + e_{\text{dense}},\ \mathrm{PE}_{\text{outer}},\ [\,t_{\text{depth}}; t_{\text{mask}}; e_{\text{sparse}}\,]\big).
$$

The post-attention mask token modulates a $4\times$-upscaled embedding (two stride-2 transposed convolutions with LayerNorm and GELU, $256 \to 64 \to 32$ channels) through a 3-layer hypernetwork MLP, producing an auxiliary low-resolution mask by inner product:

$$
\hat{M}_{\text{aux}} = \mathrm{MLP}_{\text{hyper}}\big(t_{\text{mask}}^{\text{out}}\big) \cdot \mathrm{Upscale}(E') \in \mathbb{R}^{B \times 1 \times 256 \times 256},
$$

while the post-attention depth token is read out by a dedicated 3-layer MLP (hidden dim 256):

$$
\hat{d} = \mathrm{MLP}_{\text{depth}}\big(t_{\text{depth}}^{\text{out}}\big) \in \mathbb{R}^{B \times 1}.
$$

The decoder always operates in single-mask mode (no multi-mask ambiguity logic). The auxiliary mask output serves two purposes: it anchors the token set's attention to the prompted object's spatial extent (implicit localization pressure on the shared transformer), and its errors drive the corrective-click sampling of the phase-2 iterative loop (Sec. 3.6.3).

### 3.5.3 Depth Transformer Variants

The decoder's transformer is pluggable with three variants:

- **TwoWay** (main configuration): SAM's two-way transformer — 2 layers, 8 heads, MLP dim 2048, attention downsample rate 2 — enabling weight transfer from SAM (Sec. 3.5.4).
- **Dual-attention**: a variant with SAM-compatible interface in which each layer's *token self-attention* step is replaced by PAM + CAM enhancement of the image features (zero-initialized gates, residual, LayerNorm), followed by the standard token→image cross-attention, token MLP, and image→token cross-attention steps, and a final token→image attention. Because its layer structure diverges from SAM's, it trains from random initialization.
- **Simple** (lower-bound baseline): the prompt encoder and decoder are removed entirely; depth is regressed by global average pooling of $E$ followed by a $256 \to 1024 \to 256 \to 1$ MLP. This variant has no prompt or mask conditioning and quantifies the value of object-grounded decoding.

### 3.5.4 Initialization from SAM

Training a prompt-decoder stack from scratch on 1.3K images would overfit severely. The depth branch is therefore initialized from SAM's pre-trained weights with a structure-aware mapping:

| Depth-branch module | Initialized from |
|---|---|
| outer prompt encoder | SAM prompt encoder (identical structure, full copy) |
| two-way transformer | SAM mask-decoder transformer (full copy; skipped for the dual-attention variant) |
| depth token | SAM IoU token (same shape) |
| mask token | first of SAM's 4 mask tokens |
| output upscaling | SAM output upscaling (full copy) |
| hypernetwork MLP | SAM's first hypernetwork MLP |
| depth regression MLP | random (no SAM counterpart) |

This transfers SAM's prompt-grounding and attention behavior wholesale; only the final scalar readout must be learned from data. The initialization is applied at model construction and re-applied at the start of phase 2 after loading the phase-1 checkpoint, guaranteeing that depth training always starts from the SAM prior rather than from whatever untouched state the phase-1 checkpoint carries.

## 3.6 Iterative Error-Driven Prompt Refinement

Single-click prompting rarely yields precise masks, and a depth branch conditioned on an imprecise mask inherits its errors. Following interactive-segmentation practice, each optimization step simulates a user progressively correcting the model over $T$ sub-iterations ($T = 8$). Crucially, the expensive image encoding runs **once per step**; all sub-iterations reuse the cached embedding $E$, so the marginal cost of refinement is decoder-only.

### 3.6.1 Error-Based Prompt Generator

After each sub-iteration, the predicted binary mask is compared against the resolution-matched ground truth (GT downsampled to the $256 \times 256$ prediction grid by nearest-neighbor interpolation; prediction binarized at logit 0) to extract two error regions:

$$
\Omega^{-} = \{x : M^{\text{gt}}(x)=1 \wedge \hat{M}(x)=0\} \quad \text{(missed foreground)}, \qquad
\Omega^{+} = \{x : M^{\text{gt}}(x)=0 \wedge \hat{M}(x)=1\} \quad \text{(false positive)}.
$$

Per image, **one positive click** is sampled uniformly from $\Omega^{-}$ and **one negative click** uniformly from $\Omega^{+}$. Both samplings have deterministic fallback chains for the degenerate cases:

- *positive*: $\Omega^{-}$ empty → sample from the prediction–GT overlap; overlap empty → sample from GT foreground; mask empty → image center;
- *negative*: $\Omega^{+}$ empty → sample background inside the GT bounding box dilated by 3 px (keeping the corrective click informative, near the object boundary); none available → any background pixel; none → origin.

Sampled grid coordinates are rescaled to original image space by the per-sample grid-to-image ratio and **appended** to the running prompt set with labels $(1, 0)$. The prompt sequence thus grows monotonically — $N, N{+}2, N{+}4, \dots$ — mimicking an accumulating interaction history; the model must remain consistent under an ever-denser prompt context.

### 3.6.2 Phase-1 Loop: Iterative Mask Refinement

Phase 1 trains the mask branch (and the encoder adapters) under the refinement loop:

> **for** $t = 1, \dots, T$:
> 1. run the mask branch with the accumulated prompts $\mathcal{P}_t$ and optional mask prompt $M^{\text{prev}}_t$, obtaining $(M^{\text{low}}_t, \hat{s}_t, \hat{M}_t)$ at full resolution;
> 2. accumulate the segmentation and IoU-regression losses (Sec. 3.7);
> 3. if $t < T$: sample corrective clicks from $\hat{M}_t$'s errors, append to get $\mathcal{P}_{t+1}$; with probability $p_{\text{mask}} = 0.5$ set $M^{\text{prev}}_{t+1} = \mathrm{sg}[M^{\text{low}}_t]$ (the detached previous logit map), otherwise no mask prompt.

The stochastic mask carry-over trains the decoder to operate both with and without its own prior prediction as dense context — matching SAM's interactive inference regime where a previous mask may or may not be fed back. Losses are averaged over the $T$ sub-iterations and backpropagated in a single optimizer step; gradients flow through all sub-iterations into the shared decoder weights and encoder adapters (prompt sampling itself is gradient-free).

### 3.6.3 Phase-2 Loop: Iterative Depth Refinement

Phase 2 inverts the roles: the mask branch — now frozen — runs **once** to produce $M^{\text{low}}$, which is fixed for the entire step as the depth branch's dense conditioning; the depth branch then iterates:

> run mask branch once → $M^{\text{low}}$; initialize the depth prompt set empty: $\mathcal{Q}_1 = \varnothing$;
> **for** $t = 1, \dots, T$:
> 1. run the depth branch with dense prompt $M^{\text{low}}$ and sparse prompts $\mathcal{Q}_t$, obtaining $(\hat{d}_t, \hat{M}_{\text{aux},t})$;
> 2. accumulate the depth loss; compute the auxiliary-mask Dice against GT as a *monitored diagnostic* (not optimized);
> 3. if $t < T$: downsample $\hat{M}_{\text{aux},t}$ (bilinear) and the GT (nearest) to the $256^2$ grid, sample corrective clicks from the *auxiliary decoder mask's* errors, rescale to image space, and append to get $\mathcal{Q}_{t+1}$.

Two properties of this loop deserve emphasis. First, the corrective clicks are driven by the *depth decoder's own* localization errors, not the frozen mask branch's — the loop supervises the depth branch's grounding directly. Second, because $\mathcal{Q}_1 = \varnothing$, the first sub-iteration is conditioned on the mask alone, and later sub-iterations add $2, 4, \dots, 2(T{-}1)$ clicks: each step exposes the depth token to a curriculum of localization certainty, from mask-only conditioning to densely clicked objects. This matches the deployment spectrum — at inference the depth branch may be queried with the mask alone (our evaluation protocol, Sec. 3.10) or with any number of user clicks.

In the non-iterative degenerate case ($T = 1$, and always for the prompt-free *simple* depth variant), the loop collapses to a single depth forward conditioned on the mask prompt together with the initial centroid points, mirroring the validation-time protocol (Sec. 3.7).

## 3.7 Training Objectives

All mask-level losses operate on full-resolution logits against original-resolution binary GT.

**Dice loss** (soft, on probabilities):

$$
\mathcal{L}_{\text{dice}}(\hat{M}, M^{\text{gt}}) = 1 - \frac{1}{B}\sum_b \frac{2\sum_x \sigma(\hat{M}_b)\, M^{\text{gt}}_b + \varepsilon}{\sum_x \sigma(\hat{M}_b) + \sum_x M^{\text{gt}}_b + \varepsilon}.
$$

**IoU-regression loss.** SAM's IoU head is kept calibrated under domain shift by regressing it onto the *actual* IoU of the current prediction, computed gradient-free at threshold 0.5:

$$
\mathcal{L}_{\text{iou}} = \big\| \hat{s} - \mathrm{IoU}_{0.5}\big(\mathrm{sg}[\hat{M}], M^{\text{gt}}\big) \big\|_2^2 .
$$

**Boundary loss (optional, off in the main configuration).** A boundary-band Dice term computable on demand for jagged-edge suppression: the GT's inner boundary band $M^{\text{gt}} - \mathrm{erode}_k(M^{\text{gt}})$ (morphological erosion via min-pooling, half-width 3) restricts the Dice computation to boundary pixels; the term is added to the mask loss with a configurable weight.

**Depth loss.** Plain MSE on the normalized scalar: $\mathcal{L}_{\text{depth}} = \| \hat{d} - d^{\text{gt}} \|_2^2$.

**Phase-1 objective** (iterative form; the non-iterative $T{=}1$ form is the same without averaging):

$$
\mathcal{L}_{\text{phase1}} = \frac{1}{T} \sum_{t=1}^{T} \Big[ \mathcal{L}_{\text{dice}}\big(\hat{M}_t, M^{\text{gt}}\big) + \mathcal{L}_{\text{iou},t} \Big],
$$

with the boundary term added inside the bracket when enabled. No depth term appears: the depth branch is frozen and never executed in phase 1.

**Phase-2 objective:**

$$
\mathcal{L}_{\text{phase2}} = \lambda_d \cdot \frac{1}{T} \sum_{t=1}^{T} \big\| \hat{d}_t - d^{\text{gt}} \big\|_2^2, \qquad \lambda_d = 1.
$$

Only the depth term is optimized in phase 2. The auxiliary decoder-mask Dice and the (frozen) inner mask branch's Dice are logged as diagnostics — the former tracks whether the depth branch retains object grounding, the latter verifies the frozen mask branch's stationarity — but contribute no gradient. Averaging over sub-iterations (rather than supervising only the final iterate) weights all conditioning densities equally, from mask-only to densely-clicked.

**Validation objectives and model selection.** Validation runs non-iteratively with the initial centroid prompts only: phase 1 selects checkpoints by $\mathcal{L}_{\text{dice}} + \mathcal{L}_{\text{iou}}$ (including the boundary term when enabled), phase 2 by $\lambda_d \mathcal{L}_{\text{depth}}$, each on the held-out set.

## 3.8 Two-Phase Training Protocol

Optimization is decoupled into two sequential phases with disjoint trainable sets:

| Component | Phase 1 | Phase 2 |
|---|---|---|
| DINOv3 backbone | frozen | frozen |
| DPT head + adapters + projection | **trainable** | frozen |
| SAM prompt encoder + mask decoder | **trainable** | frozen |
| outer prompt encoder | frozen | **trainable** |
| depth-mask decoder | frozen | **trainable** |

Both phases use AdamW over the trainable parameters only, learning rate $10^{-5}$, StepLR halving every 10 epochs, 50 epochs per phase, batch size 8. Phase 2 is initialized from the best phase-1 checkpoint (selected by validation mask loss), after which the depth branch is re-initialized from SAM (Sec. 3.5.4).

The rationale for decoupling is twofold. First, depth-regression gradients at initialization are large and uninformative (the depth readout is random); letting them flow into shared encoder adapters would distort the segmentation representation the depth branch itself depends on. Second, freezing the mask branch in phase 2 makes the conditioning distribution $p(M^{\text{low}} \mid I, \mathcal{P})$ stationary, turning depth learning into a well-posed regression on a fixed representation rather than a moving-target problem.

**Checkpointing and resumption.** Each phase persists three checkpoint streams — best-by-validation, latest (with optimizer and scheduler state, enabling exact resumption), and periodic snapshots every 10 epochs — each tagged with the adapter-type metadata used to diagnose configuration mismatches at load time. Checkpoint loading is tolerant (`strict=False`) with explicit reporting of missing/unexpected keys, and transparently remaps legacy parameter names, enabling cross-configuration transfer of the shared modules.

## 3.9 Decoupled Dual-DPT Layers for Mask and Depth

### 3.9.1 Motivation

The encoder of Sec. 3.3 feeds *both* branches from a single fused embedding built from one layer set $\mathcal{L}$. This imposes a hidden compromise: segmentation favors early, boundary-preserving layers, whereas monocular depth cues — perspective, occlusion ordering, object–context relations — concentrate in middle-to-late transformer blocks. A single $\mathcal{L}$ can only trade one branch's preference against the other's. We therefore extend the encoder with an optional **dual-head mode** that decouples the layer selection per branch while still executing the frozen backbone exactly once.

### 3.9.2 Architecture

Two layer sets are configured — $\mathcal{L}_m$ for the mask branch, $\mathcal{L}_d$ for the depth branch — each required to be strictly ascending and duplicate-free, matching the backbone's extraction-order contract. A single backbone pass extracts the sorted union, which a pure index-dispatch function routes to two fully independent encoding streams:

$$
\begin{aligned}
\mathcal{U} &= \mathrm{sort}\big(\mathcal{L}_m \cup \mathcal{L}_d\big), \qquad
\{F^{(\ell)}\}_{\ell \in \mathcal{U}} = \mathrm{ViT}_{\text{frozen}}(I), \\
E_m &= \Phi_m\big(\{F^{(\ell)}\}_{\ell \in \mathcal{L}_m}\big), \qquad
E_d = \Phi_d\big(\{F^{(\ell)}\}_{\ell \in \mathcal{L}_d}\big),
\end{aligned}
$$

where each stream $\Phi_{\{m,d\}}$ owns a private DPT head, private adapter pair, and private projection, producing two SAM-compatible embeddings $E_m, E_d \in \mathbb{R}^{B \times 256 \times 64 \times 64}$. The mask branch decodes from $E_m$, the depth branch from $E_d$; each branch selects its embedding at entry, so every encoder call site — including the iterative loops, which encode once and reuse across $T$ sub-iterations — is unchanged. The added cost is one extra head-adapter-projection stream, small relative to the 12-block backbone (< 5% additional forward compute); the backbone, dominating both memory and FLOPs, is never re-executed.

### 3.9.3 Training and Transfer Semantics

The dual-head mode integrates into the two-phase schedule with per-branch freezing: in phase 1, $\Phi_d$ is frozen together with the depth branch (producing no gradient or activation-storage overhead); in phase 2, it joins the outer prompt encoder and depth-mask decoder in the trainable set, while $\Phi_m$ remains frozen. The depth branch thus learns its own view of the frozen backbone precisely when — and only when — depth supervision exists.

To avoid retraining phase 1 when enabling the mode, phase 2 **warm-starts** the depth-side adapters and projection from their phase-1-trained mask-side counterparts, transferring the learned domain adaptation. The depth-side DPT head is deliberately left randomly initialized: whenever $\mathcal{L}_d \neq \mathcal{L}_m$ — the very situation the mode exists for — its per-layer projections have different layer semantics (and possibly different cardinality), so weight copying would be structurally meaningless. The warm-start is unconditional at phase-2 entry, since phase 1 by construction never trains the depth-side stream.

The design is strictly backward compatible: with the mode disabled, the parameter set, checkpoint format, and computation are byte-identical to the single-head model, and legacy single-head checkpoints load directly into the mask-side stream. Cross-mode checkpoint mismatches (dual-head weights loaded without the flag, or vice versa) are detected by exact module-name matching and rejected explicitly, preventing the failure mode of a depth decoder silently regressing depth from an untrained embedding.

## 3.10 Inference and Evaluation Protocol

**Interactive inference.** At test time the mask branch supports the same iterative protocol as training — accumulated corrective clicks with optional mask-prompt carry-over (disabled by default at evaluation) — with a configurable number of refinement rounds. As in training, evaluation-time corrective clicks are sampled from the prediction's error regions against the ground-truth mask, i.e., the protocol simulates an oracle user; reported iterative results therefore measure refinement capacity under idealized feedback rather than free-running inference. The depth branch is then queried **once**, conditioned on the final low-resolution mask alone (no depth-branch point prompts): this evaluates the deployment-relevant regime in which the user interacts with the segmentation and depth is derived automatically, and is the regime the mask-only end of the phase-2 conditioning curriculum (Sec. 3.6.3) prepares.

**Metrics.** Segmentation is measured at original resolution after 0.5-thresholding of the sigmoid: Dice, IoU, boundary IoU (IoU restricted to 2-px-half-width inner boundary bands, isolating edge quality from region quality), precision, and recall — reported as means and medians. IoU-head calibration is measured as the MAE between predicted and actual IoU. Depth predictions are rescaled to the original signed scale ($\times s_d$) and evaluated by MAE, MSE, RMSE, and the coefficient of determination $R^2$, plus per-depth-class MAE/RMSE breakdowns and prediction-vs-GT / residual-vs-GT scatter analyses that expose class-conditional bias. Phase-1 and phase-2 checkpoints are evaluated under the identical protocol, which quantifies the (expected null) effect of depth training on the frozen segmentation.

**Feature-space analysis.** For qualitative analysis of what the multi-level fusion learns, we visualize per-layer backbone features and the pre-projection fused DPT representation by per-image PCA to three components (optionally foreground-restricted using a mask), rendered with sigmoid contrast scaling. The same visualization can be hooked into phase-1 validation at a configurable epoch interval on a fixed validation batch, providing a training-time monitor of representation drift. Layer-wise PCA maps make the early-vs-late representational division of labor directly visible and motivated the early-start layer selection.

## 3.11 Implementation Details

All experiments use the SAM ViT-B checkpoint (`sam_vit_b_01ec64`) and the DINOv3 ViT-B/16 LVD-1689M checkpoint. The main configuration is: DINOv3 encoder with DPT multi-level fusion, $\mathcal{L} = \{0, 3, 6, 9\}$; Mona adapters (factor 8); TwoWay depth transformer; two-phase training ("both"), 50 epochs per phase; batch size 8; AdamW with default weight decay, lr $10^{-5}$, StepLR $\times 0.5$ every 10 epochs; $T = 8$ refinement sub-iterations; up to $k_{\max} = 3$ initial centroid prompts; mask-prompt carry-over probability $p_{\text{mask}} = 0.5$; depth loss weight $\lambda_d = 1$; depth normalization scale $s_d = 100$; boundary loss disabled. No photometric or geometric data augmentation is applied — the prompt-synthesis randomness of the iterative loop (error-region click sampling and stochastic mask carry-over) is the only stochasticity beyond batch shuffling. Training set: 1,345 triplets; validation set: 343. Data loading uses 4 workers with pinned memory; validation and model selection run after every epoch.
