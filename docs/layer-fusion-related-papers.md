# 视觉大模型层融合策略相关论文综述

> 调研时间：2026-07
> 关联项目：DINOSAM（Depth-aware Segment Anything Model）
> 本文档覆盖两大方向：MoE 路由机制与视觉基础模型层融合方案，每篇附主要工作和对 DINOSAM 的启发。

---

## 目录

- [1. MoE 稀疏路由与负载均衡](#1-moe-稀疏路由与负载均衡)
- [2. 可微 Top-K 选择算子](#2-可微-top-k-选择算子)
- [3. 层级选择与跳层分析](#3-层级选择与跳层分析)
- [4. MoE 表征坍缩与 Expert Choice](#4-moe-表征坍缩与-expert-choice)
- [5. Pipeline 级组合（无内部层融合）](#5-pipeline-级组合无内部层融合)
- [6. 显式多尺度融合模块](#6-显式多尺度融合模块)
- [7. 分层骨干（生成多尺度特征，不主动融合）](#7-分层骨干生成多尺度特征不主动融合)
- [8. 无多层融合的单尺度设计](#8-无多层融合的单尺度设计)
- [9. MoE 与层选择融合](#9-moe-与层选择融合)
- [10. 稠密层加权融合](#10-稠密层加权融合)
- [11. 方案对比总结](#11-方案对比总结)
- [12. 对 DINOSAM 的启发与改进路线](#12-对-dinosam-的启发与改进路线)

### Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
- **作者**: Noam Shazeer, Azalia Mirhoseini, et al.
- **发表**: ICLR 2017
- **链接**: https://arxiv.org/abs/1701.06538
- **概述**: MoE 领域的奠基性工作。提出稀疏门控混合专家层，每个 token 仅路由到 top-k 个专家，使模型容量可扩展至万亿参数而不增加推理成本。引入了 importance-balancing loss（门控值的平方变异系数）防止路由坍缩，以及 noisy top-k gating（加入高斯噪声）促进探索。发现零初始化 W_g 和 W_noise 矩阵对训练稳定性至关重要。

### Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
- **作者**: William Fedus, Barret Zoph, Noam Shazeer
- **发表**: ICML 2022
- **链接**: https://arxiv.org/abs/2101.03961
- **概述**: 将 MoE 简化为 top-1 路由（每个 token 只选一个专家），证明 top-1 与 top-k 质量相当但计算更高效。定义了负载均衡损失 `L_aux = α · N · Σ(f_i · P_i)`，通过 sweep 确定 α=1e-2 为最优。引入 expert capacity factor 控制每个专家处理的 token 上限。发现选择性使用 float32 精度处理 router logits 可稳定 hard-switching 决策。在 C4 数据集上达到万亿参数规模。

### Loss-Free Balancing for Mixture-of-Experts
- **作者**: Wang et al.
- **发表**: 2024
- **链接**: https://arxiv.org/abs/2408.15664
- **概述**: 提出无需辅助 loss 的负载均衡方法。核心思想：在 top-k 选择之前，对 router logits 施加动态更新的 expert-wise bias（基于近期负载统计）。当某专家过载时降低其 bias，欠载时提高 bias。在 3B 参数模型、200B token 训练上验证，比辅助 loss 方法性能更好且负载更均衡。消除了辅助 loss 梯度与任务梯度的干扰。

### A Survey on Mixture of Experts in Large Language Models
- **作者**: 综述论文
- **发表**: 2024
- **链接**: https://arxiv.org/abs/2407.06204
- **概述**: 系统梳理了 MoE 在 LLM 中的发展脉络，涵盖路由策略（token-choice vs expert-choice）、负载均衡技术、训练稳定性问题。指出 X-MoE 发现的表征坍缩问题——门控机制导致隐表示向专家质心聚集。讨论了 expert capacity factor 的调节方法和 token dropping 现象（ST-MoE 证明 fine-tuning 阶段 10-15% 的 dropping 不影响性能）。

---

## 2. 可微 Top-K 选择算子

### Fast Differentiable Sorting and Topk
- **作者**: Michael E. Sander, Joachim Puigcerver, Jos Djolonga, Gabriel Peyré, Mathieu Blondel
- **发表**: ICML 2023
- **链接**: https://arxiv.org/abs/2302.01425
- **概述**: 提出同时满足可微性和稀疏性的 top-k 算子。将 top-k 选择建模为排列多面体（permutahedron）上的线性规划，通过 p-norm 正则化（1<p<2）实现精确稀疏。化简为等渗优化问题，可用 PAV 算法或 GPU 友好的 Dykstra 算法求解。区别于 DSelect-k（仅推理时稀疏）和 SOFT top-k（可微但稠密）。成功应用于 MoE router、权重剪枝、ViT fine-tuning。36 次引用，被 TGR-MoE (2026) 和 SoftJAX (2026) 采用。

---

## 3. 层级选择与跳层分析

### Sensitivity Analysis of Block Skipping in Deep Neural Networks
- **作者**: Korol et al.
- **发表**: arXiv preprint, 2025
- **链接**: https://arxiv.org/abs/2505.17626
- **概述**: 系统研究了 ResNet 中跳层的配置敏感性。发现在 ResNet-110 中跳过 37/54 个 block 时，最差和最好配置之间精度差距高达 49.61%。提出 design-time sensitivity analysis 方法——逐 block 评估精度影响来排序，优于 L2-norm、Fisher、Hessian 和随机搜索。但关键限制：整个方法依赖 Stochastic Depth 训练使模型具备跳层鲁棒性；未经验证 SD 训练时优势不确定。仅在 CNN + CIFAR 上测试，未涉及 ViT 或密集预测任务。

---

## 4. MoE 表征坍缩与 Expert Choice

### X-MoE: Exploring Mixture-of-Experts with Representation Collapse
- **作者**: Zewen Chi, et al.
- **发表**: 2022
- **概述**: 发现标准 MoE 门控机制导致表征坍缩（representation collapse）——隐表示向量聚集在专家质心附近，降低了表征多样性。提出将隐向量投影到低维空间并 L2 归一化，使门控分数在超球面上计算，缓解坍缩问题。（注：原文因网络限制未能直接验证，信息来自综述二次引用）

### Scaling Vision with Sparse Mixture of Experts (ST-MoE)
- **作者**: Mustafa, et al. (Google)
- **发表**: 2022
- **链接**: https://arxiv.org/abs/2202.08906
- **概述**: 将 MoE 应用于视觉 Transformer（ViT），验证了稀疏 MoE 在视觉任务上的有效性。发现 fine-tuning 阶段 10-15% 的 token dropping 不影响最终性能。指出 expert capacity 过小会导致性能下降，但过大会浪费计算。

### Mixture-of-Experts with Expert Choice Routing
- **作者**: Yanping Huang, et al. (Google)
- **发表**: NeurIPS 2022
- **链接**: https://arxiv.org/abs/2202.09368
- **概述**: 提出 expert-choice routing 替代 token-choice routing。由专家选择 token 而非 token 选择专家，从根本上避免负载均衡问题（每个专家自主决定处理多少 token）。指出 token-choice routing 的负载不均衡是其根本缺陷。

---

## 5. Pipeline 级组合（无内部层融合）

### Grounded-SAM / Grounded-SAM-2
- **作者**: IDEA-Research
- **链接**: https://github.com/IDEA-Research/Grounded-Segment-Anything / https://github.com/IDEA-Research/Grounded-SAM-2
- **概述**: 将 Grounding DINO（文本→检测）与 SAM（检测→分割）串联，形成文本驱动的零样本分割系统。Grounded-SAM-2 进一步扩展到 3D 和跟踪。
- **融合方式**: 纯 pipeline 级——检测器输出 bounding box 作为 SAM 的 prompt 输入。两个模型**无共享参数、无交叉注意力、无特征图交互**。
- **对 DINOSAM 的启发**: DINOSAM 的 mask branch 和 depth branch 共享同一个 frozen encoder，已超越纯 pipeline 组合。如果未来引入更多基础模型（如 depth foundation model），可参考此方案做模块级串联而不侵入骨干。缺点是无法利用中间层特征的互补性，信息瓶颈在 prompt 传递。

---

## 6. 显式多尺度融合模块

### 6.1 Feature Pyramid Network (FPN)

- **作者**: Lin et al.
- **发表**: CVPR 2017
- **链接**: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
- **概述**: 构建自顶向下路径 + 横向连接，将骨干网络不同阶段的特征融合为统一的多尺度特征金字塔。Lateral 连接用 1×1 卷积通道对齐，融合用 element-wise sum（默认）或 average，3×3 卷积精炼输出。扩展模块 `LastLevelMaxPool`（P5→P6）、`LastLevelP6P7`（C5→P6/P7）。
- **核心机制**:
```
C5 ──1x1 conv──→ P5 ──3x3 conv──→ output
                 ↑ (upsample 2x, nearest)
C4 ──1x1 conv──→ P4 ──3x3 conv──→ output
                 ↑ (upsample 2x)
C3 ──1x1 conv──→ P3 ──3x3 conv──→ output
```
- **对 DINOSAM 的启发**: FPN 的 top-down + lateral 结构可替代 DINOSAM 当前 DPT 的简单 concat，在不同空间分辨率的特征之间融合更高效。若 DINOv3 骨干改用 hierarchical ViT（如 Swin），FPN 是天然的下游融合头。FPN 的 sum 融合保持通道数不变，DPT 的 concat 增加信息容量但参数量更大。潜在改进：在 DINOSAM 的 DPTHead 中引入 FPN 式 top-down 路径，让深层特征（语义强）逐步引导浅层特征（空间精细）。

### 6.2 DPT Head（渐进式融合）

- **作者**: Ranftl et al.
- **发表**: ICCV 2021
- **链接**: https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/segmentation/models/decode_heads/dpt_head.py
- **概述**: 提出 Dense Prediction Transformer (DPT)，将 ViT 的中间层特征通过 ReassembleBlocks + FeatureFusionBlock 进行渐进式融合，用于密集预测任务（深度估计、语义分割）。
- **核心机制**:
```python
# DINOv2 get_intermediate_layers: 提取任意指定层
features = model.get_intermediate_layers(x, n=[4, 8, 12, 16])  # 4 个中间层

# DPT 渐进融合: 从深层向浅层逐级融合
out = fusion_blocks[0](x[-1])           # 最深层
out = fusion_blocks[1](out, x[-2])      # + 次深层
out = fusion_blocks[2](out, x[-3])      # + 次浅层
out = fusion_blocks[3](out, x[-4])      # + 最浅层
```
- **ReassembleBlocks**: 将 4 个中间层投影到不同通道 [96, 192, 384, 768]，并通过 resample 对齐空间分辨率
- **FeatureFusionBlock**: 残差卷积块 + 双线性上采样，逐级融合
- **对 DINOSAM 的启发**: DINOSAM 当前使用的 `DPTSimpleHead` 是 DPT 的简化版：4 层同分辨率 → 1×1 投影 → 3×3 细化 → concat → 1×1 输出。简化版丢失了原版 DPT 的渐进融合优势（深层语义引导浅层空间）。改进方向：将 `DPTSimpleHead` 升级为带 `FeatureFusionBlock` 的完整 DPT，或至少引入 FPN 式 top-down 路径。原版 DPT 的 ReassembleBlocks 对不同分辨率特征做 resample，DINOSAM 因 ViT 所有层同分辨率而无法利用此优势，可考虑在 DINOv3 骨干中注入 pooling/stride 来制造多尺度。

### 6.3 Resize-Concat 策略

- **作者**: DINOv2 评估代码
- **链接**: https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/segmentation/models/decode_heads/linear_head.py
- **概述**: DINOv2 评估代码中的轻量级融合头（BNHead）。选取多个中间层，resize 到统一空间分辨率后直接沿通道拼接。
- **核心机制**:
```python
selected = [inputs[i] for i in layer_indices]
resized = [F.interpolate(f, size=target_size, mode='bilinear') for f in selected]
fused = torch.cat(resized, dim=1)
```
- **对 DINOSAM 的启发**: 最简单的多层融合 baseline，可作为消融实验的对照组。DINOSAM 的 `DPTSimpleHead` 本质上是 resize-concat 的增强版（多了 1×1 投影 + 3×3 细化）。如果消融实验证明 DPT 的投影/细化贡献不大，可退化到此方案以节省参数。

---

## 7. 分层骨干（生成多尺度特征，不主动融合）

### Swin Transformer

- **作者**: Liu et al. (Microsoft)
- **发表**: ICCV 2021
- **链接**: https://github.com/microsoft/Swin-Transformer
- **概述**: 提出分层 ViT 架构，通过 4 个阶段（stage）+ Patch Merging 生成 1/4, 1/8, 1/16, 1/32 分辨率的多尺度特征图。Shifted Window Attention 在保持线性复杂度的同时实现跨窗口信息流。
- **核心机制**:
```
Stage 1: patch_size=4 → 1/4 分辨率
    ↓ Patch Merging (2x2 downsample + linear)
Stage 2: → 1/8 分辨率
    ↓ Patch Merging
Stage 3: → 1/16 分辨率
    ↓ Patch Merging
Stage 4: → 1/32 分辨率
```
- **关键区分**: Swin 本身**不执行融合**——它生成多尺度特征，但实际融合交给下游头（FPN、UPerNet）。Patch Merging 是下采样而非融合。
- **对 DINOSAM 的启发**: DINOSAM 当前用 ViT-B/16（单尺度），所有层输出 64×64 特征。若将骨干替换为 Swin Transformer，天然获得多尺度特征，可配合 FPN/UPerNet 做真正的多尺度融合。但 Swin 的 shifted window 会破坏全局注意力，可能影响深度估计等需要全局上下文的任务。替代方案：在现有 ViT 骨干中插入 stride attention 或 pooling 层来制造多尺度。

---

## 8. 无多层融合的单尺度设计

### MAE (Masked Autoencoder)

- **作者**: He et al.
- **发表**: CVPR 2022
- **链接**: https://github.com/facebookresearch/mae
- **概述**: 提出掩码自编码器预训练范式。编码器仅处理可见 patch（75% masking），解码器在输入处重新引入 mask token 进行重建。
- **融合特点**: **无任何多层融合**。编码器仅返回最终层归一化输出；解码器将 mask token 拼接后通过独立 transformer blocks。
- **对 DINOSAM 的启发**: MAE 证明了单层特征 + 强预训练目标即可获得优秀的表征能力。如果 DINOv3 的预训练足够强，多层融合的边际收益可能有限。消融实验建议：对比 DPT 多层融合 vs. 仅用最后一层，量化多层融合的增益。

### SAM v1 (Segment Anything Model)

- **作者**: Kirillov et al.
- **发表**: ICCV 2023
- **链接**: https://github.com/facebookresearch/segment-anything
- **概述**: 提出通用分割基础模型。ImageEncoderViT 通过标准 transformer blocks 提取图像特征，2 层卷积 neck 输出。
- **融合特点**: **无多层融合**。所有 transformer block 顺序执行，仅最终输出经过 neck（1×1 Conv → LayerNorm → 3×3 Conv → LayerNorm），无 skip connections。
```python
for blk in self.blocks:
    x = blk(x)
x = self.neck(x)  # 简单 2 层卷积，无融合
```
- **对 DINOSAM 的启发**: SAM 本身不使用多层融合，但 DINOSAM 在使用 DINOv3 编码器时引入了 DPT 多层融合——这是 DINOSAM 的架构创新点。可以对比：DINOv3+DPT fusion vs. DINOv3+last-layer-only vs. SAM encoder（无融合）三种配置的分割+深度性能。SAM 的简单 neck 设计暗示其预训练目标（mask prediction）不需要多层特征，但深度估计可能需要。

---

## 9. MoE 与层选择融合

### 9.1 层内 MoE（Within-layer）

#### V-MoE: Scaling Vision with Sparse Mixture of Experts

- **作者**: Riquelme et al. (Google)
- **发表**: NeurIPS 2021
- **概述**: 将 MoE 引入 Vision Transformer。在每个 transformer block 中，将 FFN 替换为 MoE FFN（多个专家 + top-k 路由），显著扩大模型容量而几乎不增加计算量。
- **核心机制**:
```
每层 transformer block:
  x = x + Attention(LayerNorm(x))
  x = x + MoE_FFN(LayerNorm(x))    # MoE 替代标准 FFN

MoE_FFN:
  logits = Router(x)                 # [B*N, num_experts]
  top_k_logits, top_k_idx = logits.topk(k)
  weights = softmax(top_k_logits)
  output = Σ weights[i] * Expert_i(x)  # 仅 top-k 专家激活
```
- **对 DINOSAM 的启发**: 层内 MoE 可在不改变融合架构的前提下扩大 DINOSAM decoder 的容量。特别适用于 DepthMaskDecoder 的 MLP 层——用 MoE MLP 替代标准 MLP 可让不同深度范围由不同专家处理。实现简单，可直接替换 decoder 中的 FFN。

#### Swin-MoE

- **作者**: Liu et al.
- **发表**: CVPR 2022（含 MoE 扩展，见 Swin Transformer V2）
- **概述**: 将 MoE FFN 集成到 Swin Transformer 中，结合 hierarchical 多尺度特征与 MoE 的容量扩展。
- **对 DINOSAM 的启发**: 如果将 DINOSAM 骨干升级为 Swin，可同时获得多尺度特征和 MoE 容量扩展的双重优势。但 Swin + MoE 的训练稳定性挑战较大，不建议作为首选改进。

### 9.2 跨层 MoE 路由融合（Cross-layer）

> **注意：** 此类方案在已发表的知名论文中较少，以下更多是设计模式分析而非特定论文引用。

#### 设计思路

将 ViT 的不同 transformer 层视为"专家"，用 MoE 式路由器从 L 层中选 top-k 层进行加权融合。

```python
class MoELayerRouter(nn.Module):
    def __init__(self, num_layers, embed_dim, top_k=4):
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, num_layers),
        )
        self.layer_projs = nn.ModuleList([
            nn.Conv2d(embed_dim, embed_dim, 1) for _ in range(num_layers)
        ])

    def forward(self, layer_features):
        logits = self.gate(layer_features[-1])      # 用最后一层做路由
        top_k_logits, top_k_idx = logits.topk(k)    # 选 top-k 层
        weights = softmax(top_k_logits)              # 稀疏权重
        fused = Σ weights[k] * proj[layer_idx[k]](layer_features[idx[k]])
        return fused
```

**关键设计决策：**

| 决策点 | 选项 | 说明 |
|--------|------|------|
| 路由输入 | 最后一层 / 全局池化 / CLS token | 决定路由依赖什么信息 |
| 粒度 | per-image / per-token / per-patch | per-token 更灵活但计算量大 |
| 投影 | 无 / 1×1 conv / linear | 是否需要将各层投影到同一空间 |
| 融合方式 | 加权求和 / 拼接后投影 | 前者保持维度，后者更灵活 |
| 辅助损失 | load balancing / entropy | 防止路由坍塌到少数层 |

- **对 DINOSAM 的启发**: DINOSAM 的 DPTHead 当前对所有提取层等权 concat——没有区分哪些层对分割更重要、哪些对深度估计更重要。MoE 路由可以让 mask branch 和 depth branch 使用不同的层组合：mask branch 可能偏好深层（语义强，边界清晰），depth branch 可能偏好中层（兼顾语义和几何细节）。实现建议：为两个 branch 各加一个轻量 router，共享 encoder 的多层特征但选择不同的 top-k 组合。需要 load balancing loss 防止路由坍塌。

### 9.3 Mixture of Depths (MoD)

- **作者**: Raposo et al. (Google)
- **发表**: 2024
- **概述**: 在每层 transformer block 前加一个 router，决定每个 token 是否跳过该层（走残差旁路）。实现 per-token 的动态计算分配，在保持性能的同时减少约 50% FLOPs。
- **核心机制**:
```
每层:
  gate_logits = Router(x)          # [B, N, 1]
  top_k_mask = gate_logits.topk(capacity)

  selected_tokens = x[top_k_mask]  # 被选中的 token 走正常 block
  bypass_tokens = x[~top_k_mask]   # 未选中的 token 跳过

  y_selected = Block(selected_tokens)
  output[top_k_mask] = y_selected
  output[~top_k_mask] = bypass_tokens  # 直接传递
```
- **对 DINOSAM 的启发**: MoD 的 per-token 层选择思路可改造为对每个 patch token 选择哪些层的特征参与融合。对于 DINOSAM：前景物体区域的 token 可能需要更多层特征（复杂分割+深度），背景区域可能只需少量层。实现复杂度较高，建议作为进阶改进而非首选方案。轻量变体：不做 per-token 选择，而是 per-region 选择（将 token 按语义分组，每组选不同层）。

---

## 10. 稠密层加权融合

### Learnable Layer Weights

- **作者**: 常见模式（sentence-BERT, CLIP fine-tuning, 多个 DINOv2 下游任务）
- **概述**: 非特定论文，而是一种广泛使用的融合模式。为每个 transformer 层分配可学习权重，通过 softmax 归一化后加权求和。
- **核心机制**:
```python
class LayerWeightedSum(nn.Module):
    def __init__(self, num_layers):
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self, layer_features):
        w = F.softmax(self.weights, dim=0)
        return sum(w[i] * layer_features[i] for i in range(len(layer_features)))
```
- **对 DINOSAM 的启发**: 最简单的参数化层融合方案，仅 `num_layers` 个标量参数。可作为消融实验 baseline：对比 learnable weights vs. DPT concat vs. last-layer-only。可扩展为 per-task learnable weights：mask branch 和 depth branch 各有一组权重。缺点：稠密（所有层都参与），无法实现稀疏选择。

### Attention over Layers（层注意力）

- **概述**: 用 attention 机制为每个输入动态计算层权重。相比 learnable weights 的静态权重，attention 可根据输入内容动态调整。
- **核心机制**:
```python
layer_query = global_pool(x)                          # [B, C]
layer_keys = [global_pool(f) for f in layer_features]  # [L, B, C]
layer_attn = query @ keys.T                           # [B, L]
weights = softmax(layer_attn)                         # [B, L], 稠密但动态
fused = Σ weights[:, i] * layer_features[i]
```
- **对 DINOSAM 的启发**: 比 learnable weights 更灵活——同一模型对不同图像可使用不同的层组合。对深度估计可能有特殊价值：远景图像可能依赖不同层特征 vs. 近景图像。本质是 dense routing（所有层权重非零），可在此基础上加 top-k 截断实现稀疏化：`weights[weights < threshold] = 0`。

---

## 11. 方案对比总结

### 11.1 融合方案总览

| 方案 | 融合层级 | 稀疏性 | 参数量 | 开源代表 | 复杂度 |
|------|---------|--------|--------|---------|--------|
| Pipeline 组合 | 模型间 | N/A | 0（无融合参数） | Grounded-SAM | 低 |
| FPN | 多尺度 | 稠密 | 中 | Detectron2 | 中 |
| DPT 渐进融合 | 多层 | 稠密 | 大 | DINOv2 | 高 |
| DPTSimpleHead | 多层 | 稠密 | 中 | DINOSAM 当前 | 中 |
| Resize-Concat | 多层 | 稠密 | 小 | DINOv2 BNHead | 低 |
| 分层骨干 (Swin) | 阶段间 | N/A | N/A | Swin | N/A |
| 层内 MoE FFN | 层内 | top-k 专家 | 大 | V-MoE | 中 |
| 跨层 MoE 路由 | 多层 | **top-k 层** | 小 | 无知名开源 | 中 |
| MoD | 多层 | per-token 二值 | 小 | 部分开源 | 高 |
| Learnable Weights | 多层 | 稠密 | 极小 | 常见模式 | 低 |
| Attention over Layers | 多层 | 稠密（可扩展为稀疏） | 小 | 常见模式 | 低-中 |
| 无融合 (MAE/SAM) | 无 | N/A | 0 | MAE, SAM | 低 |

### 11.2 MoE 路由机制论文与 DINOSAM 借鉴点

| 论文 | 可借鉴点 | 在 DINOSAM 中的应用 |
|------|---------|-------------------|
| Shazeer 2017 | Importance-balancing loss, noisy top-k, 零初始化 | 防止层选择坍缩的核心机制 |
| Switch Transformer 2022 | `L_aux = α·N·Σ(f_i·P_i)`, α=1e-2, capacity factor | 辅助 loss 的具体公式和超参 |
| Loss-Free Balancing 2024 | 动态偏置替代辅助 loss | 更简洁的均衡方案备选 |
| Sander 2023 | 可微+稀疏的 top-k 算子 | 理论上最优雅的端到端方案 |
| Korol 2025 | 跳层敏感性分析 | 理解层重要性的 baseline 方法 |
| X-MoE | 表征坍缩问题与解法 | 评分网络设计时需注意的陷阱 |
| ST-MoE | MoE 在视觉任务中的验证 | 证明 MoE 机制可用于 ViT |
| Expert Choice 2022 | Expert-choice vs token-choice | 设计思路启发：层选 token 还是 token 选层 |

---

## 12. 对 DINOSAM 的启发与改进路线

### 当前架构定位

DINOSAM 使用 `DPTSimpleHead`（DPT 简化版）做 DINOv3 多层特征融合。这在上述方案中属于**稠密多层融合**的中等复杂度方案。

### 推荐的改进路线（按优先级排序）

#### 优先级 1：消融实验（低成本，高信息量）
- [ ] 对比 DPTSimpleHead vs. 仅用最后一层 vs. Learnable Layer Weights
- [ ] 对比 mask branch 和 depth branch 是否应该使用**不同的层组合**
- [ ] 量化多层融合对分割 vs. 深度估计各自的增益

#### 优先级 2：Per-Task Layer Weights（中等成本，直接收益）
- [ ] 为 mask branch 和 depth branch 各加一组 learnable layer weights
- [ ] 训练后可视化两组权重，分析两个任务分别偏好哪些层
- [ ] 如权重分布差异大，验证 top-k 稀疏选择是否可替代稠密融合

#### 优先级 3：引入 FPN top-down 路径（中等成本，架构改进）
- [ ] 在 DPTHead 中加入 top-down + lateral 连接
- [ ] 深层特征（语义强）逐步引导浅层特征（空间精细）
- [ ] 与 DPT 渐进融合对比，选择更优方案

#### 优先级 4：MoE 层路由融合（高成本，创新点）
- [ ] 实现跨层 MoE Router，per-image 选择 top-k 层
- [ ] 为两个 branch 分别路由，允许选择不同的层组合
- [ ] 加入 load balancing loss 防止路由坍塌
- [ ] 可视化路由结果，分析不同图像类型是否选择了不同层

#### 优先级 5：长期探索
- [ ] 将骨干替换为 hierarchical ViT（如 Swin），获得天然多尺度特征
- [ ] 在 decoder 中引入层内 MoE FFN，扩大容量
- [ ] 探索 MoD 式 per-token 层选择，让前景/背景 token 使用不同层特征

### 研究空白

目前没有已验证的工作直接研究"逐图像学习的 ViT 层重要性评分用于密集预测（分割/深度估计）"。DINOSAM 的层融合策略如果成功，将是该方向的首个系统性探索。
