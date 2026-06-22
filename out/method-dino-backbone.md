# Methods

## 术语表

| 规范术语 | 首次定义 | 来源变体 | 说明 |
|----------|----------|----------|------|
| DINOSAM | Depth-aware SAM (DINOSAM) | Depth-aware SAM | 首次展开后使用缩写 |
| SAM | Segment Anything Model (SAM) | — | 指 Meta 的 SAM ViT-B |
| DINOv3 | DINOv3 ViT-B/16 | DINOv3 backbone | 指预训练视觉 Transformer 骨干 |
| MoNA | Multi-scale Convolutional Adapter (MoNA) | mona, adapter A | 多尺度深度卷积适配器 |
| FCAdapter | Fully-Connected Adapter (FCAdapter) | fc, adapter B | 纯全连接瓶颈适配器 |
| DualAttnAdapter | Dual Attention Adapter (DualAttnAdapter) | dual_attn, adapter C | 双重注意力适配器 |
| PAM | Position Attention Module (PAM) | — | DualAttnAdapter 的位置注意力分支 |
| CAM | Channel Attention Module (CAM) | — | DualAttnAdapter 的通道注意力分支 |

---

## 概述

DINOSAM 在 Segment Anything Model (SAM) 的分割框架基础上引入深度预测能力，形成掩码-深度双分支联合架构。本文方法的核心在于：将 SAM 原有的 ViT-B 图像编码器替换为冻结的 DINOv3 ViT-B/16 预训练骨干，通过轻量可训练适配器将 DINOv3 的通用视觉表征对齐到 SAM 的 256 维特征空间，从而使掩码分支和深度分支可以共享同一套编码器输出，无须修改下游解码器结构。为系统性评估适配器设计对迁移效果的影响，我们设计了三种结构差异显著的适配器——多尺度卷积适配器 (MoNA)、纯全连接瓶颈适配器 (FCAdapter) 和双重注意力适配器 (DualAttnAdapter)——作为消融实验的核心变量。

本节首先描述整体架构 (3.1)，随后详述 DINOv3 编码器与投影层 (3.2)，接着系统阐述三种适配器的动机、机制与结构差异 (3.3)，最后说明两阶段训练策略与评估方法 (3.4, 3.5)。

## 3.1 整体架构

DINOSAM 由一个共享的图像编码器和两个独立的解码分支组成。掩码分支包含内部提示编码器 (Inner PromptEncoder) 和掩码解码器 (MaskDecoder)，负责根据点提示或框提示生成分割掩码；深度分支包含外部提示编码器 (Outer PromptEncoder) 和深度解码器 (DepthMaskDecoder)，负责预测逐像素深度值。两个分支共享图像编码器的输出特征，在解码阶段独立运行。

当编码器类型指定为 `dinov3` 时，构造函数实例化 DINOv3MonaEncoder 替代 SAM 原生 ViT-B 编码器。此替换对下游分支完全透明——两个解码分支的结构、参数和前向逻辑与使用 SAM 编码器时完全一致，差异仅存在于图像编码阶段。这一设计确保了消融实验的公平性：编码器变更所带来的性能差异可完全归因于骨干表征能力和适配器设计，而非下游结构差异。

## 3.2 DINOv3 编码器与特征对齐

### 3.2.1 动机

SAM 的原生 ViT-B 编码器在大规模分割数据上预训练，具备强分割先验，但其表征受限于分割任务本身的监督信号。DINOv3 ViT-B/16 经自监督预训练获得的视觉表征已被证明在多种下游任务中展现出优越的迁移性能。然而，DINOv3 的输出维度 (768) 与 SAM 的内部特征维度 (256) 不一致，且两者的图像预处理流程存在显著差异。因此，直接替换编码器并不可行，需要通过适配模块实现特征空间对齐。

### 3.2.2 骨干网络配置

DINOv3MonaEncoder 使用 DINOv3 ViT-B/16 作为冻结骨干，具体配置如下：输入图像尺寸 1024 x 1024，patch 尺寸 16 x 16，嵌入维度 768，网络深度 12 层，注意力头数 12，前馈网络比率 4。位置编码采用旋转位置编码 (RoPE)，基数设置为 100，坐标缩放因子为 2，采用独立归一化。额外配置包括 4 个存储 token、带掩码的键偏置、层缩放初始值 $1 \times 10^{-5}$，以及 BF16 层归一化。骨干网络从预训练检查点加载后，所有参数完全冻结，训练过程中不更新。

### 3.2.3 前向传播流程

与 SAM 编码器的预处理不同，DINOv3 编码器的前向传播采用以下流程。首先，输入图像直接缩放至 $1024 \times 1024$，不保持原始宽高比（SAM 编码器保持宽高比并将最长边缩放至 1024 像素后填充）。随后，按 ImageNet 均值和标准差（乘以 255 的像素值范围）进行通道归一化，这与 SAM 使用的专有归一化参数不同。

归一化后的图像输入冻结的 DINOv3 骨干网络，提取归一化后的 patch token 特征，其形状为 $(B, N, 768)$，其中 $B$ 为批量大小，$N$ 为 patch 数量。该特征序列被重塑为 $(B, H \times W, C)$ 的空间排列，其中 $H = W = \sqrt{N}$。重塑后的特征依次通过两个串联的适配器模块（见 3.3 节），再经过一个线性投影层 $\mathbb{R}^{768} \rightarrow \mathbb{R}^{256}$，将特征维度对齐到 SAM 的 256 维空间。最终输出被重塑为 $(B, 256, H, W)$ 的空间特征图，连同固定的输入尺寸 $(1024, 1024)$ 一并返回给下游解码分支。

DINOv3 编码器与 SAM 编码器的预处理差异总结如下：

| 步骤 | SAM 编码器 | DINOv3 编码器 |
|------|-----------|--------------|
| 图像缩放 | 保持宽高比，最长边 1024 px | 直接缩放至 $1024 \times 1024$ |
| 归一化参数 | SAM 专有 pixel\_mean / pixel\_std | ImageNet 均值/标准差 $\times$ 255 |
| 空间填充 | 需要填充至 $1024 \times 1024$ | 不需要 |
| 返回的 input\_size | 实际缩放后尺寸 $(h', w')$ | 固定值 $(1024, 1024)$ |

## 3.3 适配器设计

### 3.3.1 设计动机与消融逻辑

将冻结骨干的通用表征适配到特定下游任务，适配器的结构选择至关重要。现有参数高效微调 (PEFT) 方法在适配器内部引入了不同的归纳偏置：通道级瓶颈变换关注特征维度的压缩与重建，空间卷积引入局部感受野信息，注意力机制则建模全局位置或通道间的依赖关系。这些归纳偏置对稠密预测任务（分割和深度估计）的迁移效果可能具有不同影响。

为系统性评估上述因素，我们设计了三种结构差异显著的适配器，每种适配器保持相同的瓶颈压缩比（$768 / 8 = 96$）和残差连接结构，仅在核心变换机制上存在差异：MoNA 引入多尺度空间卷积，FCAdapter 仅保留通道变换作为对照基线，DualAttnAdapter 引入全局位置与通道注意力。三种适配器通过统一的注册机制实例化，通过命令行参数 `--adapter_type` 选择，共享相同的接口签名。

在 DINOv3MonaEncoder 中，两个同类型适配器以串联方式堆叠（adapter1 $\rightarrow$ adapter2），提供两级特征适配。

### 3.3.2 适配器 A：MoNA（多尺度卷积适配器）

MoNA 是默认适配器，其核心假设是：稠密预测任务受益于多尺度局部空间信息的注入。

MoNA 由层归一化、通道瓶颈变换和空间多尺度卷积三部分组成。给定输入特征 $\mathbf{x} \in \mathbb{R}^{B \times N \times 768}$，首先通过可学习缩放因子进行仿射归一化：

$$\hat{\mathbf{x}} = \text{LayerNorm}(\mathbf{x}) \odot \boldsymbol{\gamma} + \mathbf{x} \odot \boldsymbol{\gamma}_x$$

其中 $\boldsymbol{\gamma}$ 初始化为 $1 \times 10^{-6}$ 的全 1 向量，$\boldsymbol{\gamma}_x$ 初始化为全 1 向量。该初始化策略确保训练初始阶段适配器接近恒等映射，不破坏冻结骨干的预训练表征。

随后，$\hat{\mathbf{x}}$ 通过线性投影压缩至瓶颈维度：$\mathbf{z} = W_{\text{down}} \hat{\mathbf{x}} \in \mathbb{R}^{B \times N \times 96}$，其中 $W_{\text{down}} \in \mathbb{R}^{768 \times 96}$。

压缩后的特征被重塑为二维空间排列 $\mathbf{Z} \in \mathbb{R}^{B \times 96 \times H \times W}$，送入多尺度卷积模块 (MonaOp)。MonaOp 使用三路并行的深度可分离卷积，卷积核尺寸分别为 $3 \times 3$、$5 \times 5$ 和 $7 \times 7$，各路均采用 groups = $C$ 的深度卷积形式以保持参数效率。三路卷积的输出取均值后与恒等映射相加：

$$\mathbf{Z}' = \frac{1}{3}\left(\text{DWConv}_{3}(\mathbf{Z}) + \text{DWConv}_{5}(\mathbf{Z}) + \text{DWConv}_{7}(\mathbf{Z})\right) + \mathbf{Z}$$

随后通过 $1 \times 1$ 逐点卷积进行通道混合，并以第二个恒等连接输出：

$$\hat{\mathbf{Z}} = \text{Conv}_{1 \times 1}(\mathbf{Z}') + \mathbf{Z}'$$

最终，$\hat{\mathbf{Z}}$ 重塑回序列形式，经 GELU 激活、Dropout（概率 0.1）和线性投影恢复至原始维度：$\Delta \mathbf{x} = W_{\text{up}} \, \text{Dropout}(\text{GELU}(\hat{\mathbf{Z}})) \in \mathbb{R}^{B \times N \times 768}$，其中 $W_{\text{up}} \in \mathbb{R}^{96 \times 768}$。适配器的最终输出为残差形式：$\mathbf{x}' = \mathbf{x} + \Delta \mathbf{x}$。

### 3.3.3 适配器 B：FCAdapter（全连接瓶颈适配器）

FCAdapter 作为消融基线，其设计目的是隔离空间卷积的贡献——它保留了 MoNA 的全部通道变换路径，仅移除了 MonaOp 空间卷积模块。

FCAdapter 与 MoNA 共享相同的结构：可学习仿射归一化（$\boldsymbol{\gamma}$、$\boldsymbol{\gamma}_x$ 初始化策略相同）、线性下投影 $W_{\text{down}} \in \mathbb{R}^{768 \times 96}$、GELU 激活、Dropout（概率 0.1）和线性上投影 $W_{\text{up}} \in \mathbb{R}^{96 \times 768}$。其前向传播简化为：

$$\mathbf{x}' = \mathbf{x} + W_{\text{up}} \, \text{Dropout}\big(\text{GELU}(W_{\text{down}} \hat{\mathbf{x}})\big)$$

FCAdapter 接受空间尺寸参数 `hw_shapes` 但不使用，以保持与 MoNA 的接口兼容性。MoNA 与 FCAdapter 的对比直接揭示了多尺度空间卷积对稠密预测迁移效果的边际贡献。

### 3.3.4 适配器 C：DualAttnAdapter（双重注意力适配器）

DualAttnAdapter 引入全局注意力机制作为第三种归纳偏置，其设计基于 DANet 的双重注意力思想：同时建模空间位置间和通道间的长程依赖，以评估全局上下文建模能力对适配效果的影响。

DualAttnAdapter 包含两个并行分支：位置注意力模块 (PAM) 和通道注意力模块 (CAM)。

**位置注意力模块 (PAM).** 输入特征 $\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}$ 分别通过三个 $1 \times 1$ 卷积生成查询、键和值：

$$\mathbf{Q} = W_Q \mathbf{X} \in \mathbb{R}^{B \times C' \times HW}, \quad \mathbf{K} = W_K \mathbf{X} \in \mathbb{R}^{B \times C' \times HW}, \quad \mathbf{V} = W_V \mathbf{X} \in \mathbb{R}^{B \times C \times HW}$$

其中 $C' = C / 8 = 96$ 为降维后的通道数。位置注意力图和输出为：

$$\mathbf{A}_{\text{pos}} = \text{softmax}(\mathbf{Q}^\top \mathbf{K}) \in \mathbb{R}^{B \times HW \times HW}$$
$$\text{PAM}(\mathbf{X}) = \alpha \cdot (\mathbf{V} \, \mathbf{A}_{\text{pos}}^\top)$$

其中 $\alpha$ 为可学习标量参数，初始化为 0，确保训练初期 PAM 的输出为零，不干扰原始特征。

**通道注意力模块 (CAM).** 输入特征 $\mathbf{X} \in \mathbb{R}^{B \times C \times HW}$ 直接计算通道间的亲和矩阵：

$$\mathbf{E} = \mathbf{X} \, \mathbf{X}^\top \in \mathbb{R}^{B \times C \times C}$$

按照 DANet 的公式，通道注意力图通过最大值归一化计算：

$$\mathbf{A}_{\text{ch}} = \text{softmax}\big(\max(\mathbf{E}) \cdot \mathbf{1} - \mathbf{E}\big) \in \mathbb{R}^{B \times C \times C}$$

其中 $\max(\mathbf{E})$ 沿最后一个维度取最大值。通道注意力输出为：

$$\text{CAM}(\mathbf{X}) = \beta \cdot (\mathbf{A}_{\text{ch}} \, \mathbf{X})$$

其中 $\beta$ 为可学习标量参数，同样初始化为 0。

**分支融合.** 输入序列特征 $\mathbf{x} \in \mathbb{R}^{B \times N \times C}$ 首先重塑为空间特征图 $\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}$，PAM 和 CAM 并行计算后取均值，再重塑回序列形式并加上全局残差：

$$\mathbf{x}' = \mathbf{x} + \text{reshape}\left(\frac{\text{PAM}(\mathbf{X}) + \text{CAM}(\mathbf{X})}{2}\right)$$

值得注意的是，DualAttnAdapter 没有瓶颈压缩结构（PAM 内部的 $1 \times 1$ 卷积降维仅用于计算注意力权重，值投影保持原始维度），参数量和计算复杂度与 MoNA 和 FCAdapter 存在差异。这一设计选择是有意为之：消融实验的目标是评估归纳偏置类型（局部空间 vs. 全局注意力 vs. 纯通道）的影响，而非在严格等参数条件下比较。

### 3.3.5 适配器结构对比总结

| 属性 | MoNA | FCAdapter | DualAttnAdapter |
|------|------|-----------|-----------------|
| 核心机制 | 多尺度深度卷积 + 瓶颈 MLP | 纯瓶颈 MLP | 位置注意力 + 通道注意力 |
| 空间建模 | 局部多尺度 (3/5/7) | 无 | 全局 (自注意力) |
| 通道建模 | 线性瓶颈 | 线性瓶颈 | 通道自注意力 |
| 瓶颈维度 | 96 (768/8) | 96 (768/8) | 无瓶颈 (PAM 降维仅限 QK) |
| 残差连接 | 全局 | 全局 | 全局 |
| 零初始化策略 | $\boldsymbol{\gamma} = 10^{-6}$ | $\boldsymbol{\gamma} = 10^{-6}$ | $\alpha = 0, \beta = 0$ |
| 对照角色 | 默认方案 | 通道基线 | 注意力方案 |

三种适配器的消融设计遵循以下逻辑：FCAdapter 作为纯通道变换基线，隔离了空间信息的贡献；MoNA 在 FCAdapter 基础上引入局部多尺度空间信息；DualAttnAdapter 则用全局注意力替代局部卷积。三者的对比系统性地回答了一个核心问题：对于从通用视觉骨干到稠密预测任务的特征适配，何种归纳偏置最为有效。

## 3.4 训练策略

### 3.4.1 两阶段训练

DINOSAM 采用两阶段训练策略，两个阶段的目标和参数冻结策略不同。

**第一阶段（掩码训练）.** 此阶段训练分割能力。DINOv3 骨干网络完全冻结，SAM 原始 ViT-B 编码器（如存在）亦冻结，深度分支（Outer PromptEncoder 和 DepthMaskDecoder）冻结。可训练参数为：两个适配器模块（adapter1 和 adapter2）、线性投影层（$\mathbb{R}^{768} \rightarrow \mathbb{R}^{256}$）、内部提示编码器 (Inner PromptEncoder) 和掩码解码器 (MaskDecoder)。此阶段的核心目标是学习适配器参数，使 DINOv3 的冻结表征能够通过适配器和投影层对齐到 SAM 的特征空间，从而驱动掩码解码器生成准确的分割掩码。

**第二阶段（深度训练）.** 此阶段训练深度预测能力。所有第一阶段的可训练参数被冻结（包括适配器和投影层），仅 Outer PromptEncoder 和 DepthMaskDecoder 可训练。该阶段利用第一阶段已对齐的图像特征，学习从同一特征空间预测逐像素深度值。

两阶段训练策略的设计逻辑是：第一阶段专注于特征空间对齐和分割任务，确保适配器学到的表征同时保留了空间结构和语义信息；第二阶段在冻结的对齐表征上训练深度分支，验证该表征是否具备足够的深度信息。两阶段共享相同的优化器配置和学习率调度策略。

### 3.4.2 与 SAM 编码器方案的训练差异

DINOv3 编码器方案与 SAM 编码器方案共享相同的两阶段训练框架、损失函数、迭代训练逻辑和优化器配置，差异仅存在于以下方面。第一，第一阶段的可训练参数集不同：SAM 方案微调 SAM 编码器本身（或其部分层），DINOv3 方案冻结骨干、仅训练适配器和投影层。第二，图像预处理流程不同（见 3.2.3 节表格）。第三，DINOv3 方案引入了适配器类型作为额外的消融变量。上述差异均被严格控制，以确保两种编码器方案之间的性能差异可归因于编码器表征能力和适配策略的差异。

## 3.5 损失函数

掩码分支和深度分支分别使用独立的损失函数。掩码分支采用分割任务的标准监督损失；深度分支采用深度估计任务的回归损失。两个阶段的损失函数与 SAM 编码器方案完全一致，不因编码器替换而改变。[需补充：具体损失函数的数学形式——Dice loss、BCE loss 等掩码损失，以及深度回归损失的具体定义。]

## 3.6 评估方法

评估流程与 SAM 编码器方案保持一致，通过命令行参数 `--encoder dinov3 --adapter_type {mona, fc, dual_attn}` 切换编码器和适配器配置。

**分割指标.** 采用 Dice 系数、交并比 (IoU)、IoU 预测准确度和 IoU 预测平均绝对误差 (MAE) 四个指标评估掩码质量。

**深度指标.** 采用平均绝对误差 (MAE)、均方误差 (MSE)、均方根误差 (RMSE) 和决定系数 ($R^2$) 四个指标评估深度预测精度。此外，按深度值范围对样本分类，报告各深度类别的 MAE，以评估模型在不同深度区间的预测稳定性。

**推理协议.** 支持迭代推理，即将前一轮的预测掩码作为额外提示输入下一轮推理，逐步精化分割结果。评估同时报告第一阶段（仅掩码）和第二阶段（掩码 + 深度）的性能，以量化深度分支训练对分割性能的影响。

**数据集.** 所有实验使用与 SAM 编码器方案相同的 DepthDataset。数据集按深度值分文件夹组织，每个样本由 RGB 图像（.jpg）和对应的深度标注图（.png）配对组成。

---

**假设或缺失输入：**

1. 具体损失函数的数学形式（Dice loss、BCE loss、深度回归损失的具体定义）需要从代码中补充。
2. 适配器参数量的精确数值对比（三种适配器的可训练参数计数）可作为实现细节补充。
3. 数据集的具体规模、类别数和划分方式需要补充。
4. 优化器类型、学习率、调度策略的具体数值需要补充。
5. DINOv3 预训练检查点的来源和预训练数据集信息需要补充。

**论据-证据映射：**

| 论断 | 证据 | 状态 |
|------|------|------|
| DINOv3 骨干冻结，仅适配器可训练 | 代码 depth\_sam.py:114-118 冻结策略 | 已支持 |
| 三种适配器结构差异显著 | MonaOp 多尺度卷积 vs. 纯 MLP vs. PAM+CAM | 已支持 |
| 消融设计隔离了归纳偏置类型的影响 | FCAdapter 作为通道基线的对照设计 | 已支持 |
| 预处理流程差异不影响公平比较 | 两种编码器方案共享下游结构 | 已支持 |
| 适配器对深度和分割均有效 | 需要实验结果 | 需补充证据 |

**结构选择说明：**

1. 按"概述-编码器-适配器-训练-评估"的逻辑链组织，从整体到局部，避免在适配器细节中迷失全局视角。
2. 将三种适配器集中在 3.3 节描述并以对比表收尾，使消融设计的逻辑一目了然。
3. 预处理差异以表格形式呈现而非散布于正文中，提高信息密度和可检索性。
4. 训练策略独立成节，强调 DINOv3 方案与 SAM 方案的受控差异，为后续实验结果的归因分析提供基础。
