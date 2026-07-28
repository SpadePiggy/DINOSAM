# Method

## 1 问题形式化：三维细胞定位

本文的任务是显微镜图像中细胞的三维空间定位。给定一张输入图像 $I \in \mathbb{R}^{H \times W \times 3}$ 和一个用户指定的点击坐标 $(x, y)$，模型需要同时输出该位置的二值分割掩码 $\hat{M} \in \{0, 1\}^{H \times W}$ 与离焦深度 $\hat{z} \in [-100, 100]$（单位 μm，负值表示焦平面上方），二者共同构成细胞的三维位置：

$$(\hat{M}, \hat{z}) = \mathcal{F}(I \mid x, y)$$

数据集按深度类别以文件夹组织（`f100`, `f90`, …, `0`, …, `100`），每个文件夹内存放该深度下的细胞图像及其 GT 分割掩码。训练时深度标签除以 100 归一化至 $[-1, 1]$，评估时恢复原始标度。

## 2 核心洞察：以 Mask 作为 Prompt 引导深度推理

上述形式化的一个根本困难在于：仅凭稀疏的点击坐标 $(x, y)$ 直接回归全局深度值，模型缺乏空间约束——它无法判断点击所指向的区域边界在何处，因此不知道应对图像中哪一块区域进行深度估计。然而这一问题恰好可以通过分割掩码来解决：掩码精确圈定了被选中的细胞区域，告诉模型"这是需要估计深度的目标"。一旦有了这一空间范围作为条件，深度回归从全局歧义退化为区域条件回归——图像中不同区域对应不同的离焦程度，模型仅需对指定区域做推断。

这引出了一个自然的架构选择：**将分割掩码作为深度分支的稠密空间提示（dense prompt）**，使深度预测以 mask 为条件。

## 3 实现途径：TwoWayTransformer 实现 Mask 对深度的监督

要将上述直觉转化为可训练的模型，需要一个能将 mask 信息编码为条件信号、并与图像特征进行深度交互的机制。SAM 中的 TwoWayTransformer \cite{sam} 恰好提供了这一能力：它通过 prompt tokens 与 image features 之间的双向交叉注意力，实现"prompt 指定关注区域 → 从图像特征中提取区域信息 → 图像特征反向受 prompt 约束"的交互循环。

基于此，本文在 SAM 架构之上构建双分支模型 **DepthSam**。模型包含两条前向通路，共享同一个图像特征编码器。

**分割分支**沿用 SAM 原生设计：图像编码器生成特征图 $\mathbf{E} \in \mathbb{R}^{B \times 256 \times 64 \times 64}$，prompt encoder 将用户点击编码为稀疏嵌入 $\mathbf{S}_{\text{mask}} \in \mathbb{R}^{B \times K \times 256}$，mask decoder 通过 TwoWayTransformer 的交叉注意力从 $\mathbf{E}$ 中提取被 prompt 指定的区域并输出分割掩码 $\hat{M}$ 和对应的低分辨率掩码 $\mathbf{M}_{\text{low}} \in \mathbb{R}^{B \times 1 \times 256 \times 256}$。

**深度分支**是本文的核心扩展。它拥有独立的 outer prompt encoder 和 DepthMaskDecoder。分割分支产出的 $\mathbf{M}_{\text{low}}$ 经梯度切断（detach）后作为 mask prompt 注入 outer prompt encoder，与可选的 point prompt 一起编码为稀疏嵌入 $\mathbf{S}_{\text{depth}}$ 和密集嵌入 $\mathbf{D}_{\text{depth}}$：

$$\mathbf{S}_{\text{depth}}, \mathbf{D}_{\text{depth}} = \text{PromptEnc}_{\text{outer}}(\mathbf{M}_{\text{low}}, \mathbf{p}_{\text{depth}})$$

其中 $\mathbf{p}_{\text{depth}} = \{(x_k, y_k, \ell_k)\}_{k=1}^{K}$ 为点击坐标及标签（$\ell_k \in \{0, 1\}$）。

DepthMaskDecoder 构建两组可学习的输出 token——depth token $\mathbf{t}_d \in \mathbb{R}^{256}$ 和 mask token $\mathbf{t}_m \in \mathbb{R}^{256}$——与 $\mathbf{S}_{\text{depth}}$ 拼接为 $\mathbf{T} \in \mathbb{R}^{B \times (K+2) \times 256}$，送入 TwoWayTransformer 与图像特征 $\mathbf{E} + \mathbf{D}_{\text{depth}}$ 进行双向交叉注意力：

$$\mathbf{T}', \mathbf{E}' = \text{TwoWayTransformer}(\mathbf{E} + \mathbf{D}_{\text{depth}},\; \mathbf{E}_{\text{pe}},\; \mathbf{T})$$

TwoWayTransformer 的核心是两轮交叉注意力：(1) tokens 通过交叉注意力从图像特征中查询空间信息，(2) 图像特征通过反向交叉注意力接收 tokens 的语义约束，两层之间穿插 tokens 的自注意力。Transformer 输出的 depth token 经 3 层 MLP 回归标量深度值 $\hat{z}$：

$$\hat{z} = \text{MLP}_{\text{depth}}(\mathbf{T}'_{[:,0,:]}) \in \mathbb{R}$$

mask token 则经 output hypernetworks 生成辅助 decoder mask，用于迭代 prompt 精化（见 §7.4）。

深度分支的权重初始化利用 SAM 预训练的先验：outer prompt encoder 从 SAM inner prompt encoder 完整复制；transformer、output upscaling、output hypernetworks 从 SAM mask decoder 对应组件迁移；depth token 继承 iou token 的初始嵌入；仅 depth prediction head 保持随机初始化（在 SAM 中无对应结构）。

Simple 模式作为消融基线：深度分支退化为 $\hat{z} = \text{MLP}(\text{AvgPool}(\mathbf{E}))$，完全跳过 TwoWayTransformer 和 mask prompt，用于验证 mask 引导的交叉注意力是否必要。

## 4 特征提取的不足：用 DINOv3 替换 SAM Image Encoder

上述架构初步建立后，训练中暴露出一个问题：SAM 的原生图像编码器（ViT-B）虽然分割能力强，但对显微镜图像中纹理变化、离焦程度等深度相关视觉线索的表征不足——SAM encoder 擅长判断"物体边界在哪里"，但不擅长感知"这个区域看起来离焦了多少"。而细胞这种结构相对简单的目标，分割难度远低于 SAM 的训练分布，其编码器存在能力冗余。

基于这一观察，本文用 DINOv3 ViT-B/16 \cite{dinov3} 替换 SAM 原生图像编码器。DINOv3 通过大规模自监督预训练获得了丰富的中层视觉表示（纹理、材质、结构），更适合捕获深度估计所需的细粒度外观变化。替换后的编码器被完全冻结，仅通过可训练的轻量级 adapter 进行特征适配。

由于 DINOv3 的 patch token 维度为 768，而 SAM mask decoder 期望 256 维的 prompt embedding，直接替换存在维度不匹配。为此引入双层串联 adapter：每个 adapter 为残差瓶颈模块，对输入 token 序列 $\mathbf{X} \in \mathbb{R}^{B \times N \times 768}$ 进行特征变换：

$$\mathbf{X}' = \gamma \cdot \text{LayerNorm}(\mathbf{X}) + \gamma_x \cdot \mathbf{X}$$

$$\mathbf{X}_{\text{down}} = \mathbf{W}_1 \mathbf{X}' \quad \in \mathbb{R}^{B \times N \times 96}$$

$$\mathbf{X}_{\text{spatial}} = \Phi(\mathbf{X}_{\text{down}}) \quad \text{（空间变换，见下文）}$$

$$\mathbf{X}_{\text{out}} = \mathbf{X} + \mathbf{W}_2 \cdot \text{Dropout}(\text{GELU}(\mathbf{X}_{\text{spatial}}))$$

其中 $\gamma, \gamma_x$ 是可学习标量（初始值 $10^{-6}$ 和 1.0），门控机制使网络在训练初期接近 identity 映射、逐步激活 adapter 的变换能力。两个 adapter 串联后经线性投影 $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{768 \times 256}$ 将特征映射至 SAM 兼容空间：

$$\mathbf{Z} = \text{Adapter}_2(\text{Adapter}_1(\mathbf{X})) \cdot \mathbf{W}_{\text{proj}} \in \mathbb{R}^{B \times 4096 \times 256}$$

reshape 为 $\mathbb{R}^{B \times 256 \times 64 \times 64}$，与 SAM image encoder 输出格式完全一致。

三种 adapter 变体的区别在于空间变换 $\Phi(\cdot)$ 的设计：

- **Mona**（本文默认）：在瓶颈维度上执行多尺度深度可分离卷积并取均值，再经 $1 \times 1$ 卷积投影，与 identity 相加构成残差：
  $$\Phi(\mathbf{X}) = \mathbf{X} + \text{Conv}_{1\times1}\!\left(\frac{1}{3}\sum_{k \in \{3,5,7\}} \text{DWConv}_k(\mathbf{X})\right)$$
  其中 $\text{DWConv}_k$ 为 $k \times k$ 深度可分离卷积。多尺度卷积并行捕获不同感受野的局部空间结构。

- **FCAdapter**：无空间操作，$\Phi(\mathbf{X}) = \mathbf{X}$（纯通道 MLP）。参数量最小，但缺乏空间上下文建模。

- **DualAttnAdapter**：$\Phi(\mathbf{X}) = \frac{1}{2}(\text{PAM}(\mathbf{X}) + \text{CAM}(\mathbf{X}))$，即 DANet 的位置注意力与通道注意力并行后取均值，再与 identity 做单次全局残差。

实验表明 Mona 在分割与深度两项指标上综合最优，后续设计以 Mona 为默认 adapter。

## 5 分割退化与 DPT 多层融合

用 DINOv3 替换 SAM encoder 解决了深度特征提取问题，却引入了新困难：DINOv3 的单层（最后一层）特征虽然富含语义，但经过 12 层 ViT 的逐层抽象后空间定位精度下降，导致细胞分割相对 SAM encoder baseline 出现退化——模型"理解了图像中有细胞"，但边界定位不够精确。

受 SegDINO \cite{segdino} 中 DPT 思想的启发，本文从 DINOv3 的多个中间层提取特征并进行融合，以同时保留浅层的空间细节和深层的语义判别力。

设选定的中间层索引集合为 $\mathcal{L} = \{l_1, l_2, \dots, l_n\}$（$n = |\mathcal{L}|$），从 DINOv3 backbone 一次性提取各层 patch token：

$$\{\mathbf{F}_{\ell}\}_{\ell \in \mathcal{L}} = \text{Backbone}(I),\quad \mathbf{F}_{\ell} \in \mathbb{R}^{B \times 4096 \times 768}$$

各层 token 序列 reshape 为空间特征图 $\mathbf{S}_{\ell} \in \mathbb{R}^{B \times 768 \times 64 \times 64}$，经逐层独立投影与精炼：

$$\mathbf{R}_{\ell} = \text{Conv}_{3\times3}\big(\text{Conv}_{1\times1}(\mathbf{S}_{\ell})\big) \in \mathbb{R}^{B \times 256 \times 64 \times 64},\quad \forall \ell \in \mathcal{L}$$

其中 $\text{Conv}_{1\times1}$ 将 768 维降至 256 维，$\text{Conv}_{3\times3}$ 捕获局部空间关系。所有精炼后的特征图在通道维拼接，经 $1 \times 1$ 卷积融合回 768 维：

$$\mathbf{F}_{\text{DPT}} = \text{Conv}_{1\times1}\big([\mathbf{R}_{l_1}; \mathbf{R}_{l_2}; \dots; \mathbf{R}_{l_n}]\big) \in \mathbb{R}^{B \times 768 \times 64 \times 64}$$

随后 $\mathbf{F}_{\text{DPT}}$ 展平为 token 序列，经 adapter 串联和投影输出 $(B, 256, 64, 64)$。由于 DINOv3 ViT 所有中间层处于同一空间尺度（patch size 16, 输入 1024 → $64 \times 64$），DPT 头无需 Resize/Upsample 操作，架构简洁。

## 6 任务偏好差异与双层选择

在对 DPT 融合层的系统性消融中（固定 $n=4$，$n\_sub=8$ 迭代评估，0\_sam 测试集），发现了一个此前未被关注的现象：

| $\mathcal{L}$ | Dice ↑ | Depth MAE ↓ |
|---|---|---|
| $\{2, 5, 8, 11\}$ | 0.9587 | 6.18 |
| $\{0, 3, 6, 9\}$ | 0.9593 | **6.10** |
| $\{3, 6, 9, 11\}$ | **0.9637** | 7.06 |
| $\{0, 1, \dots, 11\}$ | 0.9554 | 7.40 |

消融结果揭示了一个关键规律：**分割与深度回归对 DINOv3 中间层的最优偏好不一致。** 分割受益于深层语义特征（$\{3,6,9,11\}$ 取得最高 Dice，0.9637），深度回归则更依赖浅层纹理和结构信息（$\{0,3,6,9\}$ 取得最低 MAE，6.10）。全部 12 层融合引入过多冗余，两项均垫底——融合质量取决于层选择策略，而非层数。

基于这一发现，本文提出**任务解耦的双分支层选择方案**（设计通过 Review，实现与实验验证进行中）：mask 分支与 depth 分支各自使用独立的 DPT 头，处理不同的 DINOv3 中间层组合。设分割分支的层集合为 $\mathcal{L}_{\text{mask}} = \{3, 6, 9, 11\}$，深度分支的层集合为 $\mathcal{L}_{\text{depth}} = \{0, 3, 6, 9\}$。Backbone 仅前向一次：取并集 $\mathcal{L}_{\text{union}} = \mathcal{L}_{\text{mask}} \cup \mathcal{L}_{\text{depth}}$ 一次性提取特征，再按并集索引分发给两个分支各自的 DPTHead：

$$\mathbf{F}_{\text{mask}} = \text{DPTHead}(\{\mathbf{F}_{\ell}\}_{\ell \in \mathcal{L}_{\text{mask}}})$$

$$\mathbf{F}_{\text{depth}} = \text{DPTHead}_{\text{depth}}(\{\mathbf{F}_{\ell}\}_{\ell \in \mathcal{L}_{\text{depth}}})$$

两支随后各自独立完成 adapter 串联和投影，产生两张不同的 image embedding $\mathbf{E}_{\text{mask}}, \mathbf{E}_{\text{depth}} \in \mathbb{R}^{B \times 256 \times 64 \times 64}$，分别送入分割分支和深度分支。Phase 1 冻结 depth 侧的 DPT 头与 adapter，Phase 2 将其解冻并与深度解码器一同训练；Phase 2 启动时 adapter1/2_depth 和 projection_depth 从 mask 分支对应组件 warm-start 复制权重，dpt_head_depth 保持随机初始化。

## 7 训练策略

### 7.1 功能耦合与两阶段解耦

上述架构中存在根本性的功能耦合：深度分支的输入直接依赖分割分支的输出。outer prompt encoder 以 $\mathbf{M}_{\text{low}}$（分割分支产出的低分辨率掩码）为 mask prompt——若分割尚未收敛、$\mathbf{M}_{\text{low}}$ 粗糙或残缺，深度分支收到的条件信号包含错误的空间信息，深度梯度将建立在错误的输入条件之上。

这一耦合排除了端到端联合训练的可能。联合训练早期，分割质量不足导致 $\mathbf{M}_{\text{low}}$ 近乎噪声；深度分支在此条件下产生的梯度虽被 detach 阻断无法直接回传 mask 分支，但会通过共享的图像特征编码器扰动训练方向。同时，dice loss 与 MSE loss 在梯度量级和优化方向上差异显著，联合优化无法兼顾两者的收敛需求。

**两阶段解耦训练**的策略：首先独立训练模型的全部 mask 分支参数，待分割收敛后冻结所有 mask 分支组件；随后在固定的高质量 $\mathbf{M}_{\text{low}}$ 条件下，仅训练深度分支。功能耦合仅存在于前向推理路径中，训练动态完全解耦。

### 7.2 Phase 1：分割能力训练

**目标**：建立高质量分割能力，为 Phase 2 提供稳定的 mask prompt 基础。

**参数状态**：冻结 DINOv3 backbone 和深度分支全部参数（outer prompt encoder、DepthMaskDecoder；双头模式下额外冻结 depth 侧的 DPT 头与 adapter）；仅 adapter1/2、projection、SAM prompt encoder、SAM mask decoder 参与训练。

**损失函数**：

$$\mathcal{L}_{\text{P1}} = \mathcal{L}_{\text{dice}} + \mathcal{L}_{\text{iou}}$$

Dice Loss 以 sigmoid 概率 $P_{b,i,j} = \sigma(\hat{M}_{b,i,j})$ 与 GT 二值掩码 $T_{b,i,j}$ 计算集合相似度：

$$\mathcal{L}_{\text{dice}} = 1 - \frac{1}{B}\sum_{b=1}^{B} \frac{2\sum_{i,j} P_{b,i,j} \cdot T_{b,i,j} + \varepsilon}{\sum_{i,j} P_{b,i,j} + \sum_{i,j} T_{b,i,j} + \varepsilon}$$

IoU 预测损失以 MSE 将 mask decoder 输出的置信度评分 $\hat{r}_b$ 与真实 IoU 对齐：

$$\mathcal{L}_{\text{iou}} = \frac{1}{B}\sum_{b=1}^{B} \big(\hat{r}_b - \text{IoU}(\hat{M}_b, M_b)\big)^2$$

训练时采用迭代式 prompt 精化（§7.4）：分割分支每轮接收更新的 prompt 后重新预测 mask，所有 sub-iteration 的 loss 累积取均值。优化器为 AdamW（lr=$10^{-5}$），StepLR（每 10 epoch ×0.5），共 50 epoch。

### 7.3 Phase 2：深度预测能力训练

**目标**：在冻结的高质量 mask prompt 条件下，训练深度分支的条件回归能力。

**参数状态**：先冻结全部参数，再仅解冻 outer prompt encoder 和 DepthMaskDecoder（双头模式下额外解冻 depth 侧 DPT 头与 adapter）。mask 分支全部组件维持冻结。

**加载与初始化**：从 Phase 1 产出的 `phase1_best.pt` 用 `strict=False` 加载。Phase 1 中深度分支从未接收有效梯度更新，其保存的随机初始值无保留价值——Phase 2 启动时用 SAM 预训练权重重新初始化 outer prompt encoder 和 DepthMaskDecoder，仅 depth prediction head 保持随机初始化。

**损失函数**：Phase 2 仅优化深度 MSE：

$$\mathcal{L}_{\text{P2}} = \frac{1}{B}\sum_{b=1}^{B} (\hat{z}_b - z_b)^2$$

其中 $\hat{z}_b, z_b \in [-1, 1]$ 为归一化深度值。分割 loss 仅在 log 间隔 batch 上计算用于监控，不参与梯度——mask 分支权重已冻结，无法产生有效参数更新。depth decoder 输出的辅助 mask 仅用于迭代式点采样（§7.4），其 dice loss 同样不进入优化目标。

优化器配置与 Phase 1 完全一致（AdamW, StepLR, 50 epoch），Phase 2 从 epoch 0 独立启动学习率调度。

### 7.4 迭代式 Prompt 精化：充分训练 Prompt-Image 交互

SAM 的核心能力在于 prompt encoder 与 mask decoder 之间的交互——给定 prompt，模型通过 TwoWayTransformer 从图像特征中定位指定区域。若训练时始终使用固定 prompt（如单一正点），模型仅学到"单点→完整 mask"的静态映射，从未经历过"不完美预测 → 根据误差补充 prompt → 修正预测"的信号，迭代精化能力未能被充分训练。

为此引入迭代式 Prompt 精化（$n\_sub = 8$）。每轮 sub-iteration 中，当前预测 mask 与 GT mask 对比后，从两类误差区域各采样一个点作为下一轮的额外 prompt：

- **正点**（$\ell = 1$）：从"GT 有、预测无"的遗漏前景像素中随机采样；
- **负点**（$\ell = 0$）：从"预测有、GT 无"的误报背景像素中随机采样。

采样以均匀随机方式从误差区域的所有像素中选取。新点坐标从 $256 \times 256$ 的 mask 空间经线性缩放映射回原始图像坐标，附加到现有 prompt 集合。同时以 $p = 0.5$ 的概率将上一轮的 mask logits 作为 mask_input 供下一轮使用，使模型学会利用已有预测辅助后续精化。评估时 $p = 0$ 保持确定性。

两阶段的迭代模式因冻结语义不同而具有本质区别：

**Phase 1** 迭代完全发生在分割分支内部。每轮用更新后的 prompt 重新运行 SAM forward_mask，所有 sub-iteration 的 dice loss 和 IoU loss 累积取均值作为优化目标。模型通过 8 轮"预测→误差检测→补充 prompt→重新预测"学会计逐步将粗糙的初始 mask 精化为准确结果。

**Phase 2** 迭代在深度分支中以不同模式进行。分割分支权重已冻结，仅运行一次输出固定的 $\mathbf{M}_{\text{low}}$ 作为 mask prompt——在全部 sub-iteration 中复用。深度分支自身迭代 $n\_sub$ 次：每轮 forward_depth 同时输出深度预测 $\hat{z}$ 和辅助 decoder mask。**辅助 mask 的关键功能是为下一轮提供误差采样的依据**——decoder mask 与 GT mask 对比后采样新的正/负点，附加到深度分支的 point prompt 坐标集，下一轮深度分支使用更新后的 prompt 重新推理。深度 MSE loss 在各轮间累积取均值构成唯一优化目标，decoder mask 的 dice loss 仅做日志监控、不参与梯度。

这一设计构成了一种**自监督闭环**：depth decoder 自身输出的辅助 mask 质量决定了下一轮 prompt 的精度，prompt 精度又制约深度回归的条件信息质量——模型被迫在优化深度的同时，学会输出准确的辅助 mask 以改进条件推理。

## 8 推理

推理时用户提供点击 $(x, y)$，模型执行如下流程：

1. DINOv3 encoder 单次前向生成 image embeddings（双头模式下同时生成 $\mathbf{E}_{\text{mask}}$ 和 $\mathbf{E}_{\text{depth}}$）
2. 分割分支以初始点击为 prompt，迭代 $n\_sub = 8$ 次精化 mask，每轮从误差区自动采样新 prompt
3. 精化终态的 $\mathbf{M}_{\text{low}}$（detach）与累积的全部 point prompts 送入深度分支 outer prompt encoder
4. DepthMaskDecoder 以 mask prompt + point prompt 为条件，经 TwoWayTransformer 交叉注意力回归 $\hat{z}$
5. 单次前向同时输出 full-res mask $\hat{M}$ 和深度预测 $\hat{z}$

深度分支在推理时不进行自身迭代——mask 分支的迭代精化已提供充分的条件信息，深度分支仅需一次条件前向完成回归。

---

## 引用（占位）

\begin{itemize}
    \item \texttt{\textbackslash cite\{sam\}}：A. Kirillov et al., "Segment Anything," ICCV 2023.
    \item \texttt{\textbackslash cite\{dinov3\}}：M. Oquab et al., "DINOv3: All are worth 1 word," arXiv 2025.
    \item \texttt{\textbackslash cite\{segdino\}}：B. Xu et al., "SegDINO: Boosting DINO with Semi-Supervised Semantic Segmentation," 2024.（TBD：若 DPT 引用需替换为 Ranftl et al. "Vision Transformers for Dense Prediction," ICCV 2021.）
\end{itemize}
