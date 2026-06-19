# DINO Feature Adapter — 可配置适配器架构设计

**日期**: 2026-06-19
**分支**: DINO
**状态**: 待实现

---

## 1. 背景与目标

当前 DINOv3 编码器的特征适配由硬编码的两个 Mona 层完成：

```
DINOv3 ViT-B/16 (frozen) → Mona × 2 → Linear Projection (768→256) → SAM embedding
```

目标：将适配器改造为**通过配置可选**，支持三种独立的适配器类型，便于对比实验和后续扩展。

## 2. 适配器类型

| 名称 | 配置值 | 描述 |
|------|--------|------|
| Mona | `"mona"` | 现有实现：多尺度深度可分离卷积 + bottleneck MLP |
| FC | `"fc"` | 纯全连接层：bottleneck MLP，无空间操作 |
| Dual Attention | `"dual_attn"` | DANet 风格：PAM（空间注意力）+ CAM（通道注意力），并行求和融合 |

三种适配器互斥，每次只使用一种（两层串联）。

## 3. 架构设计

### 3.1 文件结构

```
dinosam/model/
├── adapters/
│   ├── __init__.py              # get_adapter() 工厂函数
│   ├── fc_adapter.py            # 全连接层适配器
│   └── dual_attn_adapter.py     # 双重注意力适配器（内联 PAM + CAM 实现）
├── mona.py                      # 保留不动，mona 适配器直接复用
├── dinov3_encoder.py            # 改为工厂创建适配器
├── depth_sam.py                 # 传递 adapter_type 参数
└── ...
```

### 3.2 统一接口

所有适配器实现相同签名，可互换使用：

```python
class XxxAdapter(nn.Module):
    def __init__(self, in_dim: int, factor: int = 8):
        ...

    def forward(self, x: Tensor, hw_shapes: tuple | None = None) -> Tensor:
        """Args: x (B, N, C), hw_shapes (H, W). Returns: (B, N, C)"""
        ...
```

输入/输出均为 `(B, N, C)` 的序列特征。需要 2D 操作的适配器内部通过 `hw_shapes` 自行 reshape。

### 3.3 工厂函数

```python
# adapters/__init__.py
def get_adapter(name: str, in_dim: int, factor: int = 8) -> nn.Module:
    if name == "mona":
        from ..mona import Mona
        return Mona(in_dim, factor)
    elif name == "fc":
        return FCAdapter(in_dim, factor)
    elif name == "dual_attn":
        return DualAttnAdapter(in_dim, factor)
    else:
        raise ValueError(f"Unknown adapter type: {name}")
```

---

## 4. 各适配器详细设计

### 4.1 Mona（直接复用）

工厂对 `"mona"` 类型**直接返回 `Mona` 实例**，不做包装。`Mona` 类签名已符合统一接口，无需适配层。

```python
# 工厂中：
if name == "mona":
    return Mona(in_dim, factor)  # 直接返回，不包装
```

内部结构（来自现有 `Mona`）：
```
LayerNorm*γ + identity*γx → Linear(dim→bottleneck) → MonaOp(多尺度深度可分离卷积)
→ GELU → Dropout → Linear(bottleneck→dim) → 残差连接
```

### 4.2 FCAdapter

纯全连接层瓶颈结构，无空间卷积，不依赖 `hw_shapes`。

```python
class FCAdapter(nn.Module):
    def __init__(self, in_dim, factor=8):
        super().__init__()
        bottleneck = in_dim // factor
        self.norm = nn.LayerNorm(in_dim)
        self.project1 = nn.Linear(in_dim, bottleneck)
        self.project2 = nn.Linear(bottleneck, in_dim)
        self.dropout = nn.Dropout(0.1)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax
        x = self.project1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.project2(x)
        return identity + x
```

与 Mona 的区别：去掉了 `MonaOp`（多尺度深度可分离卷积），仅保留 linear → GELU → dropout → linear 的通道变换。

### 4.3 DualAttnAdapter

内联实现 DANet 的 PAM（Position Attention Module）和 CAM（Channel Attention Module），并行计算后逐元素求和，再加全局残差。不依赖外部 DANet 包。

**关键设计决策**：PAM/CAM 内部**不包含残差连接**（不同于 DANet 原版），由外层的 `DualAttnAdapter.forward()` 统一添加一个全局残差。这避免了 PAM 和 CAM 各自加残差后求和导致的 identity 信号双加（`2x` 而非 `x`）问题，保证初始化时等价于恒等映射。

**数据流**：
```
(B,N,C) → reshape → (B,C,H,W) ─┬─ PAM (no residual) → attn_pam ─┐
                                └─ CAM (no residual) → attn_cam ─┴─ (attn_pam + attn_cam) / 2 + identity → reshape → (B,N,C)
```

**PAM_Module（空间注意力）**：
- 三个 1×1 卷积：query (C→C/factor), key (C→C/factor), value (C→C)，降维比例由 `factor` 参数控制
- 计算 `softmax(query^T × key)` 得到 (N, N) 空间注意力图，每个位置聚合所有位置的特征
- 输出：`γ × (value × attention^T).reshape`（仅 attention 分支，无残差），γ 初始化为 0

**CAM_Module（通道注意力）**：
- 无额外卷积，直接从原始特征计算
- `softmax(max_energy − x·x^T)` 得到 (C, C) 通道注意力图（使用 max 减法做数值稳定）
- 输出：`β × (attention × x).reshape`（仅 attention 分支，无残差），β 初始化为 0

**可学习的缩放参数初始化为 0**，保证训练初期 attention 分支输出为 0，加上全局残差后等价于恒等映射。与 Mona/FCAdapter 的 `γ=1e-6` 初始化策略目的一致。

**内存注意**：PAM 为 64×64 特征图计算 `(4096, 4096)` 注意力矩阵（~16M floats，fp32 ≈ 64MB）。两个 adapter 各有一份，batch_size=4 时总开销约 500MB–1GB。在 <8GB VRAM 的 GPU 上可能需要减小 batch_size。

```python
class PAM_Module(nn.Module):
    """Position Attention Module — no internal residual."""
    def __init__(self, in_dim, factor=8):
        super().__init__()
        reduction = in_dim // factor
        self.query_conv = nn.Conv2d(in_dim, reduction, 1)
        self.key_conv = nn.Conv2d(in_dim, reduction, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        query = self.query_conv(x).view(B, -1, N).permute(0, 2, 1)  # (B, N, C//factor)
        key = self.key_conv(x).view(B, -1, N)                         # (B, C//factor, N)
        attn = torch.softmax(torch.bmm(query, key), dim=-1)           # (B, N, N)
        value = self.value_conv(x).view(B, -1, N)                     # (B, C, N)
        out = torch.bmm(value, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out  # no residual here


class CAM_Module(nn.Module):
    """Channel Attention Module — no internal residual."""
    def __init__(self, in_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        query = x.view(B, C, N)                                       # (B, C, N)
        key = x.view(B, C, N).permute(0, 2, 1)                        # (B, N, C)
        energy = torch.bmm(query, key)                                 # (B, C, C)
        energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attn = torch.softmax(energy, dim=-1)
        out = torch.bmm(attn, query).view(B, C, H, W)
        return self.gamma * out  # no residual here


class DualAttnAdapter(nn.Module):
    def __init__(self, in_dim, factor=8):
        super().__init__()
        self.pam = PAM_Module(in_dim, factor)
        self.cam = CAM_Module(in_dim)

    def forward(self, x, hw_shapes=None):
        if hw_shapes is None:
            raise ValueError(
                "DualAttnAdapter requires hw_shapes for 2D reshape. "
                "Got hw_shapes=None."
            )
        B, N, C = x.shape
        H, W = hw_shapes
        identity = x
        x_2d = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        attn_out = (self.pam(x_2d) + self.cam(x_2d)) / 2.0
        out = attn_out.permute(0, 2, 3, 1).reshape(B, N, C)
        return identity + out  # single global residual

---

## 5. 现有代码改动

### 5.1 `dinosam/model/dinov3_encoder.py`

```python
# 改动前
from .mona import Mona
...
self.mona1 = Mona(embed_dim, factor=8)
self.mona2 = Mona(embed_dim, factor=8)

# 改动后
from .adapters import get_adapter
...
self.adapter1 = get_adapter(adapter_type, embed_dim, factor=8)
self.adapter2 = get_adapter(adapter_type, embed_dim, factor=8)
```

新增构造函数参数 `adapter_type: str = "mona"`。`forward()` 中 `self.mona1/mona2` 调用改为 `self.adapter1/adapter2`。

### 5.2 `dinosam/model/depth_sam.py`

`DepthSam.__init__` 新增 `adapter_type` 参数，传递给 `DINOv3MonaEncoder`：

```python
def __init__(self, sam, encoder_type="sam", dinov3_checkpoint=None, adapter_type="mona"):
    ...
    self.dinov3_encoder = DINOv3MonaEncoder(
        checkpoint_path=dinov3_checkpoint,
        img_size=image_size,
        embed_dim=768,
        out_dim=256,
        adapter_type=adapter_type,
    )
```

### 5.3 `dinosam/train/trainer.py`

`_build_model_and_data` 中传递参数：

```python
model = DepthSam(
    sam,
    encoder_type=args.encoder,
    dinov3_checkpoint=args.dinov3_checkpoint,
    adapter_type=args.adapter_type,
)
```

### 5.4 `dinosam/evaluate.py`

同样在 `DepthSam` 实例化时传递 `adapter_type`。

### 5.5 `dinosam/model/mona.py`

**不做任何改动**，保留原文件以确保向后兼容。

---

## 6. Checkpoint 兼容性

**Key 名变更**：旧代码使用 `mona1.*` / `mona2.*`，新代码统一使用 `adapter1.*` / `adapter2.*`。由于 `"mona"` 类型直接返回 `Mona` 实例（无包装），state_dict key 结构保持一致，仅有前缀不同。

| 旧 key | 新 key |
|--------|--------|
| `mona1.project1.weight` | `adapter1.project1.weight` |
| `mona1.adapter_conv.conv1.weight` | `adapter1.adapter_conv.conv1.weight` |
| `mona1.gamma` | `adapter1.gamma` |
| `mona2.*` | `adapter2.*` |

**影响**：已有 branch 上训练的 checkpoint **无法直接加载**，需要简单的 key 前缀 remap。

**迁移路径**：
- 加载旧 checkpoint 前，对 `model_state_dict` 做 `mona1.` → `adapter1.`、`mona2.` → `adapter2.` 的字符串替换
- 仅 remap `model_state_dict`，不动 `optimizer_state_dict` / `scheduler_state_dict`
- 跨 adapter 类型的 checkpoint 不兼容（如 `mona` 的权重无法加载到 `fc` 模型）
- 建议：在 DINO 分支上从头训练各 adapter 类型；remap 仅用于加载已有 `mona` checkpoint 做对比验证

**设计说明**：刻意统一为 `adapter1/adapter2` 命名，而非保留 `mona1/mona2`。语义上更清晰——属性名反映了角色（第一个/第二个适配器）而非类型（Mona）。

---

## 7. CLI 与配置

新增 `--adapter_type` 参数：

```python
parser.add_argument("--adapter_type", type=str, default="mona",
                    choices=["mona", "fc", "dual_attn"],
                    help="Feature adapter type for DINOv3 encoder (default: mona)")
```

**有效性校验**：`--adapter_type` 仅在 `--encoder dinov3` 时生效。当 `--encoder sam` 且 `--adapter_type` 被显式指定为非默认值时，应打印 warning 提示用户该参数被忽略。

向后兼容：默认值为 `"mona"`，已有训练/评估命令无需修改。

**两个入口都需要添加**：`train/trainer.py` 和 `evaluate.py` 的 argparse 块各自添加此参数。

训练命令示例：

```bash
# 使用 FC adapter
python -m dinosam.train.trainer ... --encoder dinov3 --adapter_type fc

# 使用 Dual Attention adapter
python -m dinosam.train.trainer ... --encoder dinov3 --adapter_type dual_attn
```

---

## 8. 成功标准

1. `--adapter_type mona` 训练结果与当前 DINO 分支完全一致（向后兼容验证）
2. `--adapter_type fc` 可正常训练和评估
3. `--adapter_type dual_attn` 可正常训练和评估
4. 所有三种 adapter 的 checkpoint 可正常保存和加载（round-trip）
5. 旧 `mona1/mona2` 格式的 checkpoint 经 remap 后可正常加载
