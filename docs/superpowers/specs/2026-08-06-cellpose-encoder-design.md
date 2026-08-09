# Cellpose Encoder — 将 Cellpose 特征映射为 DINO-SAM Encoder 兼容格式

## 动机

现有 encoder（SAM、DINOv3、ResNet、Swin-B）均基于通用视觉 backbone，不包含
细胞形态先验知识。Cellpose 3.x 的 `eval()` 输出梯度流向场（dP）、细胞概率图
（cellprob）和欧拉积分终点（p），携带了丰富的细胞分割先验。将其作为 encoder
输入特征，有望提升下游 SAM mask decoder 和 depth decoder 在光学显微镜细胞
图像上的性能。

## 架构

```
                     Images (B, 3, 1024, 1024)
                            │
              ┌─────────────┴─────────────┐
              │  CellposeModel.eval()     │  ← FROZEN
              │  (cyto3 / livecell / ...) │
              └─────────────┬─────────────┘
                            │
       ┌────────────────────┼────────────────────┐
       │                    │                    │
  dP (B,2,1024,1024)  cellprob (B,1,1024,1024)  p (B,2,1024,1024)
       │                    │                    │
       └────────────────────┼────────────────────┘
                            │
                     Concat → (B, 5, 1024, 1024)
                            │
              ┌─────────────┴─────────────┐
              │  Trainable Downsample     │
              │  Head (4× stride-2 Conv)  │
              └─────────────┬─────────────┘
                            │
                   (B, 256, 64, 64)
```

### 可训练下采样头

```
Conv2d(5→64,  k=4,s=2,p=1) + BN + ReLU  → (B,  64, 512, 512)
Conv2d(64→128,k=4,s=2,p=1) + BN + ReLU  → (B, 128, 256, 256)
Conv2d(128→256,k=4,s=2,p=1) + BN + ReLU → (B, 256, 128, 128)
Conv2d(256→256,k=4,s=2,p=1) + BN + ReLU → (B, 256,  64,  64)
Conv2d(256→256,k=1)                     → (B, 256,  64,  64)
```

每层 stride=2 将分辨率减半：1024→512→256→128→64。

## 接口兼容性

```python
class CellposeEncoder(nn.Module):
    def __init__(self, model_type="cyto3"):
        ...
    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W) raw [0, 255] range
        Returns:
            image_embeddings: (B, 256, 64, 64)
            input_size: (1024, 1024)  固定尺寸，cellpose 内部处理 resize
        """
```

与其他 encoder 完全一致的接口：`forward()` 接受原始图像，返回
`(image_embeddings, input_size)`。

## 系统集成

### DepthSam 改动

```python
# dinosam/model/depth_sam.py __init__
elif encoder_type == "cellpose":
    self.cellpose_encoder = CellposeEncoder(model_type=cellpose_model)

# encode_images
elif self.encoder_type == "cellpose":
    return self.cellpose_encoder(images)
```

### CLI 改动

`evaluate.py` 和 `train.py` 的 `--encoder` 参数新增 `cellpose` 可选值，
新增 `--cellpose_model` 参数（默认 `cyto3`）指定 cellpose 模型类型。

## 涉及文件

| 文件 | 操作 |
|------|------|
| `dinosam/model/cellpose_encoder.py` | 新建 — CellposeEncoder 类 |
| `dinosam/model/__init__.py` | 添加 CellposeEncoder 导出 |
| `dinosam/model/depth_sam.py` | 新增 `encoder_type="cellpose"` 分支，新增 `cellpose_model` 参数 |
| `dinosam/evaluate.py` | `--encoder` 新增 `cellpose`，新增 `--cellpose_model` |
| `scripts/train.py` | 同上（如有 encoder 参数） |

## 关键设计决策

1. **Cellpose 完全冻结**：eval() 权重锁定，不参与训练。仅下采样头可训练。
2. **5 通道输入**：dP(2) + cellprob(1) + p(2)。经测试这 3 个输出是 cellpose
   最核心的中间表示，携带了边界、流向和概率信息。
3. **cellpose 与 SAM image_encoder 串联**：cellpose 替代 SAM image_encoder
   而非并联。`encode_images` 中走 cellpose 分支时跳过 SAM preprocess +
   image_encoder。
4. **预处理**：cellpose eval() 内部已处理 resize 和归一化，Encoder 只需
   传入原始 [0,255] 图像。
5. **diameter 自适应**：CellposeModel.eval() 默认 `diameter=None` 自动估计
   细胞直径，适合不同放大倍数的图像。
