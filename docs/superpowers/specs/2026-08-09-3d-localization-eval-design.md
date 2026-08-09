# 3D Localization Evaluation — Design Spec

**Date:** 2026-08-09
**Branch:** feature/3d-localization-eval (from feature/no-gt-prompts)

## Motivation

当前 `evaluate.py` 只评估分割质量（Dice/IoU）和图像级深度误差（MAE/MSE），缺少对细胞三维定位精度的直接测量。论文 method 部分定义了完整的 3D 定位计算流程：掩码质心提供 (X,Y) + 深度分支提供 Z。评估代码需要对应输出 per-cell 的 3D 定位误差指标。

## Scope

仅修改 `dinosam/evaluate.py`：

- 新增 2 个辅助函数：`_find_cells`、`_match_cells`
- 在 `evaluate()` 的 per-sample 循环内增加细胞提取 + 匈牙利匹配 + 3D 指标收集
- 扩展 `print_metrics()` 打印 3D 定位统计
- `main()` 新增 `--pixel_size_um` 可选参数

## Cell Definition

连通域（connected component）面积 > **100 pixels** 视为一个细胞。预测掩码和 GT 掩码各自独立提取连通域。

## Matching: Hungarian + IoU

1. 构造 n_pred × n_gt 的 cost 矩阵，`cost[i,j] = 1 - IoU(pred_cell_i, gt_cell_j)`
2. `scipy.optimize.linear_sum_assignment(cost)` 求最优匹配
3. 过滤 IoU < **0.3** 的匹配对

## Metrics (per matched cell pair)

| 指标 | 计算 | 说明 |
|------|------|------|
| `cell_xy_error_px` | sqrt((pred_x − gt_x)² + (pred_y − gt_y)²) | 像素 |
| `cell_xy_error_um` | xy_err_px × pixel_size_um | 微米，需传 `--pixel_size_um` |
| `cell_z_error` | abs(pred_z − gt_z) × DEPTH_SCALE | 微米 |
| `cell_3d_error` | sqrt(xy_err_um² + z_err²) | 微米，需传 `--pixel_size_um` |

每个 matched cell pair 产生一组上述值，累积为全局数组，最终报告 mean / median / std。

## New CLI Argument

```
--pixel_size_um FLOAT  微米/像素标定参数（optional）
```

不传时：仅报告 `cell_xy_error_px` 和 `cell_z_error`，µm 和 3D 误差标记为 N/A。

## Output Format

```
  3D Localization (per cell):
    XY Centroid Error (px):   mean=X.XX  median=X.XX  std=X.XX
    XY Centroid Error (µm):   mean=X.XX  median=X.XX  std=X.XX  (pixel_size=X.XXX)
    Z Depth Error (µm):       mean=X.XX  median=X.XX  std=X.XX
    3D Euclidean Error (µm):  mean=X.XX  median=X.XX  std=X.XX
    Matched cells:            N / M GT (XX.X%)
    (Predicted cells: X, GT cells: Y, Unmatched pred: X, Unmatched GT: X)
```

## Functions

### `_find_cells(mask: np.ndarray, min_area: int = 100) -> list[dict]`

```python
# Input:  (H, W) binary numpy array
# Output: list of {"centroid": (x, y), "area": int, "mask": (H, W) bool}
```

1. `scipy.ndimage.label(mask > 0)` 获取连通域
2. 遍历每个 component，面积 > min_area 则保留
3. centroid = (xs.mean(), ys.mean())

### `_match_cells(pred_cells, gt_cells, iou_thresh=0.3) -> list[tuple]`

```python
# Input:  pred_cells list, gt_cells list
# Output: [(pred_idx, gt_idx, iou), ...]
```

1. n_pred==0 或 n_gt==0 → 返回空列表
2. 双层循环构造 IoU 矩阵 → cost = 1 − IoU
3. `linear_sum_assignment(cost)` → 行-列匹配
4. 过滤 iou <= iou_thresh 的 pair

### `evaluate()` 修改

在现有 per-sample 循环末尾（line 183 前），增加：

```python
# ---- 3D cell-level metrics ----
pred_np = pred_i.cpu().numpy().astype(bool)
gt_np = gt_i.cpu().numpy().astype(bool)
pred_cells = _find_cells(pred_np, min_area=100)
gt_cells = _find_cells(gt_np, min_area=100)
matches = _match_cells(pred_cells, gt_cells, iou_thresh=0.3)

for pi, gi, iou in matches:
    px, py = pred_cells[pi]["centroid"]
    gx, gy = gt_cells[gi]["centroid"]
    err_px = np.sqrt((px - gx)**2 + (py - gy)**2)
    all_cell_xy_px.append(err_px)
    if pixel_size_um is not None:
        err_um = err_px * pixel_size_um
        all_cell_xy_um.append(err_um)
    all_cell_z.append(depth_mae)  # |pred_z - gt_z|
    if pixel_size_um is not None:
        err_3d = np.sqrt((err_px * pixel_size_um)**2 + depth_mae**2)
        all_cell_3d.append(err_3d)

all_matched += len(matches)
total_gt_cells += len(gt_cells)
total_pred_cells += len(pred_cells)
```

### `evaluate()` 签名变更

新增参数 `pixel_size_um: float = None`。

返回值新增字段：`cell_xy_error_px`, `cell_xy_error_um`, `cell_z_error`, `cell_3d_error`, `n_matched`, `n_gt_cells`, `n_pred_cells`。

### `print_metrics()` 修改

在 Depth R² 行之后新增 3D 定位 section。

## Dependencies

- `scipy.ndimage.label` — 已有（dataset 中已使用）
- `scipy.optimize.linear_sum_assignment` — 新增 import
- 无需新增 pip 依赖

## Verification

1. 在测试集上跑 `python dinosam/evaluate.py`，确认 3D metrics 正常输出
2. 用 `--pixel_size_um 0.325` 测试 µm 转换输出
3. 不传 `--pixel_size_um` 时仅输出像素误差和 Z 误差
