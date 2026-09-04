#!/usr/bin/env python3
"""扫描 evaluate.py 存下的 pred/gt PNG，输出"真值可疑"清单。

只读不改：仅统计 pred 与 gt 的分歧程度，按 IoU 升序列出最可疑的样本，
不修改任何真值文件。判定阈值仅用于排序和筛选，不做替换。

用法:
    python scripts/scan_gt_suspicious.py \
        --eval_dir /data3/LXT/DINOv3-SAM/model_sam/SAM/eval_ite8_100/phase2 \
        --out /tmp/suspicious.csv

输出 CSV 列:
    folder, image, iou, dice, gt_fg_ratio, pred_fg_ratio,
    gt_cells, pred_cells, suspicious, reason
"""

import argparse
import csv
import os

import numpy as np
from PIL import Image
from scipy import ndimage


# ---- 可疑判定阈值（仅排序/筛选用，不触发任何替换） ----
IOU_SUSPICIOUS = 0.7          # IoU 低于此值 → pred 与 GT 分歧大
CELL_DIFF_SUSPICIOUS = 2      # 连通块数量差超过此值 → 细胞数对不上
MIN_CELL_AREA = 100           # 与 evaluate.CELL_MIN_AREA 对齐，过滤噪点连通块


def load_binary(path):
    """读 PNG，二值化（>127 为前景），返回 (H, W) bool 数组。"""
    arr = np.array(Image.open(path).convert("L"))
    return arr > 127


def count_cells(binary, min_area=MIN_CELL_AREA):
    """统计二值图里面积大于 min_area 的连通块数量。"""
    labeled, n = ndimage.label(binary)
    if n == 0:
        return 0
    areas = ndimage.sum(binary, labeled, index=range(1, n + 1))
    return int((areas > min_area).sum())


def metrics_for_pair(gt_path, pred_path):
    """算单个样本的 IoU/Dice/面积比/连通块数。"""
    gt = load_binary(gt_path)
    pred = load_binary(pred_path)

    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    iou = float(inter / union) if union > 0 else 1.0
    dice = float(2 * inter / (gt.sum() + pred.sum() + 1e-7))

    gt_fg = float(gt.sum())
    pred_fg = float(pred.sum())
    # 面积比：1.0=完全一致；<1 pred 偏小（可能漏检或 GT 多标）；>1 pred 偏大
    fg_ratio = pred_fg / gt_fg if gt_fg > 0 else float("inf")

    return {
        "iou": iou,
        "dice": dice,
        "gt_fg_ratio": gt_fg,
        "pred_fg_ratio": pred_fg,
        "gt_cells": count_cells(gt),
        "pred_cells": count_cells(pred),
    }


def is_suspicious(m):
    """根据阈值判定是否可疑，返回 (是否可疑, 原因字符串)。"""
    reasons = []
    if m["iou"] < IOU_SUSPICIOUS:
        reasons.append(f"IoU={m['iou']:.3f}<{IOU_SUSPICIOUS}")
    if abs(m["gt_cells"] - m["pred_cells"]) > CELL_DIFF_SUSPICIOUS:
        reasons.append(f"cell_diff={abs(m['gt_cells']-m['pred_cells'])}>{CELL_DIFF_SUSPICIOUS}")
    return (len(reasons) > 0, ";".join(reasons) if reasons else "")


def scan(eval_dir, out_path):
    """扫描 eval_dir 下所有 *_gt.png，配对 *_pred.png，输出可疑清单 CSV。"""
    rows = []
    for folder in sorted(os.listdir(eval_dir)):
        folder_path = os.path.join(eval_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            if not fname.endswith("_gt.png"):
                continue
            base = fname[:-len("_gt.png")]
            pred_path = os.path.join(folder_path, base + "_pred.png")
            gt_path = os.path.join(folder_path, fname)
            if not os.path.exists(pred_path):
                print(f"[WARN] 缺 pred: {pred_path}")
                continue
            m = metrics_for_pair(gt_path, pred_path)
            susp, reason = is_suspicious(m)
            rows.append({
                "folder": folder,
                "image": base,
                **m,
                "suspicious": "Y" if susp else "",
                "reason": reason,
            })

    # 按 IoU 升序：最可疑的（分歧最大）在最前
    rows.sort(key=lambda r: r["iou"])

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "folder", "image", "iou", "dice",
            "gt_fg_ratio", "pred_fg_ratio", "gt_cells", "pred_cells",
            "suspicious", "reason",
        ])
        w.writeheader()
        w.writerows(rows)

    n_susp = sum(1 for r in rows if r["suspicious"] == "Y")
    print(f"扫描完成: 共 {len(rows)} 个样本, 可疑 {n_susp} 个")
    print(f"  平均 IoU: {np.mean([r['iou'] for r in rows]):.4f}" if rows else "  无样本")
    top5 = [f"{r['folder']}/{r['image']}={r['iou']:.3f}" for r in rows[:5]]
    print(f"  IoU 最小 5 个: {top5}")
    print(f"已保存: {out_path}")


def main():
    p = argparse.ArgumentParser(description="扫描 pred/gt 输出可疑真值清单（只读不改）")
    p.add_argument("--eval_dir", required=True,
                   help="evaluate.py --save_masks_dir 下的 phase1/ 或 phase2/ 子目录")
    p.add_argument("--out", required=True, help="输出 CSV 路径")
    args = p.parse_args()
    scan(args.eval_dir, args.out)


if __name__ == "__main__":
    main()
