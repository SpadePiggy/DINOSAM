"""Extract evaluation metrics from all model eval logs and output an Excel file.

Parses eval_iter[1|8]/log.txt (DPT models) and eval_*.log (OPT models) under
/data3/LXT/DINOv3-SAM/model_DINO/, then writes a multi-sheet Excel summary.
"""

import re
import os
import sys
from pathlib import Path
from collections import defaultdict

BASE = Path("/data3/LXT/DINOv3-SAM/model_DINO")

# ── Regex patterns ──────────────────────────────────────────────────────────
RE_DATASET = re.compile(r"Test dataset:.*/(\w+_test)")
RE_SAMPLES = re.compile(r"Test samples:\s+(\d+)")
RE_EVAL_TYPE = re.compile(r"Iterative eval:\s+n_sub=(\d+)")
RE_PHASE = re.compile(r"(?:Phase (\d) Model|============================================================$)")

RE_DICE = re.compile(r"Segmentation Dice:\s+([\d.]+)")
RE_IOU = re.compile(r"Segmentation IoU:\s+([\d.]+)")
RE_BOUNDIOU = re.compile(r"Boundary IoU:\s+([\d.]+)")
RE_PRECISION = re.compile(r"Precision:\s+([\d.]+)")
RE_RECALL = re.compile(r"Recall:\s+([\d.]+)")
RE_DEPTH_MSE = re.compile(r"Depth MSE:\s+([\d.]+)")
RE_DEPTH_MAE = re.compile(r"Depth MAE:\s+([\d.]+)")
RE_DEPTH_RMSE = re.compile(r"Depth RMSE:\s+([\d.]+)")
RE_DEPTH_R2 = re.compile(r"Depth R²:\s+([\-\d.]+)")
RE_DEPTH_PRED_MEAN = re.compile(r"Depth Pred:\s+mean=([\-\d.]+)")
RE_DEPTH_GT_MEAN = re.compile(r"Depth GT:\s+mean=([\-\d.]+)")

# Comparison section
RE_CMP_DICE = re.compile(r"Dice:\s+([\d.]+) -> ([\d.]+)")
RE_CMP_IOU = re.compile(r"IoU:\s+([\d.]+) -> ([\d.]+)")
RE_CMP_BOUNDIOU = re.compile(r"BoundIoU:\s+([\d.]+) -> ([\d.]+)")
RE_CMP_PRECISION = re.compile(r"Precision:\s+([\d.]+) -> ([\d.]+)")
RE_CMP_RECALL = re.compile(r"Recall:\s+([\d.]+) -> ([\d.]+)")
RE_CMP_DEPTH_MAE = re.compile(r"Depth MAE:\s+([\d.]+) -> ([\d.]+)")


def parse_log(log_path: Path) -> dict | None:
    """Parse a single eval log file. Returns a dict with extracted metrics."""
    try:
        text = log_path.read_text()
    except Exception as e:
        print(f"  [WARN] Cannot read {log_path}: {e}", file=sys.stderr)
        return None

    result = {
        "file": str(log_path.relative_to(BASE)),
        "dataset": None,
        "samples": None,
        "n_sub": None,
        "phase1": {},
        "phase2": {},
        "comparison": {},
    }

    # Header
    m = RE_DATASET.search(text)
    if m:
        result["dataset"] = m.group(1)
    m = RE_SAMPLES.search(text)
    if m:
        result["samples"] = int(m.group(1))
    m = RE_EVAL_TYPE.search(text)
    if m:
        result["n_sub"] = int(m.group(1))

    # Extract per-phase metrics
    # Strategy: find "Phase 1 Model" and "Phase 2 Model" blocks
    phase1_match = re.search(r"Phase 1 Model", text)
    phase2_match = re.search(r"Phase 2 Model", text)
    cmp_match = re.search(r"Phase 1 vs Phase 2 Comparison", text)

    def extract_phase(section: str) -> dict:
        """Extract metrics from a phase section."""
        metrics = {}
        for name, pattern in [
            ("dice", RE_DICE), ("iou", RE_IOU), ("boundary_iou", RE_BOUNDIOU),
            ("precision", RE_PRECISION), ("recall", RE_RECALL),
            ("depth_mse", RE_DEPTH_MSE), ("depth_mae", RE_DEPTH_MAE),
            ("depth_rmse", RE_DEPTH_RMSE), ("depth_r2", RE_DEPTH_R2),
            ("depth_pred_mean", RE_DEPTH_PRED_MEAN), ("depth_gt_mean", RE_DEPTH_GT_MEAN),
        ]:
            m = pattern.search(section)
            if m:
                val = float(m.group(1))
                metrics[name] = val
        return metrics

    if phase1_match:
        end1 = phase2_match.start() if phase2_match else len(text)
        result["phase1"] = extract_phase(text[phase1_match.start():end1])

    if phase2_match:
        end2 = cmp_match.start() if cmp_match else len(text)
        result["phase2"] = extract_phase(text[phase2_match.start():end2])

    if cmp_match:
        cmp_section = text[cmp_match.start():]
        for name, pattern in [
            ("dice_p1", RE_CMP_DICE), ("iou_p1", RE_CMP_IOU),
            ("boundary_iou_p1", RE_CMP_BOUNDIOU), ("precision_p1", RE_CMP_PRECISION),
            ("recall_p1", RE_CMP_RECALL), ("depth_mae_p1", RE_CMP_DEPTH_MAE),
        ]:
            m = pattern.search(cmp_section)
            if m:
                result["comparison"][name + "_from"] = float(m.group(1))
                result["comparison"][name + "_to"] = float(m.group(2))

    # If no per-phase metrics extracted, try global patterns (fallback)
    return result


def discover_logs() -> list[Path]:
    """Discover all eval log files under BASE."""
    logs = []

    # 1. DPT models: eval_iter1/log.txt and eval_iter8/log.txt
    for d in sorted(BASE.glob("DPT_*")):
        for sub in ["eval_iter1", "eval_iter8"]:
            p = d / sub / "log.txt"
            if p.exists():
                logs.append(p)

    # 2. OPT models with eval_*.log at top level
    for d in sorted(BASE.glob("*_OPT")):
        for pat in ["eval_iter.log", "eval_noniter.log"]:
            p = d / pat
            if p.exists():
                logs.append(p)

    # 3. OPT models with eval_iter8/eval.log and eval_noiter/eval.log
    for d in sorted(BASE.glob("*_OPT")):
        for sub in ["eval_iter8", "eval_noiter"]:
            p = d / sub / "eval.log"
            if p.exists():
                logs.append(p)

    # 4. OPT models with eval_masks_iter/eval_iter.log etc
    for d in sorted(BASE.glob("*_OPT")):
        for sub in ["eval_masks_iter", "eval_masks_noniter"]:
            for fname in ["eval_iter.log", "eval_noniter.log"]:
                p = d / sub / fname
                if p.exists():
                    logs.append(p)

    return sorted(set(logs))


def classify_eval(log_path: Path) -> tuple[str, str]:
    """Return (model_name, eval_type) from path.

    eval_type is one of: 'iter8', 'iter1', 'noniter', 'noiter'
    """
    rel = str(log_path.relative_to(BASE))
    parent = log_path.parent.name
    model = rel.split("/")[0]

    if parent == "eval_iter8":
        return model, "iter8"
    elif parent == "eval_iter1":
        return model, "iter1"
    elif parent == "eval_noiter" or parent == "eval_masks_noniter":
        return model, "noiter"

    # Direct .log files
    fname = log_path.name
    if "iter" in fname and "noniter" not in fname:
        return model, "iter8"
    elif "noniter" in fname:
        return model, "noiter"

    return model, "unknown"


def main():
    logs = discover_logs()
    print(f"Found {len(logs)} eval log files.\n")

    rows = []
    for lp in logs:
        model, etype = classify_eval(lp)
        print(f"Parsing: {lp.relative_to(BASE)}  →  model={model}, eval={etype}")
        data = parse_log(lp)
        if data is None:
            continue

        rows.append({
            "model": model,
            "eval_type": etype,
            "dataset": data["dataset"],
            "samples": data["samples"],
            "n_sub": data["n_sub"],
            **{f"p1_{k}": v for k, v in data["phase1"].items()},
            **{f"p2_{k}": v for k, v in data["phase2"].items()},
        })

    if not rows:
        print("No eval data found!", file=sys.stderr)
        sys.exit(1)

    # ── Build Excel ──────────────────────────────────────────────────────
    try:
        import openpyxl
    except ImportError:
        print("openpyxl not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl", "-q"])
        import openpyxl

    wb = openpyxl.Workbook()

    # Sheet 1: All metrics (long format)
    ws = wb.active
    ws.title = "Eval Summary"

    # Define columns
    columns = [
        "Model", "Eval Type", "Dataset", "Samples", "n_sub",
        "P1 Dice", "P1 IoU", "P1 BoundIoU", "P1 Precision", "P1 Recall",
        "P1 Depth MAE", "P1 Depth RMSE", "P1 Depth R²",
        "P2 Dice", "P2 IoU", "P2 BoundIoU", "P2 Precision", "P2 Recall",
        "P2 Depth MAE", "P2 Depth RMSE", "P2 Depth R²",
        "ΔDice", "ΔIoU", "ΔBoundIoU", "ΔPrecision", "ΔRecall", "ΔDepthMAE",
    ]
    key_map = {
        "P1 Dice": "p1_dice", "P1 IoU": "p1_iou", "P1 BoundIoU": "p1_boundary_iou",
        "P1 Precision": "p1_precision", "P1 Recall": "p1_recall",
        "P1 Depth MAE": "p1_depth_mae", "P1 Depth RMSE": "p1_depth_rmse",
        "P1 Depth R²": "p1_depth_r2",
        "P2 Dice": "p2_dice", "P2 IoU": "p2_iou", "P2 BoundIoU": "p2_boundary_iou",
        "P2 Precision": "p2_precision", "P2 Recall": "p2_recall",
        "P2 Depth MAE": "p2_depth_mae", "P2 Depth RMSE": "p2_depth_rmse",
        "P2 Depth R²": "p2_depth_r2",
    }

    # Header row
    for col_idx, col_name in enumerate(columns, 1):
        ws.cell(row=1, column=col_idx, value=col_name)
        ws.cell(row=1, column=col_idx).font = openpyxl.styles.Font(bold=True)

    for row_idx, r in enumerate(rows, 2):
        ws.cell(row=row_idx, column=1, value=r["model"])
        ws.cell(row=row_idx, column=2, value=r["eval_type"])
        ws.cell(row=row_idx, column=3, value=r["dataset"])
        ws.cell(row=row_idx, column=4, value=r["samples"])
        ws.cell(row=row_idx, column=5, value=r["n_sub"])

        for col_idx, col_name in enumerate(columns, 1):
            if col_name in key_map:
                val = r.get(key_map[col_name])
                if val is not None:
                    ws.cell(row=row_idx, column=col_idx, value=round(val, 4))

        # Compute deltas
        for col_name, p1k, p2k in [
            ("ΔDice", "p1_dice", "p2_dice"),
            ("ΔIoU", "p1_iou", "p2_iou"),
            ("ΔBoundIoU", "p1_boundary_iou", "p2_boundary_iou"),
            ("ΔPrecision", "p1_precision", "p2_precision"),
            ("ΔRecall", "p1_recall", "p2_recall"),
            ("ΔDepthMAE", "p1_depth_mae", "p2_depth_mae"),
        ]:
            col_idx = columns.index(col_name) + 1
            v1 = r.get(p1k)
            v2 = r.get(p2k)
            if v1 is not None and v2 is not None:
                ws.cell(row=row_idx, column=col_idx, value=round(v2 - v1, 4))

    # Auto-width
    for col_idx, col_name in enumerate(columns, 1):
        max_len = len(col_name)
        for row_idx in range(2, len(rows) + 2):
            val = ws.cell(row=row_idx, column=col_idx).value
            if val is not None:
                max_len = max(max_len, len(str(val)))
        ws.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = max_len + 2

    # Sheets 2+: Pivot tables by dataset — mask & depth metrics

    # Group by dataset
    for sheet_name, dataset_filter in [("Mask-0_sam", "0_sam_test"), ("Mask-1724", "1724_test")]:
        ws_mask = wb.create_sheet(sheet_name)

        mask_cols = ["Model", "Eval", "P1 Dice", "P2 Dice", "ΔDice",
                     "P1 IoU", "P2 IoU", "ΔIoU",
                     "P1 BoundIoU", "P2 BoundIoU", "ΔBoundIoU"]
        for ci, cn in enumerate(mask_cols, 1):
            ws_mask.cell(row=1, column=ci, value=cn)
            ws_mask.cell(row=1, column=ci).font = openpyxl.styles.Font(bold=True)

        filtered = [r for r in rows if r["dataset"] == dataset_filter]
        for ri, r in enumerate(filtered, 2):
            ws_mask.cell(row=ri, column=1, value=r["model"])
            ws_mask.cell(row=ri, column=2, value=r["eval_type"])
            for key_dst, key_src in [
                ("P1 Dice", "p1_dice"), ("P2 Dice", "p2_dice"),
                ("P1 IoU", "p1_iou"), ("P2 IoU", "p2_iou"),
                ("P1 BoundIoU", "p1_boundary_iou"), ("P2 BoundIoU", "p2_boundary_iou"),
            ]:
                ci = mask_cols.index(key_dst) + 1
                v = r.get(key_src)
                if v is not None:
                    ws_mask.cell(row=ri, column=ci, value=round(v, 4))
            # Deltas
            for delta_name, p1k, p2k in [
                ("ΔDice", "p1_dice", "p2_dice"),
                ("ΔIoU", "p1_iou", "p2_iou"),
                ("ΔBoundIoU", "p1_boundary_iou", "p2_boundary_iou"),
            ]:
                ci = mask_cols.index(delta_name) + 1
                v1, v2 = r.get(p1k), r.get(p2k)
                if v1 is not None and v2 is not None:
                    ws_mask.cell(row=ri, column=ci, value=round(v2 - v1, 4))

        for ci in range(1, len(mask_cols) + 1):
            max_len = len(mask_cols[ci - 1])
            for ri in range(2, len(filtered) + 2):
                v = ws_mask.cell(row=ri, column=ci).value
                if v is not None:
                    max_len = max(max_len, len(str(v)))
            ws_mask.column_dimensions[openpyxl.utils.get_column_letter(ci)].width = max_len + 2

    # Sheet 3: Depth metrics by model × eval type
    for sheet_name, dataset_filter in [("Depth-0_sam", "0_sam_test"), ("Depth-1724", "1724_test")]:
        ws_depth = wb.create_sheet(sheet_name)

        depth_cols = ["Model", "Eval", "P1 MAE", "P2 MAE", "ΔMAE",
                      "P1 RMSE", "P2 RMSE", "ΔRMSE",
                      "P1 R²", "P2 R²", "ΔR²"]
        for ci, cn in enumerate(depth_cols, 1):
            ws_depth.cell(row=1, column=ci, value=cn)
            ws_depth.cell(row=1, column=ci).font = openpyxl.styles.Font(bold=True)

        filtered = [r for r in rows if r["dataset"] == dataset_filter]
        for ri, r in enumerate(filtered, 2):
            ws_depth.cell(row=ri, column=1, value=r["model"])
            ws_depth.cell(row=ri, column=2, value=r["eval_type"])
            for key_dst, key_src in [
                ("P1 MAE", "p1_depth_mae"), ("P2 MAE", "p2_depth_mae"),
                ("P1 RMSE", "p1_depth_rmse"), ("P2 RMSE", "p2_depth_rmse"),
                ("P1 R²", "p1_depth_r2"), ("P2 R²", "p2_depth_r2"),
            ]:
                ci = depth_cols.index(key_dst) + 1
                v = r.get(key_src)
                if v is not None:
                    ws_depth.cell(row=ri, column=ci, value=round(v, 4))
            for delta_name, p1k, p2k in [
                ("ΔMAE", "p1_depth_mae", "p2_depth_mae"),
                ("ΔRMSE", "p1_depth_rmse", "p2_depth_rmse"),
                ("ΔR²", "p1_depth_r2", "p2_depth_r2"),
            ]:
                ci = depth_cols.index(delta_name) + 1
                v1, v2 = r.get(p1k), r.get(p2k)
                if v1 is not None and v2 is not None:
                    ws_depth.cell(row=ri, column=ci, value=round(v2 - v1, 4))

        for ci in range(1, len(depth_cols) + 1):
            max_len = len(depth_cols[ci - 1])
            for ri in range(2, len(filtered) + 2):
                v = ws_depth.cell(row=ri, column=ci).value
                if v is not None:
                    max_len = max(max_len, len(str(v)))
            ws_depth.column_dimensions[openpyxl.utils.get_column_letter(ci)].width = max_len + 2

    # Save
    out_path = Path(__file__).resolve().parent.parent / "out" / "eval_metrics.xlsx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    print(f"\nSaved: {out_path}")
    print(f"  Models: {len(set(r['model'] for r in rows))}")
    print(f"  Eval logs: {len(rows)}")


if __name__ == "__main__":
    main()
