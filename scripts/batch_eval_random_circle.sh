#!/bin/bash
# Batch evaluation: run random_circle eval on all DPT models.
#
# Usage:
#   ./scripts/batch_eval_random_circle.sh [--force] [--gpus 0,1,2,3]
#
# Scans /data3/LXT/DINOv3-SAM/model_DINO/DPT_*/ for trained checkpoints,
# extracts config from config.txt, and launches evaluate.py --prompt_mode random_circle
# on each. Skips models that already have eval_random_circle/eval.log.
set -euo pipefail

MODEL_BASE="/data3/LXT/DINOv3-SAM/model_DINO"
PYTHONPATH="/home/LXT/lxt/DINOSAM/DINOSAM"
EVAL_SCRIPT="$PYTHONPATH/dinosam/evaluate.py"
DINOV3_CKPT="/data3/LXT/data/checkpoint/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
PROMPT_MODE="random_circle"

FORCE=false
GPUS=(0 1 2 3)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=true; shift ;;
        --gpus) IFS=',' read -r -a GPUS <<< "$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Collect models to run
declare -A MODEL_GPU
GPU_IDX=0
TOTAL=0
SKIPPED=0

for dir in "$MODEL_BASE"/DPT_*/; do
    name=$(basename "$dir")
    cfg="$dir/config.txt"

    # Must have checkpoints
    [[ -f "$dir/phase1_best.pt" ]] || continue
    [[ -f "$dir/phase2_best.pt" ]] || continue

    # Skip if already done (unless --force)
    if [[ "$FORCE" != "true" ]] && [[ -f "$dir/eval_random_circle/eval.log" ]]; then
        echo "[SKIP] $name — already evaluated"
        ((SKIPPED++))
        continue
    fi

    # Extract config: val_dir -> test_dir, dpt_layers, dpt_layers_depth
    test_dir=$(grep -oP '--val_dir\s+\K\S+' "$cfg" 2>/dev/null)
    dpt_layers=$(grep -oP '--dpt_layers\s+\K\S+' "$cfg" 2>/dev/null)
    dpt_layers_depth=$(grep -oP '--dpt_layers_depth\s+\K\S+' "$cfg" 2>/dev/null)

    [[ -n "$test_dir" ]] || { echo "[WARN] $name: no val_dir in config, skipping"; continue; }
    [[ -n "$dpt_layers" ]] || { echo "[WARN] $name: no dpt_layers in config, skipping"; continue; }

    # Assign GPU (round-robin)
    gpu="${GPUS[$GPU_IDX]}"
    GPU_IDX=$(( (GPU_IDX + 1) % ${#GPUS[@]} ))

    # Build args
    DPT_LAYERS_DEPTH_ARG=""
    if [[ -n "$dpt_layers_depth" ]]; then
        DPT_LAYERS_DEPTH_ARG="--dpt_layers_depth $dpt_layers_depth"
    fi

    mkdir -p "$dir/eval_random_circle"

    CUDA_VISIBLE_DEVICES="$gpu" nohup env PYTHONPATH="$PYTHONPATH" \
        python "$EVAL_SCRIPT" \
        --test_dir "$test_dir" \
        --encoder dinov3 \
        --adapter_type dpt_simple \
        --dpt_layers "$dpt_layers" \
        $DPT_LAYERS_DEPTH_ARG \
        --depth_transformer_type twoway \
        --dinov3_checkpoint "$DINOV3_CKPT" \
        --phase1_checkpoint "$dir/phase1_best.pt" \
        --phase2_checkpoint "$dir/phase2_best.pt" \
        --save_masks_dir "$dir/eval_random_circle" \
        --prompt_mode "$PROMPT_MODE" \
        > "$dir/eval_random_circle/eval.log" 2>&1 &

    echo "[RUN] $name (GPU $gpu, PID $!) — layers=$dpt_layers${dpt_layers_depth:+ depth=$dpt_layers_depth} test=$test_dir"
    ((TOTAL++))
done

echo ""
echo "============================================"
echo "  Launched: $TOTAL   Skipped: $SKIPPED"
echo "  GPU assignment: round-robin across ${GPUS[*]}"
echo "============================================"
echo ""
echo "Monitor:"
echo "  tail -f $MODEL_BASE/DPT_*/eval_random_circle/eval.log"
echo ""
echo "Collect results:"
echo "  for d in $MODEL_BASE/DPT_*/eval_random_circle/eval.log; do"
echo "    name=\$(basename \$(dirname \$(dirname \$d)))"
echo "    dice=\$(grep 'Segmentation Dice' \$d | tail -1 | grep -oP '[\d.]+' | head -1)"
echo "    mae=\$(grep 'Depth MAE' \$d | tail -1 | grep -oP '[\d.]+' | head -1)"
echo "    echo \"\$name | Dice=\$dice | MAE=\$mae\""
echo "  done"
