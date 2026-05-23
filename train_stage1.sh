#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

PATCH_SIZE=${PATCH_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-1}
EPOCHS=${EPOCHS:-150}
GPU_IDS=${GPU_IDS:-0}
PRECISION=${PRECISION:-32}
SAVE_TOP_K=${SAVE_TOP_K:-3}
EMA=${EMA:-0}
EMA_DECAY=${EMA_DECAY:-0.9999}
MONITOR=${MONITOR:-val_psnr}
LOSS_TYPE=${LOSS_TYPE:-l1}
MSE_WEIGHT=${MSE_WEIGHT:-0.05}
TASK_CONDITIONING=${TASK_CONDITIONING:-0}
SIPL_LITE=${SIPL_LITE:-0}
SIPL_START_ALPHA=${SIPL_START_ALPHA:-0.5}
SIPL_REFINE_WEIGHT=${SIPL_REFINE_WEIGHT:-0.5}
RUN_NAME="stage1-p${PATCH_SIZE}-bs${BATCH_SIZE}-${MONITOR}"
CKPT_DIR="checkpoints/${RUN_NAME}"

echo "=== Stage 1: Training with validation ==="
echo "Patch size: $PATCH_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Save top-k: $SAVE_TOP_K"
echo "EMA: $EMA"
echo "Monitor: $MONITOR"
echo "Loss type: $LOSS_TYPE"
echo "MSE weight: $MSE_WEIGHT"
echo "Task conditioning: $TASK_CONDITIONING"
echo "SIPL-lite: $SIPL_LITE"
echo "SIPL start alpha: $SIPL_START_ALPHA"
echo "SIPL refine weight: $SIPL_REFINE_WEIGHT"

mkdir -p "$CKPT_DIR"
if compgen -G "$CKPT_DIR/promptir-epoch*.ckpt" > /dev/null; then
    echo "ERROR: Existing checkpoints found in $CKPT_DIR"
    echo "Move or delete that directory before rerunning to avoid averaging stale checkpoints."
    exit 1
fi

TRAIN_ARGS=(
    --gpu_ids "$GPU_IDS"
    --epochs "$EPOCHS"
    --precision "$PRECISION"
    --batch_size "$BATCH_SIZE"
    --patch_size "$PATCH_SIZE"
    --ckpt_dir "$CKPT_DIR"
    --save_top_k "$SAVE_TOP_K"
    --monitor "$MONITOR"
    --loss_type "$LOSS_TYPE"
    --mse_weight "$MSE_WEIGHT"
)
if [ "$EMA" = "1" ]; then
    TRAIN_ARGS+=(--ema --ema_decay "$EMA_DECAY")
fi
if [ "$TASK_CONDITIONING" = "1" ]; then
    TRAIN_ARGS+=(--task_conditioning)
fi
if [ "$SIPL_LITE" = "1" ]; then
    TRAIN_ARGS+=(--sipl_lite --sipl_start_alpha "$SIPL_START_ALPHA" --sipl_refine_weight "$SIPL_REFINE_WEIGHT")
fi

python train_hw4.py "${TRAIN_ARGS[@]}"

BEST_CKPT=$(python - "$CKPT_DIR" "$MONITOR" <<'PY'
import re
import sys
from pathlib import Path

ckpts = list(Path(sys.argv[1]).glob("promptir-epoch*.ckpt"))
metric = sys.argv[2]
reverse = metric == "val_psnr"
missing = float("-inf") if reverse else float("inf")
def metric_value(path):
    match = re.search(rf'{metric}=([0-9]+(?:\.[0-9]+)?)', path.name)
    return float(match.group(1)) if match else missing
for path in sorted(ckpts, key=metric_value, reverse=reverse)[:1]:
    print(path)
PY
)
if [ -z "$BEST_CKPT" ]; then
    echo "ERROR: No checkpoint found in $CKPT_DIR"
    exit 1
fi

BEST_EPOCH=$(basename "$BEST_CKPT" | grep -oP 'epoch=\K[0-9]+' | head -1)
if [ -z "$BEST_EPOCH" ]; then
    BEST_EPOCH=$(basename "$BEST_CKPT" | sed 's/.*epoch\([0-9]*\)\.ckpt/\1/')
fi
echo "Best checkpoint: $BEST_CKPT"
echo "Best epoch: $BEST_EPOCH"

echo ""
echo "=== Original inference ==="
python inference.py "$BEST_CKPT" --output "submission/stage1-p${PATCH_SIZE}-original.zip"

echo ""
echo "=== TTA inference ==="
python inference.py "$BEST_CKPT" --tta --output "submission/stage1-p${PATCH_SIZE}-tta.zip"

echo ""
echo "=== Averaging top checkpoints ==="
mapfile -t AVG_CKPTS < <(python - "$CKPT_DIR" "$SAVE_TOP_K" "$MONITOR" <<'PY'
import re
import sys
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
limit = int(sys.argv[2])
metric = sys.argv[3]
reverse = metric == "val_psnr"
ckpts = list(ckpt_dir.glob("promptir-epoch*.ckpt"))
missing = float("-inf") if reverse else float("inf")
def metric_value(path):
    match = re.search(rf'{metric}=([0-9]+(?:\.[0-9]+)?)', path.name)
    return float(match.group(1)) if match else missing
for path in sorted(ckpts, key=metric_value, reverse=reverse)[:limit]:
    print(path)
PY
)
AVG_CKPT="checkpoints/${RUN_NAME}-avg${SAVE_TOP_K}.ckpt"
python average_checkpoints.py "${AVG_CKPTS[@]}" --output "$AVG_CKPT"

echo ""
echo "=== Averaged TTA inference ==="
python inference.py "$AVG_CKPT" --tta --output "submission/stage1-p${PATCH_SIZE}-avg${SAVE_TOP_K}-tta.zip"

echo ""
echo "=== Done ==="
echo "Original: submission/stage1-p${PATCH_SIZE}-original.zip"
echo "TTA: submission/stage1-p${PATCH_SIZE}-tta.zip"
echo "Averaged TTA: submission/stage1-p${PATCH_SIZE}-avg${SAVE_TOP_K}-tta.zip"

echo ""
echo "=== Saving to GitHub ==="
git add -A
git commit -m "Auto-save: stage1 patch ${PATCH_SIZE} complete (best=${BEST_EPOCH})" || echo "(nothing to commit)"
git pull --rebase
git push
