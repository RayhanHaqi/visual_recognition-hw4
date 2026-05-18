#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

PATCH_SIZE=${PATCH_SIZE:-192}
BATCH_SIZE=${BATCH_SIZE:-2}
EPOCHS=${EPOCHS:-150}
GPU_IDS=${GPU_IDS:-0}
PRECISION=${PRECISION:-32}
RUN_NAME="stage1-p${PATCH_SIZE}-bs${BATCH_SIZE}"
CKPT_DIR="checkpoints/${RUN_NAME}"

echo "=== Stage 1: Training with validation ==="
echo "Patch size: $PATCH_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"

python train_hw4.py \
    --gpu_ids "$GPU_IDS" \
    --epochs "$EPOCHS" \
    --precision "$PRECISION" \
    --batch_size "$BATCH_SIZE" \
    --patch_size "$PATCH_SIZE" \
    --ckpt_dir "$CKPT_DIR"

BEST_CKPT=$(ls "$CKPT_DIR"/promptir-epoch*.ckpt 2>/dev/null | head -1)
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
echo "=== Done ==="
echo "Original: submission/stage1-p${PATCH_SIZE}-original.zip"
echo "TTA: submission/stage1-p${PATCH_SIZE}-tta.zip"

echo ""
echo "=== Saving to GitHub ==="
git add -A
git commit -m "Auto-save: stage1 patch ${PATCH_SIZE} complete (best=${BEST_EPOCH})" || echo "(nothing to commit)"
git pull --rebase
git push
