#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Stage 1: Training with validation to find best epoch ==="
python train_hw4.py --gpu_ids 1 --epochs 150 --precision 32 --batch_size 8

# Find best epoch from checkpoint filename
BEST_CKPT=$(ls checkpoints/promptir-epoch*.ckpt 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    echo "ERROR: No checkpoint found"
    exit 1
fi

BEST_EPOCH=$(basename "$BEST_CKPT" | grep -oP 'epoch=\K[0-9]+' | head -1)
if [ -z "$BEST_EPOCH" ]; then
    BEST_EPOCH=$(basename "$BEST_CKPT" | sed 's/.*epoch\([0-9]*\)\.ckpt/\1/')
fi
echo "Best epoch: $BEST_EPOCH"

echo ""
echo "=== Stage 1 submission ==="
python inference.py "$BEST_CKPT" --output "submission/stage1-epoch${BEST_EPOCH}.zip"

echo ""
echo "=== Stage 2: Retraining on all data (train+val) for $BEST_EPOCH epochs ==="
python train_hw4.py --gpu_ids 1 --epochs "$BEST_EPOCH" --merge_val --no_val --precision 32 --batch_size 8 --lr 5e-5

STAGE2_CKPT=$(ls checkpoints/promptir-epoch*.ckpt 2>/dev/null | head -1)
if [ -z "$STAGE2_CKPT" ]; then
    echo "ERROR: No stage 2 checkpoint found"
    exit 1
fi

echo ""
echo "=== Stage 2 submission ==="
python inference.py "$STAGE2_CKPT" --output "submission/stage2-epoch${BEST_EPOCH}-merged.zip"

echo ""
echo "=== Done ==="
echo "Stage 1: submission/stage1-epoch${BEST_EPOCH}.zip"
echo "Stage 2: submission/stage2-epoch${BEST_EPOCH}-merged.zip"

echo ""
echo "=== Saving to GitHub ==="
git pull --rebase
git add -A
git commit -m "Auto-save: two-stage training complete (best=${BEST_EPOCH})" || echo "(nothing to commit)"
git push
