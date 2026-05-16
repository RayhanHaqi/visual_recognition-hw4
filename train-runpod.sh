#!/bin/bash
# Full RunPod pipeline: setup → train (two-stage) → inference → git push → kill pod.
# Usage: bash train-runpod.sh [bs] [lr] [epochs]
#   bs=8  lr=2e-4  epochs=150

BS=${1:-4}
LR=${2:-2e-4}
EPOCHS=${3:-150}
RUN_NAME="bs${BS}_lr${LR}_ep${EPOCHS}_runpod"

echo "=================================================="
echo "  RUNPOD PIPELINE: $RUN_NAME"
echo "=================================================="

if [ -z "$GH_TOKEN" ]; then
    read -rp "GitHub token: " GH_TOKEN
    echo
fi

if [ -n "$GH_TOKEN" ]; then
    echo "https://Rayhan:${GH_TOKEN}@github.com" > ~/.git-credentials
fi

echo "[1/5] Setup..."
python setup.py && \

echo "" && \
echo "[2/5] Stage 1: Training with validation..." && \
python train_hw4.py \
    --gpu_ids 0 --epochs "$EPOCHS" --precision 32 \
    --batch_size "$BS" --lr "$LR" && \

BEST_CKPT=$(ls checkpoints/promptir-epoch*.ckpt 2>/dev/null | head -1) && \
if [ -z "$BEST_CKPT" ]; then echo "ERROR: No checkpoint"; exit 1; fi && \
BEST_EPOCH=$(basename "$BEST_CKPT" | sed 's/promptir-epoch\([0-9]*\)\.ckpt/\1/') && \
echo "Best epoch: $BEST_EPOCH" && \

echo "" && \
echo "[3/5] Stage 2: Retraining on all data for ${BEST_EPOCH} epochs..." && \
python train_hw4.py \
    --gpu_ids 0 --epochs "$BEST_EPOCH" --merge_val --no_val --precision 32 \
    --batch_size "$BS" --lr "$LR" && \

echo "" && \
echo "[4/5] Generating submissions..." && \
python inference.py "$BEST_CKPT" --output "submission/${RUN_NAME}_stage1-epoch${BEST_EPOCH}.zip" && \

STAGE2_CKPT=$(ls checkpoints/promptir-epoch*.ckpt 2>/dev/null | head -1) && \
python inference.py "$STAGE2_CKPT" --output "submission/${RUN_NAME}_stage2-epoch${BEST_EPOCH}-merged.zip" && \
rm -rf ./checkpoints/* && \

echo "" && \
echo "[5/5] Saving to GitHub..." && \
git add -A && \
git commit -m "Auto-save: Done training $RUN_NAME" && \
git pull --rebase && \
git push && \

echo "Pipeline done!"

sleep 60

if [ -z "$RUNPOD_POD_ID" ]; then
    echo "RUNPOD_POD_ID not set — skipping pod deletion."
else
    echo "Deleting pod $RUNPOD_POD_ID..."
    runpodctl remove pod "$RUNPOD_POD_ID"
fi
