#!/usr/bin/env bash
# Desnow specialist on RTX 4090 only (skip broken / missing RTX 5090).
# Usage after hard power-off + SSH works:
#   cd ~/Rayhan/selectedtopics/HW4 && git pull
#   bash scripts/run22_desnow_4090.sh
set -euo pipefail

cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-4}
PATCH_SIZE=${PATCH_SIZE:-256}
RUN_NAME=${RUN_NAME:-run22-desnow-specialist}

echo "=== Select RTX 4090 (ignore 5090) ==="
GPU_INDEX=$(python - <<'PY'
import sys
try:
    import torch
except ImportError:
    print("ERROR: torch not found", file=sys.stderr)
    sys.exit(1)
if not torch.cuda.is_available():
    print("ERROR: CUDA not available", file=sys.stderr)
    sys.exit(1)
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    print(f"  GPU {i}: {name}", file=sys.stderr)
    if "4090" in name:
        print(i)
        sys.exit(0)
# Single-GPU box after 5090 depowered: use GPU 0 if it is the only 4090
if torch.cuda.device_count() == 1 and "4090" in torch.cuda.get_device_name(0):
    print(0)
    sys.exit(0)
print("ERROR: no RTX 4090 found", file=sys.stderr)
sys.exit(1)
PY
)

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export GPU_IDS=0
unset DISABLE_CUDNN

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (logical cuda:0)"
echo "Run: ${RUN_NAME}  epochs=${EPOCHS}  batch=${BATCH_SIZE}  patch=${PATCH_SIZE}"
echo ""

CKPT_DIR="checkpoints/${RUN_NAME}"
if compgen -G "${CKPT_DIR}/promptir-epoch*.ckpt" > /dev/null; then
    echo "ERROR: checkpoints already exist in ${CKPT_DIR}"
    echo "Rename RUN_NAME or move that folder before a fresh train."
    exit 1
fi

RUN_NAME="${RUN_NAME}" \
DE_TYPE=desnow \
PATCH_SIZE="${PATCH_SIZE}" \
BATCH_SIZE="${BATCH_SIZE}" \
EPOCHS="${EPOCHS}" \
LOSS_TYPE=l1_mse \
MSE_WEIGHT=0.05 \
MONITOR=val_psnr_desnow \
SAVE_TOP_K=5 \
GRADIENT_CHECKPOINTING=full \
PRECISION=32 \
bash train_stage1.sh
