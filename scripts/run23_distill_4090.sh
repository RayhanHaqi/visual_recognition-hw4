#!/usr/bin/env bash
# Distill single PromptIR on 4090 (after route + make_distill_targets).
set -euo pipefail

cd "$(dirname "$0")/.."

DISTILL_DIR="${DISTILL_DIR:-PromptIR/data/Distill}"
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-4}

if [ ! -f "${DISTILL_DIR}/labels.txt" ]; then
    echo "ERROR: run make_distill_targets.py first (${DISTILL_DIR}/labels.txt missing)"
    exit 1
fi

GPU_INDEX=$(python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("ERROR: CUDA not available")
for i in range(torch.cuda.device_count()):
    if "4090" in torch.cuda.get_device_name(i):
        print(i)
        raise SystemExit
if torch.cuda.device_count() == 1 and "4090" in torch.cuda.get_device_name(0):
    print(0)
    raise SystemExit
sys.exit("ERROR: no RTX 4090 found")
PY
)

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export GPU_IDS=0
unset DISABLE_CUDNN

RUN_NAME=${RUN_NAME:-run23-single-promptir-distilled} \
DISTILL_DIR="${DISTILL_DIR}" \
DE_TYPE="desnow derain" \
PATCH_SIZE=256 \
BATCH_SIZE="${BATCH_SIZE}" \
EPOCHS="${EPOCHS}" \
LOSS_TYPE=l1_mse \
MSE_WEIGHT=0.05 \
MONITOR=val_psnr \
SAVE_TOP_K=5 \
GRADIENT_CHECKPOINTING=full \
PRECISION=32 \
bash train_stage1.sh
