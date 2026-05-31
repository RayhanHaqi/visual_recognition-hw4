#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

PATCH_SIZE=${PATCH_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-1}
EPOCHS=${EPOCHS:-150}
GPU_IDS=${GPU_IDS:-0}
PRECISION=${PRECISION:-32}
MODEL_DIM=${MODEL_DIM:-48}
NUM_BLOCKS=${NUM_BLOCKS:-4,6,6,8}
NUM_REFINEMENT_BLOCKS=${NUM_REFINEMENT_BLOCKS:-4}
SAVE_TOP_K=${SAVE_TOP_K:-5}
CKPT_EVERY_N_EPOCHS=${CKPT_EVERY_N_EPOCHS:-5}
EMA=${EMA:-0}
EMA_DECAY=${EMA_DECAY:-0.9999}
MONITOR=${MONITOR:-val_psnr}
LOSS_TYPE=${LOSS_TYPE:-l1_mse}
MSE_WEIGHT=${MSE_WEIGHT:-0.05}
TASK_CONDITIONING=${TASK_CONDITIONING:-0}
SIPL_LITE=${SIPL_LITE:-0}
SIPL_START_ALPHA=${SIPL_START_ALPHA:-0.5}
SIPL_REFINE_WEIGHT=${SIPL_REFINE_WEIGHT:-0.5}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-none}
COMPILE=${COMPILE:-0}
DERAIN_OVERSAMPLE=${DERAIN_OVERSAMPLE:-1}
RAIN_LOSS_WEIGHT=${RAIN_LOSS_WEIGHT:-1.0}
PAIR_MIX_PROB=${PAIR_MIX_PROB:-0.0}
PAIR_MIX_ALPHA=${PAIR_MIX_ALPHA:-1.2}
HARD_PATCH_PROB=${HARD_PATCH_PROB:-0.0}
HARD_PATCH_TAU=${HARD_PATCH_TAU:-2.0}
COLOR_AUG_PROB=${COLOR_AUG_PROB:-0.0}
FORCE_AUG_NO_IDENTITY=${FORCE_AUG_NO_IDENTITY:-0}
AUX_DENOISE=${AUX_DENOISE:-0}
AUX_DENOISE_SIGMAS=${AUX_DENOISE_SIGMAS:-15,25,50}
DE_TYPE=${DE_TYPE:-desnow derain}
DISTILL_DIR=${DISTILL_DIR:-}
RUN_NAME=${RUN_NAME:-stage1-p${PATCH_SIZE}-bs${BATCH_SIZE}-${LOSS_TYPE}${MSE_WEIGHT//./}}
CKPT_DIR="checkpoints/${RUN_NAME}"

echo "=== Stage 1: Training with validation ==="
echo "Patch size: $PATCH_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Save top-k: $SAVE_TOP_K"
echo "Checkpoint every N epochs: $CKPT_EVERY_N_EPOCHS"
echo "Model dim: $MODEL_DIM"
echo "Num blocks: $NUM_BLOCKS"
echo "Num refinement blocks: $NUM_REFINEMENT_BLOCKS"
echo "EMA: $EMA"
echo "Monitor: $MONITOR"
echo "Loss type: $LOSS_TYPE"
echo "MSE weight: $MSE_WEIGHT"
echo "Task conditioning: $TASK_CONDITIONING"
echo "SIPL-lite: $SIPL_LITE"
echo "Gradient checkpointing: $GRADIENT_CHECKPOINTING"
echo "Compile: $COMPILE"
echo "Derain oversample: ${DERAIN_OVERSAMPLE}x"
echo "Rain loss weight: ${RAIN_LOSS_WEIGHT}"
echo "PairMix prob: ${PAIR_MIX_PROB}"
echo "PairMix alpha: ${PAIR_MIX_ALPHA}"
echo "Hard patch prob: ${HARD_PATCH_PROB}"
echo "Hard patch tau: ${HARD_PATCH_TAU}"
echo "Color aug prob: ${COLOR_AUG_PROB}"
echo "Force aug no identity: ${FORCE_AUG_NO_IDENTITY}"
echo "Aux denoise: ${AUX_DENOISE}"
echo "Aux denoise sigmas: ${AUX_DENOISE_SIGMAS}"
echo "DE_TYPE: ${DE_TYPE}"
echo "Distill dir: ${DISTILL_DIR:-<none>}"
echo "Run name: $RUN_NAME"

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
    --model_dim "$MODEL_DIM"
    --num_blocks "$NUM_BLOCKS"
    --num_refinement_blocks "$NUM_REFINEMENT_BLOCKS"
    --ckpt_dir "$CKPT_DIR"
    --save_top_k "$SAVE_TOP_K"
    --every_n_epochs "$CKPT_EVERY_N_EPOCHS"
    --monitor "$MONITOR"
    --loss_type "$LOSS_TYPE"
    --mse_weight "$MSE_WEIGHT"
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING"
    --derain_oversample "$DERAIN_OVERSAMPLE"
    --rain_loss_weight "$RAIN_LOSS_WEIGHT"
    --pair_mix_prob "$PAIR_MIX_PROB"
    --pair_mix_alpha "$PAIR_MIX_ALPHA"
    --hard_patch_prob "$HARD_PATCH_PROB"
    --hard_patch_tau "$HARD_PATCH_TAU"
    --color_aug_prob "$COLOR_AUG_PROB"
    --aux_denoise "$AUX_DENOISE"
    --aux_denoise_sigmas "$AUX_DENOISE_SIGMAS"
    --de_type $DE_TYPE
)
if [ -n "$DISTILL_DIR" ]; then
    TRAIN_ARGS+=(--distill_dir "$DISTILL_DIR")
fi
if [ "$EMA" = "1" ]; then
    TRAIN_ARGS+=(--ema --ema_decay "$EMA_DECAY")
fi
if [ "$TASK_CONDITIONING" = "1" ]; then
    TRAIN_ARGS+=(--task_conditioning)
fi
if [ "$SIPL_LITE" = "1" ]; then
    TRAIN_ARGS+=(--sipl_lite --sipl_start_alpha "$SIPL_START_ALPHA" --sipl_refine_weight "$SIPL_REFINE_WEIGHT")
fi
if [ "$COMPILE" = "1" ]; then
    TRAIN_ARGS+=(--compile)
fi
if [ "$FORCE_AUG_NO_IDENTITY" = "1" ]; then
    TRAIN_ARGS+=(--force_aug_no_identity)
fi

python train_hw4.py "${TRAIN_ARGS[@]}"

INFER_ARGS=(
    --model_dim "$MODEL_DIM"
    --num_blocks "$NUM_BLOCKS"
    --num_refinement_blocks "$NUM_REFINEMENT_BLOCKS"
)

BEST_CKPT=$(python - "$CKPT_DIR" "$MONITOR" <<'PY'
import re
import sys
from pathlib import Path

ckpts = list(Path(sys.argv[1]).glob("promptir-epoch*.ckpt"))
metric = sys.argv[2]
reverse = metric.startswith("val_psnr")
missing = float("-inf") if reverse else float("inf")
def metric_value(path):
    match = re.search(rf'{re.escape(metric)}=([0-9]+(?:\.[0-9]+)?)', path.name)
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
python inference.py "$BEST_CKPT" "${INFER_ARGS[@]}" --output "submission/${RUN_NAME}-original.zip"

echo ""
echo "=== TTA inference ==="
python inference.py "$BEST_CKPT" "${INFER_ARGS[@]}" --tta --output "submission/${RUN_NAME}-tta.zip"

echo ""
echo "=== Averaging top checkpoints ==="
mapfile -t AVG_CKPTS < <(python - "$CKPT_DIR" "$SAVE_TOP_K" "$MONITOR" <<'PY'
import re
import sys
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
limit = int(sys.argv[2])
metric = sys.argv[3]
reverse = metric.startswith("val_psnr")
ckpts = list(ckpt_dir.glob("promptir-epoch*.ckpt"))
missing = float("-inf") if reverse else float("inf")
def metric_value(path):
    match = re.search(rf'{re.escape(metric)}=([0-9]+(?:\.[0-9]+)?)', path.name)
    return float(match.group(1)) if match else missing
for path in sorted(ckpts, key=metric_value, reverse=reverse)[:limit]:
    print(path)
PY
)
AVG_CKPT="checkpoints/${RUN_NAME}-avg${SAVE_TOP_K}.ckpt"
python average_checkpoints.py "${AVG_CKPTS[@]}" --output "$AVG_CKPT"

echo ""
echo "=== Averaged TTA inference ==="
python inference.py "$AVG_CKPT" "${INFER_ARGS[@]}" --tta --output "submission/${RUN_NAME}-avg${SAVE_TOP_K}-tta.zip"

echo ""
echo "=== Done ==="
echo "Original: submission/${RUN_NAME}-original.zip"
echo "TTA: submission/${RUN_NAME}-tta.zip"
echo "Averaged TTA: submission/${RUN_NAME}-avg${SAVE_TOP_K}-tta.zip"

echo ""
echo "=== Saving to GitHub ==="
git add -A
git commit -m "Auto-save: ${RUN_NAME} complete (best=${BEST_EPOCH})" || echo "(nothing to commit)"
git pull --rebase
git push
