#!/usr/bin/env bash
# After hard power-off: infer derain (if needed) then train desnow on 4090 only.
#   cd ~/Rayhan/selectedtopics/HW4 && git pull
#   bash scripts/resume_strategy_c_4090.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Step 0: GPU check ==="
nvidia-smi -L || true
echo ""

if [ ! -f "submission/run21-derain-specialist-avg5-tta.zip" ]; then
    echo "=== Step 1: run21 inference ==="
    bash scripts/run21_infer_4090.sh
else
    echo "=== Step 1: skip (run21 avg5 TTA zip exists) ==="
fi

echo ""
echo "=== Step 2: run22 desnow train (~$((${EPOCHS:-100}))*5min, 4090 only) ==="
bash scripts/run22_desnow_4090.sh

echo ""
echo "=== Next (after run22 finishes) ==="
echo "  mkdir -p routes && printf '%s\\n' \$(printf 'rain\\n%.0s' {1..50}) \$(printf 'snow\\n%.0s' {1..50}) > routes/test_rain_snow.txt"
echo "  python route_specialists.py --rain_zip submission/run21-derain-specialist-avg5-tta.zip \\"
echo "    --snow_zip submission/run22-desnow-specialist-avg5-tta.zip \\"
echo "    --route_file routes/test_rain_snow.txt \\"
echo "    --output submission/specialist-routed-avg5-tta.zip"
echo "  python make_distill_targets.py --zip submission/specialist-routed-avg5-tta.zip \\"
echo "    --test_dir PromptIR/data/Test/degraded --output_dir PromptIR/data/Distill \\"
echo "    --route_file routes/test_rain_snow.txt"
echo "  DISTILL_DIR=PromptIR/data/Distill RUN_NAME=run23-single-promptir-distilled EPOCHS=100 BATCH_SIZE=4 \\"
echo "    bash scripts/run22_desnow_4090.sh  # edit: use train_stage1 with DE_TYPE=desnow derain + DISTILL_DIR"
