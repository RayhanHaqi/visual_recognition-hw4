#!/usr/bin/env bash
# Finish run21 derain inference on RTX 4090 (training already done).
# Usage:
#   cd ~/Rayhan/selectedtopics/HW4 && git pull
#   bash scripts/run21_infer_4090.sh
set -euo pipefail

cd "$(dirname "$0")/.."

CKPT_DIR="checkpoints/run21-derain-specialist"
if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: missing ${CKPT_DIR} — train run21 first or restore checkpoints."
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

BEST_CKPT=$(python - "$CKPT_DIR" <<'PY'
import re
import sys
from pathlib import Path

metric = "val_psnr_derain"
ckpts = list(Path(sys.argv[1]).glob("promptir-epoch*.ckpt"))
missing = float("-inf")

def score(path):
    m = re.search(rf"{metric}=([0-9]+(?:\.[0-9]+)?)", path.name)
    return float(m.group(1)) if m else missing

for path in sorted(ckpts, key=score, reverse=True)[:1]:
    print(path)
PY
)

if [ -z "$BEST_CKPT" ]; then
    echo "ERROR: no checkpoint in ${CKPT_DIR}"
    exit 1
fi

echo "GPU: CUDA_VISIBLE_DEVICES=${GPU_INDEX}"
echo "Checkpoint: ${BEST_CKPT}"

python inference.py "$BEST_CKPT" --device cuda:0 \
    --output submission/run21-derain-specialist-original.zip

python inference.py "$BEST_CKPT" --device cuda:0 --tta \
    --output submission/run21-derain-specialist-tta.zip

mapfile -t AVG_CKPTS < <(python - "$CKPT_DIR" <<'PY'
import re
import sys
from pathlib import Path

metric = "val_psnr_derain"
ckpts = list(Path(sys.argv[1]).glob("promptir-epoch*.ckpt"))
missing = float("-inf")

def score(path):
    m = re.search(rf"{metric}=([0-9]+(?:\.[0-9]+)?)", path.name)
    return float(m.group(1)) if m else missing

for path in sorted(ckpts, key=score, reverse=True)[:5]:
    print(path)
PY
)

AVG_OUT="checkpoints/run21-derain-specialist-avg5.ckpt"
python average_checkpoints.py "${AVG_CKPTS[@]}" --output "$AVG_OUT"

python inference.py "$AVG_OUT" --device cuda:0 --tta \
    --output submission/run21-derain-specialist-avg5-tta.zip

echo ""
echo "Done. Use submission/run21-derain-specialist-avg5-tta.zip for routing."
