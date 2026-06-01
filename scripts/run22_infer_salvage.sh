#!/usr/bin/env bash
# Inference-only: reuse run22 desnow checkpoints (e.g. 5090 train stopped ~epoch 112).
# Does NOT retrain. Run on 4090 after SSH recovery.
#
# Usage:
#   cd ~/Rayhan/selectedtopics/HW4 && git pull
#   CKPT_DIR=checkpoints/run22-desnow-specialist-5090 bash scripts/run22_infer_salvage.sh
#   # or:
#   CKPT_DIR=checkpoints/run22-desnow-specialist bash scripts/run22_infer_salvage.sh
set -euo pipefail

cd "$(dirname "$0")/.."

CKPT_DIR="${CKPT_DIR:-checkpoints/run22-desnow-specialist-5090}"
MONITOR="${MONITOR:-val_psnr_desnow}"
RUN_TAG="${RUN_TAG:-run22-desnow-specialist-salvage}"
TOP_K="${TOP_K:-5}"

if ! compgen -G "${CKPT_DIR}/promptir-epoch*.ckpt" > /dev/null; then
    echo "ERROR: no checkpoints in ${CKPT_DIR}"
    echo "List dirs: ls -la checkpoints/"
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

echo "Checkpoint dir: ${CKPT_DIR}"
echo "Monitor metric: ${MONITOR}"
echo "GPU: CUDA_VISIBLE_DEVICES=${GPU_INDEX}"
echo ""

BEST_CKPT=$(python - "$CKPT_DIR" "$MONITOR" <<'PY'
import re
import sys
from pathlib import Path

ckpt_dir, metric = sys.argv[1], sys.argv[2]
ckpts = list(Path(ckpt_dir).glob("promptir-epoch*.ckpt"))
missing = float("-inf")

def score(path):
    m = re.search(rf"{re.escape(metric)}=([0-9]+(?:\.[0-9]+)?)", path.name)
    return float(m.group(1)) if m else missing

for path in sorted(ckpts, key=score, reverse=True)[:1]:
    print(path)
PY
)

echo "Best checkpoint: ${BEST_CKPT}"
ls -lh "${CKPT_DIR}"/promptir-epoch*.ckpt | tail -10

python inference.py "$BEST_CKPT" --device cuda:0 \
    --output "submission/${RUN_TAG}-original.zip"

python inference.py "$BEST_CKPT" --device cuda:0 --tta \
    --output "submission/${RUN_TAG}-tta.zip"

mapfile -t AVG_CKPTS < <(python - "$CKPT_DIR" "$MONITOR" "$TOP_K" <<'PY'
import re
import sys
from pathlib import Path

ckpt_dir, metric, top_k = sys.argv[1], sys.argv[2], int(sys.argv[3])
ckpts = list(Path(ckpt_dir).glob("promptir-epoch*.ckpt"))
missing = float("-inf")

def score(path):
    m = re.search(rf"{re.escape(metric)}=([0-9]+(?:\.[0-9]+)?)", path.name)
    return float(m.group(1)) if m else missing

for path in sorted(ckpts, key=score, reverse=True)[:top_k]:
    print(path)
PY
)

AVG_OUT="checkpoints/${RUN_TAG}-avg${TOP_K}.ckpt"
python average_checkpoints.py "${AVG_CKPTS[@]}" --output "$AVG_OUT"

python inference.py "$AVG_OUT" --device cuda:0 --tta \
    --output "submission/${RUN_TAG}-avg${TOP_K}-tta.zip"

echo ""
echo "Done. For routing use:"
echo "  submission/${RUN_TAG}-avg${TOP_K}-tta.zip"
echo ""
echo "If val_psnr_desnow in filenames is low (<27), consider retraining on 4090:"
echo "  bash scripts/run22_desnow_4090.sh"
