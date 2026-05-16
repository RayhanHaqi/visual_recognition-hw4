#!/bin/bash
set -euo pipefail

DIR="$(dirname "$0")"
RUN_NAME="$(date +%Y%m%d_%H%M)_runpod"

echo "=== [0/5] Install dependencies ==="
pip install -q -r "$DIR/requirements.txt"
pip install -q gdown

echo "=== [1/5] Clone PromptIR ==="
if [ ! -d "$DIR/PromptIR" ]; then
    git clone https://github.com/va1shn9v/PromptIR.git "$DIR/PromptIR"
fi

echo "=== [2/5] Download dataset ==="
cd "$DIR"
python -c "
import os, zipfile, shutil
from pathlib import Path
ROOT = Path('.').resolve()
data_dir = ROOT / 'data' / 'hw4_realse_dataset'
if not (data_dir.exists() and len(list(data_dir.rglob('*.png'))) > 0):
    zip_path = ROOT / 'hw4_dataset.zip'
    if not zip_path.exists():
        import gdown
        gdown.download('https://drive.google.com/uc?id=1bEIU9TZVQa-AF_z6JkOKaGp4wYGnqQ8w', str(zip_path), quiet=False)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(ROOT / 'data')
    zip_path.unlink(missing_ok=True)
    for child in list((ROOT / 'data').iterdir()):
        if child.is_dir() and child.name != 'hw4_realse_dataset':
            nested = child / 'hw4_realse_dataset'
            if nested.is_dir():
                shutil.move(str(nested), str(ROOT / 'data' / 'hw4_realse_dataset'))
            elif (child / 'train').is_dir():
                shutil.move(str(child), str(ROOT / 'data' / 'hw4_realse_dataset'))
            else:
                for item in child.iterdir():
                    shutil.move(str(item), str(ROOT / 'data' / item.name))
                child.rmdir()
    print('Dataset ready.')
else:
    print('Dataset already exists.')
"

echo "=== [3/5] Organize data into PromptIR format ==="
python "$DIR/prepare_data.py"

echo "=== [4/5] Stage 1: Training with validation ==="
python "$DIR/train_hw4.py" --gpu_ids 0 --epochs 150 --precision 32 --batch_size 8

BEST_CKPT=$(ls "$DIR/checkpoints/promptir-epoch"*.ckpt 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    echo "ERROR: Stage 1 produced no checkpoint"
    exit 1
fi
BEST_EPOCH=$(basename "$BEST_CKPT" | sed 's/promptir-epoch\([0-9]*\)\.ckpt/\1/')
echo "Best epoch: $BEST_EPOCH"

echo "=== [5/5] Stage 2: Retraining on all data for $BEST_EPOCH epochs ==="
python "$DIR/train_hw4.py" --gpu_ids 0 --epochs "$BEST_EPOCH" --merge_val --no_val --precision 32 --batch_size 8

echo "=== Inference: both stages ==="
python "$DIR/inference.py" "$BEST_CKPT" --output "$DIR/submission/${RUN_NAME}_stage1-epoch${BEST_EPOCH}.zip"

STAGE2_CKPT=$(ls "$DIR/checkpoints/promptir-epoch"*.ckpt 2>/dev/null | head -1)
python "$DIR/inference.py" "$STAGE2_CKPT" --output "$DIR/submission/${RUN_NAME}_stage2-epoch${BEST_EPOCH}-merged.zip"

echo ""
echo "=== Done ==="
echo "Stage 1: submission/${RUN_NAME}_stage1-epoch${BEST_EPOCH}.zip"
echo "Stage 2: submission/${RUN_NAME}_stage2-epoch${BEST_EPOCH}-merged.zip"

cd "$DIR"
git add -A
git commit -m "RunPod: done training $RUN_NAME" && git push origin main || echo "git push skipped"

RUNPOD_POD_ID="${RUNPOD_POD_ID:-}"
if [ -n "$RUNPOD_POD_ID" ]; then
    echo "Shutting down pod $RUNPOD_POD_ID..."
    runpodctl remove pod "$RUNPOD_POD_ID"
fi
