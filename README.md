# HW4: Image Restoration (PromptIR)

NYCU Visual Recognition, Spring 2026

## Task
Restore degraded images (rain streaks, snow overlay) using a single PromptIR model.

## Setup
```bash
pip install -r requirements.txt
python prepare_data.py
```

## Training
```bash
python train_hw4.py --batch_size 8 --epochs 150 --lr 2e-4 --num_gpus 1
```

Recommended current experiment:
```bash
bash train_stage1.sh
```

If the larger patch run OOMs:
```bash
PATCH_SIZE=192 BATCH_SIZE=1 bash train_stage1.sh
```

## Inference & Submission
```bash
python inference.py checkpoints/promptir-epoch150.ckpt
```
Outputs `submission/pred.npz` with 100 restored images named `0.png`..`99.png`.

Generate a TTA submission from the same checkpoint:
```bash
python inference.py checkpoints/promptir-epoch150.ckpt --tta --output submission/stage1-tta.zip
```

## Links
- Slides: https://docs.google.com/presentation/d/12UUbuSWaAdC6sip9-5OXC2bM_RPSs5TacbyDxkQMQ9o
- Dataset: https://drive.google.com/drive/folders/1Q4qLPMCKdjn-iGgXV_8wujDmvDpSI1ul
- Competition: https://www.codabench.org/competitions/16215/

## Constraints
- Single model for both rain + snow
- No external data, no pretrained weights
- PromptIR architecture required

## Performance Snapshot

| Submission | Public PSNR | Notes |
| --- | ---: | --- |
| **ensemble-diverse.zip** | **31.84** | **Final selected submission, R9+R8+R11avg5+R11 prediction ensemble** |
| stage1-p256-tta-run9-l1mse.zip | 31.77 | Run 9 single-model best (L1+MSE $\lambda$=0.05) |
| ensemble-r9-r8-tta.zip | 31.73 | Run 9 + Run 8 TTA ensemble |
| stage1-p256-avg5-tta-run9-l1mse.zip | 31.76 | Run 9 top-5 checkpoint averaging TTA |
| stage1-p256-tta-run14-rainw125.zip | 31.69 | Run 14 rain loss weight 1.25 (did not improve) |
| stage1-p256-bs3-bf16-mixed-l1_mse0025-tta.zip | 31.39 | Run 13 bf16+bs3+$\lambda$=0.025 (failed recipe)
| stage1-p256-avg3-tta.zip | 31.73 | p256, PSNR-monitored top-3 checkpoint averaging, best epoch 149 |
| stage1-p256-tta.zip | 31.73 | p256, PSNR-monitored, best epoch 149 |
| stage1-p384-avg3-tta.zip | 31.50 | p384, EMA, top-3 checkpoint averaging, best epoch 109 |
| stage1-p384-tta.zip | 31.49 | p384, EMA, best epoch 109 |
| stage1-p192-tta.zip | 31.35 | p192, best epoch 144 |
| stage1-p192-original.zip | 30.93 | p192 no TTA |
| stage1-epoch149.zip | 30.52 | Initial baseline p128 |

Final selected submission: `submission/ensemble-diverse.zip` at **31.84 PSNR** (equal-weight prediction ensemble of Run 9 TTA + Run 8 TTA + Run 11 avg5 TTA + Run 11 TTA). Single-model best remains Run 9 at 31.77. All training-based improvements (loss weighting, EMA, task conditioning, bf16, batch size) failed to surpass this ensemble.
