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
| **stage1-p256-tta-run9-l1mse.zip** | **31.77** | **Final selected submission, L1+MSE ($\lambda$=0.05), p256, PSNR-monitored, best epoch 139** |
| stage1-p256-avg5-tta-run9-l1mse.zip | 31.76 | Run 9 top-5 checkpoint averaging TTA |
| stage1-p256-original-run9-l1mse.zip | 31.31 | Run 9 no TTA |
| stage1-p256-tta-run12-l1mse.zip | 31.72 | Run 12 L1+MSE ($\lambda$=0.10), p256, best epoch 139 |
| stage1-p256-avg5-tta-run12-l1mse.zip | 31.72 | Run 12 top-5 checkpoint averaging TTA |
| stage1-p256-original-run12-l1mse.zip | 31.31 | Run 12 no TTA |
| stage1-p256-tta-run11-l1mse-ema.zip | 31.72 | Run 11 L1+MSE ($\lambda$=0.05) + EMA |
| stage1-p256-avg5-tta-run11-l1mse-ema.zip | 31.73 | Run 11 EMA + top-5 averaging |
| stage1-p256-tta-run10-task.zip | 31.19 | Run 10 task-conditioned TTA |
| stage1-p256-avg3-tta.zip | 31.73 | p256, PSNR-monitored top-3 checkpoint averaging, best epoch 149 |
| stage1-p256-tta.zip | 31.73 | p256, PSNR-monitored, best epoch 149 |
| stage1-p384-avg3-tta.zip | 31.50 | p384, EMA, top-3 checkpoint averaging, best epoch 109 |
| stage1-p384-tta.zip | 31.49 | p384, EMA, best epoch 109 |
| stage1-p192-tta.zip | 31.35 | p192, best epoch 144 |
| stage1-p192-original.zip | 30.93 | p192 no TTA |
| stage1-epoch149.zip | 30.52 | Initial baseline p128 |

Final selected submission: `submission/stage1-p256-tta-run9-l1mse.zip` at **31.77 PSNR**. Run 9 (L1+MSE $\lambda$=0.05) remains the best configuration. Run 12 ($\lambda$=0.10) matched Run 11 at 31.72 but did not surpass the Run 9 baseline. Run 10 (task conditioning, 31.19) was the worst performer.
