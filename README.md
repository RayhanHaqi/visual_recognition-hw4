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
| stage1-p192-tta.zip | 31.35 | Current best, patch 192, best epoch 144, submitted 2026-05-19 |
| stage1-p192-tta.zip | 31.35 | 200-epoch rerun, same best epoch 144, submitted 2026-05-20 |
| stage1-p192-original.zip | 30.93 | Same checkpoint without TTA, submitted 2026-05-19 |
| stage1-p192-original.zip | 30.93 | 200-epoch rerun, same best epoch 144, submitted 2026-05-20 |
| stage1-epoch149.zip | 30.52 | Best known public score, submitted 2026-05-17 |
| stage1-epoch149.zip | 30.23 | Current scripts, submitted 2026-05-18 |
| stage2-epoch149-merged.zip | 21.69 | Merged-data Stage 2 degraded performance |

Current final candidate: `submission/stage1-p192-tta.zip` with public PSNR 31.35.
The 200-epoch p192 rerun did not improve public PSNR over the 150-epoch run.
