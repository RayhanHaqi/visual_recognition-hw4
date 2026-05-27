# Run 13 + Prediction Ensemble — Deadline-Efficiency Design

**Date:** 2026-05-27
**Constraint:** One full RTX 4090 training run remaining before June 2 deadline.
**Baseline:** Run 9 at 31.77 public PSNR (L1+MSE, lambda=0.05, p256, PSNR-monitored, best epoch 139).

## Goal

Maximize probability of exceeding 31.77 PSNR given one training run plus zero-GPU-time CPU ensembling of existing top submissions.

## Design

### Phase 1: Prediction Ensemble (CPU-only, immediate)

Averaging predictions from high-scoring checkpoints with diverse training configurations can exploit complementary restoration patterns.

**Candidates for ensemble:**

| Submission | Public PSNR | Configuration |
|---|---|---|
| Run 9 TTA | 31.77 | L1+0.05MSE, best epoch 139 |
| Run 9 avg5 TTA | 31.76 | Run 9 top-5 checkpoint avg |
| Run 8 TTA | 31.73 | L1 loss, PSNR monitor, epoch 149 |
| Run 11 avg5 TTA | 31.73 | L1+0.05MSE + EMA, epoch 129 |
| Run 12 TTA | 31.72 | L1+0.10MSE, epoch 139 |

**Ensemble combinations (all CPU-only, no GPU needed):**
- **Conservative:** Run 9 TTA + Run 9 avg5 TTA (most correlated, safest)
- **Diverse:** Run 9 TTA + Run 8 TTA + Run 11 avg5 TTA + Run 12 TTA
- **Top-2:** Run 9 TTA + Run 8 TTA

**Mechanism:** Extract `pred.npz` from each submission zip, average corresponding arrays element-wise (clamp to [0,255]), and zip into new submission. No checkpoint needed.

**Decision rule:** Submit to CodaBench only if quota permits (each submission reveals public score). Diverse ensemble runs highest risk of blur but also highest chance of exploiting complementary strengths.

### Phase 2: Run 13 Training (single 4090 run)

Train with `LOSS_TYPE=l1_mse`, `MSE_WEIGHT=0.025`, p256, PSNR-monitored, 150 epochs.

**Rationale:**
- `MSE_WEIGHT=0.00` (pure L1): Run 8 scored 31.73
- `MSE_WEIGHT=0.05`: Run 9 scored 31.77
- `MSE_WEIGHT=0.10`: Run 12 scored 31.72
- `MSE_WEIGHT=0.025`: untested — only remaining sensible point in the loss sweep

**Command:**
```
python train_hw4.py \
  --gpu_ids 0 --epochs 150 --precision 32 --batch_size 1 --patch_size 256 \
  --ckpt_dir checkpoints/run13-p256-l1mse0025 \
  --save_top_k 5 --monitor val_psnr \
  --loss_type l1_mse --mse_weight 0.025
```

**After training:**
1. Generate original, TTA, and avg5 TTA submissions
2. Submit TTA to CodaBench first
3. Submit avg5 TTA only if quota permits

**Decision rules:**

| Condition | Action |
|---|---|
| val_psnr_best < Run 9's 30.749 | Still generate TTA sub; submit only if quota comfortable |
| Public TTA > 31.77 | Replace Run 9 as final submission |
| Public TTA 31.74–31.77 | Keep both candidates noted; Run 9 remains safer |
| Public TTA < 31.74 | Discard Run 13 for final |

### Phase 3: Ensemble from Run 13

If Run 13 trains successfully, also CPU-ensemble Run 13 outputs into existing candidates (e.g., Run 9 TTA + Run 13 TTA). This is zero-risk: if it scores lower than best single, discard it.

## What Not To Do

- EMA (already lost in Runs 7, 11)
- Task conditioning (collapsed in Run 10, 31.19)
- Stage 2 merged-data retraining (consistently worse)
- SIPL-lite inference (lost in Run 11)
- p384, p320 (worse than p256)
- Charbonnier loss (equivalent to L1, no known gain)
- SWA / BN update_bn (PromptIR has no BN layers; EMA already lost)
- Scheduler changes (need more than one run)

## Fallback

Run 9 at 31.77 remains the final submission unless new public score beats it.

## Submission Quota Strategy

Submit in this priority order to maximize information per submission:

1. **Run 13 TTA** (new candidate, highest priority)
2. **Diverse prediction ensemble** (zero GPU, plausible 0.02–0.05 gain)
3. **Run 13 avg5 TTA** (only if #1 looks promising locally)
4. **Run 9 + Run 13 TTA ensemble** (CPU, zero risk to keep)
