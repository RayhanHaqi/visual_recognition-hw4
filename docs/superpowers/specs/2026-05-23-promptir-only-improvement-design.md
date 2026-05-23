# PromptIR-Only Improvement Design

## Goal

Improve HW4 CodaBench PSNR while staying inside the assignment rule that PromptIR must remain the model. The design avoids direct replacement with Restormer, NAFNet, MPRNet, Uformer, TransWeather, diffusion, or other non-PromptIR backbones.

## Current Evidence

- Current best public score is `31.73` from `stage1-p256-tta.zip` and `stage1-p256-avg3-tta.zip`.
- p256 PSNR-monitored training improved over the older p256 run, but TTA and checkpoint averaging now appear saturated.
- All train/val/test images are `256x256`, so larger `PATCH_SIZE` values do not add context.
- Current training uses pure `L1Loss`; public metric is PSNR.
- `HW4TrainDataset` returns `de_id` for rain/snow, but `train_hw4.py` currently ignores it.
- The assignment slides require PromptIR, but allow component/module modifications.

## Scope

This design permits only PromptIR-compatible changes:

- Keep `net.model.PromptIR` as the core restoration network.
- Add lightweight wrapper modules around PromptIR when needed.
- Use only provided degraded/clean pairs and labels derivable from provided paths.
- Do not use external data, pretrained weights, pretrained feature extractors, GAN/perceptual losses, CLIP, DINO, VGG, or non-PromptIR replacement backbones.

## Approach 1: Measurement First

Add separate rain and snow validation PSNR/L1 logging before changing model behavior. The current single aggregate metric hides whether gains or regressions come from derain, desnow, or both. The validation dataset can return `de_id` based on the task directory, and `validation_step` can log `val_psnr_derain`, `val_psnr_desnow`, `val_loss_derain`, and `val_loss_desnow`.

This is low risk and helps reject experiments that improve one degradation by sacrificing the other.

## Approach 2: Task-Aware PromptIR Conditioning

Use the known training label `de_id` to provide explicit rain/snow conditioning while keeping PromptIR as the restoration backbone. The wrapper applies a small learnable per-task affine modulation to the input image before PromptIR and optionally a small per-task residual bias after PromptIR.

At training and validation, the label is known from the dataset. At test time, the label is unknown, so inference should support three modes:

- `blind`: current baseline, no conditioning if disabled.
- `task=derain` or `task=desnow`: force one task condition.
- `task=both`: run both task conditions and average outputs.

The first leaderboard-safe submission should use `task=both` if validation shows the conditioned model improves aggregate PSNR. This avoids needing a separate classifier and preserves the single-model constraint.

## Approach 3: PSNR-Aligned Losses

Add a configurable loss option:

- `l1`: current baseline.
- `charbonnier`: smooth L1-like loss often used in restoration.
- `l1_mse`: `L1 + mse_weight * MSE`, where `mse_weight` defaults to `0.05`.

The goal is to align training slightly better with PSNR without overfitting to pixel-wise MSE. This is low-risk and independent of task conditioning.

## Approach 4: SIPL-Lite PromptIR Wrapper

True SIPL modifies internal feature fusion using privileged clean-image information. Because PromptIR source may be ignored or absent in this checkout, the near-deadline PromptIR-compliant version should be a wrapper that does not replace the backbone.

SIPL-lite uses shared PromptIR weights in two passes:

1. First pass restores the degraded image: `restored0 = PromptIR(x)`.
2. During training, the second pass input is a blend of first-pass output and clean image: `second_input = (1 - alpha) * restored0.detach() + alpha * clean`, where `alpha` decays to zero.
3. During inference, the second pass input is `restored0` because clean GT is unavailable.
4. Training loss is `loss(restored0, clean) + refine_weight * loss(restored1, clean)`.

This is not a full SIPL reproduction, but it tests the key inference idea: make PromptIR robust to refining its own preliminary restoration. It is more expensive because it performs two PromptIR passes.

## Experiment Priority

1. Per-task validation logging.
2. Task-aware conditioning with `task=both` inference.
3. `charbonnier` and `l1_mse` loss experiments on the known p256 recipe.
4. SIPL-lite two-pass refinement if the first two changes do not produce enough improvement.

## Success Criteria

- Local validation aggregate `val_psnr` improves by at least `0.05 dB` over the p256 PSNR-monitored curve before CodaBench submission.
- Per-task metrics do not show a collapse in either derain or desnow.
- Generated zips still contain exactly `pred.npz` with 100 keys, `uint8`, shape `(3, 256, 256)`.
- PromptIR remains the core model and all code/report wording clearly describes changes as PromptIR modules/wrappers.

## Non-Goals

- No non-PromptIR backbone replacement.
- No external data or pretrained weights.
- No synthetic extra data from outside the provided dataset.
- No GAN, VGG, CLIP, DINO, or other pretrained perceptual guidance.
- No rerun of failed Stage 2 merged-data training from scratch.

## Risks

- Task conditioning can overfit the visible validation split and fail on hidden/public if test distribution differs.
- `task=both` doubles inference cost, though only 100 test images are required.
- SIPL-lite doubles training compute and may degrade by repeatedly over-cleaning already-restored images.
- Local validation PSNR has imperfect correlation with public CodaBench; use it as a filter, not final proof.
