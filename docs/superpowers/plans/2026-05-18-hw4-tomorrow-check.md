# HW4 Tomorrow Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume HW4 tomorrow, verify the current training output, submit the best candidate to CodaBench, and prepare the final E3 deliverables.

**Architecture:** The current pipeline trains PromptIR from scratch for all-in-one derain + desnow restoration, generates CodaBench zips containing `pred.npz`, and tracks performance using validation loss plus CodaBench PSNR. Stage 1 with validation is currently the safer candidate because previous Stage 2 merged-data retraining degraded public PSNR.

**Tech Stack:** Python, PyTorch Lightning, PromptIR, TensorBoard logs, CodaBench submission zip format.

---

## Current Status As Of 2026-05-18

- Training is currently running with the current scripts. Do not interrupt it unless it crashes or clearly stalls.
- Homework 4 deadline: 2026-06-02 23:59.
- CodaBench submission format: zip file can have any name, but it must contain exactly `pred.npz`.
- `pred.npz` must contain keys `0.png` through `99.png`; each value should be `uint8` with shape `(3, H, W)`.
- Dataset is prepared locally: raw train has `3200` degraded + `3200` clean images, test has `100` images.
- PromptIR split is prepared: `1440` train + `160` val per rain/snow type.
- Existing valid CodaBench-format zips:
  - `submission/stage1-epoch09.zip`
  - `submission/stage1-epoch149.zip`
  - `submission/stage2-epoch09-merged.zip`
  - `submission/stage2-epoch149-merged.zip`
- Previous notes indicate Stage 1 epoch 149 got about `30.52` PSNR, while Stage 2 degraded to about `24.93` PSNR.
- No report PDF was found yet.
- README still needs a `Performance Snapshot` section.

## Important Rule For Tomorrow

- Treat CodaBench as the source of truth.
- If Stage 2 is worse again, use the best Stage 1 zip for the final competition submission.
- Do not assume merged train+val Stage 2 is better just because it uses more data.

### Task 1: Check Whether Training Finished

**Files:**
- Read: `log/hw4_promptir/`
- Read: `submission/`
- Read: `checkpoints/`

- [ ] **Step 1: Check GPU and active process state**

Run:
```bash
nvidia-smi
```

Expected:
```text
If training is still running, GPU memory/utilization should show the Python training process.
If training finished, no training Python process should be consuming most GPU memory.
```

- [ ] **Step 2: Check new submission artifacts**

Run:
```bash
ls -lh submission
```

Expected:
```text
Look for a new stage1 or stage2 zip created after the current training run.
```

- [ ] **Step 3: Check checkpoint artifacts if they still exist**

Run:
```bash
ls -lh checkpoints
```

Expected:
```text
There may be one or more PromptIR .ckpt files if training has not cleaned them up.
```

### Task 2: Validate Any New Submission Zip Before Upload

**Files:**
- Read: `submission/*.zip`

- [ ] **Step 1: Run the zip validation command**

Replace `<ZIP>` with the newest candidate zip.

Run:
```bash
python3 - <<'PY'
from pathlib import Path
import sys
import zipfile
import numpy as np

zip_path = Path('<ZIP>')
with zipfile.ZipFile(zip_path) as zf:
    names = zf.namelist()
    print('entries:', names)
    assert names == ['pred.npz'], names
    with zf.open('pred.npz') as f:
        npz = np.load(f)
        keys = sorted(npz.files, key=lambda x: int(x.split('.')[0]))
        print('count:', len(keys))
        print('first:', keys[:3])
        print('last:', keys[-3:])
        assert len(keys) == 100
        assert keys[0] == '0.png'
        assert keys[-1] == '99.png'
        for key in keys:
            arr = npz[key]
            assert arr.dtype == np.uint8, (key, arr.dtype)
            assert arr.ndim == 3, (key, arr.shape)
            assert arr.shape[0] == 3, (key, arr.shape)
print('valid:', zip_path)
PY
```

Expected:
```text
entries: ['pred.npz']
count: 100
first: ['0.png', '1.png', '2.png']
last: ['97.png', '98.png', '99.png']
valid: <ZIP>
```

### Task 3: Submit And Compare On CodaBench

**Files:**
- Upload: best candidate zip from `submission/`

- [ ] **Step 1: Submit the strongest known Stage 1 candidate first**

Use:
```text
submission/stage1-epoch149.zip
```

Expected:
```text
CodaBench run finishes and reports a public PSNR score.
```

- [ ] **Step 2: Submit the newest Stage 1 candidate from the current run if it exists**

Use:
```text
The newest submission/stage1-epoch*.zip created by the current training run.
```

Expected:
```text
Record the CodaBench public PSNR score and compare it against 30.52.
```

- [ ] **Step 3: Submit Stage 2 only after Stage 1 has a score**

Use:
```text
The newest submission/stage2-epoch*-merged.zip created by the current training run.
```

Expected:
```text
If Stage 2 score is lower than Stage 1, keep Stage 1 as final.
If Stage 2 score is higher than Stage 1, keep Stage 2 as final.
```

- [ ] **Step 4: Add the chosen best submission to the leaderboard**

Expected:
```text
The best public PSNR appears on the CodaBench leaderboard.
```

### Task 4: Record Results Locally

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`

- [ ] **Step 1: Update the run table in `AGENTS.md`**

Add a new row with:
```markdown
| Run 3 | <stage1_psnr> | <stage2_psnr_or_dash> | <best_epoch> | Current scripts, checked on CodaBench 2026-05-19 |
```

- [ ] **Step 2: Add README performance snapshot**

Add this section to `README.md` with real scores:
```markdown
## Performance Snapshot

| Submission | Public PSNR | Notes |
| --- | ---: | --- |
| Stage 1 epoch 149 | 30.52 | Strongest known local candidate before 2026-05-19 check |
| Current run Stage 1 | <score> | CodaBench public score from 2026-05-19 |
| Current run Stage 2 | <score> | Use only if better than Stage 1 |
```

### Task 5: Start Final Report

**Files:**
- Create: `report/<STUDENT_ID>_HW4.md` or draft directly in PDF tooling
- Final E3 file required: `<STUDENT_ID>_HW4.pdf`

- [ ] **Step 1: Draft the required report sections**

Use these sections:
```markdown
# Homework 4: Image Restoration with PromptIR

## Introduction

## Method

## PromptIR Architecture

## Training Details

## Experiments and Results

## Discussion

## References
```

- [ ] **Step 2: Include the required PromptIR discussion**

The report must explain:
```text
PromptIR key design/contribution, why it is suitable for all-in-one restoration, any modifications made, and proper citation of the PromptIR paper/GitHub source.
```

- [ ] **Step 3: Include the Stage 2 finding**

Write the finding clearly:
```text
Merged-data Stage 2 did not consistently improve CodaBench PSNR. The likely reason is that removing validation removes model-selection feedback, and training for the same selected epoch on merged data can still overfit or shift the model away from the best public/private distribution. Therefore, the final submission uses the stage with the best CodaBench score.
```

### Task 6: Prepare E3 Zip Later

**Files:**
- Create: `<STUDENT_ID>_HW4.zip`

- [ ] **Step 1: Include code and report only**

Expected zip contents:
```text
<STUDENT_ID>_HW4.pdf
*.py files
README.md
requirements.txt
train_twostage.sh
Any needed source folders, excluding data and checkpoints
```

- [ ] **Step 2: Exclude forbidden files**

Do not include:
```text
data/
PromptIR/data/
checkpoints/
*.ckpt
*.pth
raw dataset files
```

## Final Decision Rule

- Final CodaBench submission should be the zip with the best public PSNR after tomorrow's comparison.
- If scores are close, prefer the Stage 1 validation-selected model because it has already shown better stability than Stage 2.
- If the current training run fails, fall back to `submission/stage1-epoch149.zip`.
