# Run 13 + Prediction Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create prediction ensembles from existing top submissions and train Run 13 (MSE_WEIGHT=0.025) to attempt beating Run 9's 31.77 PSNR.

**Architecture:** Three phases: (1) CPU-only prediction ensemble from existing submission zips, (2) single 4090 training run with MSE_WEIGHT=0.025, (3) Run 13 inference + cross-run ensembles. Phases 1 and 2 are independent and can run in parallel.

**Tech Stack:** Python, numpy, torch, train_hw4.py, inference.py

---

### Task 1: Create Prediction Ensemble Script

**Files:**
- Create: `ensemble_predictions.py`

- [ ] **Step 1: Write the ensemble script**

```python
import argparse
import zipfile
import tempfile
import os
import numpy as np

def load_pred_npz(zip_path):
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmp)
        return dict(np.load(os.path.join(tmp, 'pred.npz')))

def save_pred_npz(pred_dict, output_path):
    os.makedirs("submission", exist_ok=True)
    npz_path = "submission/pred.npz"
    np.savez_compressed(npz_path, **pred_dict)
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(npz_path, "pred.npz")
    os.remove(npz_path)
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zips", nargs="+", help="Submission zip files to ensemble")
    parser.add_argument("--output", required=True, help="Output zip path")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Per-zip weights (default: equal)")
    args = parser.parse_args()

    if args.weights is not None and len(args.weights) != len(args.zips):
        raise ValueError("Number of weights must match number of zips")

    all_preds = [load_pred_npz(z) for z in args.zips]
    keys = sorted(all_preds[0].keys())
    for pred in all_preds:
        if sorted(pred.keys()) != keys:
            raise ValueError("Zip keys do not match across submissions")

    weights = args.weights or [1.0 / len(args.zips)] * len(args.zips)

    ensemble = {}
    for key in keys:
        stacked = np.stack([p[key].astype(np.float64) for p in all_preds], axis=0)
        weighted = np.average(stacked, axis=0, weights=weights)
        ensemble[key] = np.rint(np.clip(weighted, 0, 255)).astype(np.uint8)

    save_pred_npz(ensemble, args.output)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit ensemble script**

```bash
git add ensemble_predictions.py
git commit -m "feat: add prediction ensemble script for CPU-only submission averaging"
git push
```

---

### Task 2: Generate Prediction Ensembles from Existing Submissions

- [ ] **Step 1: Generate conservative ensemble (Run 9 TTA + avg5 TTA)**

```bash
python ensemble_predictions.py \
  submission/stage1-p256-tta-run9-l1mse.zip \
  submission/stage1-p256-avg5-tta-run9-l1mse.zip \
  --output submission/ensemble-run9-tta-avg5.zip
```

- [ ] **Step 2: Generate top-2 ensemble (Run 9 TTA + Run 8 TTA)**

```bash
python ensemble_predictions.py \
  submission/stage1-p256-tta-run9-l1mse.zip \
  submission/stage1-p256-tta.zip \
  --output submission/ensemble-r9-r8-tta.zip
```

- [ ] **Step 3: Generate diverse ensemble (Run 9 TTA + Run 8 TTA + Run 11 avg5 + Run 12 TTA)**

```bash
python ensemble_predictions.py \
  submission/stage1-p256-tta-run9-l1mse.zip \
  submission/stage1-p256-tta.zip \
  submission/stage1-p256-avg5-tta-run11-l1mse-ema.zip \
  submission/stage1-p256-tta-run12-l1mse.zip \
  --output submission/ensemble-diverse.zip
```

- [ ] **Step 4: Commit ensemble zips**

```bash
git add submission/ensemble-run9-tta-avg5.zip submission/ensemble-r9-r8-tta.zip submission/ensemble-diverse.zip
git commit -m "feat: add prediction ensemble submissions (conservative, top2, diverse)"
git push
```

---

### Task 3: Run 13 Training (4090 only)

> **Run this on the 4090 machine, NOT this workspace.**

- [ ] **Step 1: Launch Run 13 training**

```bash
python train_hw4.py \
  --gpu_ids 0 --epochs 150 --precision 32 --batch_size 1 --patch_size 256 \
  --ckpt_dir checkpoints/run13-p256-l1mse0025 \
  --save_top_k 5 --monitor val_psnr \
  --loss_type l1_mse --mse_weight 0.025
```

**Expected:** Training completes in ~6-8 hours. Note the best `val_psnr` from the terminal log. Refer to Run 9's best `val_psnr=30.749` (from TensorBoard logs) as the comparison threshold.

- [ ] **Step 2: Record Run 13 best val_psnr and best epoch in the terminal after training**

Run 13 best `val_psnr` at the PSNR-monitored best epoch should be printed in the progress bar. If it is clearly below 30.749, proceed with caution (still generate TTA submission but weight Run 9 higher).

---

### Task 4: Run 13 Inference (4090 machine)

> **Run these on the 4090 machine after training completes.**

- [ ] **Step 1: Original inference**

```bash
# Find best checkpoint
BEST_CKPT=$(ls -1 checkpoints/run13-p256-l1mse0025/promptir-epoch*.ckpt | sort -t= -k2 -n | tail -1)  # approximate; manually identify from epoch log
python inference.py "$BEST_CKPT" --output submission/run13-p256-original.zip
```

- [ ] **Step 2: TTA inference**

```bash
python inference.py "$BEST_CKPT" --tta --output submission/run13-p256-tta.zip
```

- [ ] **Step 3: Top-5 checkpoint averaging**

```bash
python -c "
import re
from pathlib import Path
ckpt_dir = Path('checkpoints/run13-p256-l1mse0025')
ckpts = list(ckpt_dir.glob('promptir-epoch*.ckpt'))
def psnr(path):
    m = re.search(r'val_psnr=([\d.]+)', path.name)
    return float(m.group(1)) if m else -float('inf')
top5 = sorted(ckpts, key=psnr, reverse=True)[:5]
for p in top5:
    print(p)
"
# Then run:
python average_checkpoints.py <top5_paths> --output checkpoints/run13-p256-l1mse0025-avg5.ckpt
```

- [ ] **Step 4: Averaged-5 TTA inference**

```bash
python inference.py checkpoints/run13-p256-l1mse0025-avg5.ckpt --tta --output submission/run13-p256-avg5-tta.zip
```

---

### Task 5: Run 13 + Run 9 TTA Ensemble (CPU, any machine)

> **Can run anywhere after Run 13 TTA zip is available.**

- [ ] **Step 1: Ensemble Run 9 TTA + Run 13 TTA**

```bash
python ensemble_predictions.py \
  submission/stage1-p256-tta-run9-l1mse.zip \
  submission/run13-p256-tta.zip \
  --output submission/ensemble-r9-r13-tta.zip
```

- [ ] **Step 2: Ensemble Run 9 TTA + Run 13 avg5 TTA**

```bash
python ensemble_predictions.py \
  submission/stage1-p256-tta-run9-l1mse.zip \
  submission/run13-p256-avg5-tta.zip \
  --output submission/ensemble-r9-r13avg5-tta.zip
```

---

### Task 6: Submit to CodaBench in Priority Order

Submit in this exact order to maximize information gain. Stop early if quota exhausted.

**Priority order:**

1. `submission/run13-p256-tta.zip` — highest priority new candidate
2. `submission/ensemble-diverse.zip` — no GPU needed, plausible gain
3. `submission/run13-p256-avg5-tta.zip` — only if step 1 val_psnr looks promising
4. `submission/ensemble-r9-r13-tta.zip` — zero-risk CPU ensemble

**Decision after scores are returned:**

| Best public PSNR | Action |
|---|---|
| > 31.77 | Replace Run 9 as final submission. Update README, AGENTS.md, report. |
| 31.74–31.77 | Keep Run 9 final. Note Run 13 score in results table. |
| < 31.74 | Discard Run 13. Keep Run 9 final. |

---

### Task 7: Update Docs with Final Results

> **After all submissions scored.**

- [ ] **Step 1: Update AGENTS.md results table** — add Run 13 row and ensemble rows
- [ ] **Step 2: Update README.md performance snapshot** — add new entries, update final selection if changed
- [ ] **Step 3: Update report/report.tex** — add analysis of lambda=0.025 results
- [ ] **Step 4: Recompile report PDF**

```bash
latexmk -pdf report/report.tex -outdir=report -interaction=nonstopmode
```

- [ ] **Step 5: Copy final E3 PDF**

```bash
cp report/report.pdf ../313540001_HW4.pdf
```

- [ ] **Step 6: Run tests**

```bash
python -m unittest discover -s tests -v
```

- [ ] **Step 7: Commit and push all doc updates**

```bash
git add -f AGENTS.md README.md report/report.tex report/report.pdf 313540001_HW4.pdf
git commit -m "docs: add Run 13 + ensemble final results"
git push
```
