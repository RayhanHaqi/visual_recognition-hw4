# PromptIR-Only Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PromptIR-compliant mechanisms to improve HW4 restoration PSNR without replacing the PromptIR backbone.

**Architecture:** Keep `net.model.PromptIR` as the core model. Add measurement, configurable losses, optional task-aware conditioning, and optional SIPL-lite two-pass refinement around the existing PromptIR training/inference pipeline.

**Tech Stack:** Python, PyTorch, Lightning, torchvision, NumPy, unittest, existing HW4 scripts.

---

## File Structure

- Modify `hw4_dataset.py`: return degradation IDs for validation and test where requested.
- Modify `train_hw4.py`: add per-task validation metrics, configurable losses, task-aware wrapper selection, and optional SIPL-lite training.
- Modify `inference.py`: add optional task-conditioned inference mode for conditioned checkpoints.
- Modify `train_stage1.sh`: pass new CLI flags through environment variables while preserving current defaults.
- Create or modify tests under `tests/`: verify metrics inputs, loss formulas, conditioning behavior, and inference CLI compatibility.
- Modify `README.md` and `AGENTS.md` only after experiments produce new validated results.

Do not edit `PromptIR/` unless execution confirms the source is tracked and the user explicitly chooses the invasive SIPL path. The default plan uses wrappers in tracked HW4 files.

## Task 1: Add Per-Task Validation Labels And Metrics

**Files:**
- Modify: `hw4_dataset.py`
- Modify: `train_hw4.py`
- Test: `tests/test_hw4_dataset.py`
- Test: `tests/test_train_hw4_metrics.py`

- [ ] **Step 1: Add failing dataset test for validation degradation IDs**

Append this test to `tests/test_hw4_dataset.py`:

```python
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from hw4_dataset import HW4ValDataset


class HW4ValDatasetDegradationIdTest(unittest.TestCase):
    def _write_rgb(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)).save(path)

    def test_val_dataset_returns_rain_and_snow_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_rgb(root / "Val" / "Derain" / "rainy" / "rain-1.png")
            self._write_rgb(root / "Val" / "Derain" / "gt" / "rain_clean-1.png")
            self._write_rgb(root / "Val" / "Desnow" / "snowy" / "snow-1.png")
            self._write_rgb(root / "Val" / "Desnow" / "gt" / "snow_clean-1.png")

            dataset = HW4ValDataset(str(root))
            ids = {dataset[i][0]: dataset[i][1] for i in range(len(dataset))}

        self.assertEqual(ids["rain-1.png"], 1)
        self.assertEqual(ids["snow-1.png"], 0)
```

- [ ] **Step 2: Run failing dataset test**

Run:

```bash
python -m unittest tests.test_hw4_dataset.HW4ValDatasetDegradationIdTest -v
```

Expected before implementation: fail because `HW4ValDataset.__getitem__` currently returns `(name, degraded, clean)` and has no `de_id`.

- [ ] **Step 3: Modify validation dataset return format**

Change `HW4ValDataset` in `hw4_dataset.py` so samples store `de_id`:

```python
for task in ["Derain", "Desnow"]:
    de_id = 1 if task == "Derain" else 0
    degraded_dir = os.path.join(val_dir, task, "rainy" if task == "Derain" else "snowy")
    gt_dir = os.path.join(val_dir, task, "gt")
    if os.path.isdir(degraded_dir):
        for name in sorted(os.listdir(degraded_dir)):
            if name.endswith('.png'):
                prefix = "rain_clean-" if task == "Derain" else "snow_clean-"
                idx = name.split("-")[-1]
                clean_path = os.path.join(gt_dir, prefix + idx)
                if os.path.exists(clean_path):
                    self.samples.append((os.path.join(degraded_dir, name), clean_path, name, de_id))
```

Change `__getitem__`:

```python
degraded_path, clean_path, name, de_id = self.samples[idx]
degraded = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
clean = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)
return name, de_id, self.toTensor(degraded), self.toTensor(clean)
```

- [ ] **Step 4: Update validation step batch unpacking and logging**

In `train_hw4.py`, change `validation_step` to accept `de_id`:

```python
def validation_step(self, batch, batch_idx):
    name, de_id, degrad_patch, clean_patch = batch
    restored = self.net(degrad_patch)
    loss = self.loss_fn(restored, clean_patch)
    mse = torch.mean((torch.clamp(restored, 0, 1) - clean_patch) ** 2)
    psnr = -10 * torch.log10(torch.clamp(mse, min=1e-10))
    self.log("val_loss", loss, on_epoch=True, prog_bar=True)
    self.log("val_psnr", psnr, on_epoch=True, prog_bar=True)

    is_snow = de_id == 0
    is_rain = de_id == 1
    if torch.any(is_snow):
        self.log("val_loss_desnow", loss, on_epoch=True, batch_size=1)
        self.log("val_psnr_desnow", psnr, on_epoch=True, batch_size=1)
    if torch.any(is_rain):
        self.log("val_loss_derain", loss, on_epoch=True, batch_size=1)
        self.log("val_psnr_derain", psnr, on_epoch=True, batch_size=1)
    return loss
```

- [ ] **Step 5: Add focused test for batch unpacking expectation**

Create `tests/test_train_hw4_metrics.py`:

```python
import unittest

import torch


class ValidationBatchShapeTest(unittest.TestCase):
    def test_validation_batch_has_name_de_id_degraded_clean(self):
        batch = (["rain-1.png"], torch.tensor([1]), torch.zeros(1, 3, 16, 16), torch.ones(1, 3, 16, 16))
        name, de_id, degraded, clean = batch
        self.assertEqual(name[0], "rain-1.png")
        self.assertEqual(int(de_id[0]), 1)
        self.assertEqual(tuple(degraded.shape), (1, 3, 16, 16))
        self.assertEqual(tuple(clean.shape), (1, 3, 16, 16))
```

- [ ] **Step 6: Run tests**

Run:

```bash
python -m unittest tests.test_hw4_dataset.HW4ValDatasetDegradationIdTest tests.test_train_hw4_metrics.ValidationBatchShapeTest -v
```

Expected: both tests pass.

- [ ] **Step 7: Run full unit test suite**

Run:

```bash
python -m unittest discover -s tests -v
```

Expected: all tests pass.

Checkpoint only. Do not commit unless the user explicitly asks.

## Task 2: Add Configurable PSNR-Oriented Losses

**Files:**
- Modify: `train_hw4.py`
- Test: `tests/test_train_hw4_losses.py`

- [ ] **Step 1: Write failing loss tests**

Create `tests/test_train_hw4_losses.py`:

```python
import unittest

import torch

from train_hw4 import build_loss_fn


class TrainLossTest(unittest.TestCase):
    def test_charbonnier_zero_error_is_near_zero(self):
        loss_fn = build_loss_fn("charbonnier", mse_weight=0.05)
        pred = torch.zeros(1, 3, 4, 4)
        target = torch.zeros(1, 3, 4, 4)
        self.assertLess(float(loss_fn(pred, target)), 1e-2)

    def test_l1_mse_matches_expected_formula(self):
        loss_fn = build_loss_fn("l1_mse", mse_weight=0.25)
        pred = torch.tensor([0.0, 1.0])
        target = torch.tensor([1.0, 1.0])
        expected = torch.mean(torch.abs(pred - target)) + 0.25 * torch.mean((pred - target) ** 2)
        self.assertAlmostEqual(float(loss_fn(pred, target)), float(expected), places=6)

    def test_l1_matches_torch_l1(self):
        loss_fn = build_loss_fn("l1", mse_weight=0.05)
        pred = torch.tensor([0.0, 1.0])
        target = torch.tensor([1.0, 1.0])
        self.assertAlmostEqual(float(loss_fn(pred, target)), 0.5, places=6)
```

- [ ] **Step 2: Run failing loss tests**

Run:

```bash
python -m unittest tests.test_train_hw4_losses -v
```

Expected before implementation: fail because `build_loss_fn` is not defined.

- [ ] **Step 3: Implement loss builder**

Add near the top of `train_hw4.py` after imports:

```python
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class L1MSELoss(nn.Module):
    def __init__(self, mse_weight):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.l1(pred, target) + self.mse_weight * self.mse(pred, target)


def build_loss_fn(loss_type, mse_weight):
    if loss_type == "l1":
        return nn.L1Loss()
    if loss_type == "charbonnier":
        return CharbonnierLoss()
    if loss_type == "l1_mse":
        return L1MSELoss(mse_weight=mse_weight)
    raise ValueError(f"Unsupported loss type: {loss_type}")
```

Change model init:

```python
def __init__(self, lr=2e-4, warmup_epochs=15, max_epochs=150, loss_type="l1", mse_weight=0.05):
    super().__init__()
    self.net = PromptIR(decoder=True)
    self.loss_fn = build_loss_fn(loss_type, mse_weight)
```

Add parser args:

```python
parser.add_argument('--loss_type', choices=['l1', 'charbonnier', 'l1_mse'], default='l1')
parser.add_argument('--mse_weight', type=float, default=0.05)
```

Pass to model:

```python
model = PromptIRModel(
    lr=args.lr,
    warmup_epochs=args.warmup,
    max_epochs=args.epochs,
    loss_type=args.loss_type,
    mse_weight=args.mse_weight,
)
```

- [ ] **Step 4: Run loss tests**

Run:

```bash
python -m unittest tests.test_train_hw4_losses -v
```

Expected: all loss tests pass.

- [ ] **Step 5: Run compile and full tests**

Run:

```bash
python -m py_compile train_hw4.py
python -m unittest discover -s tests -v
```

Expected: compile succeeds and all tests pass.

Checkpoint only. Do not commit unless the user explicitly asks.

## Task 3: Add Task-Aware PromptIR Wrapper

**Files:**
- Modify: `train_hw4.py`
- Modify: `inference.py`
- Test: `tests/test_promptir_task_conditioning.py`

- [ ] **Step 1: Write failing wrapper tests**

Create `tests/test_promptir_task_conditioning.py`:

```python
import unittest

import torch
import torch.nn as nn

from train_hw4 import TaskConditionedRestorer


class IdentityNet(nn.Module):
    def forward(self, x):
        return x


class TaskConditionedRestorerTest(unittest.TestCase):
    def test_conditioning_preserves_shape(self):
        model = TaskConditionedRestorer(IdentityNet())
        x = torch.zeros(2, 3, 8, 8)
        de_id = torch.tensor([0, 1])
        y = model(x, de_id)
        self.assertEqual(tuple(y.shape), (2, 3, 8, 8))

    def test_unconditioned_path_matches_backbone_shape(self):
        model = TaskConditionedRestorer(IdentityNet(), enabled=False)
        x = torch.zeros(1, 3, 8, 8)
        y = model(x, torch.tensor([1]))
        self.assertEqual(tuple(y.shape), tuple(x.shape))
```

- [ ] **Step 2: Run failing wrapper tests**

Run:

```bash
python -m unittest tests.test_promptir_task_conditioning -v
```

Expected before implementation: fail because `TaskConditionedRestorer` is not defined.

- [ ] **Step 3: Implement wrapper**

Add to `train_hw4.py` after loss builders:

```python
class TaskConditionedRestorer(nn.Module):
    def __init__(self, backbone, enabled=True):
        super().__init__()
        self.backbone = backbone
        self.enabled = enabled
        self.input_scale = nn.Embedding(2, 3)
        self.input_bias = nn.Embedding(2, 3)
        self.output_bias = nn.Embedding(2, 3)
        nn.init.zeros_(self.input_scale.weight)
        nn.init.zeros_(self.input_bias.weight)
        nn.init.zeros_(self.output_bias.weight)

    def forward(self, x, de_id=None):
        if not self.enabled or de_id is None:
            return self.backbone(x)
        de_id = de_id.to(device=x.device, dtype=torch.long).view(-1)
        scale = self.input_scale(de_id).view(-1, 3, 1, 1)
        bias = self.input_bias(de_id).view(-1, 3, 1, 1)
        out_bias = self.output_bias(de_id).view(-1, 3, 1, 1)
        conditioned = x * (1.0 + scale) + bias
        return self.backbone(conditioned) + out_bias
```

Change `PromptIRModel.__init__` to accept `task_conditioning=False`:

```python
self.net = TaskConditionedRestorer(PromptIR(decoder=True), enabled=task_conditioning)
```

Change training and validation forward calls:

```python
restored = self.net(degrad_patch, de_id)
```

- [ ] **Step 4: Add parser flag**

Add to `train_hw4.py` parser:

```python
parser.add_argument('--task_conditioning', action='store_true')
```

Pass to model:

```python
task_conditioning=args.task_conditioning,
```

- [ ] **Step 5: Update inference checkpoint loading for wrapped checkpoints**

In `inference.py`, add `TaskConditionedRestorer` import guarded by existing path setup:

```python
from train_hw4 import TaskConditionedRestorer
```

Add parser arg:

```python
parser.add_argument('--task_mode', choices=['none', 'derain', 'desnow', 'both'], default='none')
```

Construct model:

```python
use_task_conditioning = any(key.startswith('backbone.') or key.startswith('input_scale.') for key in state_dict)
base_model = PromptIR(decoder=True)
model = TaskConditionedRestorer(base_model, enabled=use_task_conditioning) if use_task_conditioning else base_model
```

For `task_mode=both`, restore twice and average:

```python
if args.task_mode == 'both':
    rain_id = torch.ones(img.shape[0], dtype=torch.long, device=device)
    snow_id = torch.zeros(img.shape[0], dtype=torch.long, device=device)
    restored = 0.5 * (restore_image(model, padded, use_tta=use_tta, de_id=rain_id) + restore_image(model, padded, use_tta=use_tta, de_id=snow_id))
```

This requires updating `restore_image` to accept `de_id=None` and call `model(img, de_id)` only when `de_id` is not `None`.

- [ ] **Step 6: Run wrapper tests and compile**

Run:

```bash
python -m unittest tests.test_promptir_task_conditioning -v
python -m py_compile train_hw4.py inference.py
```

Expected: tests pass and compile succeeds.

- [ ] **Step 7: Run full tests**

Run:

```bash
python -m unittest discover -s tests -v
```

Expected: all tests pass.

Checkpoint only. Do not commit unless the user explicitly asks.

## Task 4: Wire Experiment Flags Through `train_stage1.sh`

**Files:**
- Modify: `train_stage1.sh`
- Test: `tests/test_train_stage1_script.py`

- [ ] **Step 1: Add script default assertions**

Extend `tests/test_train_stage1_script.py` with assertions that the script exposes these environment variables:

```python
def test_stage1_script_exposes_promptir_only_experiment_flags(self):
    text = self.script_path.read_text()
    self.assertIn('LOSS_TYPE=${LOSS_TYPE:-l1}', text)
    self.assertIn('MSE_WEIGHT=${MSE_WEIGHT:-0.05}', text)
    self.assertIn('TASK_CONDITIONING=${TASK_CONDITIONING:-0}', text)
    self.assertIn('SIPL_LITE=${SIPL_LITE:-0}', text)
    self.assertIn('SIPL_START_ALPHA=${SIPL_START_ALPHA:-0.5}', text)
    self.assertIn('SIPL_REFINE_WEIGHT=${SIPL_REFINE_WEIGHT:-0.5}', text)
```

- [ ] **Step 2: Run failing script test**

Run:

```bash
python -m unittest tests.test_train_stage1_script -v
```

Expected before implementation: fail because new defaults are absent.

- [ ] **Step 3: Add env vars and pass args**

In `train_stage1.sh`, add defaults near existing env vars:

```bash
LOSS_TYPE=${LOSS_TYPE:-l1}
MSE_WEIGHT=${MSE_WEIGHT:-0.05}
TASK_CONDITIONING=${TASK_CONDITIONING:-0}
SIPL_LITE=${SIPL_LITE:-0}
SIPL_START_ALPHA=${SIPL_START_ALPHA:-0.5}
SIPL_REFINE_WEIGHT=${SIPL_REFINE_WEIGHT:-0.5}
```

Add to printed config:

```bash
echo "Loss type: $LOSS_TYPE"
echo "MSE weight: $MSE_WEIGHT"
echo "Task conditioning: $TASK_CONDITIONING"
echo "SIPL-lite: $SIPL_LITE"
echo "SIPL start alpha: $SIPL_START_ALPHA"
echo "SIPL refine weight: $SIPL_REFINE_WEIGHT"
```

Add to `TRAIN_ARGS`:

```bash
--loss_type "$LOSS_TYPE"
--mse_weight "$MSE_WEIGHT"
```

Add conditional flag:

```bash
if [ "$TASK_CONDITIONING" = "1" ]; then
    TRAIN_ARGS+=(--task_conditioning)
fi
if [ "$SIPL_LITE" = "1" ]; then
    TRAIN_ARGS+=(--sipl_lite --sipl_start_alpha "$SIPL_START_ALPHA" --sipl_refine_weight "$SIPL_REFINE_WEIGHT")
fi
```

- [ ] **Step 4: Run script tests and syntax check**

Run:

```bash
python -m unittest tests.test_train_stage1_script -v
bash -n train_stage1.sh
```

Expected: tests pass and syntax check succeeds.

Checkpoint only. Do not commit unless the user explicitly asks.

## Task 5: Add SIPL-Lite Two-Pass Training As Optional Experiment

**Files:**
- Modify: `train_hw4.py`
- Modify: `inference.py`
- Test: `tests/test_sipl_lite.py`

- [ ] **Step 1: Write failing SIPL-lite tests**

Create `tests/test_sipl_lite.py`:

```python
import unittest

import torch
import torch.nn as nn

from train_hw4 import sipl_blend_alpha, sipl_second_input


class SIPLLiteTest(unittest.TestCase):
    def test_alpha_decays_to_zero(self):
        self.assertAlmostEqual(sipl_blend_alpha(epoch=0, max_epochs=100, start_alpha=0.5), 0.5, places=6)
        self.assertAlmostEqual(sipl_blend_alpha(epoch=100, max_epochs=100, start_alpha=0.5), 0.0, places=6)

    def test_training_second_input_blends_clean(self):
        restored = torch.zeros(1, 3, 4, 4)
        clean = torch.ones(1, 3, 4, 4)
        blended = sipl_second_input(restored, clean, alpha=0.25, training=True)
        self.assertAlmostEqual(float(blended.mean()), 0.25, places=6)

    def test_inference_second_input_uses_restored_only(self):
        restored = torch.zeros(1, 3, 4, 4)
        clean = torch.ones(1, 3, 4, 4)
        blended = sipl_second_input(restored, clean, alpha=0.25, training=False)
        self.assertAlmostEqual(float(blended.mean()), 0.0, places=6)
```

- [ ] **Step 2: Run failing SIPL-lite tests**

Run:

```bash
python -m unittest tests.test_sipl_lite -v
```

Expected before implementation: fail because helper functions are undefined.

- [ ] **Step 3: Implement SIPL-lite helpers**

Add to `train_hw4.py`:

```python
def sipl_blend_alpha(epoch, max_epochs, start_alpha):
    if max_epochs <= 0:
        return 0.0
    progress = min(max(epoch / max_epochs, 0.0), 1.0)
    return start_alpha * (1.0 - progress)


def sipl_second_input(restored, clean, alpha, training):
    if not training or clean is None:
        return restored.detach()
    return (1.0 - alpha) * restored.detach() + alpha * clean
```

- [ ] **Step 4: Add optional SIPL-lite model behavior**

Add parser args:

```python
parser.add_argument('--sipl_lite', action='store_true')
parser.add_argument('--sipl_start_alpha', type=float, default=0.5)
parser.add_argument('--sipl_refine_weight', type=float, default=0.5)
```

Add model init fields:

```python
self.sipl_lite = sipl_lite
self.sipl_start_alpha = sipl_start_alpha
self.sipl_refine_weight = sipl_refine_weight
```

Change `training_step`:

```python
restored = self.net(degrad_patch, de_id)
loss = self.loss_fn(restored, clean_patch)
if self.sipl_lite:
    alpha = sipl_blend_alpha(self.current_epoch, self.max_epochs, self.sipl_start_alpha)
    second_input = sipl_second_input(restored, clean_patch, alpha=alpha, training=True)
    refined = self.net(second_input, de_id)
    loss = loss + self.sipl_refine_weight * self.loss_fn(refined, clean_patch)
```

Change `validation_step` to evaluate refined output when `self.sipl_lite` is enabled:

```python
restored = self.net(degrad_patch, de_id)
if self.sipl_lite:
    second_input = sipl_second_input(restored, clean_patch, alpha=0.0, training=False)
    restored = self.net(second_input, de_id)
```

- [ ] **Step 5: Update inference for SIPL-lite checkpoint flag**

Add CLI flag:

```python
parser.add_argument('--sipl_lite', action='store_true', help='Run a second refinement pass through PromptIR')
```

After first restored output, if enabled, run a second pass before uint8 conversion:

```python
if use_sipl_lite:
    restored = restore_image(model, torch.clamp(restored, 0, 1), use_tta=use_tta, de_id=de_id)
```

Make sure this second pass occurs before cropping and `tensor_to_uint8`.

- [ ] **Step 6: Run SIPL-lite tests and compile**

Run:

```bash
python -m unittest tests.test_sipl_lite -v
python -m py_compile train_hw4.py inference.py
```

Expected: tests pass and compile succeeds.

- [ ] **Step 7: Run full tests**

Run:

```bash
python -m unittest discover -s tests -v
```

Expected: all tests pass.

Checkpoint only. Do not commit unless the user explicitly asks.

## Task 6: Experiment Commands After Implementation

Do not run these until Tasks 1-5 are implemented and tests pass.

- [ ] **Step 1: Baseline-compatible smoke help check**

Run:

```bash
python train_hw4.py --help
```

Expected: help includes `--loss_type`, `--mse_weight`, `--task_conditioning`, and `--sipl_lite`.

- [ ] **Step 2: First low-risk experiment, loss only**

Run on the training machine:

```bash
PATCH_SIZE=256 BATCH_SIZE=1 SAVE_TOP_K=5 MONITOR=val_psnr LOSS_TYPE=l1_mse MSE_WEIGHT=0.05 bash train_stage1.sh
```

Expected output submissions:

```text
submission/stage1-p256-original.zip
submission/stage1-p256-tta.zip
submission/stage1-p256-avg5-tta.zip
```

- [ ] **Step 3: Second experiment, task conditioning**

Run on the training machine:

```bash
PATCH_SIZE=256 BATCH_SIZE=1 SAVE_TOP_K=5 MONITOR=val_psnr TASK_CONDITIONING=1 bash train_stage1.sh
```

Expected: validation logs include aggregate, derain, and desnow PSNR metrics.

- [ ] **Step 4: Third experiment, task conditioning plus loss**

Run only if either Step 2 or Step 3 improves local validation:

```bash
PATCH_SIZE=256 BATCH_SIZE=1 SAVE_TOP_K=5 MONITOR=val_psnr TASK_CONDITIONING=1 LOSS_TYPE=l1_mse MSE_WEIGHT=0.05 bash train_stage1.sh
```

- [ ] **Step 5: Fourth experiment, SIPL-lite**

Run only if task conditioning is stable:

```bash
PATCH_SIZE=256 BATCH_SIZE=1 SAVE_TOP_K=3 MONITOR=val_psnr TASK_CONDITIONING=1 LOSS_TYPE=l1_mse MSE_WEIGHT=0.05 SIPL_LITE=1 bash train_stage1.sh
```

This command uses the SIPL-lite environment variables wired through `train_stage1.sh` in Task 4.

## Validation Gate

Before claiming any experiment is better:

- [ ] Extract best `val_psnr`, `val_psnr_derain`, and `val_psnr_desnow` from TensorBoard.
- [ ] Validate each generated zip with the existing zip checker pattern.
- [ ] Compare against current best public score `31.73` only after CodaBench submission.
- [ ] Update `README.md` and `AGENTS.md` only with measured scores.

## Plan Self-Review

- Spec coverage: PromptIR-only constraint, per-task metrics, task conditioning, PSNR-aligned losses, and SIPL-lite are covered by Tasks 1-5.
- Placeholder scan: no placeholder markers or unspecified implementation steps remain.
- Type consistency: `de_id` is consistently `0` for desnow and `1` for derain; conditioning uses shape `(batch,)` long tensors.
- Scope check: implementation is split into independent, testable tasks and does not require modifying ignored `PromptIR/` source.
