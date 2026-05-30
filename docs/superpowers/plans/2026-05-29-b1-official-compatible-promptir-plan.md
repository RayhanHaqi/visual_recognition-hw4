# B1 Official-Compatible PromptIR Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve HW4 PSNR with one compliant PromptIR model by adopting official PromptIR-style always-on geometric augmentation and in-domain auxiliary synthetic degradations from the provided clean images only.

**Architecture:** Keep `PromptIR(decoder=True)` as the only restoration model and keep the final checkpoint single-model. Extend the existing dataset to optionally add synthetic denoise training pairs generated from HW4 clean images, and optionally force geometric augmentation to exclude identity, matching the official PromptIR data recipe more closely without external data.

**Tech Stack:** Python, PyTorch, torchvision, Lightning, PromptIR, unittest, `train_stage1.sh`, CodaBench `.zip` submission format.

---

## File Structure

- Modify `hw4_dataset.py`: add optional forced geometric augmentation and synthetic clean-image denoise samples.
- Modify `train_hw4.py`: add CLI args for auxiliary denoise and forced augmentation.
- Modify `train_stage1.sh`: pass B1 env flags into training and expose a safe local-PC command.
- Test `tests/test_hw4_dataset.py`: verify no-identity augmentation path and synthetic denoise sample construction.
- Test `tests/test_train_stage1_script.py`: verify shell script exposes B1 flags.

## Task 1: Official-Style Forced Geometric Augmentation

**Files:**
- Modify: `hw4_dataset.py`
- Test: `tests/test_hw4_dataset.py`

- [ ] **Step 1: Write failing test for no-identity augmentation**

Append to `tests/test_hw4_dataset.py`:

```python
class OfficialAugmentationTest(unittest.TestCase):
    def test_random_augmentation_can_exclude_identity(self):
        img = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
        for _ in range(100):
            aug = random_augmentation(img, allow_identity=False)[0]
            self.assertFalse(np.array_equal(aug, img))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_hw4_dataset.OfficialAugmentationTest -v`

Expected: FAIL with `TypeError` because `allow_identity` is not implemented.

- [ ] **Step 3: Implement minimal augmentation flag**

Change `hw4_dataset.py`:

```python
def random_augmentation(*imgs, allow_identity=True):
    low = 0 if allow_identity else 1
    flag_aug = random.randint(low, 7)
    out = []
    for img in imgs:
        aug = img.copy()
        if flag_aug in (1, 3, 5, 7):
            aug = np.flipud(aug).copy()
        rot = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3}[flag_aug]
        if rot:
            aug = np.rot90(aug, rot).copy()
        out.append(aug)
    return out
```

In `HW4TrainDataset.__getitem__`, replace:

```python
degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))
```

with:

```python
allow_identity = not getattr(self.args, 'force_aug_no_identity', False)
degrad_patch, clean_patch = random_augmentation(
    *self._crop_patch(degrad_img, clean_img), allow_identity=allow_identity)
```

- [ ] **Step 4: Verify targeted test passes**

Run: `python -m unittest tests.test_hw4_dataset.OfficialAugmentationTest -v`

Expected: PASS.

## Task 2: Add In-Domain Synthetic Denoise Samples

**Files:**
- Modify: `hw4_dataset.py`
- Test: `tests/test_hw4_dataset.py`

- [ ] **Step 1: Write failing tests for synthetic denoise support**

Append to `tests/test_hw4_dataset.py`:

```python
class AuxiliaryDenoiseDatasetTest(unittest.TestCase):
    def test_aux_denoise_samples_use_hw4_clean_paths(self):
        args = argparse.Namespace(
            data_dir='PromptIR/data', de_type=['desnow', 'derain'], merge_val=False,
            derain_oversample=1, aux_denoise=1, aux_denoise_sigmas='15,25,50')
        with mock.patch.object(HW4TrainDataset, '_init_ids'):
            ds = HW4TrainDataset(args)
            ds.sample_ids = []
            ds.snow_ids = [{'clean_path': 'snow_clean-1.png', 'de_type': 0}]
            ds.rain_ids = [{'clean_path': 'rain_clean-1.png', 'de_type': 1}]
            ds._merge_ids()
        aux = [sample for sample in ds.sample_ids if sample['de_type'] == 2]
        self.assertEqual(len(aux), 6)
        self.assertEqual(sorted({sample['sigma'] for sample in aux}), [15, 25, 50])
```

If `argparse` and `mock` are not imported in the file, add:

```python
import argparse
from unittest import mock
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_hw4_dataset.AuxiliaryDenoiseDatasetTest -v`

Expected: FAIL because auxiliary denoise samples are not created.

- [ ] **Step 3: Implement minimal auxiliary denoise list creation**

Add to `hw4_dataset.py`:

```python
def parse_sigmas(value):
    return [int(part.strip()) for part in str(value).split(',') if part.strip()]
```

In `_merge_ids`, after adding snow/rain samples:

```python
if getattr(self.args, 'aux_denoise', 0) > 0:
    sigmas = parse_sigmas(getattr(self.args, 'aux_denoise_sigmas', '15,25,50'))
    clean_sources = []
    if 'desnow' in self.de_type:
        clean_sources += self.snow_ids
    if 'derain' in self.de_type:
        clean_sources += self.rain_ids
    for _ in range(getattr(self.args, 'aux_denoise', 0)):
        for sample in clean_sources:
            for sigma in sigmas:
                self.sample_ids.append({
                    'clean_path': sample['clean_path'],
                    'degraded_path': sample['clean_path'],
                    'de_type': 2,
                    'sigma': sigma,
                })
```

- [ ] **Step 4: Generate noisy patch in `__getitem__`**

Before loading paired degraded images in `__getitem__`, add branch:

```python
if de_id == 2:
    clean_img = crop_img(np.array(Image.open(sample['clean_path']).convert('RGB')), base=16)
    degrad_patch, clean_patch = self._crop_patch(clean_img, clean_img)
    allow_identity = not getattr(self.args, 'force_aug_no_identity', False)
    degrad_patch, clean_patch = random_augmentation(
        degrad_patch, clean_patch, allow_identity=allow_identity)
    noise = np.random.randn(*clean_patch.shape) * sample['sigma']
    degrad_patch = np.clip(clean_patch.astype(np.float32) + noise, 0, 255).astype(np.uint8)
else:
    degrad_img = crop_img(np.array(Image.open(sample['degraded_path']).convert('RGB')), base=16)
    clean_img = crop_img(np.array(Image.open(sample['clean_path']).convert('RGB')), base=16)
    allow_identity = not getattr(self.args, 'force_aug_no_identity', False)
    degrad_patch, clean_patch = random_augmentation(
        *self._crop_patch(degrad_img, clean_img), allow_identity=allow_identity)
```

Remove the old unconditional degraded/clean image loading block to avoid double loading.

- [ ] **Step 5: Verify targeted tests pass**

Run: `python -m unittest tests.test_hw4_dataset -v`

Expected: PASS.

## Task 3: Expose B1 Training Flags

**Files:**
- Modify: `train_hw4.py`
- Modify: `train_stage1.sh`
- Test: `tests/test_train_stage1_script.py`

- [ ] **Step 1: Write failing shell/CLI tests**

Append to `tests/test_train_stage1_script.py`:

```python
    def test_stage1_script_exposes_b1_official_recipe_flags(self):
        text = Path("train_stage1.sh").read_text()
        self.assertIn('FORCE_AUG_NO_IDENTITY=${FORCE_AUG_NO_IDENTITY:-0}', text)
        self.assertIn('AUX_DENOISE=${AUX_DENOISE:-0}', text)
        self.assertIn('AUX_DENOISE_SIGMAS=${AUX_DENOISE_SIGMAS:-15,25,50}', text)
        self.assertIn('--force_aug_no_identity', text)
        self.assertIn('--aux_denoise "$AUX_DENOISE"', text)
        self.assertIn('--aux_denoise_sigmas "$AUX_DENOISE_SIGMAS"', text)
```

In `test_train_hw4_exposes_ema_and_top_k_options`, add:

```python
self.assertIn('--force_aug_no_identity', script)
self.assertIn('--aux_denoise', script)
self.assertIn('--aux_denoise_sigmas', script)
```

- [ ] **Step 2: Run failing tests**

Run: `python -m unittest tests.test_train_stage1_script.TrainStage1ScriptTest -v`

Expected: FAIL because B1 flags are not exposed.

- [ ] **Step 3: Add train CLI args**

In `train_hw4.py`, add parser args:

```python
parser.add_argument('--force_aug_no_identity', action='store_true',
                    help='Official PromptIR-style geometric augmentation without identity')
parser.add_argument('--aux_denoise', type=int, default=0,
                    help='Repeat count for synthetic denoise samples from provided clean images')
parser.add_argument('--aux_denoise_sigmas', type=str, default='15,25,50',
                    help='Comma-separated Gaussian denoise sigmas for aux_denoise')
```

- [ ] **Step 4: Add shell env pass-through**

In `train_stage1.sh`, add defaults:

```bash
FORCE_AUG_NO_IDENTITY=${FORCE_AUG_NO_IDENTITY:-0}
AUX_DENOISE=${AUX_DENOISE:-0}
AUX_DENOISE_SIGMAS=${AUX_DENOISE_SIGMAS:-15,25,50}
```

Add to `TRAIN_ARGS`:

```bash
--aux_denoise "$AUX_DENOISE"
--aux_denoise_sigmas "$AUX_DENOISE_SIGMAS"
```

Add conditional flag:

```bash
if [ "$FORCE_AUG_NO_IDENTITY" = "1" ]; then
    TRAIN_ARGS+=(--force_aug_no_identity)
fi
```

- [ ] **Step 5: Verify tests pass**

Run: `python -m unittest tests.test_train_stage1_script.TrainStage1ScriptTest -v`

Expected: PASS.

## Task 4: Verification And Commands

**Files:**
- No production changes beyond Tasks 1-3.

- [ ] **Step 1: Run full tests**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 2: Run style check**

Run: `pycodestyle --max-line-length=120 --ignore=E402,W503 hw4_dataset.py train_hw4.py train_stage1.sh tests/test_hw4_dataset.py tests/test_train_stage1_script.py`

Expected: no output.

- [ ] **Step 3: Run shell syntax check**

Run: `bash -n train_stage1.sh`

Expected: no output.

- [ ] **Step 4: RTX 5060 Ti smoke command**

Run locally first:

```bash
cd /home/ai/Rayhan/selectedtopics/HW4 && git pull && RUN_NAME=smoke-b1-official-aug AUX_DENOISE=1 FORCE_AUG_NO_IDENTITY=1 PATCH_SIZE=192 BATCH_SIZE=1 EPOCHS=2 SAVE_TOP_K=1 GRADIENT_CHECKPOINTING=full PRECISION=32 bash train_stage1.sh
```

Expected: completes 2 epochs and writes smoke submission zips. If Blackwell/cuDNN crashes, retry with PyTorch/CUDA versions that support RTX 50-series or disable cuDNN in `train_hw4.py` before training.

- [ ] **Step 5: RTX 5060 Ti full B1 command**

If smoke passes, run:

```bash
cd /home/ai/Rayhan/selectedtopics/HW4 && git pull && RUN_NAME=run24-b1-official-aug-auxdenoise AUX_DENOISE=1 AUX_DENOISE_SIGMAS=15,25,50 FORCE_AUG_NO_IDENTITY=1 PATCH_SIZE=256 BATCH_SIZE=1 EPOCHS=150 SAVE_TOP_K=5 MONITOR=val_psnr LOSS_TYPE=l1_mse MSE_WEIGHT=0.05 GRADIENT_CHECKPOINTING=full PRECISION=32 bash train_stage1.sh
```

Expected outputs:

```text
submission/run24-b1-official-aug-auxdenoise-original.zip
submission/run24-b1-official-aug-auxdenoise-tta.zip
submission/run24-b1-official-aug-auxdenoise-avg5-tta.zip
```

## Self-Review

- Spec coverage: implements approved option A: official-compatible data/augmentation upgrade before PromptIR-large.
- Constraint check: uses only provided HW4 clean images, one PromptIR model, no pretrained weights, no external data.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: flags are `force_aug_no_identity`, `aux_denoise`, and `aux_denoise_sigmas` in dataset, train CLI, and shell script.
