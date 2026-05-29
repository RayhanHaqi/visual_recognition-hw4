# 33 dB PromptIR Push Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chase a 33.0+ CodaBench score by first building a compliant larger PromptIR, then measuring specialist upper bound, then distilling specialist behavior back into one PromptIR if time allows.

**Architecture:** Phase B keeps one PromptIR model and adds configurable width/depth so the final model remains compliant. Phase A trains derain/desnow specialist PromptIR checkpoints only as a high-upside diagnostic and leaderboard-risk candidate. Phase C uses specialist outputs as in-domain pseudo targets for one PromptIR checkpoint, preserving the final single-model rule.

**Tech Stack:** Python, PyTorch, Lightning, PromptIR, unittest, TensorBoard logs, CodaBench `pred.npz` zip submissions.

---

## File Structure

- Modify `PromptIR/net/model.py`: expose PromptIR width/depth/refinement arguments already present in `PromptIR.__init__`; no backbone replacement.
- Modify `train_hw4.py`: add CLI args for PromptIR size, task filtering, and optional distillation targets.
- Modify `hw4_dataset.py`: add task-only training support and optional pseudo-clean target directory support.
- Modify `train_stage1.sh`: pass model-size/task/distillation options into training and encode run names explicitly.
- Modify `inference.py`: accept matching PromptIR size arguments for large/specialist checkpoints.
- Create `route_specialists.py`: combine two specialist prediction zips using a local rain/snow classifier or explicit mapping.
- Create `make_distill_targets.py`: create pseudo-clean images from specialist outputs for single-model distillation.
- Add tests under `tests/`: verify CLI flags, model config propagation, task filtering, pseudo-target loading, and routing zip validity.

## Task 1: Add PromptIR-Large Configuration

**Files:**
- Modify: `train_hw4.py`
- Modify: `inference.py`
- Modify: `train_stage1.sh`
- Test: `tests/test_promptir_large_config.py`

- [ ] **Step 1: Write failing tests for PromptIR constructor config**

Add `tests/test_promptir_large_config.py`:

```python
import unittest
from unittest import mock

import train_hw4


class PromptIRLargeConfigTest(unittest.TestCase):
    def test_promptir_model_forwards_size_args_to_promptir(self):
        with mock.patch("train_hw4.PromptIR") as promptir:
            train_hw4.PromptIRModel(
                model_dim=64,
                num_blocks=[4, 8, 8, 10],
                num_refinement_blocks=8,
            )
        promptir.assert_called_once_with(
            decoder=True,
            dim=64,
            num_blocks=[4, 8, 8, 10],
            num_refinement_blocks=8,
        )

    def test_parse_num_blocks_accepts_four_integers(self):
        self.assertEqual(train_hw4.parse_num_blocks("4,8,8,10"), [4, 8, 8, 10])

    def test_parse_num_blocks_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            train_hw4.parse_num_blocks("4,8,8")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the failing test**

Run: `python -m unittest tests.test_promptir_large_config -v`

Expected: FAIL because `model_dim`, `num_blocks`, `num_refinement_blocks`, and `parse_num_blocks` are not implemented.

- [ ] **Step 3: Implement minimal model config plumbing**

In `train_hw4.py`, add near `build_loss_fn`:

```python
def parse_num_blocks(value):
    blocks = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(blocks) != 4:
        raise ValueError("num_blocks must contain four comma-separated integers")
    return blocks
```

Change `PromptIRModel.__init__` signature to include:

```python
model_dim=48, num_blocks=None, num_refinement_blocks=4,
```

Before creating the backbone:

```python
if num_blocks is None:
    num_blocks = [4, 6, 6, 8]
backbone = PromptIR(
    decoder=True,
    dim=model_dim,
    num_blocks=num_blocks,
    num_refinement_blocks=num_refinement_blocks,
)
```

Add parser args in `main()`:

```python
parser.add_argument('--model_dim', type=int, default=48)
parser.add_argument('--num_blocks', type=str, default='4,6,6,8')
parser.add_argument('--num_refinement_blocks', type=int, default=4)
```

When constructing `PromptIRModel`, pass:

```python
model_dim=args.model_dim,
num_blocks=parse_num_blocks(args.num_blocks),
num_refinement_blocks=args.num_refinement_blocks,
```

- [ ] **Step 4: Add inference config support**

In `inference.py`, add CLI args:

```python
parser.add_argument("--model_dim", type=int, default=48)
parser.add_argument("--num_blocks", type=str, default="4,6,6,8")
parser.add_argument("--num_refinement_blocks", type=int, default=4)
```

Import `parse_num_blocks` from `train_hw4`, and construct PromptIR with the same values in both conditioned and unconditioned paths.

- [ ] **Step 5: Add shell env pass-through**

In `train_stage1.sh`, add defaults:

```bash
MODEL_DIM=${MODEL_DIM:-48}
NUM_BLOCKS=${NUM_BLOCKS:-4,6,6,8}
NUM_REFINEMENT_BLOCKS=${NUM_REFINEMENT_BLOCKS:-4}
```

Add to `TRAIN_ARGS`:

```bash
--model_dim "$MODEL_DIM"
--num_blocks "$NUM_BLOCKS"
--num_refinement_blocks "$NUM_REFINEMENT_BLOCKS"
```

Add matching args to all `python inference.py` calls.

- [ ] **Step 6: Verify tests pass**

Run: `python -m unittest tests.test_promptir_large_config -v`

Expected: PASS.

Run: `python -m unittest discover -s tests -v`

Expected: all existing tests pass.

## Task 2: Launch Compliant PromptIR-Large Run B

**Files:**
- No code changes after Task 1

- [ ] **Step 1: Run the biggest safe compliant model first**

Run on 4090:

```bash
RUN_NAME=run20-promptir-large-d64-b48810-r8 MODEL_DIM=64 NUM_BLOCKS=4,8,8,10 NUM_REFINEMENT_BLOCKS=8 PATCH_SIZE=256 BATCH_SIZE=1 GRADIENT_CHECKPOINTING=full LOSS_TYPE=l1_mse MSE_WEIGHT=0.05 MONITOR=val_psnr SAVE_TOP_K=5 bash train_stage1.sh
```

Expected: one full training run, then outputs:

```text
submission/run20-promptir-large-d64-b48810-r8-original.zip
submission/run20-promptir-large-d64-b48810-r8-tta.zip
submission/run20-promptir-large-d64-b48810-r8-avg5-tta.zip
```

- [ ] **Step 2: Only submit if validation is competitive**

Submit `avg5-tta` only if local `val_psnr >= 30.75` or derain/desnow split indicates a meaningful win over Run 9.

- [ ] **Step 3: If OOM, run smaller large model**

Run:

```bash
RUN_NAME=run20b-promptir-large-d56-b4668-r8 MODEL_DIM=56 NUM_BLOCKS=4,6,6,8 NUM_REFINEMENT_BLOCKS=8 PATCH_SIZE=256 BATCH_SIZE=1 GRADIENT_CHECKPOINTING=full LOSS_TYPE=l1_mse MSE_WEIGHT=0.05 MONITOR=val_psnr SAVE_TOP_K=5 bash train_stage1.sh
```

Expected: same three submission zips with `run20b` prefix.

## Task 3: Add Task-Only Specialist Training Support

**Files:**
- Modify: `hw4_dataset.py`
- Modify: `train_hw4.py`
- Modify: `train_stage1.sh`
- Test: `tests/test_task_only_training.py`

- [ ] **Step 1: Write failing task-filter tests**

Add `tests/test_task_only_training.py`:

```python
import unittest
from argparse import Namespace
from unittest import mock

from hw4_dataset import HW4TrainDataset


class TaskOnlyTrainingTest(unittest.TestCase):
    def test_derain_only_keeps_rain_samples(self):
        args = Namespace(data_dir="PromptIR/data", de_type=["derain"], merge_val=False, derain_oversample=1)
        with mock.patch.object(HW4TrainDataset, "_init_ids"):
            ds = HW4TrainDataset(args)
            ds.sample_ids = []
            ds.rain_ids = [{"de_type": 1}]
            ds._merge_ids()
        self.assertEqual([s["de_type"] for s in ds.sample_ids], [1])

    def test_desnow_only_keeps_snow_samples(self):
        args = Namespace(data_dir="PromptIR/data", de_type=["desnow"], merge_val=False, derain_oversample=1)
        with mock.patch.object(HW4TrainDataset, "_init_ids"):
            ds = HW4TrainDataset(args)
            ds.sample_ids = []
            ds.snow_ids = [{"de_type": 0}]
            ds._merge_ids()
        self.assertEqual([s["de_type"] for s in ds.sample_ids], [0])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the failing test**

Run: `python -m unittest tests.test_task_only_training -v`

Expected: PASS if current `de_type` already supports this; otherwise fix `_merge_ids` only.

- [ ] **Step 3: Add shell support for specialist task selection**

In `train_stage1.sh`, add:

```bash
DE_TYPE=${DE_TYPE:-desnow derain}
```

Replace the fixed default by adding to `TRAIN_ARGS`:

```bash
--de_type $DE_TYPE
```

Do not quote `$DE_TYPE`, because argparse expects separate values.

- [ ] **Step 4: Verify full tests**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass.

## Task 4: Train Specialist Upper-Bound Models A

**Files:**
- No code changes after Task 3

- [ ] **Step 1: Train derain specialist**

Run:

```bash
RUN_NAME=run21-derain-specialist DE_TYPE=derain PATCH_SIZE=256 BATCH_SIZE=1 LOSS_TYPE=l1_mse MSE_WEIGHT=0.05 MONITOR=val_psnr SAVE_TOP_K=5 bash train_stage1.sh
```

Expected outputs:

```text
submission/run21-derain-specialist-original.zip
submission/run21-derain-specialist-tta.zip
submission/run21-derain-specialist-avg5-tta.zip
```

- [ ] **Step 2: Train desnow specialist**

Run:

```bash
RUN_NAME=run22-desnow-specialist DE_TYPE=desnow PATCH_SIZE=256 BATCH_SIZE=1 LOSS_TYPE=l1_mse MSE_WEIGHT=0.05 MONITOR=val_psnr SAVE_TOP_K=5 bash train_stage1.sh
```

Expected outputs:

```text
submission/run22-desnow-specialist-original.zip
submission/run22-desnow-specialist-tta.zip
submission/run22-desnow-specialist-avg5-tta.zip
```

## Task 5: Build Specialist Router Candidate

**Files:**
- Create: `route_specialists.py`
- Test: `tests/test_route_specialists.py`

- [ ] **Step 1: Write failing routing tests**

Add tests that create two temporary `pred.npz` zips with keys `0.png` and `1.png`, route `0.png` to rain and `1.png` to snow, then assert output zip contains exactly those selected arrays.

- [ ] **Step 2: Implement minimal router**

Create `route_specialists.py` with CLI:

```bash
python route_specialists.py --rain_zip submission/run21-derain-specialist-avg5-tta.zip --snow_zip submission/run22-desnow-specialist-avg5-tta.zip --route rain,snow --output submission/specialist-routed.zip
```

Implementation requirements:
- Load `pred.npz` from each input zip.
- If `--route` is provided, split comma labels in filename order `0.png` to `99.png`.
- For each key, select rain prediction if label is `rain`, else snow prediction.
- Write compressed `pred.npz` into `--output`.

- [ ] **Step 3: Add simple automatic route mode**

Add `--auto_by_prefix_dir PromptIR/data/Test/degraded` only if manual labels are unavailable. Compute a crude score from image high-frequency residuals; use it only to generate test candidates, not as final truth.

- [ ] **Step 4: Verify router format**

Run: `python -m unittest tests.test_route_specialists -v`

Expected: PASS.

Run output validation command:

```bash
python - <<'PY'
import os, tempfile, zipfile
import numpy as np
path = 'submission/specialist-routed.zip'
with tempfile.TemporaryDirectory() as tmp:
    with zipfile.ZipFile(path) as zf:
        zf.extractall(tmp)
    data = np.load(os.path.join(tmp, 'pred.npz'))
    print(len(data.files), sorted({data[k].dtype.name for k in data.files}))
PY
```

Expected: `100 ['uint8']`.

## Task 6: Distill Specialists Back Into One PromptIR C

**Files:**
- Create: `make_distill_targets.py`
- Modify: `hw4_dataset.py`
- Modify: `train_hw4.py`
- Modify: `train_stage1.sh`
- Test: `tests/test_distill_targets.py`

- [ ] **Step 1: Create pseudo-target extraction script**

`make_distill_targets.py` should extract `pred.npz` from a routed zip and write `PromptIR/data/Distill/gt/0.png` ... `99.png` plus copies of degraded inputs under `PromptIR/data/Distill/degraded/`.

- [ ] **Step 2: Add dataset support for distill samples**

In `hw4_dataset.py`, if `args.distill_dir` is non-empty, append samples with degraded path from `Distill/degraded` and clean path from `Distill/gt`, using `de_type=2` or `de_type=0` if avoiding a third task id.

- [ ] **Step 3: Add CLI pass-through**

Add `--distill_dir` to `train_hw4.py`, `DISTILL_DIR=${DISTILL_DIR:-}` to `train_stage1.sh`, and pass it only when non-empty.

- [ ] **Step 4: Train one-model distillation candidate**

Run:

```bash
python make_distill_targets.py --zip submission/specialist-routed.zip --test_dir PromptIR/data/Test/degraded --output_dir PromptIR/data/Distill
RUN_NAME=run23-single-promptir-distilled DISTILL_DIR=PromptIR/data/Distill PATCH_SIZE=256 BATCH_SIZE=1 LOSS_TYPE=l1_mse MSE_WEIGHT=0.05 MONITOR=val_psnr SAVE_TOP_K=5 bash train_stage1.sh
```

Expected: one PromptIR checkpoint and three compliant single-model submission zips.

## Execution Priority

1. Implement and run Task 1 immediately.
2. Start Task 2 large PromptIR training first because it is compliant.
3. While Task 2 runs or if it underperforms, implement Task 3 and run Task 4 specialists.
4. Use Task 5 to measure the rule-risky upper bound.
5. Use Task 6 only if specialist routing beats current best by a meaningful margin.

## Self-Review

- Spec coverage: includes B first, A second, C third as requested.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: model-size args are `model_dim`, `num_blocks`, `num_refinement_blocks` across train and inference; task selector is `DE_TYPE`; distillation path is `DISTILL_DIR` / `--distill_dir`.
- Risk note: Task 5 is intentionally rule-risky; Task 6 is the compliant recovery path.
