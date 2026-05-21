from pathlib import Path
import unittest


class TrainStage1ScriptTest(unittest.TestCase):

    def test_stage1_script_trains_once_and_emits_original_and_tta_outputs(self):
        script = Path("train_stage1.sh").read_text()

        self.assertIn("PATCH_SIZE=${PATCH_SIZE:-384}", script)
        self.assertIn("BATCH_SIZE=${BATCH_SIZE:-1}", script)
        self.assertIn("SAVE_TOP_K=${SAVE_TOP_K:-3}", script)
        self.assertIn("EMA=${EMA:-1}", script)
        self.assertIn("EMA_DECAY=${EMA_DECAY:-0.9999}", script)
        self.assertEqual(script.count("python train_hw4.py"), 1)
        self.assertIn("--patch_size \"$PATCH_SIZE\"", script)
        self.assertIn("--save_top_k \"$SAVE_TOP_K\"", script)
        self.assertIn("--ema", script)
        self.assertIn("--ema_decay \"$EMA_DECAY\"", script)
        self.assertIn("submission/stage1-p${PATCH_SIZE}-original.zip", script)
        self.assertIn("submission/stage1-p${PATCH_SIZE}-tta.zip", script)
        self.assertIn("submission/stage1-p${PATCH_SIZE}-avg${SAVE_TOP_K}-tta.zip", script)
        self.assertIn("--tta", script)
        self.assertIn("python average_checkpoints.py", script)
        self.assertIn("Existing checkpoints found", script)
        self.assertNotIn("--merge_val", script)
        self.assertNotIn("Stage 2", script)

    def test_train_hw4_exposes_ema_and_top_k_options(self):
        script = Path("train_hw4.py").read_text()

        self.assertIn("--save_top_k", script)
        self.assertIn("--ema", script)
        self.assertIn("--ema_decay", script)
        self.assertIn("EMAWeightAveraging", script)
        self.assertIn("save_top_k=args.save_top_k", script)
        self.assertNotIn("args.save_top_k = -1", script)

    def test_stage1_script_parses_val_loss_without_ckpt_suffix(self):
        script = Path("train_stage1.sh").read_text()

        self.assertIn(r"val_loss=([0-9]+(?:\.[0-9]+)?)", script)
        self.assertNotIn(r"val_loss=([0-9.]+)", script)

    def test_stage1_script_autosave_commits_before_rebase(self):
        commands = [line.strip() for line in Path("train_stage1.sh").read_text().splitlines()]

        add_index = commands.index("git add -A")
        commit_index = next(
            i for i, line in enumerate(commands)
            if line.startswith("git commit -m ")
        )
        pull_index = commands.index("git pull --rebase")
        push_index = commands.index("git push")

        self.assertLess(add_index, commit_index)
        self.assertLess(commit_index, pull_index)
        self.assertLess(pull_index, push_index)


if __name__ == "__main__":
    unittest.main()
