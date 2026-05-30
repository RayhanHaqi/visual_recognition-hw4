from pathlib import Path
import unittest


class TrainStage1ScriptTest(unittest.TestCase):

    def test_stage1_script_trains_once_and_emits_original_and_tta_outputs(self):
        script = Path("train_stage1.sh").read_text()

        self.assertIn("PATCH_SIZE=${PATCH_SIZE:-256}", script)
        self.assertIn("BATCH_SIZE=${BATCH_SIZE:-1}", script)
        self.assertIn("SAVE_TOP_K=${SAVE_TOP_K:-5}", script)
        self.assertIn("CKPT_EVERY_N_EPOCHS=${CKPT_EVERY_N_EPOCHS:-5}", script)
        self.assertIn("EMA=${EMA:-0}", script)
        self.assertIn("EMA_DECAY=${EMA_DECAY:-0.9999}", script)
        self.assertIn("MONITOR=${MONITOR:-val_psnr}", script)
        self.assertEqual(script.count("python train_hw4.py"), 1)
        self.assertIn("--patch_size \"$PATCH_SIZE\"", script)
        self.assertIn("--save_top_k \"$SAVE_TOP_K\"", script)
        self.assertIn("--every_n_epochs \"$CKPT_EVERY_N_EPOCHS\"", script)
        self.assertIn("--monitor \"$MONITOR\"", script)
        self.assertIn("--gradient_checkpointing \"$GRADIENT_CHECKPOINTING\"", script)
        self.assertIn("--compile", script)
        self.assertIn("--ema", script)
        self.assertIn("--ema_decay \"$EMA_DECAY\"", script)
        self.assertIn("submission/${RUN_NAME}-original.zip", script)
        self.assertIn("submission/${RUN_NAME}-tta.zip", script)
        self.assertIn("submission/${RUN_NAME}-avg${SAVE_TOP_K}-tta.zip", script)
        self.assertIn("--tta", script)
        self.assertIn("python average_checkpoints.py", script)
        self.assertIn("Existing checkpoints found", script)
        self.assertNotIn("--merge_val", script)
        self.assertNotIn("Stage 2", script)

    def test_train_hw4_exposes_ema_and_top_k_options(self):
        script = Path("train_hw4.py").read_text()

        self.assertIn("--save_top_k", script)
        self.assertIn("--every_n_epochs", script)
        self.assertIn("--monitor", script)
        self.assertIn("--ema", script)
        self.assertIn("--ema_decay", script)
        self.assertIn("val_psnr", script)
        self.assertIn('mode="max" if args.monitor == "val_psnr" else "min"', script)
        self.assertIn('{val_psnr:.6f}', script)
        self.assertIn("EMAWeightAveraging", script)
        self.assertIn("save_top_k=args.save_top_k", script)
        self.assertIn("every_n_epochs=args.every_n_epochs", script)
        self.assertNotIn("args.save_top_k = -1", script)
        self.assertIn("--loss_type", script)
        self.assertIn("--mse_weight", script)
        self.assertIn("--task_conditioning", script)
        self.assertIn("--sipl_lite", script)
        self.assertIn("--gradient_checkpointing", script)
        self.assertIn("--compile", script)
        self.assertIn("--rain_loss_weight", script)
        self.assertIn("CheckpointedSequential", script)
        self.assertIn("checkpoint_sequential", script)
        self.assertIn("torch.backends.cudnn.benchmark = True", script)
        self.assertIn('--force_aug_no_identity', script)
        self.assertIn('--aux_denoise', script)
        self.assertIn('--aux_denoise_sigmas', script)

    def test_stage1_script_exposes_promptir_only_experiment_flags(self):
        text = Path("train_stage1.sh").read_text()
        self.assertIn('LOSS_TYPE=${LOSS_TYPE:-l1_mse}', text)
        self.assertIn('MSE_WEIGHT=${MSE_WEIGHT:-0.05}', text)
        self.assertIn('TASK_CONDITIONING=${TASK_CONDITIONING:-0}', text)
        self.assertIn('SIPL_LITE=${SIPL_LITE:-0}', text)
        self.assertIn('SIPL_START_ALPHA=${SIPL_START_ALPHA:-0.5}', text)
        self.assertIn('SIPL_REFINE_WEIGHT=${SIPL_REFINE_WEIGHT:-0.5}', text)
        self.assertIn('GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-none}', text)
        self.assertIn('COMPILE=${COMPILE:-0}', text)
        self.assertIn('RAIN_LOSS_WEIGHT=${RAIN_LOSS_WEIGHT:-1.0}', text)
        self.assertIn('PAIR_MIX_PROB=${PAIR_MIX_PROB:-0.0}', text)
        self.assertIn('HARD_PATCH_PROB=${HARD_PATCH_PROB:-0.0}', text)
        self.assertIn('--gradient_checkpointing "$GRADIENT_CHECKPOINTING"', text)

    def test_stage1_script_exposes_b1_official_recipe_flags(self):
        text = Path("train_stage1.sh").read_text()

        self.assertIn('FORCE_AUG_NO_IDENTITY=${FORCE_AUG_NO_IDENTITY:-0}', text)
        self.assertIn('AUX_DENOISE=${AUX_DENOISE:-0}', text)
        self.assertIn('AUX_DENOISE_SIGMAS=${AUX_DENOISE_SIGMAS:-15,25,50}', text)
        self.assertIn('--force_aug_no_identity', text)
        self.assertIn('--aux_denoise "$AUX_DENOISE"', text)
        self.assertIn('--aux_denoise_sigmas "$AUX_DENOISE_SIGMAS"', text)

    def test_stage1_script_parses_metric_without_ckpt_suffix(self):
        script = Path("train_stage1.sh").read_text()

        self.assertIn(r"rf'{metric}=([0-9]+(?:\.[0-9]+)?)'", script)
        self.assertIn('missing = float("-inf") if reverse else float("inf")', script)
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
