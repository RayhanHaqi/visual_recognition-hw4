from pathlib import Path
import unittest


class TrainStage1ScriptTest(unittest.TestCase):

    def test_stage1_script_trains_once_and_emits_original_and_tta_outputs(self):
        script = Path("train_stage1.sh").read_text()

        self.assertIn("PATCH_SIZE=${PATCH_SIZE:-192}", script)
        self.assertIn("BATCH_SIZE=${BATCH_SIZE:-2}", script)
        self.assertEqual(script.count("python train_hw4.py"), 1)
        self.assertIn("--patch_size \"$PATCH_SIZE\"", script)
        self.assertIn("submission/stage1-p${PATCH_SIZE}-original.zip", script)
        self.assertIn("submission/stage1-p${PATCH_SIZE}-tta.zip", script)
        self.assertIn("--tta", script)
        self.assertNotIn("--merge_val", script)
        self.assertNotIn("Stage 2", script)

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
