import unittest
from pathlib import Path
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

    def test_inference_exposes_matching_model_size_options(self):
        text = Path("inference.py").read_text()

        self.assertIn("--model_dim", text)
        self.assertIn("--num_blocks", text)
        self.assertIn("--num_refinement_blocks", text)
        self.assertIn("parse_num_blocks", text)

    def test_stage1_script_passes_model_size_to_train_and_inference(self):
        text = Path("train_stage1.sh").read_text()

        self.assertIn("MODEL_DIM=${MODEL_DIM:-48}", text)
        self.assertIn("NUM_BLOCKS=${NUM_BLOCKS:-4,6,6,8}", text)
        self.assertIn("NUM_REFINEMENT_BLOCKS=${NUM_REFINEMENT_BLOCKS:-4}", text)
        self.assertIn('--model_dim "$MODEL_DIM"', text)
        self.assertIn('--num_blocks "$NUM_BLOCKS"', text)
        self.assertIn('--num_refinement_blocks "$NUM_REFINEMENT_BLOCKS"', text)


if __name__ == "__main__":
    unittest.main()
