import unittest

import torch

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


if __name__ == "__main__":
    unittest.main()
