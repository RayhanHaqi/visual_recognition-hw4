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


if __name__ == "__main__":
    unittest.main()
