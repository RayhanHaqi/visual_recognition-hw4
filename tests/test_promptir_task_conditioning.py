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

    def test_unconditioned_matches_backbone_shape(self):
        model = TaskConditionedRestorer(IdentityNet(), enabled=False)
        x = torch.zeros(1, 3, 8, 8)
        y = model(x, torch.tensor([1]))
        self.assertEqual(tuple(y.shape), tuple(x.shape))

    def test_disabled_ignores_label_and_passes_through(self):
        model = TaskConditionedRestorer(IdentityNet(), enabled=False)
        x = torch.zeros(1, 3, 8, 8)
        y = model(x, de_id=None)
        self.assertEqual(tuple(y.shape), tuple(x.shape))


if __name__ == "__main__":
    unittest.main()
