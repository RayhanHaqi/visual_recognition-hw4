import unittest

import torch

from inference import (
    TTA_MODES,
    apply_tta_transform,
    invert_tta_transform,
    restore_image,
)


class InferenceTTATest(unittest.TestCase):

    def test_tta_transforms_round_trip_to_original_tensor(self):
        img = torch.arange(1 * 3 * 5 * 7).reshape(1, 3, 5, 7)

        self.assertEqual(len(TTA_MODES), 8)
        for mode in TTA_MODES:
            transformed = apply_tta_transform(img, mode)
            restored = invert_tta_transform(transformed, mode)
            self.assertTrue(torch.equal(restored, img), mode)

    def test_restore_image_averages_all_tta_predictions(self):
        class AddOneModel(torch.nn.Module):
            def forward(self, x):
                return x + 1

        img = torch.zeros(1, 3, 5, 7)

        restored = restore_image(AddOneModel(), img, use_tta=True)

        self.assertEqual(restored.shape, img.shape)
        self.assertTrue(torch.allclose(restored, torch.ones_like(img)))


if __name__ == "__main__":
    unittest.main()
