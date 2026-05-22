import unittest

import numpy as np
import torch

from inference import (
    TTA_MODES,
    apply_tta_transform,
    invert_tta_transform,
    restore_image,
    tensor_to_uint8,
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

    def test_tensor_to_uint8_rounds_instead_of_flooring(self):
        restored = torch.tensor([[[[0.0, 0.5, 1.0 / 255.0, 254.6 / 255.0, 1.0]]]])

        converted = tensor_to_uint8(restored)

        self.assertEqual(converted.dtype, np.uint8)
        self.assertEqual(converted.tolist(), [[[0, 128, 1, 255, 255]]])


if __name__ == "__main__":
    unittest.main()
