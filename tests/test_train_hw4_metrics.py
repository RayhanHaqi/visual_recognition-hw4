import unittest

import torch


class ValidationBatchShapeTest(unittest.TestCase):
    def test_validation_batch_has_name_de_id_degraded_clean(self):
        batch = ("rain-1.png", 1, torch.zeros(1, 3, 16, 16), torch.ones(1, 3, 16, 16))
        name, de_id, degraded, clean = batch
        self.assertEqual(name, "rain-1.png")
        self.assertEqual(de_id, 1)
        self.assertEqual(tuple(degraded.shape), (1, 3, 16, 16))
        self.assertEqual(tuple(clean.shape), (1, 3, 16, 16))


if __name__ == "__main__":
    unittest.main()
