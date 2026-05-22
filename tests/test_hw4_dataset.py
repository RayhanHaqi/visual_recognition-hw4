import unittest
from unittest.mock import patch

import numpy as np

from hw4_dataset import random_augmentation


class HW4DatasetTest(unittest.TestCase):

    def test_random_augmentation_supports_identity_mode(self):
        img = np.arange(12).reshape(3, 4)

        with patch("hw4_dataset.random.randint", return_value=0):
            augmented, = random_augmentation(img)

        self.assertTrue(np.array_equal(augmented, img))


if __name__ == "__main__":
    unittest.main()
