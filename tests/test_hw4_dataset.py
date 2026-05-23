import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from hw4_dataset import HW4ValDataset, random_augmentation


class HW4DatasetTest(unittest.TestCase):

    def test_random_augmentation_supports_identity_mode(self):
        img = np.arange(12).reshape(3, 4)

        with patch("hw4_dataset.random.randint", return_value=0):
            augmented, = random_augmentation(img)

        self.assertTrue(np.array_equal(augmented, img))


class HW4ValDatasetDegradationIdTest(unittest.TestCase):
    def _write_rgb(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)).save(path)

    def test_val_dataset_returns_rain_and_snow_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_rgb(root / "Val" / "Derain" / "rainy" / "rain-1.png")
            self._write_rgb(root / "Val" / "Derain" / "gt" / "rain_clean-1.png")
            self._write_rgb(root / "Val" / "Desnow" / "snowy" / "snow-1.png")
            self._write_rgb(root / "Val" / "Desnow" / "gt" / "snow_clean-1.png")

            dataset = HW4ValDataset(str(root))
            ids = {dataset[i][0]: dataset[i][1] for i in range(len(dataset))}

        self.assertEqual(ids["rain-1.png"], 1)
        self.assertEqual(ids["snow-1.png"], 0)


if __name__ == "__main__":
    unittest.main()
