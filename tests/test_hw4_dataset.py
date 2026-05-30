import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import numpy as np
from PIL import Image

from hw4_dataset import HW4TrainDataset, HW4ValDataset, random_augmentation


class HW4DatasetTest(unittest.TestCase):

    def test_random_augmentation_supports_identity_mode(self):
        img = np.arange(12).reshape(3, 4)

        with patch("hw4_dataset.random.randint", return_value=0):
            augmented, = random_augmentation(img)

        self.assertTrue(np.array_equal(augmented, img))


class OfficialAugmentationTest(unittest.TestCase):
    def test_random_augmentation_can_exclude_identity(self):
        img = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)

        for _ in range(100):
            aug = random_augmentation(img, allow_identity=False)[0]
            self.assertFalse(np.array_equal(aug, img))


class AuxiliaryDenoiseDatasetTest(unittest.TestCase):
    def test_aux_denoise_samples_use_hw4_clean_paths(self):
        args = argparse.Namespace(
            data_dir='PromptIR/data', de_type=['desnow', 'derain'], merge_val=False,
            derain_oversample=1, aux_denoise=1, aux_denoise_sigmas='15,25,50')

        with mock.patch.object(HW4TrainDataset, '_init_ids'), \
                mock.patch.object(HW4TrainDataset, '_merge_ids'):
            ds = HW4TrainDataset(args)
        ds.sample_ids = []
        ds.snow_ids = [{'clean_path': 'snow_clean-1.png', 'de_type': 0}]
        ds.rain_ids = [{'clean_path': 'rain_clean-1.png', 'de_type': 1}]
        HW4TrainDataset._merge_ids(ds)

        aux = [sample for sample in ds.sample_ids if sample['de_type'] == 2]
        self.assertEqual(len(aux), 6)
        self.assertEqual(sorted({sample['sigma'] for sample in aux}), [15, 25, 50])


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
