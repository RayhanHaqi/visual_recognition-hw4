import unittest
from argparse import Namespace
from unittest import mock

from hw4_dataset import HW4TrainDataset


class TaskOnlyTrainingTest(unittest.TestCase):
    def test_derain_only_keeps_rain_samples(self):
        args = Namespace(
            data_dir="PromptIR/data",
            de_type=["derain"],
            merge_val=False,
            derain_oversample=1,
            aux_denoise=0,
            distill_dir="",
        )
        with mock.patch.object(HW4TrainDataset, "_init_ids"):
            ds = HW4TrainDataset(args)
            ds.sample_ids = []
            ds.rain_ids = [{"de_type": 1, "degraded_path": "a", "clean_path": "b"}]
            ds.snow_ids = []
            HW4TrainDataset._merge_ids(ds)
        self.assertEqual([s["de_type"] for s in ds.sample_ids], [1])

    def test_desnow_only_keeps_snow_samples(self):
        args = Namespace(
            data_dir="PromptIR/data",
            de_type=["desnow"],
            merge_val=False,
            derain_oversample=1,
            aux_denoise=0,
            distill_dir="",
        )
        with mock.patch.object(HW4TrainDataset, "_init_ids"):
            ds = HW4TrainDataset(args)
            ds.sample_ids = []
            ds.snow_ids = [{"de_type": 0, "degraded_path": "a", "clean_path": "b"}]
            ds.rain_ids = []
            HW4TrainDataset._merge_ids(ds)
        self.assertEqual([s["de_type"] for s in ds.sample_ids], [0])


if __name__ == "__main__":
    unittest.main()
