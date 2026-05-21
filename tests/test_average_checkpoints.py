import tempfile
import unittest
from pathlib import Path

import torch

from average_checkpoints import average_checkpoints


class AverageCheckpointsTest(unittest.TestCase):

    def test_average_checkpoints_writes_compact_model_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt_a = tmp_path / "a.ckpt"
            ckpt_b = tmp_path / "b.ckpt"
            output = tmp_path / "avg.ckpt"

            torch.save({
                "state_dict": {
                    "net.weight": torch.tensor([1.0, 3.0]),
                    "net.bias": torch.tensor([5.0]),
                    "net.counter": torch.tensor([7], dtype=torch.int64),
                }
            }, ckpt_a)
            torch.save({
                "state_dict": {
                    "net.weight": torch.tensor([3.0, 7.0]),
                    "net.bias": torch.tensor([9.0]),
                    "net.counter": torch.tensor([11], dtype=torch.int64),
                }
            }, ckpt_b)

            average_checkpoints([ckpt_a, ckpt_b], output)

            averaged = torch.load(output, map_location="cpu")
            self.assertEqual(set(averaged), {"weight", "bias", "counter"})
            self.assertTrue(torch.equal(averaged["weight"], torch.tensor([2.0, 5.0])))
            self.assertTrue(torch.equal(averaged["bias"], torch.tensor([7.0])))
            self.assertTrue(torch.equal(averaged["counter"], torch.tensor([7], dtype=torch.int64)))

    def test_average_checkpoints_rejects_mismatched_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt_a = tmp_path / "a.ckpt"
            ckpt_b = tmp_path / "b.ckpt"
            output = tmp_path / "avg.ckpt"

            torch.save({"state_dict": {"net.weight": torch.tensor([1.0])}}, ckpt_a)
            torch.save({"state_dict": {"net.other": torch.tensor([1.0])}}, ckpt_b)

            with self.assertRaisesRegex(ValueError, "keys do not match"):
                average_checkpoints([ckpt_a, ckpt_b], output)


if __name__ == "__main__":
    unittest.main()
