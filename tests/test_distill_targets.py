import os
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from make_distill_targets import extract_distill_targets


class DistillTargetsTest(unittest.TestCase):
    def test_extract_writes_degraded_gt_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_dir = os.path.join(tmp, "test")
            out_dir = os.path.join(tmp, "Distill")
            os.makedirs(test_dir, exist_ok=True)
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(
                os.path.join(test_dir, "0.png"))

            npz_path = os.path.join(tmp, "pred.npz")
            gt = np.full((3, 16, 16), 255, dtype=np.uint8)
            np.savez_compressed(npz_path, **{"0.png": gt})
            zip_path = os.path.join(tmp, "routed.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(npz_path, "pred.npz")

            route_file = os.path.join(tmp, "routes.txt")
            with open(route_file, "w", encoding="utf-8") as f:
                f.write("rain\n")

            count = extract_distill_targets(zip_path, test_dir, out_dir, route_file)
            self.assertEqual(count, 1)
            self.assertTrue(os.path.exists(os.path.join(out_dir, "degraded", "0.png")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "gt", "0.png")))
            with open(os.path.join(out_dir, "labels.txt"), encoding="utf-8") as f:
                self.assertEqual(f.read().strip(), "1")


if __name__ == "__main__":
    unittest.main()
