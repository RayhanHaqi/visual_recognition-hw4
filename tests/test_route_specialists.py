import os
import tempfile
import unittest
import zipfile

import numpy as np

from route_specialists import load_pred_npz, route_predictions, save_pred_npz


class RouteSpecialistsTest(unittest.TestCase):
    def _make_zip(self, path, pred_dict):
        npz_path = os.path.join(os.path.dirname(path), "pred.npz")
        np.savez_compressed(npz_path, **pred_dict)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(npz_path, "pred.npz")
        os.remove(npz_path)

    def test_route_selects_rain_or_snow_per_key(self):
        rain = {f"{i}.png": np.full((3, 4, 4), 10, dtype=np.uint8) for i in range(2)}
        snow = {f"{i}.png": np.full((3, 4, 4), 200, dtype=np.uint8) for i in range(2)}
        labels = ["rain", "snow"]
        routed = route_predictions(rain, snow, labels)
        self.assertTrue(np.array_equal(routed["0.png"], rain["0.png"]))
        self.assertTrue(np.array_equal(routed["1.png"], snow["1.png"]))

    def test_route_zip_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            rain_zip = os.path.join(tmp, "rain.zip")
            snow_zip = os.path.join(tmp, "snow.zip")
            out_zip = os.path.join(tmp, "routed.zip")
            route_file = os.path.join(tmp, "routes.txt")
            self._make_zip(rain_zip, {"0.png": np.ones((3, 2, 2), dtype=np.uint8)})
            self._make_zip(snow_zip, {"0.png": np.zeros((3, 2, 2), dtype=np.uint8)})
            with open(route_file, "w", encoding="utf-8") as f:
                f.write("rain\n")

            from route_specialists import main
            import sys
            argv = [
                "route_specialists.py",
                "--rain_zip", rain_zip,
                "--snow_zip", snow_zip,
                "--route_file", route_file,
                "--output", out_zip,
            ]
            old = sys.argv
            try:
                sys.argv = argv
                main()
            finally:
                sys.argv = old

            data = load_pred_npz(out_zip)
            self.assertEqual(data["0.png"].dtype, np.uint8)
            self.assertTrue(np.array_equal(data["0.png"], np.ones((3, 2, 2), dtype=np.uint8)))


if __name__ == "__main__":
    unittest.main()
