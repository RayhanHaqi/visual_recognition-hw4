import argparse
import zipfile
import tempfile
import os
import numpy as np


def load_pred_npz(zip_path):
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmp)
        return dict(np.load(os.path.join(tmp, 'pred.npz')))


def save_pred_npz(pred_dict, output_path):
    os.makedirs("submission", exist_ok=True)
    npz_path = "submission/pred.npz"
    np.savez_compressed(npz_path, **pred_dict)
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(npz_path, "pred.npz")
    os.remove(npz_path)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zips", nargs="+", help="Submission zip files to ensemble")
    parser.add_argument("--output", required=True, help="Output zip path")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Per-zip weights (default: equal)")
    args = parser.parse_args()

    if args.weights is not None and len(args.weights) != len(args.zips):
        raise ValueError("Number of weights must match number of zips")

    all_preds = [load_pred_npz(z) for z in args.zips]
    keys = sorted(all_preds[0].keys())
    for pred in all_preds:
        if sorted(pred.keys()) != keys:
            raise ValueError("Zip keys do not match across submissions")

    weights = args.weights or [1.0 / len(args.zips)] * len(args.zips)

    ensemble = {}
    for key in keys:
        stacked = np.stack([p[key].astype(np.float64) for p in all_preds], axis=0)
        weighted = np.average(stacked, axis=0, weights=weights)
        ensemble[key] = np.rint(np.clip(weighted, 0, 255)).astype(np.uint8)

    save_pred_npz(ensemble, args.output)


if __name__ == "__main__":
    main()
