import argparse
import os
import shutil
import tempfile
import zipfile

import numpy as np
from PIL import Image

from route_specialists import load_pred_npz, load_route_labels


def _chw_to_hwc(arr):
    if arr.ndim == 3 and arr.shape[0] == 3:
        return np.transpose(arr, (1, 2, 0))
    return arr


def extract_distill_targets(zip_path, test_dir, output_dir, route_file):
    pred = load_pred_npz(zip_path)
    labels = load_route_labels(route_file)
    keys = sorted(pred.keys())
    if len(labels) != len(keys):
        raise ValueError(
            f"Route file has {len(labels)} labels but zip has {len(keys)} predictions"
        )

    degraded_dir = os.path.join(output_dir, "degraded")
    gt_dir = os.path.join(output_dir, "gt")
    os.makedirs(degraded_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    de_ids = []
    for key, label in zip(keys, labels):
        stem = os.path.splitext(key)[0]
        src_degraded = os.path.join(test_dir, f"{stem}.png")
        if not os.path.exists(src_degraded):
            raise FileNotFoundError(f"Missing test degraded image: {src_degraded}")

        shutil.copy2(src_degraded, os.path.join(degraded_dir, key))
        gt_arr = _chw_to_hwc(pred[key])
        if gt_arr.dtype != np.uint8:
            gt_arr = np.rint(np.clip(gt_arr, 0, 255)).astype(np.uint8)
        Image.fromarray(gt_arr).save(os.path.join(gt_dir, key))
        de_ids.append("1" if label == "rain" else "0")

    labels_path = os.path.join(output_dir, "labels.txt")
    with open(labels_path, "w", encoding="utf-8") as f:
        f.write("\n".join(de_ids) + "\n")

    return len(keys)


def main():
    parser = argparse.ArgumentParser(description="Build distillation targets from routed predictions")
    parser.add_argument("--zip", required=True, help="Routed specialist submission zip")
    parser.add_argument("--test_dir", default="PromptIR/data/Test/degraded")
    parser.add_argument("--output_dir", default="PromptIR/data/Distill")
    parser.add_argument("--route_file", required=True, help="rain/snow label file (same as routing)")
    args = parser.parse_args()

    count = extract_distill_targets(args.zip, args.test_dir, args.output_dir, args.route_file)
    print(f"Wrote {count} distill pairs to {args.output_dir}")


if __name__ == "__main__":
    main()
