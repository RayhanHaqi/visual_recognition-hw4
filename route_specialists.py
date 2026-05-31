import argparse
import os
import tempfile
import zipfile

import numpy as np


def load_pred_npz(zip_path):
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)
        return dict(np.load(os.path.join(tmp, "pred.npz")))


def save_pred_npz(pred_dict, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = os.path.join(tmp, "pred.npz")
        np.savez_compressed(npz_path, **pred_dict)
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(npz_path, "pred.npz")
    print(f"Saved: {output_path}")


def load_route_labels(route_file):
    labels = []
    with open(route_file, encoding="utf-8") as f:
        for line in f:
            label = line.strip().lower()
            if not label:
                continue
            if label not in ("rain", "snow"):
                raise ValueError(f"Invalid route label '{label}' in {route_file}; use rain or snow")
            labels.append(label)
    if not labels:
        raise ValueError(f"No route labels found in {route_file}")
    return labels


def route_predictions(rain_pred, snow_pred, labels):
    keys = sorted(rain_pred.keys())
    if sorted(snow_pred.keys()) != keys:
        raise ValueError("Rain and snow zip keys do not match")

    if len(labels) != len(keys):
        raise ValueError(
            f"Route file has {len(labels)} labels but predictions have {len(keys)} images"
        )

    routed = {}
    for key, label in zip(keys, labels):
        routed[key] = rain_pred[key] if label == "rain" else snow_pred[key]
    return routed


def main():
    parser = argparse.ArgumentParser(description="Route derain/desnow specialist predictions into one zip")
    parser.add_argument("--rain_zip", required=True, help="Derain specialist submission zip")
    parser.add_argument("--snow_zip", required=True, help="Desnow specialist submission zip")
    parser.add_argument("--route_file", required=True, help="One rain/snow label per test image line")
    parser.add_argument("--output", required=True, help="Output submission zip path")
    args = parser.parse_args()

    rain_pred = load_pred_npz(args.rain_zip)
    snow_pred = load_pred_npz(args.snow_zip)
    labels = load_route_labels(args.route_file)
    routed = route_predictions(rain_pred, snow_pred, labels)
    save_pred_npz(routed, args.output)


if __name__ == "__main__":
    main()
