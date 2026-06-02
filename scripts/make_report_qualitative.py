#!/usr/bin/env python3
"""Build restoration_examples.pdf/png for the HW4 report (validation pairs)."""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HW4_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VAL_ROOT = os.path.join(HW4_ROOT, "PromptIR", "data", "Val")
REPORT_DIR = os.path.join(HW4_ROOT, "report")

SAMPLES = (
    ("Derain", "rainy", "rain-1005.png", "gt", "rain_clean-1005.png", "Rain"),
    ("Desnow", "snowy", "snow-100.png", "gt", "snow_clean-100.png", "Snow"),
)


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default=REPORT_DIR,
        help="Directory for restoration_examples.pdf/png",
    )
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2))
    for row, (task, deg_dir, deg_name, gt_dir, gt_name, label) in enumerate(SAMPLES):
        deg_path = os.path.join(VAL_ROOT, task, deg_dir, deg_name)
        gt_path = os.path.join(VAL_ROOT, task, gt_dir, gt_name)
        if not os.path.isfile(deg_path) or not os.path.isfile(gt_path):
            # fallback: first file in folder
            deg_folder = os.path.join(VAL_ROOT, task, deg_dir)
            gt_folder = os.path.join(VAL_ROOT, task, gt_dir)
            deg_name = sorted(os.listdir(deg_folder))[0]
            gt_name = sorted(os.listdir(gt_folder))[0]
            deg_path = os.path.join(deg_folder, deg_name)
            gt_path = os.path.join(gt_folder, gt_name)

        degraded = load_rgb(deg_path)
        clean = load_rgb(gt_path)
        axes[row, 0].imshow(degraded)
        axes[row, 0].set_title(f"{label}: degraded input")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(clean)
        axes[row, 1].set_title(f"{label}: ground-truth clean")
        axes[row, 1].axis("off")

    fig.suptitle("Validation restoration examples ($256 \\times 256$)", fontsize=11)
    plt.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    pdf_path = os.path.join(args.out_dir, "restoration_examples.pdf")
    png_path = os.path.join(args.out_dir, "restoration_examples.png")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=150)
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
