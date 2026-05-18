import argparse
import os
import sys
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "PromptIR"))

from hw4_dataset import HW4TestDataset
from net.model import PromptIR

TTA_MODES = (
    "identity",
    "hflip",
    "vflip",
    "hvflip",
    "transpose",
    "transpose_hflip",
    "transpose_vflip",
    "transpose_hvflip",
)


def pad_to_multiple(img, base=16):
    # img: (1, C, H, W) from DataLoader with batch_size=1
    b, c, h, w = img.shape
    pad_h = (base - h % base) % base
    pad_w = (base - w % base) % base
    if pad_h == 0 and pad_w == 0:
        return img, h, w
    return torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='reflect'), h, w


def apply_tta_transform(img, mode):
    if mode.startswith("transpose"):
        img = img.transpose(-2, -1)
    if "hflip" in mode:
        img = torch.flip(img, dims=(-1,))
    if "vflip" in mode:
        img = torch.flip(img, dims=(-2,))
    return img.contiguous()


def invert_tta_transform(img, mode):
    if "vflip" in mode:
        img = torch.flip(img, dims=(-2,))
    if "hflip" in mode:
        img = torch.flip(img, dims=(-1,))
    if mode.startswith("transpose"):
        img = img.transpose(-2, -1)
    return img.contiguous()


def restore_image(model, img, use_tta=False):
    if not use_tta:
        return model(img)

    restored = []
    for mode in TTA_MODES:
        aug_img = apply_tta_transform(img, mode)
        aug_restored = model(aug_img)
        restored.append(invert_tta_transform(aug_restored, mode))
    return torch.stack(restored, dim=0).mean(dim=0)


def inference(ckpt_path, test_dir, output_path, device="cuda", use_tta=False):
    print(f"Checkpoint: {ckpt_path}")
    print(f"Test: {test_dir}")
    print(f"Output: {output_path}")
    print(f"TTA: {use_tta}")

    model = PromptIR(decoder=True)
    ckpt = torch.load(ckpt_path, map_location=device)

    if "state_dict" in ckpt:
        state_dict = {}
        for k, v in ckpt["state_dict"].items():
            if k.startswith("net."):
                state_dict[k[4:]] = v
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    dataset = HW4TestDataset(test_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Write pred.npz to a temp location, then zip it
    os.makedirs("submission", exist_ok=True)
    npz_path = "submission/pred.npz"

    pred_dict = {}
    with torch.no_grad():
        for name, img in tqdm(loader, desc="Inference"):
            name = name[0]
            img = img.to(device)

            original_h, original_w = img.shape[2], img.shape[3]
            padded, h, w = pad_to_multiple(img)

            restored = restore_image(model, padded, use_tta=use_tta)
            restored = restored[:, :, :original_h, :original_w]

            restored = torch.clamp(restored, 0, 1)
            restored = (restored.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

            pred_dict[f"{name}.png"] = restored

    np.savez_compressed(npz_path, **pred_dict)
    print(f"Wrote {len(pred_dict)} images to {npz_path}")

    # Zip pred.npz into the output archive
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(npz_path, "pred.npz")
    print(f"Submission zip: {output_path}")

    # Clean up temp npz
    os.remove(npz_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--test_dir", type=str, default="PromptIR/data/Test/degraded")
    parser.add_argument("--output", type=str, default=None,
                        help="Output zip path (default: submission/<ckpt_name>.zip)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tta", action="store_true", help="Use 8-way test-time augmentation")
    args = parser.parse_args()

    if args.output is None:
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output = f"submission/{ckpt_name}.zip"

    inference(args.checkpoint, args.test_dir, args.output, args.device, use_tta=args.tta)
