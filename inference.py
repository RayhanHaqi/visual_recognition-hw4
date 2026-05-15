import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "PromptIR"))

from hw4_dataset import HW4TestDataset
from net.model import PromptIR


def pad_to_multiple(img, base=16):
    """Pad image tensor (C, H, W) to be divisible by base."""
    _, h, w = img.shape
    pad_h = (base - h % base) % base
    pad_w = (base - w % base) % base
    if pad_h == 0 and pad_w == 0:
        return img, h, w
    return torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='reflect'), h, w


def inference(ckpt_path, test_dir, output_path, device="cuda"):
    print(f"Checkpoint: {ckpt_path}")
    print(f"Test: {test_dir}")
    print(f"Output: {output_path}")

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

    pred_dict = {}
    with torch.no_grad():
        for name, img in tqdm(loader, desc="Inference"):
            name = name[0]
            img = img.to(device)

            original_h, original_w = img.shape[2], img.shape[3]
            padded, h, w = pad_to_multiple(img)

            restored = model(padded)

            # Unpad to original size
            restored = restored[:, :, :original_h, :original_w]

            restored = torch.clamp(restored, 0, 1)
            restored = (restored.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

            pred_dict[f"{name}.png"] = restored

    np.savez_compressed(output_path, **pred_dict)
    print(f"Saved {len(pred_dict)} images to {output_path}")

    # Verify
    verify = np.load(output_path)
    print(f"Verification: {len(verify.keys())} keys")
    k0 = list(verify.keys())[0]
    print(f"  {k0}: shape={verify[k0].shape}, dtype={verify[k0].dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--test_dir", type=str, default="PromptIR/data/Test/degraded")
    parser.add_argument("--output", type=str, default="submission/pred.npz")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    inference(args.checkpoint, args.test_dir, args.output, args.device)
