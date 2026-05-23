import argparse
import os
import sys
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "PromptIR"))
sys.path.insert(0, os.path.dirname(__file__))

from hw4_dataset import HW4TestDataset
from net.model import PromptIR
from train_hw4 import TaskConditionedRestorer

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


def restore_image(model, img, use_tta=False, de_id=None):
    if not use_tta:
        return model(img) if de_id is None else model(img, de_id=de_id)

    restored = []
    for mode in TTA_MODES:
        aug_img = apply_tta_transform(img, mode)
        aug_restored = model(aug_img) if de_id is None else model(aug_img, de_id=de_id)
        restored.append(invert_tta_transform(aug_restored, mode))
    return torch.stack(restored, dim=0).mean(dim=0)


def tensor_to_uint8(img):
    arr = torch.clamp(img.squeeze(0), 0, 1).cpu().numpy()
    return np.rint(arr * 255).clip(0, 255).astype(np.uint8)


def _has_conditioning_keys(state_dict):
    return any(key in state_dict for key in ("input_scale.weight", "backbone.conv_in.weight"))


def inference(ckpt_path, test_dir, output_path, device="cuda", use_tta=False, task_mode="none",
              use_sipl_lite=False):
    print(f"Checkpoint: {ckpt_path}")
    print(f"Test: {test_dir}")
    print(f"Output: {output_path}")
    print(f"TTA: {use_tta}")
    print(f"Task mode: {task_mode}")
    print(f"SIPL-lite: {use_sipl_lite}")

    ckpt = torch.load(ckpt_path, map_location=device)

    if "state_dict" in ckpt:
        state_dict = {}
        for k, v in ckpt["state_dict"].items():
            if k.startswith("net."):
                state_dict[k[4:]] = v
    else:
        state_dict = ckpt

    has_cond = _has_conditioning_keys(state_dict)
    if has_cond:
        base_model = PromptIR(decoder=True)
        model = TaskConditionedRestorer(base_model, enabled=True)
    else:
        model = PromptIR(decoder=True)

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    dataset = HW4TestDataset(test_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Write pred.npz to a temp location, then zip it
    os.makedirs("submission", exist_ok=True)
    npz_path = "submission/pred.npz"

    rain_id = torch.ones(1, dtype=torch.long, device=device)
    snow_id = torch.zeros(1, dtype=torch.long, device=device)

    pred_dict = {}
    with torch.no_grad():
        for name, img in tqdm(loader, desc="Inference"):
            name = name[0]
            img = img.to(device)

            original_h, original_w = img.shape[2], img.shape[3]
            padded, h, w = pad_to_multiple(img)

            if has_cond and task_mode == "both":
                r = restore_image(model, padded, use_tta=use_tta, de_id=rain_id)
                s = restore_image(model, padded, use_tta=use_tta, de_id=snow_id)
                restored = 0.5 * (r + s)
            elif has_cond and task_mode == "derain":
                restored = restore_image(model, padded, use_tta=use_tta, de_id=rain_id)
            elif has_cond and task_mode == "desnow":
                restored = restore_image(model, padded, use_tta=use_tta, de_id=snow_id)
            else:
                restored = restore_image(model, padded, use_tta=use_tta)

            if use_sipl_lite:
                restored_clamped = torch.clamp(restored, 0, 1)
                de_id_for_second = rain_id if (has_cond and task_mode == "derain") else (snow_id if (has_cond and task_mode == "desnow") else None)
                restored = restore_image(model, restored_clamped, use_tta=use_tta, de_id=de_id_for_second)

            restored = restored[:, :, :original_h, :original_w]

            restored = tensor_to_uint8(restored)

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
    parser.add_argument("--task_mode", choices=["none", "derain", "desnow", "both"], default="none",
                        help="Task conditioning mode for conditioned checkpoints")
    parser.add_argument("--sipl_lite", action="store_true",
                        help="Run a second refinement pass through PromptIR")
    args = parser.parse_args()

    if args.output is None:
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output = f"submission/{ckpt_name}.zip"

    inference(args.checkpoint, args.test_dir, args.output, args.device,
              use_tta=args.tta, task_mode=args.task_mode, use_sipl_lite=args.sipl_lite)
