import os
import random
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor


def crop_img(img, base=16):
    h, w = img.shape[:2]
    return img[:h - h % base, :w - w % base]


def random_augmentation(*imgs, allow_identity=True):
    low = 0 if allow_identity else 1
    flag_aug = random.randint(low, 7)
    out = []
    for img in imgs:
        aug = img.copy()
        if flag_aug in (1, 3, 5, 7):
            aug = np.flipud(aug).copy()
        rot = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3}[flag_aug]
        if rot:
            aug = np.rot90(aug, rot).copy()
        out.append(aug)
    return out


def parse_sigmas(value):
    return [int(part.strip()) for part in str(value).split(',') if part.strip()]


def paired_color_augmentation(*imgs, prob=0.2):
    if random.random() >= prob:
        return list(imgs)
    gamma = random.uniform(0.85, 1.15)
    brightness = random.uniform(-0.05, 0.05)
    contrast = random.uniform(0.9, 1.1)
    out = []
    for img in imgs:
        aug = img.astype(np.float32) / 255.0
        aug = np.clip(aug ** gamma, 0, 1)
        aug = np.clip(aug * contrast + brightness, 0, 1)
        aug = (aug * 255).astype(img.dtype)
        out.append(aug)
    return out


class HW4TrainDataset(Dataset):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.de_type = args.de_type
        self.merge_val = getattr(args, 'merge_val', False)
        self.sample_ids = []
        self._init_ids()
        self._merge_ids()
        self.toTensor = ToTensor()

    def _init_ids(self):
        splits = ["Train", "Val"] if self.merge_val else ["Train"]

        if 'desnow' in self.de_type:
            self.snow_ids = []
            for split in splits:
                snowy_dir = os.path.join(self.args.data_dir, split, "Desnow", "snowy")
                gt_dir = os.path.join(self.args.data_dir, split, "Desnow", "gt")
                if os.path.exists(snowy_dir):
                    for name in sorted(os.listdir(snowy_dir)):
                        clean_name = "snow_clean-" + name.split("snow-")[-1]
                        self.snow_ids.append({
                            "degraded_path": os.path.join(snowy_dir, name),
                            "clean_path": os.path.join(gt_dir, clean_name),
                            "de_type": 0,
                        })
            print(f"Snow samples: {len(self.snow_ids)}")

        if 'derain' in self.de_type:
            self.rain_ids = []
            for split in splits:
                rainy_dir = os.path.join(self.args.data_dir, split, "Derain", "rainy")
                gt_dir = os.path.join(self.args.data_dir, split, "Derain", "gt")
                if os.path.exists(rainy_dir):
                    for name in sorted(os.listdir(rainy_dir)):
                        clean_name = "rain_clean-" + name.split("rain-")[-1]
                        self.rain_ids.append({
                            "degraded_path": os.path.join(rainy_dir, name),
                            "clean_path": os.path.join(gt_dir, clean_name),
                            "de_type": 1,
                        })
            print(f"Rain samples: {len(self.rain_ids)}")

    def _merge_ids(self):
        if 'desnow' in self.de_type:
            self.sample_ids += self.snow_ids
        if 'derain' in self.de_type:
            oversample = getattr(self.args, 'derain_oversample', 1)
            for _ in range(oversample):
                self.sample_ids += self.rain_ids
        distill_dir = getattr(self.args, 'distill_dir', '') or ''
        if distill_dir:
            labels_path = os.path.join(distill_dir, 'labels.txt')
            degraded_dir = os.path.join(distill_dir, 'degraded')
            gt_dir = os.path.join(distill_dir, 'gt')
            if not os.path.isfile(labels_path):
                raise FileNotFoundError(f"Missing distill labels file: {labels_path}")
            with open(labels_path, encoding='utf-8') as f:
                de_ids = [line.strip() for line in f if line.strip()]
            names = sorted(
                name for name in os.listdir(degraded_dir)
                if name.lower().endswith('.png')
            )
            if len(names) != len(de_ids):
                raise ValueError(
                    f"Distill labels ({len(de_ids)}) do not match degraded images ({len(names)})"
                )
            for name, de_id in zip(names, de_ids):
                self.sample_ids.append({
                    'degraded_path': os.path.join(degraded_dir, name),
                    'clean_path': os.path.join(gt_dir, name),
                    'de_type': int(de_id),
                })
            print(f"Distill samples: {len(names)}")

        if getattr(self.args, 'aux_denoise', 0) > 0:
            sigmas = parse_sigmas(getattr(self.args, 'aux_denoise_sigmas', '15,25,50'))
            clean_sources = []
            if 'desnow' in self.de_type:
                clean_sources += self.snow_ids
            if 'derain' in self.de_type:
                clean_sources += self.rain_ids
            for _ in range(getattr(self.args, 'aux_denoise', 0)):
                for sample in clean_sources:
                    for sigma in sigmas:
                        self.sample_ids.append({
                            'clean_path': sample['clean_path'],
                            'degraded_path': sample['clean_path'],
                            'de_type': 2,
                            'sigma': sigma,
                        })
        random.shuffle(self.de_type)
        print(f"Total training samples: {len(self.sample_ids)}")
        if getattr(self.args, 'derain_oversample', 1) > 1:
            print(f"  Derain oversampling: {oversample}x "
                  f"({len(self.rain_ids)} rain -> {len(self.rain_ids) * oversample} samples)")

    def _crop_patch(self, img_1, img_2):
        H, W = img_1.shape[:2]
        tau = getattr(self.args, 'hard_patch_tau', 2.0)
        prob = getattr(self.args, 'hard_patch_prob', 0.0)

        if prob > 0 and random.random() < prob:
            residual = np.abs(img_1.astype(np.float32) - img_2.astype(np.float32))
            if residual.ndim == 3:
                residual = residual.mean(axis=2)
            res_h = residual.shape[0] - self.args.patch_size
            res_w = residual.shape[1] - self.args.patch_size
            if res_h <= 0 and res_w <= 0:
                return self._crop_patch(img_1, img_2)
            if res_h <= 0:
                res_h = 1
            if res_w <= 0:
                res_w = 1
            energy = np.zeros((res_h, res_w), dtype=np.float32)
            for hh in range(res_h):
                for ww in range(res_w):
                    energy[hh, ww] = residual[hh:hh + self.args.patch_size, ww:ww + self.args.patch_size].sum()
            energy = energy.reshape(-1)
            energy_exp = np.exp(tau * energy / (energy.max() + 1e-8))

            if energy_exp.sum() <= 0:
                return self._crop_patch(img_1, img_2)
            probs = energy_exp / energy_exp.sum()
            idx = np.random.choice(len(probs), p=probs)
            ind_H = idx // res_w
            ind_W = idx % res_w
        else:
            ind_H = random.randint(0, max(0, H - self.args.patch_size))
            ind_W = random.randint(0, max(0, W - self.args.patch_size))

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        return patch_1, patch_2

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]
        allow_identity = not getattr(self.args, 'force_aug_no_identity', False)
        if de_id == 2:
            clean_img = crop_img(np.array(Image.open(sample['clean_path']).convert('RGB')), base=16)
            degrad_patch, clean_patch = self._crop_patch(clean_img, clean_img)
            degrad_patch, clean_patch = random_augmentation(
                degrad_patch, clean_patch, allow_identity=allow_identity)
            noise = np.random.randn(*clean_patch.shape) * sample['sigma']
            degrad_patch = np.clip(clean_patch.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        else:
            degrad_img = crop_img(np.array(Image.open(sample["degraded_path"]).convert('RGB')), base=16)
            clean_img = crop_img(np.array(Image.open(sample["clean_path"]).convert('RGB')), base=16)
            degrad_patch, clean_patch = random_augmentation(
                *self._crop_patch(degrad_img, clean_img), allow_identity=allow_identity)
        color_prob = getattr(self.args, 'color_aug_prob', 0.0)
        degrad_patch, clean_patch = paired_color_augmentation(degrad_patch, clean_patch, prob=color_prob)
        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)
        clean_name = os.path.splitext(os.path.basename(sample["clean_path"]))[0]
        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)


class HW4ValDataset(Dataset):

    def __init__(self, data_dir):
        super().__init__()
        self.samples = []
        val_dir = os.path.join(data_dir, "Val")
        for task in ["Derain", "Desnow"]:
            de_id = 1 if task == "Derain" else 0
            degraded_dir = os.path.join(val_dir, task, "rainy" if task == "Derain" else "snowy")
            gt_dir = os.path.join(val_dir, task, "gt")
            if os.path.isdir(degraded_dir):
                for name in sorted(os.listdir(degraded_dir)):
                    if name.endswith('.png'):
                        prefix = "rain_clean-" if task == "Derain" else "snow_clean-"
                        idx = name.split("-")[-1]
                        clean_path = os.path.join(gt_dir, prefix + idx)
                        if os.path.exists(clean_path):
                            self.samples.append((os.path.join(degraded_dir, name), clean_path, name, de_id))
        self.toTensor = ToTensor()
        print(f"Val samples: {len(self.samples)}")

    def __getitem__(self, idx):
        degraded_path, clean_path, name, de_id = self.samples[idx]
        degraded = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        clean = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)
        return name, de_id, self.toTensor(degraded), self.toTensor(clean)

    def __len__(self):
        return len(self.samples)


class HW4TestDataset(Dataset):

    def __init__(self, test_dir):
        super().__init__()
        self.degraded_paths = []
        if os.path.isdir(test_dir):
            for name in sorted(os.listdir(test_dir)):
                if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.degraded_paths.append(os.path.join(test_dir, name))
        self.toTensor = ToTensor()
        print(f"Test images: {len(self.degraded_paths)}")

    def __getitem__(self, idx):
        path = self.degraded_paths[idx]
        name = os.path.splitext(os.path.basename(path))[0]
        img = crop_img(np.array(Image.open(path).convert('RGB')), base=16)
        return name, self.toTensor(img)

    def __len__(self):
        return len(self.degraded_paths)
