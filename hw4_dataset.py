import os
import random
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor


def crop_img(img, base=16):
    h, w = img.shape[:2]
    return img[:h - h % base, :w - w % base]


def random_augmentation(*imgs):
    flag_aug = random.randint(0, 7)
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
        random.shuffle(self.de_type)
        print(f"Total training samples: {len(self.sample_ids)}")
        if getattr(self.args, 'derain_oversample', 1) > 1:
            print(f"  Derain oversampling: {oversample}x ({len(self.rain_ids)} rain -> {len(self.rain_ids) * oversample} samples)")

    def _crop_patch(self, img_1, img_2):
        H, W = img_1.shape[:2]
        ind_H = random.randint(0, max(0, H - self.args.patch_size))
        ind_W = random.randint(0, max(0, W - self.args.patch_size))
        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        return patch_1, patch_2

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]
        degrad_img = crop_img(np.array(Image.open(sample["degraded_path"]).convert('RGB')), base=16)
        clean_img = crop_img(np.array(Image.open(sample["clean_path"]).convert('RGB')), base=16)
        degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))
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
