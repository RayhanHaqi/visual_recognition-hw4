import os
import random
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor


def crop_img(img, base=16):
    """Crop image to be divisible by base."""
    h, w = img.shape[:2]
    h = h - h % base
    w = w - w % base
    return img[:h, :w]


def random_augmentation(*imgs):
    """Apply the same random hflip + rotation to all images."""
    flag_aug = random.randint(1, 7)
    out = []
    for img in imgs:
        aug = img.copy()
        if flag_aug in (1, 3, 5, 7):
            aug = np.flipud(aug).copy()
        rot = {1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3}[flag_aug]
        if rot:
            aug = np.rot90(aug, rot).copy()
        out.append(aug)
    return out


class HW4TrainDataset(Dataset):
    """Dataset for HW4 image restoration with rain + snow."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.de_type = args.de_type

        self.de_dict = {'desnow': 0, 'derain': 1}

        self.sample_ids = []
        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        # Snow degraded images
        if 'desnow' in self.de_type:
            snowy_dir = os.path.join(self.args.data_dir, "Train", "Desnow", "snowy")
            gt_dir = os.path.join(self.args.data_dir, "Train", "Desnow", "gt")
            if os.path.exists(snowy_dir):
                names = sorted(os.listdir(snowy_dir))
                self.snow_ids = []
                for name in names:
                    clean_name = "snow_clean-" + name.split("snow-")[-1]
                    self.snow_ids.append({
                        "degraded_path": os.path.join(snowy_dir, name),
                        "clean_path": os.path.join(gt_dir, clean_name),
                        "de_type": 0,
                    })
                print(f"Snow samples: {len(self.snow_ids)}")

        # Rain images
        if 'derain' in self.de_type:
            rainy_dir = os.path.join(self.args.data_dir, "Train", "Derain", "rainy")
            gt_dir = os.path.join(self.args.data_dir, "Train", "Derain", "gt")
            if os.path.exists(rainy_dir):
                names = sorted(os.listdir(rainy_dir))
                self.rain_ids = []
                for name in names:
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
            self.sample_ids += self.rain_ids
        random.shuffle(self.de_type)
        print(f"Total training samples: {len(self.sample_ids)}")

    def _crop_patch(self, img_1, img_2):
        H, W = img_1.shape[:2]
        ind_H = random.randint(0, max(0, H - self.args.patch_size))
        ind_W = random.randint(0, max(0, W - self.args.patch_size))
        patch_1 = img_1[ind_H:ind_H + self.args.patch_size,
                         ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size,
                         ind_W:ind_W + self.args.patch_size]
        return patch_1, patch_2

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        degrad_img = crop_img(np.array(Image.open(sample["degraded_path"]).convert('RGB')), base=16)
        clean_img = crop_img(np.array(Image.open(sample["clean_path"]).convert('RGB')), base=16)

        degrad_patch, clean_patch = random_augmentation(
            *self._crop_patch(degrad_img, clean_img)
        )

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)

        clean_name = os.path.splitext(os.path.basename(sample["clean_path"]))[0]

        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)


class HW4TestDataset(Dataset):
    """Dataset for HW4 test inference (degraded images only)."""

    def __init__(self, test_dir):
        super().__init__()
        self.degraded_paths = []
        if os.path.isdir(test_dir):
            names = sorted(os.listdir(test_dir))
            for name in names:
                if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.degraded_paths.append(os.path.join(test_dir, name))

        self.toTensor = ToTensor()
        print(f"Test images: {len(self.degraded_paths)}")

    def __getitem__(self, idx):
        path = self.degraded_paths[idx]
        name = os.path.splitext(os.path.basename(path))[0]

        img = crop_img(np.array(Image.open(path).convert('RGB')), base=16)
        img = self.toTensor(img)

        return name, img

    def __len__(self):
        return len(self.degraded_paths)
