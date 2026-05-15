

class HW4ValDataset(Dataset):
    """Validation dataset with paired degraded + clean images."""

    def __init__(self, data_dir):
        super().__init__()
        self.samples = []
        val_dir = os.path.join(data_dir, "Val")
        for task in ["Derain", "Desnow"]:
            degraded_dir = os.path.join(val_dir, task, "rainy" if task == "Derain" else "snowy")
            gt_dir = os.path.join(val_dir, task, "gt")
            if os.path.isdir(degraded_dir):
                for name in sorted(os.listdir(degraded_dir)):
                    if name.endswith('.png'):
                        prefix = "rain_clean-" if task == "Derain" else "snow_clean-"
                        idx = name.split("-")[-1]
                        clean_name = prefix + idx
                        clean_path = os.path.join(gt_dir, clean_name)
                        if os.path.exists(clean_path):
                            self.samples.append((
                                os.path.join(degraded_dir, name),
                                clean_path,
                                name,
                            ))
        self.toTensor = ToTensor()
        print(f"Val samples: {len(self.samples)}")

    def __getitem__(self, idx):
        degraded_path, clean_path, name = self.samples[idx]
        degraded = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        clean = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)
        return name, self.toTensor(degraded), self.toTensor(clean)

    def __len__(self):
        return len(self.samples)