import os
import shutil
import random


def prepare_train_data(src_root, dst_root, val_ratio=0.1):
    """Organize HW4 dataset into PromptIR-compatible format.

    Creates:
      data/Train/Derain/gt/    -> rain clean images
      data/Train/Derain/rainy/ -> rain degraded images
      data/Train/Desnow/gt/    -> snow clean images
      data/Train/Desnow/snowy/ -> snow degraded images
    """

    train_src = os.path.join(src_root, "train")
    degraded_src = os.path.join(train_src, "degraded")
    clean_src = os.path.join(train_src, "clean")

    rain_dst_gt = os.path.join(dst_root, "Train", "Derain", "gt")
    rain_dst_rainy = os.path.join(dst_root, "Train", "Derain", "rainy")
    snow_dst_gt = os.path.join(dst_root, "Train", "Desnow", "gt")
    snow_dst_snowy = os.path.join(dst_root, "Train", "Desnow", "snowy")

    val_rain_gt = os.path.join(dst_root, "Val", "Derain", "gt")
    val_rain_rainy = os.path.join(dst_root, "Val", "Derain", "rainy")
    val_snow_gt = os.path.join(dst_root, "Val", "Desnow", "gt")
    val_snow_snowy = os.path.join(dst_root, "Val", "Desnow", "snowy")

    for d in [rain_dst_gt, rain_dst_rainy, snow_dst_gt, snow_dst_snowy,
              val_rain_gt, val_rain_rainy, val_snow_gt, val_snow_snowy]:
        os.makedirs(d, exist_ok=True)

    # Separate rain and snow files
    rain_names = []
    snow_names = []
    for fname in sorted(os.listdir(degraded_src)):
        if fname.startswith("rain-"):
            idx = fname.replace("rain-", "").replace(".png", "")
            clean_name = f"rain_clean-{idx}.png"
            if os.path.exists(os.path.join(clean_src, clean_name)):
                rain_names.append((fname, clean_name))
        elif fname.startswith("snow-"):
            idx = fname.replace("snow-", "").replace(".png", "")
            clean_name = f"snow_clean-{idx}.png"
            if os.path.exists(os.path.join(clean_src, clean_name)):
                snow_names.append((fname, clean_name))

    print(f"Found {len(rain_names)} rain pairs, {len(snow_names)} snow pairs")

    random.seed(42)
    random.shuffle(rain_names)
    random.shuffle(snow_names)

    n_rain_val = int(len(rain_names) * val_ratio)
    n_snow_val = int(len(snow_names) * val_ratio)

    rain_train = rain_names[n_rain_val:]
    rain_val = rain_names[:n_rain_val]
    snow_train = snow_names[n_snow_val:]
    snow_val = snow_names[:n_snow_val]

    print(f"Rain: {len(rain_train)} train, {len(rain_val)} val")
    print(f"Snow: {len(snow_train)} train, {len(snow_val)} val")

    # Copy rain train
    for deg_name, clean_name in rain_train:
        shutil.copy2(os.path.join(degraded_src, deg_name),
                     os.path.join(rain_dst_rainy, deg_name))
        shutil.copy2(os.path.join(clean_src, clean_name),
                     os.path.join(rain_dst_gt, clean_name))

    # Copy rain val
    for deg_name, clean_name in rain_val:
        shutil.copy2(os.path.join(degraded_src, deg_name),
                     os.path.join(val_rain_rainy, deg_name))
        shutil.copy2(os.path.join(clean_src, clean_name),
                     os.path.join(val_rain_gt, clean_name))

    # Copy snow train
    for deg_name, clean_name in snow_train:
        shutil.copy2(os.path.join(degraded_src, deg_name),
                     os.path.join(snow_dst_snowy, deg_name))
        shutil.copy2(os.path.join(clean_src, clean_name),
                     os.path.join(snow_dst_gt, clean_name))

    # Copy snow val
    for deg_name, clean_name in snow_val:
        shutil.copy2(os.path.join(degraded_src, deg_name),
                     os.path.join(val_snow_snowy, deg_name))
        shutil.copy2(os.path.join(clean_src, clean_name),
                     os.path.join(val_snow_gt, clean_name))

    print("Data preparation complete.")


def prepare_test_data(src_root, dst_root):
    """Copy test images."""
    test_src = os.path.join(src_root, "test", "degraded")
    test_dst = os.path.join(dst_root, "Test", "degraded")
    os.makedirs(test_dst, exist_ok=True)

    for fname in sorted(os.listdir(test_src)):
        shutil.copy2(os.path.join(test_src, fname),
                     os.path.join(test_dst, fname))

    print(f"Copied {len(os.listdir(test_src))} test images.")


if __name__ == "__main__":
    src = "data/hw4_realse_dataset"
    dst = "PromptIR/data"

    prepare_train_data(src, dst)
    prepare_test_data(src, dst)
