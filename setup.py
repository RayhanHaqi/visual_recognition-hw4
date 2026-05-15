import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_FILE_ID = "1bEIU9TZVQa-AF_z6JkOKaGp4wYGnqQ8w"
PROMPTIR_REPO = "https://github.com/va1shn9v/PromptIR.git"


def install_requirements():
    req = ROOT / "requirements.txt"
    if not req.exists():
        return
    print("Installing requirements...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=False)
    _upgrade_pytorch_if_blackwell()


def _upgrade_pytorch_if_blackwell():
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    cap = torch.cuda.get_device_capability()
    if cap is not None and cap[0] >= 12:
        print(f"RTX 50xx detected (sm_{cap[0]}{cap[1]}). Installing PyTorch nightly for Blackwell...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--pre", "torch", "torchvision",
             "--index-url", "https://download.pytorch.org/whl/nightly/cu128",
             "--force-reinstall"],
            check=False,
        )
        print("PyTorch nightly installed.")


def clone_promptir():
    promptir_dir = ROOT / "PromptIR"
    if promptir_dir.exists():
        print("PromptIR already exists, skipping clone.")
        return
    print(f"Cloning PromptIR from {PROMPTIR_REPO}...")
    subprocess.run(["git", "clone", PROMPTIR_REPO, str(promptir_dir)], check=True)
    print("PromptIR cloned.")


def download_dataset():
    data_dir = ROOT / "data" / "hw4_realse_dataset"
    if data_dir.exists() and len(list(data_dir.rglob("*.png"))) > 0:
        print("Dataset already exists, skipping download.")
        return

    zip_path = ROOT / "hw4_dataset.zip"
    if not zip_path.exists():
        print(f"Downloading dataset from Google Drive (id={DATASET_FILE_ID})...")
        subprocess.run(
            ["gdown", f"https://drive.google.com/uc?id={DATASET_FILE_ID}", "-O", str(zip_path)],
            check=True,
        )
    else:
        print("Dataset zip already downloaded, skipping.")

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(ROOT / "data")
    zip_path.unlink(missing_ok=True)

    # If extracted into a nested directory, flatten
    for child in list((ROOT / "data").iterdir()):
        if child.is_dir() and child.name != "hw4_realse_dataset":
            nested = child / "hw4_realse_dataset"
            if nested.is_dir():
                shutil.move(str(nested), str(ROOT / "data" / "hw4_realse_dataset"))
            elif (child / "train").is_dir():
                shutil.move(str(child), str(ROOT / "data" / "hw4_realse_dataset"))
            else:
                for item in child.iterdir():
                    shutil.move(str(item), str(ROOT / "data" / item.name))
                child.rmdir()

    print("Dataset extracted.")


def prepare_data():
    print("Organizing data into PromptIR format...")
    subprocess.run([sys.executable, str(ROOT / "prepare_data.py")], check=True)


def sanity_check():
    train_dir = ROOT / "data" / "hw4_realse_dataset" / "train"
    test_dir = ROOT / "data" / "hw4_realse_dataset" / "test"

    if train_dir.exists():
        n_degraded = len(list((train_dir / "degraded").glob("*.png"))) if (train_dir / "degraded").exists() else 0
        n_clean = len(list((train_dir / "clean").glob("*.png"))) if (train_dir / "clean").exists() else 0
        print(f"Train: {n_degraded} degraded, {n_clean} clean images (expected 3200 each)")
    else:
        print(f"MISSING: {train_dir}")

    if test_dir.exists():
        n_test = len(list((test_dir / "degraded").glob("*.png"))) if (test_dir / "degraded").exists() else 0
        print(f"Test: {n_test} images (expected 100)")
    else:
        print(f"MISSING: {test_dir}")

    promptir_data = ROOT / "PromptIR" / "data"
    if promptir_data.exists():
        n_train = len(list(promptir_data.rglob("*.png")))
        print(f"PromptIR data: {n_train} images ready for training")


def main():
    print("=== HW4 Setup ===")
    os.makedirs(ROOT / "data", exist_ok=True)
    install_requirements()
    clone_promptir()
    download_dataset()
    prepare_data()
    sanity_check()
    print("Setup complete.")
    print()
    print("To train: python train_hw4.py --batch_size 8 --epochs 150 --lr 2e-4 --num_gpus 1")


if __name__ == "__main__":
    main()
