import argparse
import os
import sys

# Add PromptIR to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PromptIR'))

# Blackwell (RTX 50xx): disable Triton JIT, enable Tensor Cores
os.environ.setdefault("TRITON_INTERPRET", "1")

import torch

torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from hw4_dataset import HW4TrainDataset, HW4ValDataset
from utils.schedulers import LinearWarmupCosineAnnealingLR
from net.model import PromptIR


class PromptIRModel(pl.LightningModule):
    def __init__(self, lr=2e-4, warmup_epochs=15, max_epochs=150):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = nn.L1Loss()
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        name, degrad_patch, clean_patch = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=self.warmup_epochs,
            max_epochs=self.max_epochs,
        )
        return [optimizer], [scheduler]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup', type=int, default=15)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_ids', type=str, default='1', help='GPU IDs (e.g. "0", "1", "0,1")')
    parser.add_argument('--data_dir', type=str, default='PromptIR/data')
    parser.add_argument('--de_type', nargs='+', default=['desnow', 'derain'])
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--precision', type=str, default='32', help='16-mixed or 32')
    parser.add_argument('--no_val', action='store_true', help='Skip validation (faster training)')
    parser.add_argument('--merge_val', action='store_true', help='Merge val into train (stage 2)')
    args = parser.parse_args()

    print("Training configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    trainset = HW4TrainDataset(args)
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )

    if args.no_val:
        valloader = None
    else:
        valset = HW4ValDataset(args.data_dir)
        valloader = DataLoader(
            valset,
            batch_size=1,  # full images, not patches
            pin_memory=True,
            shuffle=False,
            num_workers=2,
        )

    model = PromptIRModel(lr=args.lr, warmup_epochs=args.warmup, max_epochs=args.epochs)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        every_n_epochs=10,
        save_top_k=-1 if args.no_val else 1,
        monitor=None if args.no_val else "val_loss",
        mode="min",
        filename="promptir-{epoch:02d}-{val_loss:.4f}",
    )

    logger = TensorBoardLogger(save_dir=args.log_dir, name="hw4_promptir")

    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=gpu_ids,
        strategy="auto",
        precision=args.precision,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader)


if __name__ == '__main__':
    main()
