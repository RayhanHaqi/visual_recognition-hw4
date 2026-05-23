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
from lightning.pytorch.callbacks import EMAWeightAveraging, ModelCheckpoint

from hw4_dataset import HW4TrainDataset, HW4ValDataset
from utils.schedulers import LinearWarmupCosineAnnealingLR
from net.model import PromptIR


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class L1MSELoss(nn.Module):
    def __init__(self, mse_weight):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.l1(pred, target) + self.mse_weight * self.mse(pred, target)


def build_loss_fn(loss_type, mse_weight):
    if loss_type == "l1":
        return nn.L1Loss()
    if loss_type == "charbonnier":
        return CharbonnierLoss()
    if loss_type == "l1_mse":
        return L1MSELoss(mse_weight=mse_weight)
    raise ValueError(f"Unsupported loss type: {loss_type}")


def sipl_blend_alpha(epoch, max_epochs, start_alpha):
    if max_epochs <= 0:
        return 0.0
    progress = min(max(epoch / max_epochs, 0.0), 1.0)
    return start_alpha * (1.0 - progress)


def sipl_second_input(restored, clean, alpha, training):
    if not training or clean is None:
        return restored.detach()
    return (1.0 - alpha) * restored.detach() + alpha * clean


class TaskConditionedRestorer(nn.Module):
    def __init__(self, backbone, enabled=True):
        super().__init__()
        self.backbone = backbone
        self.enabled = enabled
        self.input_scale = nn.Embedding(2, 3)
        self.input_bias = nn.Embedding(2, 3)
        self.output_bias = nn.Embedding(2, 3)
        nn.init.zeros_(self.input_scale.weight)
        nn.init.zeros_(self.input_bias.weight)
        nn.init.zeros_(self.output_bias.weight)

    def forward(self, x, de_id=None):
        if not self.enabled or de_id is None:
            return self.backbone(x)
        de_id = de_id.to(device=x.device, dtype=torch.long).view(-1)
        scale = self.input_scale(de_id).view(-1, 3, 1, 1)
        bias = self.input_bias(de_id).view(-1, 3, 1, 1)
        out_bias = self.output_bias(de_id).view(-1, 3, 1, 1)
        conditioned = x * (1.0 + scale) + bias
        return self.backbone(conditioned) + out_bias


class PromptIRModel(pl.LightningModule):
    def __init__(self, lr=2e-4, warmup_epochs=15, max_epochs=150, loss_type="l1", mse_weight=0.05,
                 task_conditioning=False, sipl_lite=False, sipl_start_alpha=0.5, sipl_refine_weight=0.5):
        super().__init__()
        self.net = TaskConditionedRestorer(PromptIR(decoder=True), enabled=task_conditioning)
        self.loss_fn = build_loss_fn(loss_type, mse_weight)
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.task_conditioning = task_conditioning
        self.sipl_lite = sipl_lite
        self.sipl_start_alpha = sipl_start_alpha
        self.sipl_refine_weight = sipl_refine_weight
        self.save_hyperparameters()

    def forward(self, x, de_id=None):
        return self.net(x, de_id=de_id)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch, de_id=de_id)
        loss = self.loss_fn(restored, clean_patch)
        if self.sipl_lite:
            alpha = sipl_blend_alpha(self.current_epoch, self.max_epochs, self.sipl_start_alpha)
            second_input = sipl_second_input(restored, clean_patch, alpha=alpha, training=True)
            refined = self.net(second_input, de_id=de_id)
            loss = loss + self.sipl_refine_weight * self.loss_fn(refined, clean_patch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        name, de_id, degrad_patch, clean_patch = batch
        restored = self.net(degrad_patch, de_id=de_id)
        if self.sipl_lite:
            second_input = sipl_second_input(restored, clean_patch, alpha=0.0, training=False)
            restored = self.net(second_input, de_id=de_id)
        loss = self.loss_fn(restored, clean_patch)
        mse = torch.mean((torch.clamp(restored, 0, 1) - clean_patch) ** 2)
        psnr = -10 * torch.log10(torch.clamp(mse, min=1e-10))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_psnr", psnr, on_epoch=True, prog_bar=True)

        is_snow = de_id == 0
        is_rain = de_id == 1
        if torch.any(is_snow):
            self.log("val_loss_desnow", loss, on_epoch=True, batch_size=1)
            self.log("val_psnr_desnow", psnr, on_epoch=True, batch_size=1)
        if torch.any(is_rain):
            self.log("val_loss_derain", loss, on_epoch=True, batch_size=1)
            self.log("val_psnr_derain", psnr, on_epoch=True, batch_size=1)
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
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup', type=int, default=15)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU IDs (e.g. "0", "1", "0,1")')
    parser.add_argument('--data_dir', type=str, default='PromptIR/data')
    parser.add_argument('--de_type', nargs='+', default=['desnow', 'derain'])
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--precision', type=str, default='32', help='16-mixed or 32')
    parser.add_argument('--no_val', action='store_true', help='Skip validation (faster training)')
    parser.add_argument('--merge_val', action='store_true', help='Merge val into train (stage 2)')
    parser.add_argument('--save_top_k', type=int, default=1, help='Number of best validation checkpoints to keep')
    parser.add_argument('--monitor', choices=['val_loss', 'val_psnr'], default='val_loss')
    parser.add_argument('--ema', action='store_true', help='Use EMA weights for validation/checkpointing')
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--loss_type', choices=['l1', 'charbonnier', 'l1_mse'], default='l1',
                        help='Training loss type')
    parser.add_argument('--mse_weight', type=float, default=0.05,
                        help='Weight of MSE term when loss_type is l1_mse')
    parser.add_argument('--task_conditioning', action='store_true',
                        help='Enable per-task affine conditioning for rain/snow')
    parser.add_argument('--sipl_lite', action='store_true',
                        help='Enable SIPL-lite two-pass refinement training')
    parser.add_argument('--sipl_start_alpha', type=float, default=0.5,
                        help='Starting blend alpha for SIPL-lite, decays to 0')
    parser.add_argument('--sipl_refine_weight', type=float, default=0.5,
                        help='Weight of refinement pass loss in SIPL-lite')
    args = parser.parse_args()
    if args.no_val and args.save_top_k not in (-1, 0, 1):
        raise ValueError("--no_val supports --save_top_k only -1, 0, or 1 because no validation metric is available")

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

    model = PromptIRModel(lr=args.lr, warmup_epochs=args.warmup, max_epochs=args.epochs,
                          loss_type=args.loss_type, mse_weight=args.mse_weight,
                          task_conditioning=args.task_conditioning,
                          sipl_lite=args.sipl_lite, sipl_start_alpha=args.sipl_start_alpha,
                          sipl_refine_weight=args.sipl_refine_weight)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        every_n_epochs=5,
        save_top_k=args.save_top_k,
        monitor=None if args.no_val else args.monitor,
        mode="max" if args.monitor == "val_psnr" else "min",
        filename="promptir-{epoch:02d}-{val_loss:.6f}-{val_psnr:.6f}",
    )
    callbacks = [checkpoint_callback]
    if args.ema:
        callbacks.append(EMAWeightAveraging(decay=args.ema_decay))

    logger = TensorBoardLogger(save_dir=args.log_dir, name="hw4_promptir")

    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=gpu_ids,
        strategy="auto",
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader)


if __name__ == '__main__':
    main()
