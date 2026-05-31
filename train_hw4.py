import argparse
import os
import sys

# Add PromptIR to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PromptIR'))

# Blackwell (RTX 50xx): disable Triton JIT, enable Tensor Cores
os.environ.setdefault("TRITON_INTERPRET", "1")

import torch

torch.set_float32_matmul_precision('high')
if os.environ.get('DISABLE_CUDNN', '0') == '1':
    torch.backends.cudnn.enabled = False
else:
    torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint_sequential

import time

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EMAWeightAveraging, ModelCheckpoint, TQDMProgressBar

from hw4_dataset import HW4TrainDataset, HW4ValDataset
from utils.schedulers import LinearWarmupCosineAnnealingLR
from net.model import PromptIR, PromptGenBlock, TransformerBlock


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class CharbonnierMSELoss(nn.Module):
    def __init__(self, mse_weight=0.05, eps=1e-3):
        super().__init__()
        self.mse_weight = mse_weight
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        diff = pred - target
        charbonnier = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return charbonnier + self.mse_weight * self.mse(pred, target)


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
    if loss_type == "charbonnier_mse":
        return CharbonnierMSELoss(mse_weight=mse_weight)
    raise ValueError(f"Unsupported loss type: {loss_type}")


def parse_num_blocks(value):
    blocks = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(blocks) != 4:
        raise ValueError("num_blocks must contain four comma-separated integers")
    return blocks


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


class CheckpointedSequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        return checkpoint_sequential(self.blocks, len(self.blocks), x, use_reentrant=False)


def apply_gradient_checkpointing(model, mode):
    if mode == "none":
        return
    target_attrs = {
        "highres": ["encoder_level1", "decoder_level1", "refinement"],
        "full": ["encoder_level1", "encoder_level2", "encoder_level3",
                 "latent", "decoder_level1", "decoder_level2", "decoder_level3",
                 "refinement"],
    }[mode]
    for attr in target_attrs:
        if hasattr(model, attr):
            seq = getattr(model, attr)
            if isinstance(seq, nn.Sequential) and len(seq) > 0:
                setattr(model, attr, CheckpointedSequential(seq))


def _patch_promptir_decoder_dims(model, dim):
    level2 = int(dim * 2 ** 1)
    level3 = int(dim * 2 ** 2)
    level4 = int(dim * 2 ** 3)
    prompt1_dim = 64
    prompt2_dim = 128
    prompt3_dim = 320

    model.prompt1 = PromptGenBlock(prompt_dim=prompt1_dim, prompt_len=5, prompt_size=64, lin_dim=level2)
    model.prompt2 = PromptGenBlock(prompt_dim=prompt2_dim, prompt_len=5, prompt_size=32, lin_dim=level3)
    model.prompt3 = PromptGenBlock(prompt_dim=prompt3_dim, prompt_len=5, prompt_size=16, lin_dim=level4)

    model.reduce_chan_level3 = nn.Conv2d(level2 + level3, level3, kernel_size=1, bias=False)
    model.noise_level3 = TransformerBlock(
        dim=level4 + prompt3_dim, num_heads=4, ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='WithBias')
    model.reduce_noise_level3 = nn.Conv2d(level4 + prompt3_dim, level3, kernel_size=1, bias=False)

    model.noise_level2 = TransformerBlock(
        dim=level3 + prompt2_dim, num_heads=4, ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='WithBias')
    model.reduce_noise_level2 = nn.Conv2d(level3 + prompt2_dim, level3, kernel_size=1, bias=False)

    model.noise_level1 = TransformerBlock(
        dim=level2 + prompt1_dim, num_heads=4, ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='WithBias')
    model.reduce_noise_level1 = nn.Conv2d(level2 + prompt1_dim, level2, kernel_size=1, bias=False)


def build_promptir(decoder=True, model_dim=48, num_blocks=None, num_refinement_blocks=4):
    if num_blocks is None:
        num_blocks = [4, 6, 6, 8]
    model = PromptIR(
        decoder=decoder,
        dim=model_dim,
        num_blocks=num_blocks,
        num_refinement_blocks=num_refinement_blocks,
    )
    if decoder and model_dim != 48:
        _patch_promptir_decoder_dims(model, model_dim)
    return model


class PromptIRModel(pl.LightningModule):
    def __init__(
            self, lr=2e-4, warmup_epochs=15, max_epochs=150, loss_type="l1", mse_weight=0.05,
            task_conditioning=False, sipl_lite=False, sipl_start_alpha=0.5, sipl_refine_weight=0.5,
            gradient_checkpointing="none", rain_loss_weight=1.0,
            pair_mix_prob=0.0, pair_mix_alpha=1.2,
            model_dim=48, num_blocks=None, num_refinement_blocks=4):
        super().__init__()
        backbone = build_promptir(
            decoder=True,
            model_dim=model_dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks)
        apply_gradient_checkpointing(backbone, gradient_checkpointing)
        self.net = TaskConditionedRestorer(backbone, enabled=task_conditioning)
        self.loss_fn = build_loss_fn(loss_type, mse_weight)
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.task_conditioning = task_conditioning
        self.sipl_lite = sipl_lite
        self.sipl_start_alpha = sipl_start_alpha
        self.sipl_refine_weight = sipl_refine_weight
        self.rain_loss_weight = rain_loss_weight
        self.pair_mix_prob = pair_mix_prob
        self.pair_mix_alpha = pair_mix_alpha
        self._pair_mix_buffer = None
        self.save_hyperparameters()

    def forward(self, x, de_id=None):
        return self.net(x, de_id=de_id)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch

        if self.pair_mix_prob > 0 and self._pair_mix_buffer is not None:
            prev_degrad, prev_clean, prev_de_id = self._pair_mix_buffer
            same_task = prev_de_id == de_id
            if same_task and torch.rand(1).item() < self.pair_mix_prob:
                lam = float(torch.distributions.Beta(self.pair_mix_alpha, self.pair_mix_alpha).sample())
                degrad_patch = lam * degrad_patch + (1 - lam) * prev_degrad
                clean_patch = lam * clean_patch + (1 - lam) * prev_clean

        self._pair_mix_buffer = (degrad_patch.detach(), clean_patch.detach(), de_id)
        restored = self.net(degrad_patch, de_id=de_id)

        if self.rain_loss_weight != 1.0:
            bs = clean_patch.size(0)
            loss_per_sample = torch.zeros(bs, device=clean_patch.device)
            for i in range(bs):
                loss_per_sample[i] = self.loss_fn(restored[i:i + 1], clean_patch[i:i + 1])
            is_rain = (torch.tensor(de_id, device=clean_patch.device) == 1).float()
            weights = 1.0 + (self.rain_loss_weight - 1.0) * is_rain
            loss = (loss_per_sample * weights).mean()
        else:
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


class TimeEstimateProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate=1):
        super().__init__(refresh_rate=refresh_rate)
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        super().on_train_start(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if self.start_time is None:
            return

        elapsed = time.time() - self.start_time
        epoch = trainer.current_epoch
        max_epochs = trainer.max_epochs
        completed = trainer.global_step
        total = trainer.estimated_stepping_batches

        elapsed_str = self._format_seconds(elapsed)
        part1 = f"[time] epoch={epoch}/{max_epochs} elapsed={elapsed_str}"

        if completed > 0 and total:
            remaining = elapsed * max(total - completed, 0) / completed
            remaining_str = self._format_seconds(remaining)
            print(f"{part1} remaining={remaining_str}")
        else:
            print(part1)

    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        if self.start_time is None:
            return metrics

        elapsed = time.time() - self.start_time
        completed = trainer.global_step
        total = trainer.estimated_stepping_batches

        metrics["elapsed"] = self._format_seconds(elapsed)
        if completed > 0 and total:
            remaining = elapsed * max(total - completed, 0) / completed
            metrics["remaining"] = self._format_seconds(remaining)
        return metrics

    @staticmethod
    def _format_seconds(seconds):
        seconds = int(seconds)
        hours, rem = divmod(seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:d}:{seconds:02d}"


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
    parser.add_argument('--precision', type=str, default='32', help='32, 16-mixed, bf16-mixed')
    parser.add_argument('--gradient_checkpointing', choices=['none', 'highres', 'full'], default='none',
                        help='Activation checkpointing: none | highres (full-res blocks) | full (all blocks)')
    parser.add_argument('--model_dim', type=int, default=48,
                        help='PromptIR base channel dimension')
    parser.add_argument('--num_blocks', type=str, default='4,6,6,8',
                        help='PromptIR block depths as four comma-separated integers')
    parser.add_argument('--num_refinement_blocks', type=int, default=4,
                        help='PromptIR refinement block count')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile for training speed')
    parser.add_argument(
        '--disable_cudnn',
        action='store_true',
        help='Disable cuDNN (recommended for RTX 50xx / Blackwell smoke tests)',
    )
    parser.add_argument('--derain_oversample', type=int, default=1,
                        help='Duplicate rain samples N times per epoch (1=no oversampling)')
    parser.add_argument('--rain_loss_weight', type=float, default=1.0,
                        help='Multiply rain sample losses by this factor (1.0=no weighting, 1.25 recommended)')
    parser.add_argument('--pair_mix_prob', type=float, default=0.0,
                        help='Probability of same-task PairMix (0=off, 0.2 recommended)')
    parser.add_argument('--pair_mix_alpha', type=float, default=1.2,
                        help='Beta distribution alpha for PairMix lambda')
    parser.add_argument('--hard_patch_prob', type=float, default=0.0,
                        help='Probability of residual-biased hard patch sampling (0=off, 0.5 recommended)')
    parser.add_argument('--hard_patch_tau', type=float, default=2.0,
                        help='Temperature for hard patch sampling softmax')
    parser.add_argument('--color_aug_prob', type=float, default=0.0,
                        help='Probability of paired gamma/brightness/contrast augmentation (0=off, 0.2 recommended)')
    parser.add_argument('--force_aug_no_identity', action='store_true',
                        help='Official PromptIR-style geometric augmentation without identity')
    parser.add_argument('--aux_denoise', type=int, default=0,
                        help='Repeat count for synthetic denoise samples from provided clean images')
    parser.add_argument('--aux_denoise_sigmas', type=str, default='15,25,50',
                        help='Comma-separated Gaussian denoise sigmas for aux_denoise')
    parser.add_argument('--no_val', action='store_true', help='Skip validation (faster training)')
    parser.add_argument('--merge_val', action='store_true', help='Merge val into train (stage 2)')
    parser.add_argument('--save_top_k', type=int, default=1, help='Number of best validation checkpoints to keep')
    parser.add_argument('--every_n_epochs', type=int, default=5, help='Checkpoint interval in epochs')
    parser.add_argument(
        '--monitor',
        choices=['val_loss', 'val_psnr', 'val_psnr_derain', 'val_psnr_desnow'],
        default='val_loss',
    )
    parser.add_argument(
        '--distill_dir',
        type=str,
        default='',
        help='Optional dir with Distill/degraded, Distill/gt, labels.txt for pseudo-target training',
    )
    parser.add_argument('--ema', action='store_true', help='Use EMA weights for validation/checkpointing')
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--loss_type', choices=['l1', 'charbonnier', 'l1_mse', 'charbonnier_mse'], default='l1',
                        help='Training loss type')
    parser.add_argument('--mse_weight', type=float, default=0.05,
                        help='Weight of MSE term when loss_type is l1_mse or charbonnier_mse')
    parser.add_argument('--task_conditioning', action='store_true',
                        help='Enable per-task affine conditioning for rain/snow')
    parser.add_argument('--sipl_lite', action='store_true',
                        help='Enable SIPL-lite two-pass refinement training')
    parser.add_argument('--sipl_start_alpha', type=float, default=0.5,
                        help='Starting blend alpha for SIPL-lite, decays to 0')
    parser.add_argument('--sipl_refine_weight', type=float, default=0.5,
                        help='Weight of refinement pass loss in SIPL-lite')
    args = parser.parse_args()
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("  cuDNN: disabled (--disable_cudnn)")
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
                          sipl_refine_weight=args.sipl_refine_weight,
                          gradient_checkpointing=args.gradient_checkpointing,
                          rain_loss_weight=args.rain_loss_weight,
                          pair_mix_prob=args.pair_mix_prob,
                          pair_mix_alpha=args.pair_mix_alpha,
                          model_dim=args.model_dim,
                          num_blocks=parse_num_blocks(args.num_blocks),
                          num_refinement_blocks=args.num_refinement_blocks)

    if args.compile:
        import logging
        logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model)
        print("  compile: enabled (torch.compile applied)")

    psnr_monitors = {"val_psnr", "val_psnr_derain", "val_psnr_desnow"}
    monitor_mode = "max" if args.monitor in psnr_monitors else "min"
    checkpoint_filename = (
        f"promptir-{{epoch:02d}}-{{val_loss:.6f}}-{{{args.monitor}:.6f}}"
        if args.monitor != "val_loss"
        else "promptir-{epoch:02d}-{val_loss:.6f}-{val_psnr:.6f}"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        every_n_epochs=args.every_n_epochs,
        save_top_k=args.save_top_k,
        monitor=None if args.no_val else args.monitor,
        mode=monitor_mode,
        filename=checkpoint_filename,
    )
    callbacks = [checkpoint_callback, TimeEstimateProgressBar()]
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
