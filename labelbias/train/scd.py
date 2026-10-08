#!/usr/bin/env python3
"""
Asymmetric debiasing segmentation training.

Three strategies:
  1. asymmetric   – Asymmetric loss masking: boundary-region loss for the
                    biased group is zeroed out so boundary learning is driven
                    100% by the clean (unbiased) group.  Applied to BOTH CE
                    and Dice losses.

  2. style_cond   – Style-conditioned decoder (FiLM): learnable group
                    embeddings modulate encoder features so the decoder can
                    disambiguate annotation *style* from visual *content*.
                    At inference the clean-group embedding is always used.

  3. asym_style   – Both strategies simultaneously.

Supports CelebAMask-HQ and PHC datasets.

Example usage:
    # Asymmetric masking on CelebA (all folds)
    for FOLD in 0 1 2 3 4; do
        python -m labelbias.train.scd --dataset celebamask --method asymmetric \\
            --bias_ratio 1.0 --erosion_radius 15 --epochs 20 --fold $FOLD
    done

    # Style-conditioned decoder on PHC
    for FOLD in 0 1 2 3 4; do
        python -m labelbias.train.scd --dataset phc --method style_cond \\
            --bias_ratio 1.0 --epochs 20 --fold $FOLD
    done

    # Combined on CelebA
    for FOLD in 0 1 2 3 4; do
        python -m labelbias.train.scd --dataset celebamask --method asym_style \\
            --bias_ratio 0.5 --erosion_radius 15 --epochs 20 --fold $FOLD
    done
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

import segmentation_models_pytorch as smp

from labelbias.data.factory import (
    add_dataset_args, apply_dataset_defaults, create_splits, create_datasets,
    get_demographic_names, DATASET_DEFAULTS,
)
from labelbias.data.splits import get_fold_indices
from labelbias.metrics import SegmentationMetrics


# ===================================================================
# Asymmetric boundary masking
# ===================================================================

def compute_boundary_mask(masks: torch.Tensor, width: int = 2) -> torch.Tensor:
    """Binary boundary mask via morphological gradient (dilation − erosion).

    Args:
        masks:  (B, H, W) long tensor with class indices (0=bg, 1=fg).
        width:  half-width of the boundary band in pixels.

    Returns:
        (B, H, W) float tensor: 1.0 on boundary pixels, 0.0 elsewhere.
    """
    fg = (masks == 1).float().unsqueeze(1)          # (B, 1, H, W)
    k = 2 * width + 1
    dilated = F.max_pool2d(fg, kernel_size=k, stride=1, padding=width)
    eroded  = -F.max_pool2d(-fg, kernel_size=k, stride=1, padding=width)
    boundary = (dilated - eroded).clamp(min=0.0).squeeze(1)  # (B, H, W)
    return boundary


def build_asymmetric_loss_weight(
    masks: torch.Tensor,
    genders: torch.Tensor,
    biased_group_id: int,
    boundary_width: int = 2,
    interior_weight: float = 1.0,
) -> torch.Tensor:
    """Per-pixel loss weight that zeros boundary pixels for the biased group.

    Clean group:  weight = 1.0 everywhere.
    Biased group: weight = 0.0 on boundary, ``interior_weight`` on interior.

    Returns:
        (B, H, W) float tensor of weights in [0, 1].
    """
    boundary = compute_boundary_mask(masks, width=boundary_width)   # (B, H, W)
    is_biased = (genders == biased_group_id).float()[:, None, None] # (B,1,1)

    weight = torch.ones_like(masks, dtype=torch.float32)

    # Zero out boundary pixels of biased group
    weight = weight - boundary * is_biased

    # Optionally down-weight interior pixels of biased group
    if interior_weight < 1.0:
        interior_mask = (1.0 - boundary) * is_biased
        weight = weight - interior_mask * (1.0 - interior_weight)

    return weight.clamp(min=0.0)


# ===================================================================
# Asymmetric losses  (CE + Dice, both masked)
# ===================================================================

def asymmetric_ce_loss(pred_logits, targets, loss_weight):
    """Per-pixel CE weighted by ``loss_weight``."""
    ce = F.cross_entropy(pred_logits, targets, reduction='none')    # (B, H, W)
    weighted = ce * loss_weight
    loss = weighted.sum() / loss_weight.sum().clamp(min=1.0)
    per_sample = weighted.sum(dim=(1, 2)) / loss_weight.sum(dim=(1, 2)).clamp(min=1.0)
    return loss, per_sample


def asymmetric_dice_loss(pred_logits, targets, loss_weight, smooth=1.0):
    """Dice loss computed only over unmasked pixels."""
    pred_probs = F.softmax(pred_logits, dim=1)
    pred_fg   = pred_probs[:, 1]                     # (B, H, W)
    target_fg = (targets == 1).float()

    intersection = (pred_fg * target_fg * loss_weight).sum(dim=(1, 2))
    pred_sum     = (pred_fg * loss_weight).sum(dim=(1, 2))
    target_sum   = (target_fg * loss_weight).sum(dim=(1, 2))

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    per_sample = 1.0 - dice
    return per_sample.mean(), per_sample


def asymmetric_combined_loss(pred_logits, targets, loss_weight,
                             ce_weight=0.5, dice_weight=0.5):
    """Combined CE + Dice with asymmetric per-pixel masking."""
    ce_loss, ce_ps   = asymmetric_ce_loss(pred_logits, targets, loss_weight)
    d_loss,  d_ps    = asymmetric_dice_loss(pred_logits, targets, loss_weight)
    total      = ce_weight * ce_loss + dice_weight * d_loss
    per_sample = ce_weight * ce_ps   + dice_weight * d_ps
    return total, per_sample


def standard_combined_loss(pred_logits, targets, ce_weight=0.5, dice_weight=0.5):
    """Standard (symmetric) CE + Dice without masking."""
    weight = torch.ones(targets.shape, dtype=torch.float32, device=targets.device)
    return asymmetric_combined_loss(pred_logits, targets, weight,
                                    ce_weight=ce_weight, dice_weight=dice_weight)


# ===================================================================
# FiLM conditioning  (Feature-wise Linear Modulation)
# ===================================================================

class FiLMLayer(nn.Module):
    """Applies  γ(group) ⊙ x + β(group)  per channel."""

    def __init__(self, num_groups: int, channels: int):
        super().__init__()
        self.channels = channels
        if channels > 0:
            self.gamma = nn.Embedding(num_groups, channels)
            self.beta  = nn.Embedding(num_groups, channels)
            # Initialise as identity so the model starts unmodified.
            nn.init.ones_(self.gamma.weight)
            nn.init.zeros_(self.beta.weight)

    def forward(self, x: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        if self.channels == 0:
            return x
        g = self.gamma(group_ids)[:, :, None, None]   # (B, C, 1, 1)
        b = self.beta(group_ids)[:, :, None, None]
        return g * x + b


class StyleConditionedSegModel(nn.Module):
    """Wraps an SMP segmentation model with FiLM-conditioned encoder features.

    At *training* time, the true ``group_ids`` are passed so the model can
    learn separate annotation-style modes.
    At *inference* time, ``clean_group_id`` is always used to produce
    debiased predictions.
    """

    def __init__(self, seg_model: nn.Module, num_groups: int = 2):
        super().__init__()
        self.seg_model = seg_model

        encoder_channels = seg_model.encoder.out_channels
        self.films = nn.ModuleList([
            FiLMLayer(num_groups, ch) for ch in encoder_channels
        ])

    def forward(self, images: torch.Tensor,
                group_ids: torch.Tensor) -> torch.Tensor:
        features = self.seg_model.encoder(images)

        modulated = []
        for feat, film in zip(features, self.films):
            modulated.append(film(feat, group_ids))

        decoder_out = self.seg_model.decoder(modulated)
        logits = self.seg_model.segmentation_head(decoder_out)
        return logits

    def predict(self, images: torch.Tensor,
                clean_group_id: int) -> torch.Tensor:
        """Inference with the clean-group embedding only."""
        B = images.size(0)
        gids = torch.full((B,), clean_group_id,
                          dtype=torch.long, device=images.device)
        return self.forward(images, gids)


# ===================================================================
# Balanced batch sampler  (ensures both groups in every batch)
# ===================================================================

class GroupBalancedBatchSampler:
    """Batch sampler that approximately balances demographic groups."""

    def __init__(self, groups, batch_size, group1_frac=0.5,
                 seed=42, drop_last=True):
        self.groups     = np.asarray(groups)
        self.batch_size = int(batch_size)
        self.group1_frac = float(group1_frac)
        self.seed       = int(seed)
        self.drop_last  = drop_last

        self.idx0 = np.where(self.groups == 0)[0]
        self.idx1 = np.where(self.groups == 1)[0]

        n = len(self.groups)
        self._n_batches = (n // self.batch_size if drop_last
                           else int(np.ceil(n / self.batch_size)))
        self._n_batches = max(self._n_batches, 0)
        self._iter_count = 0

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self._iter_count)
        self._iter_count += 1

        if len(self.idx0) == 0 or len(self.idx1) == 0:
            all_idx = np.arange(len(self.groups))
            rng.shuffle(all_idx)
            for i in range(self._n_batches):
                s = i * self.batch_size
                yield all_idx[s:s + self.batch_size].tolist()
            return

        n1 = int(self.batch_size * self.group1_frac)
        n0 = self.batch_size - n1
        if n0 <= 0 or n1 <= 0:
            all_idx = np.arange(len(self.groups))
            rng.shuffle(all_idx)
            for i in range(self._n_batches):
                s = i * self.batch_size
                yield all_idx[s:s + self.batch_size].tolist()
            return

        for _ in range(self._n_batches):
            b0 = rng.choice(self.idx0, size=n0, replace=(len(self.idx0) < n0))
            b1 = rng.choice(self.idx1, size=n1, replace=(len(self.idx1) < n1))
            batch = np.concatenate([b0, b1])
            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self._n_batches


# ===================================================================
# Model factory
# ===================================================================

def create_base_model(args, num_classes=2):
    model_fn = {
        'unet':       smp.Unet,
        'unetpp':     smp.UnetPlusPlus,
        'deeplabv3':  smp.DeepLabV3,
        'deeplabv3p': smp.DeepLabV3Plus,
        'fpn':        smp.FPN,
        'pspnet':     smp.PSPNet,
    }[args.model]

    return model_fn(
        encoder_name=args.encoder,
        encoder_weights='imagenet' if args.pretrained else None,
        in_channels=3,
        classes=num_classes,
    )


def create_model(args):
    base = create_base_model(args, num_classes=2)
    if args.method in ('style_cond', 'asym_style'):
        return StyleConditionedSegModel(base, num_groups=2)
    return base


# ===================================================================
# Data loaders
# ===================================================================

def create_dataloaders(args, train_idx, val_idx, test_idx):
    train_ds, val_ds, test_ds = create_datasets(
        args, train_idx, val_idx, test_idx,
    )

    kw = dict(batch_size=args.batch_size,
              num_workers=args.num_workers, pin_memory=True)

    if args.balance_groups:
        genders = train_ds.get_all_genders()
        if {0, 1}.issubset(set(np.unique(genders))):
            sampler = GroupBalancedBatchSampler(
                genders, args.batch_size, seed=args.seed)
            train_loader = DataLoader(
                train_ds, batch_sampler=sampler, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)
        else:
            print('Warning: only one group present, falling back to shuffle')
            train_loader = DataLoader(train_ds, shuffle=True,
                                      drop_last=True, **kw)
    else:
        train_loader = DataLoader(train_ds, shuffle=True,
                                  drop_last=True, **kw)

    val_loader  = DataLoader(val_ds,  shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    return train_loader, val_loader, test_loader, train_ds


# ===================================================================
# Training
# ===================================================================

def train_one_epoch(model, loader, optimizer, epoch, device, args):
    model.train()
    use_asym  = args.method in ('asymmetric', 'asym_style')
    use_style = args.method in ('style_cond', 'asym_style')

    metrics = SegmentationMetrics(num_classes=2)
    running = {'seg_loss': 0.0, 'total_loss': 0.0}
    biased_losses, clean_losses = [], []
    n_batches = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch+1} Train')
    for batch in pbar:
        images  = batch['image'].to(device)
        masks   = batch['mask'].to(device)
        genders = batch['gender'].to(device)
        is_biased_flag = batch.get(
            'is_biased', torch.zeros(len(images), dtype=torch.bool))

        # ---------- forward ----------
        if use_style:
            seg_logits = model(images, group_ids=genders)
        else:
            seg_logits = model(images)

        # ---------- loss ----------
        if use_asym:
            loss_weight = build_asymmetric_loss_weight(
                masks, genders,
                biased_group_id=args.biased_group_id,
                boundary_width=args.boundary_width,
                interior_weight=args.interior_weight,
            )
            seg_loss, per_sample = asymmetric_combined_loss(
                seg_logits, masks, loss_weight,
                ce_weight=args.ce_weight,
                dice_weight=args.dice_weight,
            )
        else:
            seg_loss, per_sample = standard_combined_loss(
                seg_logits, masks,
                ce_weight=args.ce_weight,
                dice_weight=args.dice_weight,
            )

        total_loss = seg_loss

        # ---------- backward ----------
        optimizer.zero_grad()
        total_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # ---------- tracking ----------
        for sl, b in zip(per_sample, is_biased_flag):
            (biased_losses if b else clean_losses).append(sl.item())

        with torch.no_grad():
            preds = seg_logits.argmax(dim=1)
            metrics.update(preds, masks, genders.cpu(),
                           loss=seg_loss.item(),
                           per_sample_loss=per_sample.detach())

        running['seg_loss']   += seg_loss.item()
        running['total_loss'] += total_loss.item()
        n_batches += 1

        pbar.set_postfix(seg=f"{seg_loss.item():.4f}")

    results = metrics.compute()
    results['seg_loss']   = running['seg_loss']   / max(n_batches, 1)
    results['total_loss'] = running['total_loss'] / max(n_batches, 1)
    if biased_losses:
        results['biased_sample_loss'] = np.mean(biased_losses)
    if clean_losses:
        results['clean_sample_loss'] = np.mean(clean_losses)
    return results


# ===================================================================
# Evaluation
# ===================================================================

@torch.no_grad()
def evaluate(model, loader, device, args, desc='Validation'):
    model.eval()
    use_style = args.method in ('style_cond', 'asym_style')
    metrics = SegmentationMetrics(num_classes=2)

    for batch in tqdm(loader, desc=desc):
        images  = batch['image'].to(device)
        masks   = batch['mask'].to(device)
        genders = batch['gender']

        if use_style:
            eval_group = args.force_group_id if args.force_group_id is not None else args.clean_group_id
            seg_logits = model.predict(images, eval_group)
        else:
            seg_logits = model(images)

        loss, per_sample = standard_combined_loss(
            seg_logits, masks,
            ce_weight=args.ce_weight, dice_weight=args.dice_weight)

        preds = seg_logits.argmax(dim=1)
        metrics.update(preds, masks, genders,
                       loss=loss.item(), per_sample_loss=per_sample)

    return metrics.compute()


# ===================================================================
# Helpers
# ===================================================================

def log_metrics(writer, metrics, step, prefix='train'):
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.floating)):
            writer.add_scalar(f'{prefix}/{k}', v, step)


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }, path)


# ===================================================================
# Argument parsing
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Asymmetric debiasing segmentation training')

    # Dataset
    add_dataset_args(p)
    p.add_argument('--data_root', type=str,
                   default='/path/to/CelebAMask-HQ')
    p.add_argument('--img_size', type=int, default=256)

    # Splits
    p.add_argument('--splits_path', type=str,
                   default='configs/splits/cv_splits.json')
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--test_ratio', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)

    # Bias parameters
    p.add_argument('--bias_ratio', type=float, default=0.25)
    p.add_argument('--erosion_radius', type=int, default=15)
    p.add_argument('--biased_gender', type=int, default=0, choices=[0, 1])
    p.add_argument('--bias_seed', type=int, default=42)

    # Debiasing method
    p.add_argument('--method', type=str, default='asymmetric',
                   choices=['asymmetric', 'style_cond', 'asym_style'])
    p.add_argument('--boundary_width', type=int, default=2,
                   help='Half-width of boundary band (pixels) for asymmetric masking')
    p.add_argument('--clean_group_id', type=int, default=None,
                   help='Group ID used at inference (auto-detected if omitted)')
    p.add_argument('--force_group_id', type=int, default=None,
                   help='Force a specific group ID during eval_only inference (for ablations)')
    p.add_argument('--biased_group_id', type=int, default=None,
                   help='Group whose boundaries are masked (auto-detected if omitted)')
    p.add_argument('--interior_weight', type=float, default=1.0,
                   help='Loss weight for interior pixels of biased group '
                        '(1.0=full, <1 to further down-weight)')
    p.add_argument('--balance_groups', action='store_true', default=False,
                   help='Balance demographic groups within each batch')

    # Model
    p.add_argument('--model', type=str, default='unet',
                   choices=['unet', 'unetpp', 'deeplabv3', 'deeplabv3p',
                            'fpn', 'pspnet'])
    p.add_argument('--encoder', type=str, default='resnet34')
    p.add_argument('--pretrained', action='store_true', default=True)

    # Training
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--grad_clip', type=float, default=1.0)

    # Loss
    p.add_argument('--ce_weight', type=float, default=0.5)
    p.add_argument('--dice_weight', type=float, default=0.5)

    # Output
    p.add_argument('--output_dir', type=str,
                   default='/path/to/output')
    p.add_argument('--exp_name', type=str, default=None)

    # Misc
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--resume', type=str, default=None)

    args = p.parse_args()
    apply_dataset_defaults(args)

    # Auto-detect group IDs based on dataset
    if args.biased_group_id is None:
        args.biased_group_id = 1 if args.dataset == 'phc' else args.biased_gender
    if args.clean_group_id is None:
        args.clean_group_id = 1 - args.biased_group_id

    return args


# ===================================================================
# Main
# ===================================================================

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ---- Recover saved config in eval-only mode ----
    # When --eval_only is used, the CLI defaults (e.g. --method=asymmetric)
    # may not match the original training config.  Read back the saved
    # config.json AND detect the actual model type from checkpoint keys
    # (config.json may have the wrong method due to earlier bugs).
    if args.eval_only and args.exp_name is not None:
        fold_dir = (Path(args.output_dir) / 'experiments'
                    / args.exp_name / f'fold_{args.fold}')
        saved_cfg = {}
        saved_cfg_path = fold_dir / 'config.json'
        if saved_cfg_path.exists():
            with open(saved_cfg_path) as f:
                saved_cfg = json.load(f)
            # Restore keys that affect evaluation (NOT method — detected below)
            for key in ('model', 'encoder', 'pretrained',
                        'biased_group_id', 'clean_group_id', 'biased_gender',
                        'boundary_width', 'interior_weight',
                        'bias_ratio', 'erosion_radius', 'dataset',
                        'ce_weight', 'dice_weight'):
                if key in saved_cfg:
                    setattr(args, key, saved_cfg[key])
            print(f'Restored config from {saved_cfg_path}')

        # Detect actual model type from checkpoint keys (authoritative)
        ckpt_path = fold_dir / 'best_model.pt'
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location='cpu',
                              weights_only=False)
            ckpt_keys = set(ckpt['model_state_dict'].keys())
            has_films = any(k.startswith('films.') for k in ckpt_keys)
            has_seg_model = any(k.startswith('seg_model.') for k in ckpt_keys)
            if has_films and has_seg_model:
                # Distinguish asym_style vs style_cond from experiment name
                if 'asym_style' in args.exp_name or 'asym_style' in saved_cfg.get('method', ''):
                    args.method = 'asym_style'
                else:
                    args.method = 'style_cond'
            elif has_seg_model:
                args.method = 'style_cond'
            else:
                args.method = saved_cfg.get('method', 'asymmetric')
            del ckpt  # free memory
            print(f'  Detected method={args.method} from checkpoint keys'
                  f' (films={has_films}, seg_model={has_seg_model})')

    # ---- Splits ----
    splits = create_splits(args)
    train_idx, val_idx, test_idx = get_fold_indices(splits, args.fold)
    print(f'Fold {args.fold}: Train={len(train_idx)}  Val={len(val_idx)}  '
          f'Test={len(test_idx)}')

    # ---- Data ----
    train_loader, val_loader, test_loader, train_ds = create_dataloaders(
        args, train_idx, val_idx, test_idx)
    print('\n' + '=' * 60)
    if hasattr(train_ds, 'get_bias_summary'):
        print(train_ds.get_bias_summary())
    print('=' * 60)

    # ---- Model ----
    model = create_model(args).to(device)

    use_style = args.method in ('style_cond', 'asym_style')
    use_asym  = args.method in ('asymmetric', 'asym_style')

    seg_model_ref = model.seg_model if use_style else model
    total_params = sum(p.numel() for p in model.parameters())
    base_params  = sum(p.numel() for p in seg_model_ref.parameters())
    extra_params = total_params - base_params

    print(f'\nModel params: {total_params:,} total  '
          f'({extra_params:,} FiLM conditioning)')
    print(f'Method: {args.method}')
    print(f'  Biased group: {args.biased_group_id}  '
          f'Clean group: {args.clean_group_id}')
    if use_asym:
        print(f'  Boundary width: {args.boundary_width}px  '
              f'Interior weight: {args.interior_weight}')
    print()

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # ---- Experiment dir ----
    if args.exp_name is None:
        method_tag = args.method
        if use_asym:
            method_tag += f'_bw{args.boundary_width}'
        if args.dataset == 'phc':
            bias_tag = f'phc_biased_r{int(args.bias_ratio * 100)}'
        else:
            gname = 'female' if args.biased_gender == 0 else 'male'
            bias_tag = (f'bias_{gname}_r{int(args.bias_ratio * 100)}'
                        f'_e{args.erosion_radius}')
        args.exp_name = f'debias_{method_tag}_{bias_tag}'

    exp_dir = (Path(args.output_dir) / 'experiments'
               / args.exp_name / f'fold_{args.fold}')
    exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(exp_dir / 'logs'))

    # Save config
    config = vars(args).copy()
    config['bias_info'] = (train_ds.bias_info
                           if hasattr(train_ds, 'bias_info') else {})
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    if hasattr(train_ds, 'save_bias_indices'):
        train_ds.save_bias_indices(str(exp_dir / 'biased_indices.json'))

    # ---- Training loop ----
    best_val_iou = 0.0

    if not args.eval_only:
        print('Starting training...\n')

        for epoch in range(args.epochs):
            train_m = train_one_epoch(
                model, train_loader, optimizer, epoch, device, args)
            log_metrics(writer, train_m, epoch, prefix='train')

            g0, g1 = get_demographic_names(args.dataset)
            print(f"\n  Train  seg_loss: {train_m['seg_loss']:.4f}  "
                  f"IoU: {train_m['mean_iou']:.4f}")
            print(f"         {g0}-IoU: "
                  f"{train_m['female_iou_foreground']:.4f}  "
                  f"{g1}-IoU: "
                  f"{train_m['male_iou_foreground']:.4f}  "
                  f"Gap: {train_m['iou_gap']:.4f}")
            if 'biased_sample_loss' in train_m:
                print(f"         Biased loss: "
                      f"{train_m['biased_sample_loss']:.4f}  "
                      f"Clean loss: "
                      f"{train_m.get('clean_sample_loss', 0):.4f}")

            # Validation
            val_m = evaluate(model, val_loader, device, args,
                             desc='Validation')
            log_metrics(writer, val_m, epoch, prefix='val')

            print(f"  Val    loss: {val_m['loss']:.4f}  "
                  f"IoU: {val_m['mean_iou']:.4f}")
            print(f"         {g0}-IoU: "
                  f"{val_m['female_iou_foreground']:.4f}  "
                  f"{g1}-IoU: "
                  f"{val_m['male_iou_foreground']:.4f}  "
                  f"Gap: {val_m['iou_gap']:.4f}")

            writer.add_scalar('train/lr',
                              optimizer.param_groups[0]['lr'], epoch)

            # Checkpoint
            if val_m['mean_iou'] > best_val_iou:
                best_val_iou = val_m['mean_iou']
                save_checkpoint(model, optimizer, scheduler, epoch,
                                val_m, exp_dir / 'best_model.pt')
                print(f'  ** New best! IoU: {best_val_iou:.4f} **')

            save_checkpoint(model, optimizer, scheduler, epoch,
                            val_m, exp_dir / 'latest_model.pt')
            scheduler.step()

    # ---- Final test ----
    print('\n' + '=' * 60)
    print('Final Test Evaluation (clean labels)')
    print('=' * 60 + '\n')

    best_ckpt = exp_dir / 'best_model.pt'
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

    test_m = evaluate(model, test_loader, device, args, desc='Test')
    log_metrics(writer, test_m, args.epochs, prefix='test')

    g0, g1 = get_demographic_names(args.dataset)
    print('Test Results:')
    print(f"  Mean IoU:       {test_m['mean_iou']:.4f}")
    print(f"  FG IoU:         {test_m['iou_foreground']:.4f}")
    print(f"  FG Dice:        {test_m['dice_foreground']:.4f}")
    print(f"\n  {g0} IoU:     {test_m['female_iou_foreground']:.4f}  "
          f"Dice: {test_m['female_dice_foreground']:.4f}")
    print(f"  {g1} IoU:       {test_m['male_iou_foreground']:.4f}  "
          f"Dice: {test_m['male_dice_foreground']:.4f}")
    print(f"\n  IoU Gap:        {test_m['iou_gap']:.4f}")
    print(f"  Dice Gap:       {test_m['dice_gap']:.4f}")

    with open(exp_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_m, f, indent=2)

    writer.close()
    print(f'\nResults saved to: {exp_dir}')


if __name__ == '__main__':
    main()
