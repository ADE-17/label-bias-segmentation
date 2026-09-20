#!/usr/bin/env python3
"""
Debiased segmentation training with three strategies:

  1. adversarial   – Gradient reversal + gender adversary (DANN-style)
  2. fairness      – Fairness loss regularisation (DP / EO)
  3. domain_invariant – Feature alignment across genders (MMD / CORAL)

Trains on biased data (eroded masks) while actively mitigating
the bias effect through the chosen debiasing method.

Example usage:
    # Adversarial debiasing
    for FOLD in 0 1 2 3 4; do
        python train_debiased.py --method adversarial --bias_ratio 1.0 \
            --erosion_radius 15 --epochs 20 --lambda_debias 0.01 --fold $FOLD
    done

    # Fairness loss (equalized odds)
    python train_debiased.py --method fairness --fairness_mode eo \\
        --bias_ratio 0.25 --erosion_radius 15 --fold 0

    # Domain invariant (MMD)
    for FOLD in 0 1 2 3 4; do
        python train_debiased.py --method domain_invariant --domain_method mmd --bias_ratio 0.25 --erosion_radius 15 --fold $FOLD --epochs 20 --lambda_debias 0.01 --mmd_normalize --balance_gender_in_batch
    done

    for FOLD in 0 1 2 3 4; do
        python train_debiased.py --method domain_invariant --domain_method coral --bias_ratio 1.00 --erosion_radius 15 --fold $FOLD --epochs 20 --lambda_debias 0.01
    done

    for FOLD in 0 1 2 3 4; do
        python train_debiased.py --loss_mode boundary_dice --bias_ratio 0.50 --erosion_radius 15 --fold $FOLD --epochs 20 --lambda_debias 0.01
    done
"""
import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

import segmentation_models_pytorch as smp

from dataloader import CelebAMaskHQDataset, CelebAMaskHQBiasedDataset
from utils.splits import create_kfold_splits, save_splits, load_splits, get_fold_indices
from utils.metrics import SegmentationMetrics, combined_loss
from debiasing import (
    SegmentationWithDebiasing,
    grl_alpha_schedule,
)
from dataset_factory import (
    add_dataset_args, apply_dataset_defaults, create_splits, create_datasets,
    get_demographic_names, DATASET_DEFAULTS,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Debiased segmentation training')

    # Dataset selection
    add_dataset_args(parser)

    # Data
    parser.add_argument('--data_root', type=str,
                        default='/path/to/CelebAMask-HQ')
    parser.add_argument('--img_size', type=int, default=256)

    # Splits
    parser.add_argument('--splits_path', type=str,
                        default='configs/splits/cv_splits.json')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)

    # Bias parameters
    parser.add_argument('--bias_ratio', type=float, default=0.25)
    parser.add_argument('--erosion_radius', type=int, default=15)
    parser.add_argument('--biased_gender', type=int, default=0, choices=[0, 1])
    parser.add_argument('--bias_seed', type=int, default=42)

    # Debiasing method
    parser.add_argument('--method', type=str, default='adversarial',
                        choices=['adversarial', 'fairness', 'domain_invariant'])
    parser.add_argument('--lambda_debias', type=float, default=0.1,
                        help='Weight for the debiasing loss term')
    parser.add_argument('--lambda_warmup', action='store_true', default=True,
                        help='Ramp lambda_debias from 0 alongside GRL alpha schedule')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Max gradient norm (0 to disable)')

    # Adversarial-specific
    parser.add_argument('--adv_hidden', type=int, default=256)
    parser.add_argument('--adv_lr', type=float, default=5e-4,
                        help='Separate LR for the adversary head')
    parser.add_argument('--grl_warmup', type=int, default=5,
                        help='Epochs before GRL activates')
    parser.add_argument('--grl_max_alpha', type=float, default=1.0)

    # Fairness-specific
    parser.add_argument('--fairness_mode', type=str, default='both',
                        choices=['dp', 'eo', 'both'])

    # Domain-invariant-specific
    parser.add_argument('--domain_method', type=str, default='mmd_logit',
                        choices=['mmd', 'mmd_logit', 'coral'])
    parser.add_argument('--kernel_bandwidth', type=float, default=1.0)
    parser.add_argument('--mmd_normalize', action='store_true', default=False,
                        help='L2-normalize vectors before computing MMD')
    parser.add_argument('--balance_gender_in_batch', action='store_true', default=False,
                        help='Ensure each training batch contains both genders (0/1)')
    parser.add_argument('--gender_batch_group1_fraction', type=float, default=0.5,
                        help='When balancing, fraction of group=1 samples per batch')

    # Model
    parser.add_argument('--model', type=str, default='unet',
                        choices=['unet', 'unetpp', 'deeplabv3', 'deeplabv3p', 'fpn', 'pspnet'])
    parser.add_argument('--encoder', type=str, default='resnet34')
    parser.add_argument('--pretrained', action='store_true', default=True)

    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)

    # Loss
    parser.add_argument('--loss_mode', type=str, default='ce_dice',
                        choices=['ce_dice', 'ce', 'dice', 'boundary_dice'],
                        help='Segmentation loss. boundary_dice focuses only on object boundaries.')
    parser.add_argument('--ce_weight', type=float, default=0.5)
    parser.add_argument('--dice_weight', type=float, default=0.5)
    parser.add_argument('--boundary_width', type=int, default=2,
                        help='Boundary half-width (pixels) for boundary_dice loss')

    # Output
    parser.add_argument('--output_dir', type=str,
                        default='/path/to/output')
    parser.add_argument('--exp_name', type=str, default=None)

    # Misc
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()
    apply_dataset_defaults(args)
    return args


def create_base_model(args, num_classes=2):
    model_fn = {
        'unet': smp.Unet,
        'unetpp': smp.UnetPlusPlus,
        'deeplabv3': smp.DeepLabV3,
        'deeplabv3p': smp.DeepLabV3Plus,
        'fpn': smp.FPN,
        'pspnet': smp.PSPNet,
    }[args.model]

    return model_fn(
        encoder_name=args.encoder,
        encoder_weights='imagenet' if args.pretrained else None,
        in_channels=3,
        classes=num_classes,
    )


class GenderBalancedBatchSampler:
    """Batch sampler that (approximately) balances gender groups inside each batch."""

    def __init__(
        self,
        genders: np.ndarray,
        batch_size: int,
        group1_fraction: float = 0.5,
        seed: int = 42,
        drop_last: bool = True,
    ):
        self.genders = np.asarray(genders)
        self.batch_size = int(batch_size)
        self.group1_fraction = float(group1_fraction)
        self.seed = int(seed)
        self.drop_last = drop_last

        idx0 = np.where(self.genders == 0)[0]
        idx1 = np.where(self.genders == 1)[0]
        self.idx0 = idx0
        self.idx1 = idx1

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        # Precompute number of batches per epoch for __len__.
        n = len(self.genders)
        self._n_batches = n // self.batch_size if self.drop_last else int(np.ceil(n / self.batch_size))

        if self._n_batches < 1:
            self._n_batches = 0

    def __iter__(self):
        # Different shuffling each epoch/iterator invocation.
        it_seed = self.seed + getattr(self, "_iter_count", 0)
        rng = np.random.RandomState(it_seed)
        self._iter_count = getattr(self, "_iter_count", 0) + 1

        # If one group is missing, fall back to plain random batching.
        if len(self.idx0) == 0 or len(self.idx1) == 0:
            all_idx = np.arange(len(self.genders))
            rng.shuffle(all_idx)
            for i in range(self._n_batches):
                start = i * self.batch_size
                end = start + self.batch_size
                yield all_idx[start:end].tolist()
            return

        n1 = int(self.batch_size * self.group1_fraction)
        n0 = self.batch_size - n1
        if n0 <= 0 or n1 <= 0:
            # Degenerate fractions -> don't try to force balance.
            all_idx = np.arange(len(self.genders))
            rng.shuffle(all_idx)
            for i in range(self._n_batches):
                start = i * self.batch_size
                end = start + self.batch_size
                yield all_idx[start:end].tolist()
            return

        for _ in range(self._n_batches):
            batch0 = rng.choice(self.idx0, size=n0, replace=(len(self.idx0) < n0))
            batch1 = rng.choice(self.idx1, size=n1, replace=(len(self.idx1) < n1))
            batch = np.concatenate([batch0, batch1])
            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self._n_batches


def create_dataloaders(args, train_indices, val_indices, test_indices):
    train_dataset, val_dataset, test_dataset = create_datasets(
        args, train_indices, val_indices, test_indices,
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.balance_gender_in_batch:
        genders = train_dataset.get_all_genders()
        unique = set(np.unique(genders).tolist())
        if {0, 1}.issubset(unique):
            batch_sampler = GenderBalancedBatchSampler(
                genders=genders,
                batch_size=args.batch_size,
                group1_fraction=args.gender_batch_group1_fraction,
                seed=args.seed,
                drop_last=True,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        else:
            print(
                "Warning: could not balance genders (missing one group in train split). "
                "Falling back to standard shuffling."
            )
            train_loader = DataLoader(
                train_dataset, shuffle=True, drop_last=True, **loader_kwargs
            )
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, train_dataset


def build_optimizers(model, args):
    """
    Adversarial method uses two optimizers:
      - main optimizer for encoder + decoder + segmentation head
      - adversary optimizer for the gender classifier
    Other methods use a single optimizer for everything.
    """
    if args.method == 'adversarial':
        seg_params = list(model.seg_model.parameters())
        adv_params = list(model.adversary.parameters())

        seg_optimizer = torch.optim.AdamW(
            seg_params, lr=args.lr, weight_decay=args.weight_decay
        )
        adv_optimizer = torch.optim.AdamW(
            adv_params, lr=args.adv_lr, weight_decay=args.weight_decay
        )
        seg_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            seg_optimizer, T_max=args.epochs
        )
        return seg_optimizer, adv_optimizer, seg_scheduler
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        return optimizer, None, scheduler


def train_one_epoch(model, train_loader, optimizers, epoch, device, args):
    model.train()
    seg_optimizer, adv_optimizer, _ = optimizers
    metrics = SegmentationMetrics(num_classes=2)

    alpha = grl_alpha_schedule(
        epoch, args.epochs,
        warmup_epochs=args.grl_warmup,
        max_alpha=args.grl_max_alpha,
    ) if args.method == 'adversarial' else 0.0

    # Ramp lambda_debias from 0 → target using the same schedule as GRL
    if args.lambda_warmup:
        effective_lambda = args.lambda_debias * grl_alpha_schedule(
            epoch, args.epochs,
            warmup_epochs=args.grl_warmup,
            max_alpha=1.0,
        )
    else:
        effective_lambda = args.lambda_debias

    running = {'seg_loss': 0., 'debias_loss': 0., 'total_loss': 0.}
    adv_accs = []
    biased_losses, clean_losses = [], []
    n_batches = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} Train')
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        genders = batch['gender'].to(device)
        is_biased = batch.get('is_biased', torch.zeros(len(images), dtype=torch.bool))

        # Forward through debiased model
        out = model(images, genders=genders, targets=masks, alpha=alpha)
        seg_logits = out['seg_logits']
        debias_loss = out['debiasing_loss']

        # Segmentation loss
        seg_loss, per_sample_loss = combined_loss(
            seg_logits, masks,
            ce_weight=args.ce_weight,
            dice_weight=args.dice_weight,
            loss_mode=args.loss_mode,
            boundary_width=args.boundary_width,
        )

        total_loss = seg_loss + effective_lambda * debias_loss

        # Backward
        if args.method == 'adversarial' and adv_optimizer is not None:
            seg_optimizer.zero_grad()
            adv_optimizer.zero_grad()
            total_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.seg_model.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(model.adversary.parameters(), args.grad_clip)
            seg_optimizer.step()
            adv_optimizer.step()
        else:
            seg_optimizer.zero_grad()
            total_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            seg_optimizer.step()

        # Track per-sample losses by bias status
        for sl, b in zip(per_sample_loss, is_biased):
            (biased_losses if b else clean_losses).append(sl.item())

        # Metrics
        with torch.no_grad():
            preds = seg_logits.argmax(dim=1)
            metrics.update(preds, masks, genders.cpu(), loss=seg_loss.item(),
                           per_sample_loss=per_sample_loss.detach())

        running['seg_loss'] += seg_loss.item()
        running['debias_loss'] += debias_loss.item()
        running['total_loss'] += total_loss.item()
        n_batches += 1

        if 'adv_accuracy' in out:
            adv_accs.append(out['adv_accuracy'].item())

        pbar.set_postfix({
            'seg': f"{seg_loss.item():.4f}",
            'db': f"{debias_loss.item():.4f}",
        })

    results = metrics.compute()
    results['seg_loss'] = running['seg_loss'] / max(n_batches, 1)
    results['debias_loss'] = running['debias_loss'] / max(n_batches, 1)
    results['total_loss'] = running['total_loss'] / max(n_batches, 1)

    if adv_accs:
        results['adv_accuracy'] = np.mean(adv_accs)
    if args.method == 'adversarial':
        results['grl_alpha'] = alpha
    results['effective_lambda'] = effective_lambda
    if biased_losses:
        results['biased_sample_loss'] = np.mean(biased_losses)
    if clean_losses:
        results['clean_sample_loss'] = np.mean(clean_losses)

    return results


@torch.no_grad()
def evaluate(model, data_loader, device, args, desc='Validation'):
    model.eval()
    metrics = SegmentationMetrics(num_classes=2)

    pbar = tqdm(data_loader, desc=desc)
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        genders = batch['gender']

        seg_logits = model.predict(images)
        loss, per_sample_loss = combined_loss(
            seg_logits, masks,
            ce_weight=args.ce_weight,
            dice_weight=args.dice_weight,
            loss_mode=args.loss_mode,
            boundary_width=args.boundary_width,
        )

        preds = seg_logits.argmax(dim=1)
        metrics.update(preds, masks, genders, loss=loss.item(),
                       per_sample_loss=per_sample_loss)

    return metrics.compute()


def log_metrics(writer, metrics, step, prefix='train'):
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating)):
            writer.add_scalar(f'{prefix}/{key}', value, step)


def save_checkpoint(model, optimizers, epoch, metrics, path):
    seg_opt, adv_opt, scheduler = optimizers
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'seg_optimizer_state_dict': seg_opt.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }
    if adv_opt is not None:
        state['adv_optimizer_state_dict'] = adv_opt.state_dict()
    torch.save(state, path)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ---- Splits ----
    splits = create_splits(args)

    train_idx, val_idx, test_idx = get_fold_indices(splits, args.fold)
    print(f'Fold {args.fold}: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}')

    # ---- Data ----
    train_loader, val_loader, test_loader, train_dataset = create_dataloaders(
        args, train_idx, val_idx, test_idx,
    )
    print('\n' + '=' * 60)
    if hasattr(train_dataset, 'get_bias_summary'):
        print(train_dataset.get_bias_summary())
    print('=' * 60)

    # ---- Model ----
    base_model = create_base_model(args, num_classes=2)
    model = SegmentationWithDebiasing(
        base_model,
        method=args.method,
        adversary_hidden=args.adv_hidden,
        fairness_mode=args.fairness_mode,
        domain_method=args.domain_method,
        kernel_bandwidth=args.kernel_bandwidth,
        mmd_normalize=args.mmd_normalize,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    debias_params = total_params - sum(p.numel() for p in model.seg_model.parameters())
    print(f'\nModel params: {total_params:,} total  ({debias_params:,} debiasing head)')
    print(f'Method: {args.method}  |  lambda_debias: {args.lambda_debias}')
    if args.method == 'adversarial':
        print(f'  GRL warmup: {args.grl_warmup} epochs, max_alpha: {args.grl_max_alpha}')
    elif args.method == 'fairness':
        print(f'  Fairness mode: {args.fairness_mode}')
    elif args.method == 'domain_invariant':
        print(f'  Domain method: {args.domain_method}')
    print()

    # ---- Optimizers ----
    seg_optimizer, adv_optimizer, scheduler = build_optimizers(model, args)
    optimizers = (seg_optimizer, adv_optimizer, scheduler)

    # ---- Experiment dir ----
    if args.exp_name is None:
        method_tag = args.method
        if args.method == 'fairness':
            method_tag += f"_{args.fairness_mode}"
        elif args.method == 'domain_invariant':
            method_tag += f"_{args.domain_method}"
            if args.mmd_normalize:
                method_tag += "_mmdnorm"
        if args.dataset == 'phc':
            bias_tag = f"phc_biased_r{int(args.bias_ratio*100)}"
        else:
            gender_name = 'female' if args.biased_gender == 0 else 'male'
            bias_tag = f"bias_{gender_name}_r{int(args.bias_ratio*100)}_e{args.erosion_radius}"
        args.exp_name = f"debias_{method_tag}_l{args.lambda_debias}_{bias_tag}"

    exp_dir = Path(args.output_dir) / 'experiments' / args.exp_name / f'fold_{args.fold}'
    exp_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(exp_dir / 'logs'))

    # Save config
    config = vars(args).copy()
    config['bias_info'] = train_dataset.bias_info if hasattr(train_dataset, 'bias_info') else {}
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    if hasattr(train_dataset, 'save_bias_indices'):
        train_dataset.save_bias_indices(str(exp_dir / 'biased_indices.json'))

    # ---- Training loop ----
    best_val_iou = 0.0

    if not args.eval_only:
        print('Starting training...\n')

        for epoch in range(args.epochs):
            train_metrics = train_one_epoch(
                model, train_loader, optimizers, epoch, device, args,
            )
            log_metrics(writer, train_metrics, epoch, prefix='train')

            g0, g1 = get_demographic_names(args.dataset)
            # Print training summary
            print(f"\n  Train  seg_loss: {train_metrics['seg_loss']:.4f}  "
                  f"debias_loss: {train_metrics['debias_loss']:.4f}  "
                  f"IoU: {train_metrics['mean_iou']:.4f}")
            print(f"         {g0}-IoU: {train_metrics['female_iou_foreground']:.4f}  "
                  f"{g1}-IoU: {train_metrics['male_iou_foreground']:.4f}  "
                  f"Gap: {train_metrics['iou_gap']:.4f}")
            if 'adv_accuracy' in train_metrics:
                print(f"         Adv acc: {train_metrics['adv_accuracy']:.4f}  "
                      f"GRL alpha: {train_metrics.get('grl_alpha', 0):.4f}  "
                      f"eff_lambda: {train_metrics.get('effective_lambda', 0):.4f}")
            if 'biased_sample_loss' in train_metrics:
                print(f"         Biased loss: {train_metrics['biased_sample_loss']:.4f}  "
                      f"Clean loss: {train_metrics.get('clean_sample_loss', 0):.4f}")

            # Validation
            val_metrics = evaluate(model, val_loader, device, args, desc='Validation')
            log_metrics(writer, val_metrics, epoch, prefix='val')

            print(f"  Val    loss: {val_metrics['loss']:.4f}  "
                  f"IoU: {val_metrics['mean_iou']:.4f}")
            print(f"         {g0}-IoU: {val_metrics['female_iou_foreground']:.4f}  "
                  f"{g1}-IoU: {val_metrics['male_iou_foreground']:.4f}  "
                  f"Gap: {val_metrics['iou_gap']:.4f}")

            writer.add_scalar('train/lr', seg_optimizer.param_groups[0]['lr'], epoch)

            # Checkpoint
            if val_metrics['mean_iou'] > best_val_iou:
                best_val_iou = val_metrics['mean_iou']
                save_checkpoint(model, optimizers, epoch, val_metrics, exp_dir / 'best_model.pt')
                print(f'  ** New best! IoU: {best_val_iou:.4f} **')

            save_checkpoint(model, optimizers, epoch, val_metrics, exp_dir / 'latest_model.pt')

            if scheduler is not None:
                scheduler.step()

    # ---- Final test ----
    print('\n' + '=' * 60)
    print('Final Test Evaluation (clean labels)')
    print('=' * 60 + '\n')

    if (exp_dir / 'best_model.pt').exists():
        ckpt = torch.load(exp_dir / 'best_model.pt', map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

    test_metrics = evaluate(model, test_loader, device, args, desc='Test')
    log_metrics(writer, test_metrics, args.epochs, prefix='test')

    g0, g1 = get_demographic_names(args.dataset)
    print('Test Results:')
    print(f"  Mean IoU:       {test_metrics['mean_iou']:.4f}")
    print(f"  FG IoU:         {test_metrics['iou_foreground']:.4f}")
    print(f"  FG Dice:        {test_metrics['dice_foreground']:.4f}")
    print(f"\n  {g0} IoU:     {test_metrics['female_iou_foreground']:.4f}  "
          f"Dice: {test_metrics['female_dice_foreground']:.4f}")
    print(f"  {g1} IoU:       {test_metrics['male_iou_foreground']:.4f}  "
          f"Dice: {test_metrics['male_dice_foreground']:.4f}")
    print(f"\n  IoU Gap:        {test_metrics['iou_gap']:.4f}")
    print(f"  Dice Gap:       {test_metrics['dice_gap']:.4f}")

    with open(exp_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    writer.close()
    print(f'\nResults saved to: {exp_dir}')


if __name__ == '__main__':
    main()
