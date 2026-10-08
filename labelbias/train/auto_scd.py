#!/usr/bin/env python3
"""
Hybrid Structural Debiasing for Segmentation:
1. Auto-Conditioning (Warmup): Learns which group is unbiased dynamically.
2. Mixture of Experts (MoE) Adapters: Routes features between dual decoders.
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

# ---- Asymmetric losses ----
def compute_boundary_mask(masks: torch.Tensor, width: int = 2) -> torch.Tensor:
    fg = (masks == 1).float().unsqueeze(1)
    k = 2 * width + 1
    dilated = F.max_pool2d(fg, kernel_size=k, stride=1, padding=width)
    eroded  = -F.max_pool2d(-fg, kernel_size=k, stride=1, padding=width)
    boundary = (dilated - eroded).clamp(min=0.0).squeeze(1)
    return boundary

def build_asymmetric_loss_weight(masks, genders, biased_group_id, boundary_width=2, interior_weight=1.0):
    boundary = compute_boundary_mask(masks, width=boundary_width)
    is_biased = (genders == biased_group_id).float()[:, None, None]
    weight = torch.ones_like(masks, dtype=torch.float32)
    weight = weight - boundary * is_biased
    if interior_weight < 1.0:
        interior_mask = (1.0 - boundary) * is_biased
        weight = weight - interior_mask * (1.0 - interior_weight)
    return weight.clamp(min=0.0)

def asymmetric_ce_loss(pred_logits, targets, loss_weight):
    ce = F.cross_entropy(pred_logits, targets, reduction='none')
    weighted = ce * loss_weight
    loss = weighted.sum() / loss_weight.sum().clamp(min=1.0)
    per_sample = weighted.sum(dim=(1, 2)) / loss_weight.sum(dim=(1, 2)).clamp(min=1.0)
    return loss, per_sample

def asymmetric_dice_loss(pred_logits, targets, loss_weight, smooth=1.0):
    pred_probs = F.softmax(pred_logits, dim=1)
    pred_fg = pred_probs[:, 1]
    target_fg = (targets == 1).float()
    intersection = (pred_fg * target_fg * loss_weight).sum(dim=(1, 2))
    pred_sum = (pred_fg * loss_weight).sum(dim=(1, 2))
    target_sum = (target_fg * loss_weight).sum(dim=(1, 2))
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    per_sample = 1.0 - dice
    return per_sample.mean(), per_sample

def asymmetric_combined_loss(pred_logits, targets, loss_weight, ce_weight=0.5, dice_weight=0.5):
    ce_loss, ce_ps = asymmetric_ce_loss(pred_logits, targets, loss_weight)
    d_loss, d_ps = asymmetric_dice_loss(pred_logits, targets, loss_weight)
    total = ce_weight * ce_loss + dice_weight * d_loss
    per_sample = ce_weight * ce_ps + dice_weight * d_ps
    return total, per_sample

def standard_combined_loss(pred_logits, targets, ce_weight=0.5, dice_weight=0.5):
    weight = torch.ones(targets.shape, dtype=torch.float32, device=targets.device)
    return asymmetric_combined_loss(pred_logits, targets, weight, ce_weight, dice_weight)

# ---- Models ----
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

class FiLMLayer(nn.Module):
    def __init__(self, num_groups: int, channels: int):
        super().__init__()
        self.gamma = nn.Embedding(num_groups, channels)
        self.beta = nn.Embedding(num_groups, channels)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
    def forward(self, x: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        g = self.gamma(group_ids)[:, :, None, None]
        b = self.beta(group_ids)[:, :, None, None]
        return (g * x + b).contiguous()

class StyleConditionedSegModel(nn.Module):
    def __init__(self, seg_model: nn.Module, num_groups: int = 2):
        super().__init__()
        self.seg_model = seg_model
        encoder_channels = seg_model.encoder.out_channels
        self.films = nn.ModuleList([FiLMLayer(num_groups, ch) for ch in encoder_channels])
    def forward(self, images: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        features = self.seg_model.encoder(images)
        modulated = []
        for feat, film in zip(features, self.films):
            modulated.append(film(feat, group_ids))
        decoder_out = self.seg_model.decoder(modulated)
        logits = self.seg_model.segmentation_head(decoder_out)
        return logits
    def predict(self, images: torch.Tensor, clean_group_id: int) -> torch.Tensor:
        B = images.size(0)
        gids = torch.full((B,), clean_group_id, dtype=torch.long, device=images.device)
        return self.forward(images, gids)

class MoESegmentationModel(nn.Module):
    def __init__(self, args, num_groups=2):
        super().__init__()
        # Two identical bases to get distinct decoders
        expert_0 = create_base_model(args, num_classes=2)
        expert_1 = create_base_model(args, num_classes=2)
        
        # Shared Encoder
        self.encoder = expert_0.encoder
        
        # Dual Decoders + Heads
        self.decoder_0 = expert_0.decoder
        self.head_0 = expert_0.segmentation_head
        
        self.decoder_1 = expert_1.decoder
        self.head_1 = expert_1.segmentation_head
        
        # Gating Adapter
        enc_ch = self.encoder.out_channels[-1]
        self.adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(enc_ch, num_groups)
        )
    
    def forward(self, images, force_group=None):
        features = self.encoder(images)
        
        # Gate weights
        gate_logits = self.adapter(features[-1])
        if force_group is not None:
             weights = torch.zeros_like(gate_logits)
             weights[:, force_group] = 1.0
        else:
             weights = F.softmax(gate_logits, dim=-1)
             
        dec0 = self.decoder_0(features)
        logits0 = self.head_0(dec0)
        
        dec1 = self.decoder_1(features)
        logits1 = self.head_1(dec1)
        
        w0 = weights[:, 0].view(-1, 1, 1, 1)
        w1 = weights[:, 1].view(-1, 1, 1, 1)
        
        return w0 * logits0 + w1 * logits1
        
    def predict(self, images, clean_group_id: int):
        return self.forward(images, force_group=clean_group_id)

def create_model(args):
    if args.hybrid_mode == 'moe_decoders':
        return MoESegmentationModel(args, num_groups=2)
    else:
        base = create_base_model(args, num_classes=2)
        return StyleConditionedSegModel(base, num_groups=2)

# --- Sampler ---
class GroupBalancedBatchSampler:
    def __init__(self, groups, batch_size, group1_frac=0.5, seed=42, drop_last=True):
        self.groups = np.asarray(groups)
        self.batch_size = int(batch_size)
        self.group1_frac = float(group1_frac)
        self.seed = int(seed)
        self.drop_last = drop_last
        self.idx0 = np.where(self.groups == 0)[0]
        self.idx1 = np.where(self.groups == 1)[0]
        n = len(self.groups)
        self._n_batches = (n // self.batch_size if drop_last else int(np.ceil(n / self.batch_size)))
        self._n_batches = max(self._n_batches, 0)
        self._iter_count = 0

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self._iter_count)
        self._iter_count += 1
        if len(self.idx0) == 0 or len(self.idx1) == 0:
            all_idx = np.arange(len(self.groups))
            rng.shuffle(all_idx)
            for i in range(self._n_batches):
                yield all_idx[i * self.batch_size:(i + 1) * self.batch_size].tolist()
            return
        n1 = int(self.batch_size * self.group1_frac)
        n0 = self.batch_size - n1
        for _ in range(self._n_batches):
            b0 = rng.choice(self.idx0, size=n0, replace=(len(self.idx0) < n0))
            b1 = rng.choice(self.idx1, size=n1, replace=(len(self.idx1) < n1))
            batch = np.concatenate([b0, b1])
            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self._n_batches

def create_dataloaders(args, train_idx, val_idx, test_idx):
    train_ds, val_ds, test_ds = create_datasets(args, train_idx, val_idx, test_idx)
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    if args.balance_groups:
        genders = train_ds.get_all_genders()
        if {0, 1}.issubset(set(np.unique(genders))):
            sampler = GroupBalancedBatchSampler(genders, args.batch_size, seed=args.seed)
            train_loader = DataLoader(train_ds, batch_sampler=sampler, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        else:
            train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    return train_loader, val_loader, test_loader, train_ds

# --- Training ---
def train_one_epoch(model, loader, optimizer, epoch, device, args, running_group_losses):
    model.train()
    metrics = SegmentationMetrics(num_classes=2)
    running = {'seg_loss': 0.0, 'total_loss': 0.0}
    n_batches = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch+1} Train')
    
    is_warmup = (epoch < args.warmup_epochs)
    
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        genders = batch['gender'].to(device)
        
        if args.hybrid_mode == 'auto_film':
            seg_logits = model(images, group_ids=genders)
        else:
            seg_logits = model(images, force_group=None) # dynamic routing

        if is_warmup or args.clean_group_id is None:
            seg_loss, per_sample = standard_combined_loss(seg_logits, masks, ce_weight=args.ce_weight, dice_weight=args.dice_weight)
        else:
            loss_weight = build_asymmetric_loss_weight(
                masks, genders, args.biased_group_id, args.boundary_width, args.interior_weight)
            seg_loss, per_sample = asymmetric_combined_loss(
                seg_logits, masks, loss_weight, args.ce_weight, args.dice_weight)

        total_loss = seg_loss
        optimizer.zero_grad()
        total_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if is_warmup:
            for g_id in [0, 1]:
                mask_g = (genders == g_id)
                if mask_g.any():
                    running_group_losses[g_id].extend(per_sample[mask_g].detach().cpu().numpy().tolist())

        with torch.no_grad():
            preds = seg_logits.argmax(dim=1)
            metrics.update(preds, masks, genders.cpu(), loss=seg_loss.item(), per_sample_loss=per_sample.detach())

        running['seg_loss'] += seg_loss.item()
        running['total_loss'] += total_loss.item()
        n_batches += 1
        pbar.set_postfix(seg=f"{seg_loss.item():.4f}")

    results = metrics.compute()
    results['seg_loss'] = running['seg_loss'] / max(n_batches, 1)
    results['total_loss'] = running['total_loss'] / max(n_batches, 1)
    return results

@torch.no_grad()
def evaluate(model, loader, device, args, desc='Validation'):
    model.eval()
    metrics = SegmentationMetrics(num_classes=2)
    for batch in tqdm(loader, desc=desc):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        genders = batch['gender']

        if args.clean_group_id is not None:
             seg_logits = model.predict(images, args.clean_group_id)
        else:
             group_ids = genders.to(device) if args.hybrid_mode == 'auto_film' else None
             seg_logits = model(images, group_ids) if args.hybrid_mode == 'auto_film' else model(images, force_group=None)

        loss, per_sample = standard_combined_loss(seg_logits, masks, ce_weight=args.ce_weight, dice_weight=args.dice_weight)
        preds = seg_logits.argmax(dim=1)
        metrics.update(preds, masks, genders, loss=loss.item(), per_sample_loss=per_sample)
    return metrics.compute()

# --- Helpers ---
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

def parse_args():
    p = argparse.ArgumentParser()
    add_dataset_args(p)
    p.add_argument('--data_root', type=str, default='/path/to/CelebAMask-HQ')
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--splits_path', type=str, default='configs/splits/cv_splits.json')
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--test_ratio', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--bias_ratio', type=float, default=0.25)
    p.add_argument('--erosion_radius', type=int, default=15)
    p.add_argument('--biased_gender', type=int, default=0, choices=[0, 1])
    p.add_argument('--bias_seed', type=int, default=42)
    p.add_argument('--bias_mode', type=str, default='erosion', choices=['erosion', 'dilation', 'wave'])

    # Hybrid specific
    p.add_argument('--hybrid_mode', type=str, default='auto_film', choices=['auto_film', 'moe_decoders'])
    p.add_argument('--warmup_epochs', type=int, default=5, help='Epochs to select clean group based on loss discrepancy')
    p.add_argument('--boundary_width', type=int, default=2)
    p.add_argument('--interior_weight', type=float, default=1.0)
    p.add_argument('--balance_groups', action='store_true', default=False)

    # Model
    p.add_argument('--model', type=str, default='unet', choices=['unet', 'unetpp', 'deeplabv3', 'deeplabv3p', 'fpn', 'pspnet'])
    p.add_argument('--encoder', type=str, default='resnet34')
    p.add_argument('--pretrained', action='store_true', default=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--ce_weight', type=float, default=0.5)
    p.add_argument('--dice_weight', type=float, default=0.5)
    p.add_argument('--output_dir', type=str, default='/path/to/output')
    p.add_argument('--exp_name', type=str, default=None)
    p.add_argument('--device', type=str, default='cuda')

    args = p.parse_args()
    apply_dataset_defaults(args)
    return args

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Fix for CuDNN engine backward crash with ViT interpolation
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    args.clean_group_id = None
    args.biased_group_id = None

    splits = create_splits(args)
    train_idx, val_idx, test_idx = get_fold_indices(splits, args.fold)
    train_loader, val_loader, test_loader, train_ds = create_dataloaders(args, train_idx, val_idx, test_idx)
    print('\n' + '=' * 60)
    if hasattr(train_ds, 'get_bias_summary'):
        print(train_ds.get_bias_summary())
    else:
        print("Dataset has no bias summary.")
    print('=' * 60)

    model = create_model(args).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel params: {total_params:,} total')
    print(f'Hybrid Mode: {args.hybrid_mode}')
    print(f'Warmup Epochs: {args.warmup_epochs}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.exp_name is None:
        gname = 'female' if args.biased_gender == 0 else 'male'
        bias_tag = f'bias_{gname}_r{int(args.bias_ratio * 100)}_e{args.erosion_radius}' if args.dataset != 'phc' else f'phc_biased_r{int(args.bias_ratio * 100)}'
        args.exp_name = f'debias_hybrid_{args.hybrid_mode}_{bias_tag}'

    exp_dir = Path(args.output_dir) / 'experiments' / args.exp_name / f'fold_{args.fold}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(exp_dir / 'logs'))

    config = vars(args).copy()
    config['bias_info'] = getattr(train_ds, 'bias_info', {})
    with open(exp_dir / 'config.json', 'w') as f:
         json.dump(config, f, indent=2, default=str)

    if hasattr(train_ds, 'save_bias_indices'):
        train_ds.save_bias_indices(str(exp_dir / 'biased_indices.json'))

    best_val_iou = 0.0
    running_group_losses = {0: [], 1: []}

    print('Starting training...\n')
    for epoch in range(args.epochs):
        
        train_m = train_one_epoch(model, train_loader, optimizer, epoch, device, args, running_group_losses)
        log_metrics(writer, train_m, epoch, prefix='train')

        # Evaluate Warmup end condition
        if epoch == args.warmup_epochs - 1:
            mean_loss_0 = np.mean(running_group_losses[0]) if running_group_losses[0] else float('inf')
            mean_loss_1 = np.mean(running_group_losses[1]) if running_group_losses[1] else float('inf')
            print(f"\n[Warmup Terminated] End of Epoch {epoch+1}")
            print(f"Group 0 Mean Loss: {mean_loss_0:.4f} | Group 1 Mean Loss: {mean_loss_1:.4f}")
            
            # User heuristic: clean labels have lower noise, low loss = clean
            clean_g = 0 if mean_loss_0 < mean_loss_1 else 1
            biased_g = 1 - clean_g
            args.clean_group_id = clean_g
            args.biased_group_id = biased_g
            print(f" => Automatically selected Clean Group: {clean_g}")
            print(f" => Automatically selected Biased Group: {biased_g}\n")

        print(f"\n  Train  seg_loss: {train_m['seg_loss']:.4f}  IoU: {train_m['mean_iou']:.4f}")
        
        val_m = evaluate(model, val_loader, device, args, desc='Validation')
        log_metrics(writer, val_m, epoch, prefix='val')
        print(f"  Val    loss: {val_m['loss']:.4f}  IoU: {val_m['mean_iou']:.4f}")
        
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
        
        if val_m['mean_iou'] > best_val_iou and args.clean_group_id is not None:
            best_val_iou = val_m['mean_iou']
            save_checkpoint(model, optimizer, scheduler, epoch, val_m, exp_dir / 'best_model.pt')
            print(f'  ** New best! IoU: {best_val_iou:.4f} **')
            
        save_checkpoint(model, optimizer, scheduler, epoch, val_m, exp_dir / 'latest_model.pt')
        scheduler.step()

    print('\nFinal Test Evaluation')
    if (exp_dir / 'best_model.pt').exists():
        ckpt = torch.load(exp_dir / 'best_model.pt', map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

    test_m = evaluate(model, test_loader, device, args, desc='Test')
    log_metrics(writer, test_m, args.epochs, prefix='test')
    
    d_names = get_demographic_names(args.dataset)
    g0, g1 = d_names[0], d_names[-1]
    print(f"\n  {g0} IoU: {test_m['female_iou_foreground']:.4f} | {g1} IoU: {test_m['male_iou_foreground']:.4f}")
    print(f"  IoU Gap: {test_m['iou_gap']:.4f}")
    with open(exp_dir / 'test_metrics.json', 'w') as f:
         json.dump(test_m, f, indent=2)
    writer.close()

if __name__ == '__main__':
    main()
