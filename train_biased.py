#!/usr/bin/env python3
"""
Binary segmentation training with label bias injection.
Supports erosion (under-seg), dilation (over-seg), and wave (non-uniform boundary) bias.

Example usage:
for FOLD in 0 1 2 3 4; do
    python train_biased.py --bias_mode erosion --bias_ratio .50 --erosion_radius 15 --fold $FOLD --epochs 10
done

# Dilation (over-segmentation) bias
python train_biased.py --bias_mode dilation --bias_ratio .50 --erosion_radius 10 --fold 0 --epochs 10

# Wave (non-uniform boundary perturbation) bias
python train_biased.py --bias_mode wave --bias_ratio .50 --erosion_radius 8 --fold 0 --epochs 10

python aggregate_cv_results.py --exp_dir /path/to/output/experiments/bias_female_r25_e5

# Train biased model on PHC
python train_biased.py --dataset phc --epochs 20 --fold 0

# Train debiased model on PHC (MMD)
python train_debiased.py --dataset phc --method domain_invariant --domain_method mmd_logit --epochs 20 --fold 0

# Feature analysis (auto-detects dataset from experiment config)
python feature_analysis.py --dataset phc --experiments phc_biased

# CL analysis (auto-detects dataset per experiment)
python run_cl_analysis.py --experiments phc_biased --folds 0

python train_biased.py --loss_mode dice_boundary --bias_ratio 0.50 --erosion_radius 15 --fold 0 --epochs 10 --exp_name dice_boundary_bias_r50_e15
"""
import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

import segmentation_models_pytorch as smp

from dataloader import CelebAMaskHQDataset, CelebAMaskHQBiasedDataset
from utils.splits import create_kfold_splits, save_splits, load_splits, get_fold_indices
from utils.metrics import SegmentationMetrics, combined_loss, boundary_dice_loss
from dataset_factory import (
    add_dataset_args, apply_dataset_defaults, create_splits, create_datasets,
    get_demographic_names, DATASET_DEFAULTS,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train with label bias injection')
    
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
    parser.add_argument('--bias_ratio', type=float, default=0.25,
                       help='Fraction of target gender to apply bias (0.0 to 1.0)')
    parser.add_argument('--erosion_radius', type=int, default=5,
                       help='Radius for erosion/dilation or amplitude for wave perturbation')
    parser.add_argument('--biased_gender', type=int, default=0,
                       choices=[0, 1], help='Gender to bias (0=Female, 1=Male)')
    parser.add_argument('--bias_seed', type=int, default=42,
                       help='Seed for selecting biased samples')
    parser.add_argument('--bias_mode', type=str, default='erosion',
                       choices=['erosion', 'dilation', 'wave'],
                       help='Type of label bias: erosion (under-seg), dilation (over-seg), '
                            'wave (non-uniform boundary perturbation)')
    
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
                       choices=['ce_dice', 'ce', 'dice', 'boundary_dice', 'dice_boundary', 'gce', 'gce_dice', 'bootstrapping', 'bootstrapping_dice'],
                       help='Segmentation loss mode.')
    parser.add_argument('--ce_weight', type=float, default=0.5)
    parser.add_argument('--dice_weight', type=float, default=0.5)
    parser.add_argument('--boundary_width', type=int, default=2)
    parser.add_argument('--boundary_weight', type=float, default=0.5,
                       help='Weight for boundary_dice term in dice_boundary mode')
    parser.add_argument('--gce_q', type=float, default=0.7,
                       help='Robustness parameter q for Generalized Cross Entropy (0, 1]')
    parser.add_argument('--bootstrapping_beta', type=float, default=0.8,
                       help='Weight for the noisy label in Bootstrapping loss (0, 1)')
    
    # Output
    parser.add_argument('--output_dir', type=str,
                       default='/path/to/output')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')
    
    # Misc
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    apply_dataset_defaults(args)
    return args


def create_model(args, num_classes=2):
    """Create segmentation model."""
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


def create_dataloaders(args, train_indices, val_indices, test_indices):
    """Create dataloaders with bias applied to training set only."""
    train_dataset, val_dataset, test_dataset = create_datasets(
        args, train_indices, val_indices, test_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_dataset


def train_one_epoch(model, train_loader, optimizer, scheduler, device, args):
    """Train for one epoch."""
    model.train()
    metrics = SegmentationMetrics(num_classes=2)
    
    # Track biased vs non-biased sample metrics
    biased_losses = []
    clean_losses = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        genders = batch['gender']
        is_biased = batch.get('is_biased', torch.zeros(len(images), dtype=torch.bool))
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss, per_sample_loss = combined_loss(
            outputs, masks,
            ce_weight=args.ce_weight,
            dice_weight=args.dice_weight,
            loss_mode=args.loss_mode,
            boundary_width=args.boundary_width,
            boundary_weight=args.boundary_weight,
            gce_q=args.gce_q,
            bootstrapping_beta=args.bootstrapping_beta,
        )

        # loss, per_sample_loss = boundary_dice_loss(
        #     outputs, masks,
        #     width=args.boundary_width
        # )
        
        # Track losses by bias status
        for i, (sample_loss, biased) in enumerate(zip(per_sample_loss, is_biased)):
            if biased:
                biased_losses.append(sample_loss.item())
            else:
                clean_losses.append(sample_loss.item())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            metrics.update(
                preds, masks, genders,
                loss=loss.item(),
                per_sample_loss=per_sample_loss.detach()
            )
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    if scheduler is not None:
        scheduler.step()
    
    results = metrics.compute()
    
    # Add bias-specific metrics
    if biased_losses:
        results['biased_sample_loss'] = np.mean(biased_losses)
    if clean_losses:
        results['clean_sample_loss'] = np.mean(clean_losses)
    
    return results


@torch.no_grad()
def evaluate(model, data_loader, device, args, desc='Validation'):
    """Evaluate on clean labels."""
    model.eval()
    metrics = SegmentationMetrics(num_classes=2)
    
    pbar = tqdm(data_loader, desc=desc)
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        genders = batch['gender']
        
        outputs = model(images)
        
        loss, per_sample_loss = combined_loss(
            outputs, masks,
            ce_weight=args.ce_weight,
            dice_weight=args.dice_weight,
            loss_mode=args.loss_mode,
            boundary_width=args.boundary_width,
            boundary_weight=args.boundary_weight,
            gce_q=args.gce_q,
            bootstrapping_beta=args.bootstrapping_beta,
        )

        # loss, per_sample_loss = boundary_dice_loss(
        #     outputs, masks,
        #     width=args.boundary_width
        # )
        
        preds = outputs.argmax(dim=1)
        metrics.update(
            preds, masks, genders,
            loss=loss.item(),
            per_sample_loss=per_sample_loss
        )
    
    return metrics.compute()


def log_metrics(writer, metrics, step, prefix='train'):
    """Log metrics to TensorBoard."""
    for key, value in metrics.items():
        writer.add_scalar(f'{prefix}/{key}', value, step)


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Save checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }, path)


def main():
    args = parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create or load splits
    splits = create_splits(args)
    
    # Get indices
    train_indices, val_indices, test_indices = get_fold_indices(splits, args.fold)
    print(f'\nFold {args.fold}: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}')
    
    # Create dataloaders
    train_loader, val_loader, test_loader, train_dataset = create_dataloaders(
        args, train_indices, val_indices, test_indices
    )
    
    # Print bias configuration
    print('\n' + '='*50)
    if hasattr(train_dataset, 'get_bias_summary'):
        print(train_dataset.get_bias_summary())
    print('='*50 + '\n')
    
    # Create model
    model = create_model(args, num_classes=2).to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Experiment directory
    if args.exp_name is None:
        if args.dataset == 'phc':
            args.exp_name = f"phc_biased_r{int(args.bias_ratio*100)}"
        else:
            gender_name = 'female' if args.biased_gender == 0 else 'male'
            mode_tag = args.bias_mode
            args.exp_name = f"bias_{gender_name}_{mode_tag}_r{int(args.bias_ratio*100)}_e{args.erosion_radius}"
    
    exp_dir = Path(args.output_dir) / 'experiments' / args.exp_name / f'fold_{args.fold}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(exp_dir / 'logs'))
    
    # Save configuration
    config = vars(args).copy()
    config['bias_info'] = train_dataset.bias_info if hasattr(train_dataset, 'bias_info') else {}
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save biased indices for reproducibility (CelebA only)
    if hasattr(train_dataset, 'save_bias_indices'):
        train_dataset.save_bias_indices(str(exp_dir / 'biased_indices.json'))
    
    # Training loop
    best_val_iou = 0.0
    
    if not args.eval_only:
        print('\nStarting training...\n')
        
        for epoch in range(args.epochs):
            print(f'Epoch {epoch+1}/{args.epochs}')
            
            # Train
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, scheduler, device, args
            )
            log_metrics(writer, train_metrics, epoch, prefix='train')
            
            g0, g1 = get_demographic_names(args.dataset)
            print(f"Train - Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['mean_iou']:.4f}")
            print(f"  {g0} IoU: {train_metrics['female_iou_foreground']:.4f}, "
                  f"{g1} IoU: {train_metrics['male_iou_foreground']:.4f}, "
                  f"Gap: {train_metrics['iou_gap']:.4f}")
            if 'biased_sample_loss' in train_metrics:
                print(f"  Biased samples loss: {train_metrics['biased_sample_loss']:.4f}, "
                      f"Clean samples loss: {train_metrics.get('clean_sample_loss', 0):.4f}")
            
            # Validate (on clean labels)
            val_metrics = evaluate(model, val_loader, device, args, desc='Validation')
            log_metrics(writer, val_metrics, epoch, prefix='val')
            
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['mean_iou']:.4f}")
            print(f"  {g0} IoU: {val_metrics['female_iou_foreground']:.4f}, "
                  f"{g1} IoU: {val_metrics['male_iou_foreground']:.4f}, "
                  f"Gap: {val_metrics['iou_gap']:.4f}")
            
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
            
            # Save best
            if val_metrics['mean_iou'] > best_val_iou:
                best_val_iou = val_metrics['mean_iou']
                save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, exp_dir / 'best_model.pt')
                print(f'  New best! IoU: {best_val_iou:.4f}')
            
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, exp_dir / 'latest_model.pt')
            print()
    
    # Final evaluation
    print('\n' + '='*50)
    print('Final Test Evaluation (on clean labels)')
    print('='*50 + '\n')
    
    # Load best model
    if (exp_dir / 'best_model.pt').exists():
        ckpt = torch.load(exp_dir / 'best_model.pt', map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device, args, desc='Test')
    log_metrics(writer, test_metrics, args.epochs, prefix='test')
    
    print('Test Results:')
    print(f"  Mean IoU: {test_metrics['mean_iou']:.4f}")
    print(f"  Foreground IoU: {test_metrics['iou_foreground']:.4f}")
    print(f"  Foreground Dice: {test_metrics['dice_foreground']:.4f}")
    g0, g1 = get_demographic_names(args.dataset)
    print(f"\nBy Group:")
    print(f"  {g0} IoU: {test_metrics['female_iou_foreground']:.4f}, Dice: {test_metrics['female_dice_foreground']:.4f}")
    print(f"  {g1} IoU: {test_metrics['male_iou_foreground']:.4f}, Dice: {test_metrics['male_dice_foreground']:.4f}")
    print(f"\nFairness Gaps:")
    print(f"  IoU Gap: {test_metrics['iou_gap']:.4f}")
    print(f"  Dice Gap: {test_metrics['dice_gap']:.4f}")
    
    # Save test metrics
    with open(exp_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    writer.close()
    print(f'\nResults saved to: {exp_dir}')


if __name__ == '__main__':
    main()
