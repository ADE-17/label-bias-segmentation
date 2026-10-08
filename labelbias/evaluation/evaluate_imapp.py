#!/usr/bin/env python3
"""
Elegant Unified Evaluation Pipeline.

This script evaluates trained experiments in a single pass. It computes:
1. Standard Segmentation Metrics (IoU, Dice) against True and Observed test labels.
2. Label Error Rate (LER) and Signed Directional Rate (SDR) by auditing the
   observed training labels using Confident Learning (cross-fitted predictions).

   python -m labelbias.evaluation.evaluate_imapp \
    --experiments_dir /path/to/output/experiments \
    --experiments imapp_bias_50 imapp_bias_baseline imapp_bias_nobias \
    --output_dir /path/to/output/evaluations_imapp \
    --data_root configs/imapp_processed.csv \
    --dataset imapp \
    --img_size 256


Usage Examples:
    # Evaluate a single experiment
    python -m labelbias.evaluation.evaluate --experiments gce_baseline_r50_e15

    # Evaluate multiple experiments and save to custom directory
    python -m labelbias.evaluation.evaluate \\
        --experiments bootstrapped_baseline_r50_e15 gce_baseline_r50_e15 \\
        --output_dir /path/to/output/evaluations

    # Evaluate all experiments matching a pattern (if using shell expansion)
    python -m labelbias.evaluation.evaluate --experiments bias_female_*

    python -m labelbias.evaluation.evaluate \\
        --experiments /path/to/output/experiments/phc_biased_r0 \\
            /path/to/output/experiments/phc_biased_r25 \\
            /path/to/output/experiments/phc_biased_r50 \\
            /path/to/output/experiments/phc_biased_r100 \\
        --output_dir /path/to/output/evaluations_phc_old
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from labelbias.data.factory import (
    add_dataset_args, apply_dataset_defaults, create_splits,
    create_clean_eval_dataset, create_biased_eval_dataset,
    get_demographic_names
)
from labelbias.data.splits import get_fold_indices
from labelbias.audit.confident_learning_imapp import ConfidentLearningAnalyzer, create_model

def load_fold_model(fold_dir: Path, device: torch.device):
    """Load a trained model from a fold directory."""
    ckpt_path = fold_dir / 'best_model.pt'
    if not ckpt_path.exists():
        return None

    config_path = fold_dir / 'config.json'
    if not config_path.exists():
        config_path = fold_dir / 'args.json'

    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        model_name = cfg.get('model', 'unet')
        encoder = cfg.get('encoder', 'resnet34')
    else:
        model_name, encoder = 'unet', 'resnet34'

    model = create_model(model_name, encoder, num_classes=2)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    state = ckpt.get('model_state_dict', ckpt)
    # Strip debiasing wrapper keys if present
    filtered = {
        k.replace('seg_model.', ''): v
        for k, v in state.items()
        if not any(k.startswith(p) for p in (
            'adversary.', 'adv_loss_fn.', 'fairness_loss_fn.',
            'domain_loss_fn.', 'films.', 'bottleneck_film.',
        ))
    }
    try:
        model.load_state_dict(filtered)
    except RuntimeError:
        model.load_state_dict(state)

    return model.to(device).eval()

def get_experiment_config(fold_dir: Path) -> Dict:
    for name in ('config.json', 'args.json'):
        p = fold_dir / name
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return {}

def compute_raw_metrics(model, dataloader, device):
    """Compute IoU and Dice against whatever labels are in the dataloader."""
    tp = [0, 0]
    fp = [0, 0]
    fn = [0, 0]
    tn = [0, 0]
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['mask'].cpu().numpy()
            genders = batch['gender'].cpu().numpy()
            
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            for i in range(len(images)):
                p = preds[i]
                l = labels[i]
                g = genders[i]
                
                tp_cur = ((l == 1) & (p == 1)).sum()
                fp_cur = ((l == 0) & (p == 1)).sum()
                fn_cur = ((l == 1) & (p == 0)).sum()
                tn_cur = ((l == 0) & (p == 0)).sum()
                
                if g in [0, 1]:
                    tp[g] += tp_cur; fp[g] += fp_cur; fn[g] += fn_cur; tn[g] += tn_cur
                    
    def calc_metrics(t_p, f_p, f_n):
        iou = t_p / max(t_p + f_p + f_n, 1e-8)
        dice = 2 * t_p / max(2 * t_p + f_p + f_n, 1e-8)
        return float(iou), float(dice)
        
    tp_clean = tp[0]
    fp_clean = fp[0]
    fn_clean = fn[0]
    
    tp_biased = tp[1]
    fp_biased = fp[1]
    fn_biased = fn[1]
    
    iou_clean, dice_clean = calc_metrics(tp_clean, fp_clean, fn_clean)
    iou_biased, dice_biased = calc_metrics(tp_biased, fp_biased, fn_biased)
    iou_all, dice_all = calc_metrics(sum(tp), sum(fp), sum(fn))
    
    return {
        'iou': iou_all, 'dice': dice_all,
        'iou_clean': iou_clean, 'dice_clean': dice_clean,
        'iou_biased': iou_biased, 'dice_biased': dice_biased,
        'iou_gap': abs(iou_clean - iou_biased),
        'dice_gap': abs(dice_clean - dice_biased)
    }

def main():
    p = argparse.ArgumentParser(description="Unified Evaluation for Segmentation & Label Bias")
    add_dataset_args(p)
    p.add_argument('--experiments_dir', type=str, default='/path/to/output/experiments')
    p.add_argument('--experiments', type=str, nargs='+', required=True)
    p.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--output_dir', type=str, default='/path/to/output/evaluations')
    p.add_argument('--data_root', type=str, default='/path/to/CelebAMask-HQ')
    p.add_argument('--splits_path', type=str, default='configs/splits/cv_splits.json')
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--bias_threshold', type=float, default=0.02)
    args = p.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    exp_root = Path(args.experiments_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for exp_arg in args.experiments:
        exp_arg_path = Path(exp_arg)
        if exp_arg_path.is_absolute():
            exp_dir = exp_arg_path
            exp_name = exp_arg_path.name
        else:
            exp_dir = exp_root / exp_arg
            exp_name = exp_arg
            
        print(f"\\n{'='*80}")
        print(f"Evaluating Experiment: {exp_name} (Path: {exp_dir})")
        print(f"{'='*80}")
        
        if not exp_dir.exists():
            print(f"Experiment directory not found: {exp_dir}")
            continue
            
        try:
            test_metrics_true = []
            test_metrics_obs = []
            
            # Accumulators for cross-fitted validation set predictions (for CL audit)
            val_all_probs = []
            val_all_labels = []
            val_all_genders = []
            
            for fold in args.folds:
                fold_dir = exp_dir / f'fold_{fold}'
                print(f"\\n--- Fold {fold} ---")
                
                model = load_fold_model(fold_dir, device)
                if model is None:
                    print(f"  Skipped (no best_model.pt)")
                    continue
                    
                cfg = get_experiment_config(fold_dir)
                args.dataset = 'imapp'
                apply_dataset_defaults(args)
                
                # Extract bias config
                ds = args.dataset
                bias_cfg = {}
                if ds == 'phc':
                    br = cfg.get('bias_ratio', 1.0)
                    bias_cfg = {'dataset': 'phc', 'is_biased': (br > 0), 'bias_ratio': br, 'bias_seed': cfg.get('bias_seed', 42)}
                else:
                    bias_cfg = {
                        'bias_ratio': cfg.get('bias_ratio', 0),
                        'erosion_radius': cfg.get('erosion_radius', 5),
                        'biased_gender': cfg.get('biased_gender', 0),
                        'bias_seed': cfg.get('bias_seed', 42),
                        'bias_mode': cfg.get('bias_mode', 'erosion')
                    }
                
                for k, v in bias_cfg.items():
                    if hasattr(args, k): setattr(args, k, v)
                    
                splits = create_splits(args)
                train_idx, val_idx, test_idx = get_fold_indices(splits, fold)
                
                # --- 1. TEST SET EVALUATION ---
                if test_idx and len(test_idx) > 0:
                    print(f"  Evaluating Test Set ({len(test_idx)} samples)...")
                    
                    # True Labels
                    clean_test_ds = create_clean_eval_dataset(args, test_idx)
                    clean_loader = DataLoader(clean_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                    true_mets = compute_raw_metrics(model, clean_loader, device)
                    test_metrics_true.append(true_mets)
                    
                    # Observed (Biased) Labels
                    if bias_cfg.get('bias_ratio', 0) > 0 or bias_cfg.get('is_biased', False):
                        obs_test_ds = create_biased_eval_dataset(args, test_idx, bias_cfg)
                        obs_loader = DataLoader(obs_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                        obs_mets = compute_raw_metrics(model, obs_loader, device)
                        test_metrics_obs.append(obs_mets)
                else:
                    print("  No test set available. Skipping Test Evaluation.")
                    
                # --- 2. VALIDATION SET INFERENCE (FOR CL AUDIT) ---
                print(f"  Computing Cross-Fitted Val Predictions for CL ({len(val_idx)} samples)...")
                biased_val_ds = create_biased_eval_dataset(args, val_idx, bias_cfg)
                val_loader = DataLoader(biased_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                
                analyzer = ConfidentLearningAnalyzer(model, val_loader, device, num_classes=2)
                probs, labels, genders, _ = analyzer.compute_predictions()
                val_all_probs.extend(probs)
                val_all_labels.extend(labels)
                val_all_genders.extend(genders)
                
            # ==========================================
            # AGGREGATE METRICS FOR THE EXPERIMENT
            # ==========================================
            if not val_all_probs:
                print("No validation predictions to audit.")
                continue
                
            print(f"\\nAggregating Results for {exp_name}...")
            
            row = {'experiment': exp_name}
            
            # Test Metrics Averaging
            if test_metrics_true:
                for k in test_metrics_true[0].keys():
                    row[f'test_true_{k}'] = np.mean([m[k] for m in test_metrics_true])
            if test_metrics_obs:
                for k in test_metrics_obs[0].keys():
                    row[f'test_obs_{k}'] = np.mean([m[k] for m in test_metrics_obs])
                    
            # Run CL Audit on all cross-fitted predictions together
            print(f"Running Confident Learning Audit on {len(val_all_probs)} training samples...")
            
            # We temporarily mock the dataloader in Analyzer since we already have the predictions
            analyzer = ConfidentLearningAnalyzer(None, None, device, num_classes=2, bias_threshold=args.bias_threshold)
            
            thresholds = analyzer.compute_confidence_thresholds(val_all_probs, val_all_labels)
            norm_metrics = analyzer.compute_sample_normalized_metrics(val_all_probs, val_all_labels, thresholds, val_all_genders)
            
            row['cl_ler_clean'] = norm_metrics['ler_clean']
            row['cl_ler_biased'] = norm_metrics['ler_biased']
            row['cl_ler_gap'] = abs(norm_metrics['ler_clean'] - norm_metrics['ler_biased'])
            row['cl_rr'] = (norm_metrics['ler_clean'] + 1e-8) / (norm_metrics['ler_biased'] + 1e-8)
            
            row['cl_sdr_clean'] = norm_metrics['sdr_clean']
            row['cl_sdr_biased'] = norm_metrics['sdr_biased']
            row['cl_sdr_gap'] = norm_metrics['sdr_clean'] - norm_metrics['sdr_biased']
            
            for k, v in norm_metrics['counts'].items():
                row[f'cl_{k}'] = v
                
            all_results.append(row)
            
            # Save individual experiment CSV
            exp_df = pd.DataFrame([row])
            exp_csv_path = out_root / f"{exp_name}_eval.csv"
            exp_df.to_csv(exp_csv_path, index=False)
            print(f"Saved {exp_name} results to {exp_csv_path}")

        except Exception as e:
            print(f"\\n[ERROR] Failed to evaluate experiment '{exp_name}'!")
            print(f"Error details: {str(e)}")
            print("Continuing to the next experiment...\\n")
            continue

    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = out_root / "all_experiments_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\\nSaved aggregated summary to {summary_path}")

if __name__ == '__main__':
    main()
