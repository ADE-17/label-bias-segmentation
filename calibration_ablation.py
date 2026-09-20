import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

from confident_learning_renewed import create_model, ConfidentLearningAnalyzer
from dataloader import CelebAMaskHQBiasedDataset
from utils.splits import get_fold_indices
from torch.utils.data import DataLoader, Subset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/path/to/output/experiments/bias_female_r50_e15/fold_1/best_model.pt')
    parser.add_argument('--dataset', type=str, default='celeba')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--bias_ratio', type=float, default=0.5)
    return parser.parse_args()

@torch.no_grad()
def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading model...")
    model = create_model('unet', 'resnet34', num_classes=2)
    checkpoint = torch.load(args.model_path, map_location=device)
    state = checkpoint.get('model_state_dict', checkpoint)
    filtered = {}
    for k, v in state.items():
        clean_k = k.replace('seg_model.', '')
        filtered[clean_k] = v
    model.load_state_dict(filtered, strict=False)
    model = model.to(device)
    model.eval()
    
    print("Loading dataloader (fold 1 test)...")
    splits_path = 'configs/splits/cv_splits.json'
    from utils.splits import load_splits
    splits_data = load_splits(splits_path)
    _, _, test_indices = get_fold_indices(splits_data, 1)
    
    full_dataset = CelebAMaskHQBiasedDataset(
        root_dir='/path/to/CelebAMask-HQ',
        img_size=(args.img_size, args.img_size),
        bias_ratio=args.bias_ratio,
        bias_seed=42,
        biased_gender=0
    )
    test_dataset = Subset(full_dataset, test_indices)
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    
    print("Computing raw logits...")
    all_logits = []
    all_labels = []
    all_genders = []
    
    for batch in tqdm(dataloader):
        images = batch['image'].to(device)
        labels = batch['mask'].cpu().numpy()
        genders = batch['gender'].cpu().numpy()
        
        logits = model(images)
        all_logits.append(logits.cpu().numpy())
        all_labels.extend(labels)
        all_genders.extend(genders)
        
    all_labels = np.array(all_labels)
    all_genders = np.array(all_genders)
    
    temperatures = [0.1, 0.5, 0.8, 1.0, 1.5, 2.0, 5.0, 10.0]
    
    results = []
    analyzer = ConfidentLearningAnalyzer(model=None, dataloader=None, device=device, num_classes=2)
    
    print("\nStarting Temperature Scaling Ablation...")
    print("-" * 110)
    print(f"{'Temp':<6} | {'Thresh(BG/FG)':<15} | {'SDR (F/M)':<16} | {'SDR Gap':<9} | {'LER Gap':<9} | {'Excess/Deficit (Total)':<25}")
    print("-" * 110)
    
    for T in temperatures:
        all_probs = []
        for batch_logits in all_logits:
            # Apply temperature scaling
            scaled_logits = batch_logits / T
            # Convert to probabilities
            probs_tensor = F.softmax(torch.tensor(scaled_logits), dim=1)
            # Convert to (H, W, C) for Analyzer
            for i in range(len(probs_tensor)):
                all_probs.append(np.transpose(probs_tensor[i].numpy(), (1, 2, 0)))
                
        # 1. Compute thresholds
        thresholds = analyzer.compute_confidence_thresholds(all_probs, all_labels)
        
        # 2. Compute sample normalized metrics (SDR, LER)
        norm_metrics = analyzer.compute_sample_normalized_metrics(all_probs, all_labels, thresholds, all_genders)
        
        sdr_f = norm_metrics['sdr_female'] * 100
        sdr_m = norm_metrics['sdr_male'] * 100
        sdr_gap = (norm_metrics['sdr_female'] - norm_metrics['sdr_male']) * 100
        
        ler_gap = abs(norm_metrics['ler_female'] - norm_metrics['ler_male']) * 100
        
        counts = norm_metrics['counts']
        total_exc = counts['total_excess']
        total_def = counts['total_deficit']
        
        res_row = {
            'T': T,
            'thresh_bg': thresholds[0],
            'thresh_fg': thresholds[1],
            'sdr_f': sdr_f,
            'sdr_m': sdr_m,
            'sdr_gap': sdr_gap,
            'ler_gap': ler_gap,
            'total_excess': total_exc,
            'total_deficit': total_def
        }
        results.append(res_row)
        
        print(f"{T:<6.1f} | {thresholds[0]:.2f} / {thresholds[1]:.2f}     | {sdr_f:>6.1f} / {sdr_m:>6.1f} | {sdr_gap:>7.1f}   | {ler_gap:>7.2f}   | {total_exc:,} / {total_def:,}")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv('calibration_ablation_results.csv', index=False)
    print("-" * 110)
    print("\nResults saved to calibration_ablation_results.csv")

if __name__ == "__main__":
    main()
