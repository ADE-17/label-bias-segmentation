#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import pandas as pd
import segmentation_models_pytorch as smp
import umap
from scipy import stats

from dataloader_imapp import IMAPPBiasedDataset


'''
python feature_analysis_imapp.py \
  --csv_path configs/imapp_processed.csv \
  --exp_dir /path/to/output/experiments/imapp_bias_baseline/fold_0 \
  --fold 0 \
  --output_dir /path/to/output/imapp_feature_analysis

python feature_analysis_imapp.py \
  --csv_path configs/imapp_processed.csv \
  --exp_dir /path/to/output/experiments/imapp_biased_r100/fold_0 \
  --fold 0 \
  --output_dir /path/to/output/imapp_feature_analysis

'''

def parse_args():
    parser = argparse.ArgumentParser(description='Feature separability analysis for IMAPP (Clean vs Biased).')
    parser.add_argument('--csv_path', type=str, default='configs/imapp_processed.csv')
    parser.add_argument('--experiments_dir', type=str, default='/path/to/output/experiments')
    parser.add_argument('--experiments', type=str, nargs='+', required=True, help='List of experiment names')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='/path/to/output/evaluations_imapp/feature_analysis')
    return parser.parse_args()

@torch.no_grad()
def extract_features(model, dataloader, device, n_samples):
    all_feats, all_genders = [], []
    collected = 0

    for batch in tqdm(dataloader, desc='Extracting features'):
        if collected >= n_samples:
            break

        images = batch['image'].to(device)
        raw_genders = batch['gender'].numpy()
        # Map 0, 1 -> 0 (Clean) and 2, 3 -> 1 (Biased)
        genders = np.where(np.isin(raw_genders, [0, 1]), 0, 1)

        features = model.encoder(images)
        pooled = F.adaptive_avg_pool2d(features[-1], 1).flatten(1)  # (B, C)
        all_feats.append(pooled.cpu().numpy())
        all_genders.append(genders)
        collected += len(images)

    feats = np.concatenate(all_feats, axis=0)[:n_samples]
    genders = np.concatenate(all_genders, axis=0)[:n_samples]
    return feats, genders

def linear_probe(feats, labels):
    clf = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    acc_scores = cross_val_score(clf, feats, labels, cv=5, scoring='accuracy')
    return {
        'probe_acc_mean': acc_scores.mean(),
        'probe_acc_std': acc_scores.std(),
    }

def compute_advanced_metrics(feats, labels):
    feats_0 = feats[labels == 0]
    feats_1 = feats[labels == 1]
    
    if len(feats_0) == 0 or len(feats_1) == 0:
        return {'centroid_distance': 0.0, 'fisher_ratio': 0.0, 'fisher_pvalue': 1.0}
        
    mu_0 = feats_0.mean(axis=0)
    mu_1 = feats_1.mean(axis=0)
    
    dist = np.linalg.norm(mu_1 - mu_0)
    
    w = mu_1 - mu_0
    w_norm = w / (np.linalg.norm(w) + 1e-8)
    
    proj_0 = feats_0 @ w_norm
    proj_1 = feats_1 @ w_norm
    
    var_0 = np.var(proj_0, ddof=1)
    var_1 = np.var(proj_1, ddof=1)
    
    fisher_ratio = ((proj_1.mean() - proj_0.mean()) ** 2) / (var_0 + var_1 + 1e-8)
    t_stat, p_val = stats.ttest_ind(proj_0, proj_1, equal_var=False)
    
    return {
        'centroid_distance': float(dist),
        'fisher_ratio': float(fisher_ratio),
        'fisher_pvalue': float(p_val)
    }

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    master_ds = IMAPPBiasedDataset(csv_path=args.csv_path, img_size=(args.img_size, args.img_size), bias_ratio=0.0)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(range(len(master_ds))))
    _, val_idx = splits[args.fold]
    
    val_ds = Subset(master_ds, val_idx)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    all_embeddings = []

    for exp_name in args.experiments:
        print(f"\n{'='*60}\nAnalyzing Features for: {exp_name}\n{'='*60}")
        exp_dir = Path(args.experiments_dir) / exp_name / f'fold_{args.fold}'
        ckpt_path = exp_dir / 'best_model.pt'
        
        if not ckpt_path.exists():
            print(f"  [ERROR] Checkpoint not found at {ckpt_path}. Skipping.")
            continue

        model = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=2)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        model.to(device).eval()
        
        feats, genders = extract_features(model, loader, device, args.n_samples)
        
        res = {'experiment': exp_name}
        res.update(linear_probe(feats, genders))
        res.update(compute_advanced_metrics(feats, genders))
        
        if len(np.unique(genders)) > 1:
            res['silhouette'] = silhouette_score(feats, genders, metric='euclidean', sample_size=min(1000, len(feats)))
        else:
            res['silhouette'] = 0.0
            
        print(f"Probe Acc: {res['probe_acc_mean']:.3f} +- {res['probe_acc_std']:.3f}")
        print(f"Silhouette Score: {res['silhouette']:.4f}")
        print(f"Centroid Dist: {res['centroid_distance']:.3f} | Fisher Ratio: {res['fisher_ratio']:.3f} (p={res['fisher_pvalue']:.2e})")
        
        all_results.append(res)
        
        # Generate Embeddings
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
        emb_tsne = tsne.fit_transform(feats)
        
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        emb_umap = reducer.fit_transform(feats)
        
        all_embeddings.append({
            'exp_name': exp_name,
            'tsne': emb_tsne,
            'umap': emb_umap,
            'genders': genders,
            'acc': res['probe_acc_mean']
        })

    # Grid Plotting
    if all_embeddings:
        n_exp = len(all_embeddings)
        fig, axes = plt.subplots(nrows=n_exp, ncols=2, figsize=(12, 5 * n_exp))
        
        if n_exp == 1:
            axes = np.expand_dims(axes, axis=0)
            
        colors = ['#1f77b4', '#d62728']
        labels = ['Clean (Tones 0, 1)', 'Biased (Tones 2, 3)']
        
        for i, emb_dict in enumerate(all_embeddings):
            genders = emb_dict['genders']
            exp_name = emb_dict['exp_name']
            acc = emb_dict['acc']
            
            # t-SNE
            ax_tsne = axes[i, 0]
            for c in [0, 1]:
                mask = (genders == c)
                ax_tsne.scatter(emb_dict['tsne'][mask, 0], emb_dict['tsne'][mask, 1], 
                                c=colors[c], label=labels[c], alpha=0.7, s=15)
            ax_tsne.set_title(f"t-SNE: {exp_name}\nProbe Acc: {acc:.2f}")
            ax_tsne.legend()
            
            # UMAP
            ax_umap = axes[i, 1]
            for c in [0, 1]:
                mask = (genders == c)
                ax_umap.scatter(emb_dict['umap'][mask, 0], emb_dict['umap'][mask, 1], 
                                c=colors[c], label=labels[c], alpha=0.7, s=15)
            ax_umap.set_title(f"UMAP: {exp_name}\nProbe Acc: {acc:.2f}")
            ax_umap.legend()
            
        plt.tight_layout()
        plt.savefig(out_root / 'separability_grid.png', dpi=150)
        plt.close()

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(out_root / 'separability_summary.csv', index=False)
        print(f"\nSaved aggregated summary to {out_root / 'separability_summary.csv'}")
        print(f"Saved visualization grid to {out_root / 'separability_grid.png'}")

if __name__ == '__main__':
    main()
