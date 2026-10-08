#!/usr/bin/env python3
"""
Feature separability analysis across bias levels.

Hypothesis: more label bias → encoder features become more separable by gender
(the model learns gender-specific representations instead of task-relevant ones).

For each experiment checkpoint:
  1. Extract encoder features (GAP of last encoder layer) for test samples
  2. Compute separability metrics:
     - Linear probe accuracy (logistic regression on features → gender)
     - Silhouette score (cluster quality by gender)
     - Maximum Mean Discrepancy between male/female feature distributions
     - Fisher's discriminant ratio
     - AUROC of gender classifier
  3. t-SNE / PCA visualizations colored by gender
  4. Statistical tests (permutation test for significance)

Usage:
    python -m labelbias.evaluation.feature_analysis --experiments debias_domain_invariant_mmd_l0.01_bias_female_r50_e15 --fold 0 --output_dir /path/to/output/experiments/feature_analysis/debias
    python -m labelbias.evaluation.feature_analysis --experiments binary_seg_cv bias_female_r25_e15 bias_female_r50_e15 bias_female_r75_e15 bias_female_r100_e15 --output_dir /path/to/output/experiments/feature_analysis/bias_celeba
    python -m labelbias.evaluation.feature_analysis --experiments phc_biased_r0 phc_biased_r25 phc_biased_r50 phc_biased_r100
    python -m labelbias.evaluation.feature_analysis --experiments debias_domain_invariant_mmd_logit_l0.01_bias_female_r100_e15 bias_female_r100_e15 debias_domain_invariant_mmd_logit_l0.01_bias_female_r50_e15 bias_female_r50_e15
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, silhouette_score
from sklearn.model_selection import cross_val_score
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

import segmentation_models_pytorch as smp

from labelbias.data.celebamask import CelebAMaskHQDataset
from labelbias.data.splits import load_splits, get_fold_indices
from labelbias.data.factory import (
    add_dataset_args, apply_dataset_defaults, create_clean_eval_dataset,
    create_splits, get_demographic_names, DATASET_DEFAULTS, is_phc_experiment,
)


def parse_args():
    p = argparse.ArgumentParser(description='Feature separability analysis')
    add_dataset_args(p)
    p.add_argument('--experiments_dir', type=str,
                   default='/path/to/output/experiments')
    p.add_argument('--experiments', type=str, nargs='+', default=None)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--all_folds', action='store_true',
                   help='Run analysis on all folds (0-4) and aggregate results')
    p.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                   help='Which folds to use in --all_folds mode')
    p.add_argument('--data_root', type=str,
                   default='/path/to/CelebAMask-HQ')
    p.add_argument('--splits_path', type=str,
                   default='configs/splits/cv_splits.json')
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--n_samples', type=int, default=1000,
                   help='Max samples to use (balanced by gender)')
    p.add_argument('--tsne_perplexity', type=float, default=30)
    p.add_argument('--n_permutations', type=int, default=1000,
                   help='Permutation test iterations')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--output_dir', type=str, default=None)
    args = p.parse_args()
    apply_dataset_defaults(args)
    return args


def load_model(fold_dir: Path, device: torch.device):
    """Load SMP segmentation model from checkpoint."""
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

    model_fn = {
        'unet': smp.Unet, 'unetpp': smp.UnetPlusPlus,
        'deeplabv3': smp.DeepLabV3, 'deeplabv3p': smp.DeepLabV3Plus,
        'fpn': smp.FPN, 'pspnet': smp.PSPNet,
    }[model_name]

    model = model_fn(encoder_name=encoder, encoder_weights=None,
                     in_channels=3, classes=2)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)

    # Handle debiased model checkpoints (strip wrapper keys)
    filtered = {}
    for k, v in state.items():
        clean_k = k.replace('seg_model.', '')
        if any(clean_k.startswith(p) for p in
               ['adversary.', 'adv_loss_fn.', 'fairness_loss_fn.', 'domain_loss_fn.']):
            continue
        filtered[clean_k] = v

    try:
        model.load_state_dict(filtered)
    except RuntimeError:
        model.load_state_dict(state)

    return model.to(device).eval()


@torch.no_grad()
def extract_features(model, dataloader, device, n_samples):
    """Extract GAP-pooled encoder features and gender labels."""
    all_feats, all_genders = [], []
    collected = 0

    for batch in tqdm(dataloader, desc='Extracting features'):
        if collected >= n_samples:
            break

        images = batch['image'].to(device)
        genders = batch['gender'].numpy()

        features = model.encoder(images)
        pooled = F.adaptive_avg_pool2d(features[-1], 1).flatten(1)  # (B, C)
        all_feats.append(pooled.cpu().numpy())
        all_genders.append(genders)
        collected += len(images)

    feats = np.concatenate(all_feats, axis=0)[:n_samples]
    genders = np.concatenate(all_genders, axis=0)[:n_samples]
    return feats, genders


def balance_by_gender(feats, genders, max_per_group):
    """Subsample to equal male/female counts."""
    idx_f = np.where(genders == 0)[0]
    idx_m = np.where(genders == 1)[0]
    n = min(len(idx_f), len(idx_m), max_per_group)

    rng = np.random.RandomState(42)
    sel_f = rng.choice(idx_f, n, replace=False)
    sel_m = rng.choice(idx_m, n, replace=False)
    sel = np.concatenate([sel_f, sel_m])
    rng.shuffle(sel)

    return feats[sel], genders[sel]


# ---- Separability metrics ----

def linear_probe(feats, labels):
    """5-fold CV logistic regression accuracy and AUROC."""
    clf = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    acc_scores = cross_val_score(clf, feats, labels, cv=5, scoring='accuracy')
    auc_scores = cross_val_score(clf, feats, labels, cv=5, scoring='roc_auc')
    clf.fit(feats, labels)
    return {
        'probe_acc_mean': acc_scores.mean(),
        'probe_acc_std': acc_scores.std(),
        'probe_auroc_mean': auc_scores.mean(),
        'probe_auroc_std': auc_scores.std(),
    }


def compute_silhouette(feats, labels):
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(silhouette_score(feats, labels, metric='euclidean', sample_size=min(2000, len(feats))))


def compute_mmd(feats, labels, n_kernels=5, base_bw=1.0):
    """Empirical MMD² with multi-scale Gaussian kernel."""
    f0 = feats[labels == 0]
    f1 = feats[labels == 1]

    def rbf(x, y, bw):
        d = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
        return np.exp(-d / (2 * bw ** 2))

    mmd = 0.0
    for i in range(-n_kernels // 2, n_kernels // 2 + 1):
        bw = base_bw * (2 ** i)
        kxx = rbf(f0, f0, bw).mean()
        kyy = rbf(f1, f1, bw).mean()
        kxy = rbf(f0, f1, bw).mean()
        mmd += kxx + kyy - 2 * kxy
    return float(mmd)


def fishers_discriminant_ratio(feats, labels):
    """Multi-dimensional Fisher's discriminant ratio."""
    f0 = feats[labels == 0]
    f1 = feats[labels == 1]
    mu_diff = (f0.mean(axis=0) - f1.mean(axis=0)) ** 2
    var_sum = f0.var(axis=0) + f1.var(axis=0) + 1e-10
    return float((mu_diff / var_sum).mean())


def permutation_test(feats, labels, metric_fn, n_perm=1000):
    """Permutation test: is the observed metric significantly above chance?"""
    observed = metric_fn(feats, labels)
    rng = np.random.RandomState(42)
    count = 0
    for _ in range(n_perm):
        perm_labels = rng.permutation(labels)
        if metric_fn(feats, perm_labels) >= observed:
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return observed, p_value


def centroid_distance(feats, labels):
    """Euclidean distance between group centroids."""
    f0 = feats[labels == 0]
    f1 = feats[labels == 1]
    return float(np.linalg.norm(f0.mean(axis=0) - f1.mean(axis=0)))


def hotelling_t2_test(feats, labels):
    """Hotelling's T² test for multivariate mean difference."""
    f0 = feats[labels == 0]
    f1 = feats[labels == 1]
    n0, n1 = len(f0), len(f1)
    d = feats.shape[1]

    # Use PCA to reduce dimensionality if needed (T² unstable with d > n)
    max_d = min(d, n0 + n1 - 2, 50)
    if d > max_d:
        pca = PCA(n_components=max_d, random_state=42)
        feats_r = pca.fit_transform(feats)
        f0 = feats_r[labels == 0]
        f1 = feats_r[labels == 1]
        d = max_d

    mu0, mu1 = f0.mean(axis=0), f1.mean(axis=0)
    diff = mu0 - mu1

    S0 = np.cov(f0, rowvar=False)
    S1 = np.cov(f1, rowvar=False)
    S_pooled = ((n0 - 1) * S0 + (n1 - 1) * S1) / (n0 + n1 - 2)

    S_inv = np.linalg.pinv(S_pooled)
    t2 = (n0 * n1) / (n0 + n1) * diff @ S_inv @ diff

    # Convert to F-statistic
    f_stat = t2 * (n0 + n1 - d - 1) / (d * (n0 + n1 - 2))
    df1, df2 = d, n0 + n1 - d - 1
    p_value = 1.0 - stats.f.cdf(max(f_stat, 0), df1, max(df2, 1))

    return float(t2), float(f_stat), float(p_value)


def analyze_one_experiment(exp_name, feats, genders):
    """Compute all separability metrics for one experiment."""
    result = {'experiment': exp_name}

    # Linear probe
    result.update(linear_probe(feats, genders))

    # Silhouette
    result['silhouette'] = compute_silhouette(feats, genders)

    # Fisher's discriminant ratio with permutation test
    result['fisher_ratio'], result['fisher_pvalue'] = permutation_test(
        feats, genders, fishers_discriminant_ratio, n_perm=500
    )

    # Centroid distance
    result['centroid_distance'] = centroid_distance(feats, genders)

    # MMD (use subset for speed)
    n_mmd = min(300, len(feats) // 2)
    idx = np.random.RandomState(42).choice(len(feats), 2 * n_mmd, replace=False)
    result['mmd'] = compute_mmd(feats[idx], genders[idx])

    # Hotelling's T²
    t2, f_stat, p_val = hotelling_t2_test(feats, genders)
    result['hotelling_t2'] = t2
    result['hotelling_f'] = f_stat
    result['hotelling_pvalue'] = p_val

    return result


def save_individual_plots(all_features, df, output_dir, tsne_perplexity=30):
    """Save bare individual plots (no title, legend, axis labels) per experiment."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from scipy.stats import gaussian_kde

    indiv_dir = output_dir / 'individual_plots'
    indiv_dir.mkdir(parents=True, exist_ok=True)

    for exp_name, (feats, genders) in all_features.items():
        safe_name = exp_name.replace('/', '_')

        # --- t-SNE ---
        tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42,
                    init='pca', learning_rate='auto')
        emb = tsne.fit_transform(feats)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(emb[genders == 0, 0], emb[genders == 0, 1],
                   c='#E57373', alpha=0.5, s=8)
        ax.scatter(emb[genders == 1, 0], emb[genders == 1, 1],
                   c='#64B5F6', alpha=0.5, s=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.tight_layout(pad=0.1)
        plt.savefig(indiv_dir / f'tsne_{safe_name}.png', dpi=150, bbox_inches='tight')
        plt.savefig(indiv_dir / f'tsne_{safe_name}.pdf', bbox_inches='tight')
        plt.close()

        # --- PCA ---
        pca = PCA(n_components=2, random_state=42)
        emb = pca.fit_transform(feats)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(emb[genders == 0, 0], emb[genders == 0, 1],
                   c='#E57373', alpha=0.5, s=8)
        ax.scatter(emb[genders == 1, 0], emb[genders == 1, 1],
                   c='#64B5F6', alpha=0.5, s=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.tight_layout(pad=0.1)
        plt.savefig(indiv_dir / f'pca_{safe_name}.png', dpi=150, bbox_inches='tight')
        plt.savefig(indiv_dir / f'pca_{safe_name}.pdf', bbox_inches='tight')
        plt.close()

        # --- LDA ---
        lda = LinearDiscriminantAnalysis(n_components=1)
        proj = lda.fit_transform(feats, genders).ravel()

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(proj[genders == 0], bins=40, alpha=0.6, color='#E57373', density=True)
        ax.hist(proj[genders == 1], bins=40, alpha=0.6, color='#64B5F6', density=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.tight_layout(pad=0.1)
        plt.savefig(indiv_dir / f'lda_{safe_name}.png', dpi=150, bbox_inches='tight')
        plt.savefig(indiv_dir / f'lda_{safe_name}.pdf', bbox_inches='tight')
        plt.close()

    print(f'  Individual plots saved to: {indiv_dir}/')


def run_single_fold(args, exp_dirs, fold, device):
    """
    Run feature analysis for a single fold.

    Returns:
        all_features: dict of {exp_name: (feats, genders)}
        all_results: list of metric dicts
    """
    print(f'\n{"#" * 70}')
    print(f'  FOLD {fold}')
    print(f'{"#" * 70}')

    # Auto-detect dataset from first experiment config if not explicitly set
    if args.dataset == 'celebamask' and exp_dirs:
        first_cfg_path = exp_dirs[0] / f'fold_{fold}' / 'config.json'
        if first_cfg_path.exists():
            with open(first_cfg_path) as _f:
                _cfg = json.load(_f)
            if is_phc_experiment(_cfg):
                args.dataset = 'phc'
                apply_dataset_defaults(args)

    # Load shared test set
    splits = create_splits(args)
    _, _, test_indices = get_fold_indices(splits, fold)

    dataset = create_clean_eval_dataset(args, test_indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    print(f'Test set: {len(dataset)} samples\n')

    # Extract and analyze for each experiment
    all_features = {}
    all_results = []

    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        fold_dir = exp_dir / f'fold_{fold}'
        print(f'{"=" * 60}')
        print(f'{exp_name}  (fold {fold})')
        print(f'{"=" * 60}')

        model = load_model(fold_dir, device)
        if model is None:
            print('  No checkpoint, skipping.')
            continue

        feats, genders = extract_features(model, loader, device, args.n_samples)
        feats, genders = balance_by_gender(feats, genders, args.n_samples // 2)
        print(f'  Features: {feats.shape},  Female: {(genders==0).sum()},  Male: {(genders==1).sum()}')

        all_features[exp_name] = (feats, genders)

        result = analyze_one_experiment(exp_name, feats, genders)
        result['fold'] = fold
        all_results.append(result)

        print(f'  Probe acc:  {result["probe_acc_mean"]:.4f} ± {result["probe_acc_std"]:.4f}')
        print(f'  Probe AUC:  {result["probe_auroc_mean"]:.4f} ± {result["probe_auroc_std"]:.4f}')
        print(f'  Silhouette: {result["silhouette"]:.4f}')
        print(f'  Fisher:     {result["fisher_ratio"]:.4f}  (p={result["fisher_pvalue"]:.4f})')
        print(f'  MMD:        {result["mmd"]:.6f}')
        print(f'  Centroid Δ: {result["centroid_distance"]:.4f}')
        print(f'  Hotelling:  T²={result["hotelling_t2"]:.1f}  F={result["hotelling_f"]:.1f}  '
              f'p={result["hotelling_pvalue"]:.2e}')
        print()

        del model
        torch.cuda.empty_cache()

    return all_features, all_results


def make_combined_plots(all_features, df, output_dir, args):
    """Generate the combined grid plots and metric bar charts."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from scipy.stats import gaussian_kde

    n_exp = len(all_features)
    if n_exp == 0:
        return
    exp_names = list(all_features.keys())

    # --- Plot 1: t-SNE grid ---
    fig, axes = plt.subplots(1, n_exp, figsize=(6 * n_exp, 5))
    if n_exp == 1:
        axes = [axes]
    fold_label = f'fold {args.fold}' if not args.all_folds else 'all folds'
    fig.suptitle(f't-SNE of Encoder Features by Gender ({fold_label})', fontsize=14, y=1.02)

    for ax, exp_name in zip(axes, exp_names):
        feats, genders = all_features[exp_name]
        tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity, random_state=42,
                    init='pca', learning_rate='auto')
        emb = tsne.fit_transform(feats)

        ax.scatter(emb[genders == 0, 0], emb[genders == 0, 1],
                   c='#E57373', alpha=0.5, s=8, label='Female')
        ax.scatter(emb[genders == 1, 0], emb[genders == 1, 1],
                   c='#64B5F6', alpha=0.5, s=8, label='Male')

        r = df[df['experiment'] == exp_name].iloc[0]
        ax.set_title(f'{exp_name}\nProbe={r["probe_acc_mean"]:.3f}  AUC={r["probe_auroc_mean"]:.3f}',
                     fontsize=10)
        ax.legend(fontsize=8, markerscale=2)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_dir / 'tsne_grid.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- Plot 2: PCA grid ---
    fig, axes = plt.subplots(1, n_exp, figsize=(6 * n_exp, 5))
    if n_exp == 1:
        axes = [axes]
    fig.suptitle(f'PCA of Encoder Features by Gender ({fold_label})', fontsize=14, y=1.02)

    for ax, exp_name in zip(axes, exp_names):
        feats, genders = all_features[exp_name]
        pca = PCA(n_components=2, random_state=42)
        emb = pca.fit_transform(feats)

        ax.scatter(emb[genders == 0, 0], emb[genders == 0, 1],
                   c='#E57373', alpha=0.5, s=8, label='Female')
        ax.scatter(emb[genders == 1, 0], emb[genders == 1, 1],
                   c='#64B5F6', alpha=0.5, s=8, label='Male')

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title(f'{exp_name}', fontsize=10)
        ax.legend(fontsize=8, markerscale=2)

    plt.tight_layout()
    plt.savefig(output_dir / 'pca_grid.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- Plot 3: Metric comparison bar charts ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Gender Separability Metrics Across Bias Levels', fontsize=14, y=1.01)
    xp = np.arange(n_exp)

    metric_specs = [
        (axes[0, 0], 'probe_acc_mean', 'probe_acc_std', 'Linear Probe Accuracy', 'Higher = more separable'),
        (axes[0, 1], 'probe_auroc_mean', 'probe_auroc_std', 'Linear Probe AUROC', 'Higher = more separable'),
        (axes[0, 2], 'silhouette', None, 'Silhouette Score', 'Higher = better gender clustering'),
        (axes[1, 0], 'fisher_ratio', None, "Fisher's Discriminant Ratio", 'Higher = more separable'),
        (axes[1, 1], 'mmd', None, 'MMD (feature distributions)', 'Higher = more different'),
        (axes[1, 2], 'centroid_distance', None, 'Centroid Distance', 'Higher = more separated'),
    ]

    for ax, col, err_col, title, subtitle in metric_specs:
        vals = df[col].values
        errs = df[err_col].values if err_col and err_col in df.columns else None
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, n_exp))
        bars = ax.bar(xp, vals, color=colors, edgecolor='black', lw=0.5,
                      yerr=errs, capsize=4)
        ax.set_xticks(xp)
        ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
        ax.set_title(f'{title}\n({subtitle})', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        if 'probe' in col or 'auroc' in col:
            ax.axhline(0.5, color='gray', ls='--', lw=1, label='Chance')
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'separability_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- Plot 4: Feature distribution overlap (1D projection) ---
    fig, axes = plt.subplots(1, n_exp, figsize=(5 * n_exp, 4))
    if n_exp == 1:
        axes = [axes]
    fig.suptitle('LDA Projection: Feature Distribution by Gender', fontsize=14, y=1.02)

    for ax, exp_name in zip(axes, exp_names):
        feats, genders = all_features[exp_name]
        lda = LinearDiscriminantAnalysis(n_components=1)
        proj = lda.fit_transform(feats, genders).ravel()

        ax.hist(proj[genders == 0], bins=40, alpha=0.6, color='#E57373',
                label='Female', density=True)
        ax.hist(proj[genders == 1], bins=40, alpha=0.6, color='#64B5F6',
                label='Male', density=True)

        # Overlap coefficient
        x_range = np.linspace(proj.min(), proj.max(), 200)
        kde0 = gaussian_kde(proj[genders == 0])(x_range)
        kde1 = gaussian_kde(proj[genders == 1])(x_range)
        overlap = np.trapz(np.minimum(kde0, kde1), x_range)

        ax.set_title(f'{exp_name}\nOverlap coeff: {overlap:.3f}', fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlabel('LDA projection')

    plt.tight_layout()
    plt.savefig(output_dir / 'lda_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()


def print_summary_table(df):
    """Print a formatted summary table of separability metrics."""
    print('\n' + '=' * 100)
    print('FEATURE SEPARABILITY SUMMARY')
    print('  Higher probe/AUC/Fisher/MMD/centroid = more gender-separable = more biased features')
    print('  Silhouette: -1 to 1, higher = better gender clustering')
    print('=' * 100)
    header = (f'{"Experiment":<35} {"Probe":>7} {"AUC":>7} {"Silh":>7} '
              f'{"Fisher":>8} {"MMD":>10} {"Ctr-Δ":>8} {"T² p-val":>10}')
    print(header)
    print('-' * len(header))
    for _, r in df.iterrows():
        sig = '***' if r['hotelling_pvalue'] < 0.001 else '**' if r['hotelling_pvalue'] < 0.01 else '*' if r['hotelling_pvalue'] < 0.05 else 'n.s.'
        print(f'{r["experiment"]:<35} {r["probe_acc_mean"]:>6.3f} {r["probe_auroc_mean"]:>6.3f} '
              f'{r["silhouette"]:>6.3f} {r["fisher_ratio"]:>7.4f} {r["mmd"]:>9.5f} '
              f'{r["centroid_distance"]:>7.3f} {r["hotelling_pvalue"]:>8.2e} {sig:>4}')


def main():
    args = parse_args()
    exp_root = Path(args.experiments_dir)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.all_folds:
        # ---- Full-run mode: iterate over all folds ----
        folds = args.folds

        if args.experiments:
            exp_dirs = [exp_root / n for n in args.experiments]
        else:
            # Find experiments that have at least fold_0
            exp_dirs = sorted([d for d in exp_root.iterdir()
                               if d.is_dir() and (d / 'fold_0' / 'best_model.pt').exists()])

        print(f'Experiments:  {[d.name for d in exp_dirs]}')
        print(f'Folds:        {folds}')
        print(f'Samples/fold: {args.n_samples},  Device: {device}')

        output_dir = Path(args.output_dir) if args.output_dir else exp_root / 'feature_analysis'
        output_dir.mkdir(parents=True, exist_ok=True)

        all_fold_results = []  # list of dicts, one per (experiment, fold)
        last_fold_features = {}  # keep features from last fold for plots

        for fold in folds:
            # Filter to experiments that have this fold
            fold_exp_dirs = [d for d in exp_dirs
                            if (d / f'fold_{fold}' / 'best_model.pt').exists()]
            if not fold_exp_dirs:
                print(f'\nFold {fold}: no experiments with checkpoint, skipping.')
                continue

            fold_features, fold_results = run_single_fold(
                args, fold_exp_dirs, fold, device)
            all_fold_results.extend(fold_results)

            # Save per-fold CSV
            if fold_results:
                fold_df = pd.DataFrame(fold_results)
                fold_df.to_csv(output_dir / f'separability_metrics_fold{fold}.csv',
                               index=False)

                # Save per-fold individual plots
                fold_plot_dir = output_dir / f'fold_{fold}'
                fold_plot_dir.mkdir(parents=True, exist_ok=True)
                save_individual_plots(fold_features, fold_df, fold_plot_dir,
                                     tsne_perplexity=args.tsne_perplexity)

            last_fold_features = fold_features

        if not all_fold_results:
            print('No results across any fold!')
            return

        # Save all-folds raw data
        all_df = pd.DataFrame(all_fold_results)
        all_df.to_csv(output_dir / 'all_folds_raw.csv', index=False)

        # Aggregate: mean ± std across folds per experiment
        metric_cols = [c for c in all_df.columns
                       if c not in ('experiment', 'fold')]
        agg = all_df.groupby('experiment')[metric_cols].agg(['mean', 'std']).reset_index()
        agg.columns = ['_'.join(c).rstrip('_') for c in agg.columns]
        agg.to_csv(output_dir / 'summary.csv', index=False)

        # Build a clean "mean" dataframe for printing and plotting
        mean_df = all_df.groupby('experiment')[metric_cols].mean().reset_index()

        print_summary_table(mean_df)

        # Use last fold's features for combined plots
        if last_fold_features:
            make_combined_plots(last_fold_features, mean_df, output_dir, args)
            save_individual_plots(last_fold_features, mean_df, output_dir,
                                  tsne_perplexity=args.tsne_perplexity)

        print(f'\nAll outputs saved to: {output_dir}/')
        print(f'  all_folds_raw.csv                  — per-fold per-experiment data')
        print(f'  summary.csv                        — mean/std across folds')
        for fold in folds:
            print(f'  separability_metrics_fold{fold}.csv    — fold {fold} results')
            print(f'  fold_{fold}/individual_plots/        — bare plots for fold {fold}')
        print(f'  individual_plots/                  — bare plots (last fold)')
        print(f'  tsne_grid.png / pca_grid.png       — combined grid plots')
        print(f'  separability_metrics.png           — metric bar charts')
        print(f'  lda_distributions.png              — LDA projections')

    else:
        # ---- Single fold mode (original behavior) ----
        if args.experiments:
            exp_dirs = [exp_root / n for n in args.experiments]
        else:
            exp_dirs = sorted([d for d in exp_root.iterdir()
                               if d.is_dir() and (d / f'fold_{args.fold}' / 'best_model.pt').exists()])

        print(f'Experiments: {[d.name for d in exp_dirs]}')
        print(f'Fold: {args.fold},  Samples: {args.n_samples},  Device: {device}')

        all_features, all_results = run_single_fold(
            args, exp_dirs, args.fold, device)

        if not all_results:
            print('No results!')
            return

        df = pd.DataFrame(all_results)
        output_dir = Path(args.output_dir) if args.output_dir else exp_root / 'feature_analysis'
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / 'separability_metrics.csv', index=False)

        print_summary_table(df)

        # Combined plots
        make_combined_plots(all_features, df, output_dir, args)

        # Individual bare plots
        save_individual_plots(all_features, df, output_dir,
                              tsne_perplexity=args.tsne_perplexity)

        print(f'\nAll outputs saved to: {output_dir}/')
        print(f'  separability_metrics.csv')
        print(f'  tsne_grid.png')
        print(f'  pca_grid.png')
        print(f'  separability_metrics.png')
        print(f'  lda_distributions.png')
        print(f'  individual_plots/       — bare per-experiment plots')


if __name__ == '__main__':
    main()
