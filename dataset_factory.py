"""
Dataset registry / factory for switching between datasets via a single flag.

Supported datasets:
  - celebamask: CelebAMask-HQ face segmentation (foreground vs background)
  - phc:        PHC cell segmentation with synthetic demographic shortcut

Usage in scripts:
    from dataset_factory import add_dataset_args, create_datasets, create_splits

    add_dataset_args(parser)   # adds --dataset, --data_root, etc.
    args = parser.parse_args()

    splits = create_splits(args)
    train_ds, val_ds, test_ds = create_datasets(args, train_idx, val_idx, test_idx)
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from utils.splits import create_kfold_splits, save_splits, load_splits, get_fold_indices


DATASET_DEFAULTS = {
    'celebamask': {
        'data_root': '/path/to/CelebAMask-HQ',
        'img_size': 256,
        'splits_path': 'configs/splits/cv_splits.json',
    },
    'phc': {
        'data_root': '/path/to/phc_data',
        'img_size': 128,
        'splits_path': 'configs/splits/phc_splits.json',
    },
    'imapp': {
        'data_root': 'configs/imapp_processed.csv',
        'img_size': 256,
        'splits_path': 'configs/splits/imapp_splits.json',
    }
}

AVAILABLE_DATASETS = list(DATASET_DEFAULTS.keys())


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    """Add --dataset flag and dataset-agnostic data arguments to a parser."""
    parser.add_argument('--dataset', type=str, default='celebamask',
                        choices=AVAILABLE_DATASETS,
                        help='Which dataset to use')
    parser.add_argument('--tint_strength', type=float, default=0.25,
                        help='Strength of the color tint for the PHC dataset (0.0 to 1.0)')


def apply_dataset_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in data_root / img_size / splits_path defaults from dataset name
    only when the user hasn't overridden them on the command line."""
    defaults = DATASET_DEFAULTS.get(args.dataset, {})
    for key, default_val in defaults.items():
        current = getattr(args, key, None)
        # If the value is still the celebamask default (the argparse default)
        # and the user chose a different dataset, override it.
        celebamask_default = DATASET_DEFAULTS['celebamask'].get(key)
        if current is None or (args.dataset != 'celebamask' and current == celebamask_default):
            setattr(args, key, default_val)
    return args


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def create_splits(args) -> Dict:
    """Create or load K-fold splits for the selected dataset."""
    splits_path = Path(args.splits_path)
    if splits_path.exists():
        return load_splits(splits_path)

    if args.dataset == 'celebamask':
        from dataloader import CelebAMaskHQDataset
        tmp = CelebAMaskHQDataset(
            root_dir=args.data_root,
            img_size=(args.img_size, args.img_size),
            binary_segmentation=True,
        )
        stratify = tmp.get_all_genders()
        n = len(tmp)
        del tmp

    elif args.dataset == 'phc':
        from dataloader_phc import PHCDataset
        tmp = PHCDataset(root_dir=args.data_root, img_size=(args.img_size, args.img_size),
                         tint_strength=getattr(args, 'tint_strength', 0.25))
        stratify = tmp.get_all_genders()
        n = len(tmp)
        del tmp

    elif args.dataset == 'imapp':
        from dataloader_imapp import IMAPPBiasedDataset
        tmp = IMAPPBiasedDataset(csv_path=args.data_root, img_size=(args.img_size, args.img_size), bias_ratio=0.0)
        stratify = np.array(tmp.df['gender'].tolist())
        n = len(tmp)
        del tmp

    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    splits = create_kfold_splits(
        n_samples=n,
        n_folds=getattr(args, 'n_folds', 5),
        stratify_labels=stratify,
        random_seed=getattr(args, 'seed', 42),
        test_ratio=getattr(args, 'test_ratio', 0.15),
    )
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    save_splits(splits, splits_path)
    return splits


# ---------------------------------------------------------------------------
# Dataset constructors
# ---------------------------------------------------------------------------

def _celebamask_datasets(args, train_indices, val_indices, test_indices):
    """Return (train_ds, val_ds, test_ds) for CelebAMask-HQ."""
    from dataloader import CelebAMaskHQDataset, CelebAMaskHQBiasedDataset

    bias_ratio = getattr(args, 'bias_ratio', 0.0)
    is_biased = bias_ratio > 0

    if is_biased:
        train_ds = CelebAMaskHQBiasedDataset(
            root_dir=args.data_root,
            img_size=(args.img_size, args.img_size),
            binary_segmentation=True,
            indices=train_indices,
            bias_ratio=bias_ratio,
            erosion_radius=getattr(args, 'erosion_radius', 5),
            biased_gender=getattr(args, 'biased_gender', 0),
            bias_seed=getattr(args, 'bias_seed', 42),
            bias_mode=getattr(args, 'bias_mode', 'erosion'),
        )
    else:
        train_ds = CelebAMaskHQDataset(
            root_dir=args.data_root,
            img_size=(args.img_size, args.img_size),
            binary_segmentation=True,
            indices=train_indices,
        )

    val_ds = CelebAMaskHQDataset(
        root_dir=args.data_root,
        img_size=(args.img_size, args.img_size),
        binary_segmentation=True,
        indices=val_indices,
    )
    test_ds = CelebAMaskHQDataset(
        root_dir=args.data_root,
        img_size=(args.img_size, args.img_size),
        binary_segmentation=True,
        indices=test_indices,
    )
    return train_ds, val_ds, test_ds


def _phc_datasets(args, train_indices, val_indices, test_indices):
    """Return (train_ds, val_ds, test_ds) for PHC.

    Training uses biased labels for bias_ratio% of tinted samples.
    Val/test always use clean (fine) labels.
    """
    from dataloader_phc import PHCDataset, PHCCleanDataset

    bias_ratio = getattr(args, 'bias_ratio', 1.0)
    bias_seed = getattr(args, 'bias_seed', 42)

    train_ds = PHCDataset(
        root_dir=args.data_root,
        img_size=(args.img_size, args.img_size),
        indices=train_indices,
        use_biased_labels=True,
        bias_ratio=bias_ratio,
        bias_seed=bias_seed,
        tint_strength=getattr(args, 'tint_strength', 0.25),
    )
    val_ds = PHCCleanDataset(
        root_dir=args.data_root,
        img_size=(args.img_size, args.img_size),
        indices=val_indices,
        tint_strength=getattr(args, 'tint_strength', 0.25),
    )
    test_ds = PHCCleanDataset(
        root_dir=args.data_root,
        img_size=(args.img_size, args.img_size),
        indices=test_indices,
        tint_strength=getattr(args, 'tint_strength', 0.25),
    )
    return train_ds, val_ds, test_ds


def _imapp_datasets(args, train_indices, val_indices, test_indices):
    from dataloader_imapp import IMAPPBiasedDataset
    from torch.utils.data import Subset

    bias_ratio = getattr(args, 'bias_ratio', 0.0)
    biased_skin_tones = getattr(args, 'biased_skin_tones', [2, 3])

    train_ds_full = IMAPPBiasedDataset(
        csv_path=args.data_root,
        img_size=(args.img_size, args.img_size),
        bias_ratio=bias_ratio,
        biased_skin_tones=biased_skin_tones
    )
    val_ds_full = IMAPPBiasedDataset(
        csv_path=args.data_root,
        img_size=(args.img_size, args.img_size),
        bias_ratio=0.0
    )
    test_ds_full = IMAPPBiasedDataset(
        csv_path=args.data_root,
        img_size=(args.img_size, args.img_size),
        bias_ratio=0.0
    )
    
    # We must wrap them in Subsets to match the API expectation,
    # or just return the subsets.
    train_ds = Subset(train_ds_full, train_indices)
    val_ds = Subset(val_ds_full, val_indices)
    test_ds = Subset(test_ds_full, test_indices)
    
    return train_ds, val_ds, test_ds

_DATASET_BUILDERS = {
    'celebamask': _celebamask_datasets,
    'phc': _phc_datasets,
    'imapp': _imapp_datasets,
}


def create_datasets(args, train_indices, val_indices, test_indices):
    """Create (train_ds, val_ds, test_ds) for the selected dataset."""
    builder = _DATASET_BUILDERS.get(args.dataset)
    if builder is None:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    return builder(args, train_indices, val_indices, test_indices)


# ---------------------------------------------------------------------------
# Clean / biased dataset loaders (for CL analysis)
# ---------------------------------------------------------------------------

def create_clean_eval_dataset(args, indices):
    """Create a dataset that always returns clean (ground-truth) labels."""
    if args.dataset == 'celebamask':
        from dataloader import CelebAMaskHQDataset
        return CelebAMaskHQDataset(
            root_dir=args.data_root,
            img_size=(args.img_size, args.img_size),
            binary_segmentation=True,
            indices=indices,
        )
    elif args.dataset == 'phc':
        from dataloader_phc import PHCCleanDataset
        return PHCCleanDataset(
            root_dir=args.data_root,
            img_size=(args.img_size, args.img_size),
            indices=indices,
        )
    elif args.dataset == 'imapp':
        from dataloader_imapp import IMAPPBiasedDataset
        from torch.utils.data import Subset
        ds = IMAPPBiasedDataset(
            csv_path=args.data_root,
            img_size=(args.img_size, args.img_size),
            bias_ratio=0.0
        )
        return Subset(ds, indices)
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')


def create_biased_eval_dataset(args, indices, bias_cfg: Dict = None):
    """Create a dataset that returns biased labels for CL comparison.

    For CelebAMask: uses CelebAMaskHQBiasedDataset with erosion params.
    For PHC: uses PHCDataset with use_biased_labels=True (coarse labels).
    """
    if args.dataset == 'celebamask':
        from dataloader import CelebAMaskHQBiasedDataset
        if bias_cfg is None:
            bias_cfg = {}
        return CelebAMaskHQBiasedDataset(
            root_dir=args.data_root,
            img_size=(args.img_size, args.img_size),
            binary_segmentation=True,
            indices=indices,
            bias_ratio=bias_cfg.get('bias_ratio', getattr(args, 'bias_ratio', 0.0)),
            erosion_radius=bias_cfg.get('erosion_radius', getattr(args, 'erosion_radius', 5)),
            biased_gender=bias_cfg.get('biased_gender', getattr(args, 'biased_gender', 0)),
            bias_seed=bias_cfg.get('bias_seed', getattr(args, 'bias_seed', 42)),
            bias_mode=bias_cfg.get('bias_mode', getattr(args, 'bias_mode', 'erosion')),
        )
    elif args.dataset == 'phc':
        from dataloader_phc import PHCDataset
        br = (bias_cfg or {}).get('bias_ratio', getattr(args, 'bias_ratio', 1.0))
        bs = (bias_cfg or {}).get('bias_seed', getattr(args, 'bias_seed', 42))
        return PHCDataset(
            root_dir=args.data_root,
            img_size=(args.img_size, args.img_size),
            indices=indices,
            use_biased_labels=True,
            bias_ratio=br,
            bias_seed=bs,
        )
    elif args.dataset == 'imapp':
        from dataloader_imapp import IMAPPBiasedDataset
        from torch.utils.data import Subset
        if bias_cfg is None:
            bias_cfg = {}
        ds = IMAPPBiasedDataset(
            csv_path=args.data_root,
            img_size=(args.img_size, args.img_size),
            bias_ratio=bias_cfg.get('bias_ratio', getattr(args, 'bias_ratio', 0.0)),
            biased_skin_tones=bias_cfg.get('biased_skin_tones', getattr(args, 'biased_skin_tones', [2, 3]))
        )
        return Subset(ds, indices)
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')


def get_demographic_names(dataset: str) -> List[str]:
    """Return standard demographic group names for logging."""
    if dataset == 'celebamask':
        return ["Male", "Female"]
    elif dataset == 'phc':
        return ["Male", "Female"]
    elif dataset == 'imapp':
        return ["Light", "Dark"]
    return ["Group_0", "Group_1"]


def is_phc_experiment(config: Dict) -> bool:
    """Infer from saved config.json whether an experiment used PHC."""
    return config.get('dataset', 'celebamask') == 'phc'
