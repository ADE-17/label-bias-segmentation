"""
Utilities for creating and saving reproducible K-fold cross-validation splits.
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import StratifiedKFold


def create_kfold_splits(
    n_samples: int,
    n_folds: int = 5,
    stratify_labels: Optional[np.ndarray] = None,
    random_seed: int = 42,
    test_ratio: float = 0.15
) -> Dict:
    """
    Create K-fold cross-validation splits with a held-out test set.
    
    Args:
        n_samples: Total number of samples
        n_folds: Number of folds for cross-validation
        stratify_labels: Labels for stratified splitting (e.g., gender)
        random_seed: Random seed for reproducibility
        test_ratio: Ratio of data to hold out for final test set
        
    Returns:
        Dictionary containing:
            - 'test_indices': Indices for held-out test set
            - 'folds': List of dicts with 'train_indices' and 'val_indices'
            - 'metadata': Info about the split
    """
    np.random.seed(random_seed)
    
    all_indices = np.arange(n_samples)
    
    if stratify_labels is not None:
        # Stratified split for test set
        from sklearn.model_selection import train_test_split
        trainval_indices, test_indices = train_test_split(
            all_indices,
            test_size=test_ratio,
            stratify=stratify_labels,
            random_state=random_seed
        )
        trainval_labels = stratify_labels[trainval_indices]
    else:
        # Random split
        np.random.shuffle(all_indices)
        n_test = int(n_samples * test_ratio)
        test_indices = all_indices[:n_test]
        trainval_indices = all_indices[n_test:]
        trainval_labels = None
    
    # Create K folds from train+val data
    folds = []
    if trainval_labels is not None:
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        splits = list(kfold.split(trainval_indices, trainval_labels))
    else:
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        splits = list(kfold.split(trainval_indices))
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        folds.append({
            'fold': fold_idx,
            'train_indices': trainval_indices[train_idx].tolist(),
            'val_indices': trainval_indices[val_idx].tolist()
        })
    
    splits_data = {
        'test_indices': test_indices.tolist(),
        'folds': folds,
        'metadata': {
            'n_samples': n_samples,
            'n_folds': n_folds,
            'random_seed': random_seed,
            'test_ratio': test_ratio,
            'stratified': stratify_labels is not None,
            'n_test': len(test_indices),
            'n_trainval': len(trainval_indices)
        }
    }
    
    return splits_data


def save_splits(splits_data: Dict, save_path: str) -> None:
    """Save splits to a JSON file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(splits_data, f, indent=2)
    
    print(f"Splits saved to {save_path}")
    print(f"  Test set size: {splits_data['metadata']['n_test']}")
    print(f"  Number of folds: {splits_data['metadata']['n_folds']}")
    for fold in splits_data['folds']:
        print(f"  Fold {fold['fold']}: train={len(fold['train_indices'])}, val={len(fold['val_indices'])}")


def load_splits(load_path: str) -> Dict:
    """Load splits from a JSON file."""
    with open(load_path, 'r') as f:
        splits_data = json.load(f)
    
    print(f"Loaded splits from {load_path}")
    print(f"  Test set size: {splits_data['metadata']['n_test']}")
    print(f"  Number of folds: {splits_data['metadata']['n_folds']}")
    
    return splits_data


def get_fold_indices(splits_data: Dict, fold: int) -> Tuple[List[int], List[int], List[int]]:
    """
    Get train, val, and test indices for a specific fold.
    
    Args:
        splits_data: Dictionary from create_kfold_splits or load_splits
        fold: Fold index (0 to n_folds-1)
        
    Returns:
        train_indices, val_indices, test_indices
    """
    fold_data = splits_data['folds'][fold]
    return (
        fold_data['train_indices'],
        fold_data['val_indices'],
        splits_data['test_indices']
    )


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dataloader import CelebAMaskHQDataset
    
    # Create dataset to get gender labels
    dataset = CelebAMaskHQDataset(binary_segmentation=True)
    genders = dataset.get_all_genders()
    
    # Create splits
    splits = create_kfold_splits(
        n_samples=len(dataset),
        n_folds=5,
        stratify_labels=genders,
        random_seed=42,
        test_ratio=0.15
    )
    
    # Save splits
    save_splits(splits, "configs/splits/cv_splits.json")
