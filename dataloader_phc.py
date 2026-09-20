"""
PHC Dataset for segmentation with synthetic demographic shortcut.

The dataset has:
  - 651 phase-contrast microscopy cell images (128x128)
  - 3 fine annotators (fine1/2/3) → combined = clean labels
  - 3 coarse annotators (coarse1/2/3) → combined = biased labels

Synthetic demographic shortcut:
  Half the samples are assigned to "group 1" and receive a color tint
  on the input image (the tint acts as a visual shortcut a model can
  latch onto). Group 1 (tinted) receives coarse (biased) labels.
  Group 0 (original) receives fine (clean) labels.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import torchvision.transforms as transforms
import json


class PHCDataset(Dataset):
    """
    PHC cell segmentation dataset with synthetic demographic groups.

    Group 0 (no tint)  → fine (clean) labels
    Group 1 (tinted)   → coarse (biased) labels
    """

    # Amber/sepia tint that doesn't overlap with common augmentation colors
    TINT_COLOR = np.array([0.85, 0.65, 0.30])  # warm amber
    TINT_STRENGTH = 0.25

    def __init__(
        self,
        root_dir: str,
        img_size: Tuple[int, int] = (128, 128),
        binary_segmentation: bool = True,
        indices: Optional[List[int]] = None,
        group_seed: int = 42,
        tint_strength: float = 0.25,
        use_biased_labels: bool = True,
        bias_ratio: float = 1.0,
        bias_seed: int = 42,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            root_dir: Path to phc_data/ containing images/, fine1-3/, coarse1-3/
            img_size: Target (H, W) for resizing
            binary_segmentation: Always True for this dataset
            indices: Optional subset indices
            group_seed: Seed for group assignment (deterministic split)
            tint_strength: How strong the color tint is (0-1)
            use_biased_labels: If True, group 1 gets coarse labels.
                               If False, everyone gets fine labels (for clean eval).
            bias_ratio: Fraction of tinted (group 1) samples that actually
                        receive coarse labels (0.0 = none, 1.0 = all).
                        The rest of group 1 gets fine labels despite being tinted.
            bias_seed: Seed for selecting which tinted samples are biased.
            transform: Optional custom image transform
        """
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / 'images'
        self.img_size = img_size
        self.binary_segmentation = binary_segmentation
        self.tint_strength = tint_strength
        self.use_biased_labels = use_biased_labels
        self.bias_ratio = bias_ratio
        self.bias_seed = bias_seed

        all_ids = sorted([
            f.stem.replace('img_', '')
            for f in self.img_dir.glob('img_*.jpg')
        ])
        self.all_ids = all_ids

        if indices is not None:
            self.sample_ids = [all_ids[i] for i in indices]
        else:
            self.sample_ids = all_ids

        # Deterministic group assignment: split all IDs (not just subset)
        rng = np.random.RandomState(group_seed)
        perm = rng.permutation(len(all_ids))
        half = len(all_ids) // 2
        tinted_set = set(all_ids[i] for i in perm[:half])
        self.groups = {sid: (1 if sid in tinted_set else 0) for sid in all_ids}

        # Select which tinted samples actually get biased (coarse) labels
        self._biased_ids = self._select_biased_samples(tinted_set)

        # Label folder naming conventions
        self._fine_patterns = {
            'fine1': 'label_{}.png',
            'fine2': 'img_{}.png',
            'fine3': 'img_{}.png',
        }
        self._coarse_patterns = {
            'coarse1': 'img_{}.png',
            'coarse2': 'img_{}.png',
            'coarse3': 'img_{}.png',
        }

        # Bias bookkeeping for compatibility with CelebA pipeline
        n_tinted = sum(1 for s in self.sample_ids if self.groups[s] == 1)
        n_biased = sum(1 for s in self.sample_ids if s in self._biased_ids)
        self.bias_info = {
            'bias_ratio': bias_ratio,
            'biased_gender': 1,
            'biased_gender_name': 'Tinted',
            'num_biased_samples': n_biased,
            'total_target_gender': n_tinted,
        }

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform

    def _select_biased_samples(self, tinted_set: set) -> set:
        """Pick which tinted samples actually receive coarse labels."""
        if not self.use_biased_labels or self.bias_ratio <= 0:
            return set()

        tinted_ids = sorted(tinted_set)
        if self.bias_ratio >= 1.0:
            return set(tinted_ids)

        rng = np.random.RandomState(self.bias_seed)
        n_to_bias = int(len(tinted_ids) * self.bias_ratio)
        chosen = rng.choice(tinted_ids, size=n_to_bias, replace=False)
        return set(chosen)

    def _load_mask(self, sid: str, annotator_patterns: dict) -> np.ndarray:
        """Load and combine masks from multiple annotators via majority vote."""
        masks = []
        for folder, pattern in annotator_patterns.items():
            path = self.root_dir / folder / pattern.format(sid)
            if path.exists():
                m = np.array(Image.open(path).convert('L').resize(
                    self.img_size, Image.NEAREST))
                masks.append((m > 127).astype(np.uint8))

        if not masks:
            return np.zeros(self.img_size, dtype=np.uint8)

        combined = sum(masks)
        return (combined >= 2).astype(np.uint8)  # majority vote

    def _apply_tint(self, image: Image.Image) -> Image.Image:
        """Apply amber color tint to image (before transforms)."""
        arr = np.array(image).astype(np.float32) / 255.0
        tint = self.TINT_COLOR.reshape(1, 1, 3)
        blended = arr * (1 - self.tint_strength) + tint * self.tint_strength
        blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(blended)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sid = self.sample_ids[idx]
        group = self.groups[sid]

        # Load image — tint is always applied to group 1 (visual shortcut)
        img_path = self.img_dir / f'img_{sid}.jpg'
        image = Image.open(img_path).convert('RGB')

        if group == 1:
            image = self._apply_tint(image)

        # Load mask: coarse only for samples selected as biased
        is_biased = sid in self._biased_ids
        if is_biased:
            mask = self._load_mask(sid, self._coarse_patterns)
        else:
            mask = self._load_mask(sid, self._fine_patterns)

        if self.transform:
            image = self.transform(image)

        mask = torch.from_numpy(mask).long()

        return {
            'image': image,
            'mask': mask,
            'gender': torch.tensor(group, dtype=torch.long),
            'image_id': torch.tensor(int(sid), dtype=torch.long),
            'is_biased': torch.tensor(is_biased, dtype=torch.bool),
        }

    def get_gender_for_idx(self, idx: int) -> int:
        return self.groups[self.sample_ids[idx]]

    def get_all_genders(self) -> np.ndarray:
        return np.array([self.get_gender_for_idx(i) for i in range(len(self))])

    def get_num_classes(self) -> int:
        return 2

    def get_class_names(self) -> List[str]:
        return ['background', 'cell']

    def get_bias_summary(self) -> str:
        n_g0 = sum(1 for i in range(len(self)) if self.get_gender_for_idx(i) == 0)
        n_g1 = len(self) - n_g0
        n_biased = sum(1 for s in self.sample_ids if s in self._biased_ids)
        return (
            f"PHC Dataset Configuration:\n"
            f"  Total samples: {len(self)}\n"
            f"  Group 0 (no tint, fine labels): {n_g0}\n"
            f"  Group 1 (tinted): {n_g1}\n"
            f"    Biased (coarse labels): {n_biased} / {n_g1} "
            f"({self.bias_ratio*100:.0f}%)\n"
            f"    Clean  (fine labels):    {n_g1 - n_biased} / {n_g1}\n"
            f"  Tint strength: {self.tint_strength}"
        )

    def save_bias_indices(self, path: str) -> None:
        """Save the IDs of biased samples for reproducibility."""
        biased_in_subset = [
            int(s) for s in self.sample_ids if s in self._biased_ids
        ]
        data = {
            'biased_indices': sorted(biased_in_subset),
            'config': {
                'bias_ratio': self.bias_ratio,
                'bias_seed': self.bias_seed,
                'num_biased': len(biased_in_subset),
                'num_tinted': sum(1 for s in self.sample_ids if self.groups[s] == 1),
            },
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


class PHCCleanDataset(PHCDataset):
    """PHC dataset that always uses fine (clean) labels for both groups.
    Use for evaluation / CL analysis against ground truth."""

    def __init__(self, **kwargs):
        kwargs['use_biased_labels'] = False
        kwargs.setdefault('bias_ratio', 0.0)
        super().__init__(**kwargs)
