import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import torchvision.transforms as transforms
import json
from scipy import ndimage


class CelebAMaskHQDataset(Dataset):
    """
    CelebAMask-HQ Dataset for semantic segmentation with gender information.
    
    The dataset contains:
    - 30,000 high-resolution face images
    - 19 semantic segmentation classes (or binary: foreground vs background)
    - Gender and other attribute annotations
    """
    
    # All 19 segmentation classes in CelebAMask-HQ
    CLASSES = [
        'background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 
        'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 
        'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 
        'neck_l', 'neck', 'cloth'
    ]
    
    BINARY_CLASSES = ['background', 'foreground']
    
    # Mapping from filename patterns to class indices
    CLASS_MAPPING = {
        'skin': 1, 'nose': 2, 'eye_g': 3, 'l_eye': 4, 'r_eye': 5,
        'l_brow': 6, 'r_brow': 7, 'l_ear': 8, 'r_ear': 9, 'mouth': 10,
        'u_lip': 11, 'l_lip': 12, 'hair': 13, 'hat': 14, 'ear_r': 15,
        'neck_l': 16, 'neck': 17, 'cloth': 18
    }
    
    def __init__(
        self, 
        root_dir: str = "/path/to/CelebAMask-HQ",
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
        img_size: Tuple[int, int] = (512, 512),
        binary_segmentation: bool = False,
        indices: Optional[List[int]] = None
    ):
        """
        Args:
            root_dir: Root directory of CelebAMask-HQ dataset
            transform: Optional transform to be applied on images
            target_transform: Optional transform to be applied on masks
            img_size: Target size for images and masks (height, width)
            binary_segmentation: If True, convert to binary (foreground vs background)
            indices: Optional list of indices to use (for train/val/test splits)
        """
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / "CelebA-HQ-img"
        self.mask_dir = self.root_dir / "CelebAMask-HQ-mask-anno"
        self.attr_file = self.root_dir / "CelebAMask-HQ-attribute-anno.txt"
        self.img_size = img_size
        self.binary_segmentation = binary_segmentation
        
        # Load attributes
        self.attributes = self._load_attributes()
        
        # Get list of available images
        all_image_ids = sorted([
            int(f.stem) for f in self.img_dir.glob("*.jpg")
        ])
        
        # Use subset if indices provided
        if indices is not None:
            self.image_ids = [all_image_ids[i] for i in indices]
        else:
            self.image_ids = all_image_ids
        
        # Default transforms if not provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        self.target_transform = target_transform
        
    def _load_attributes(self) -> pd.DataFrame:
        """Load and parse the attribute annotation file."""
        with open(self.attr_file, 'r') as f:
            lines = f.readlines()
        
        # First line is count, second line is attribute names
        num_images = int(lines[0].strip())
        attr_names = lines[1].strip().split()
        
        # Parse attributes
        data = []
        for line in lines[2:]:
            parts = line.strip().split()
            img_name = parts[0]
            attrs = [int(x) for x in parts[1:]]
            data.append([img_name] + attrs)
        
        df = pd.DataFrame(data, columns=['image'] + attr_names)
        
        # Convert -1/1 to 0/1 for easier use
        for col in attr_names:
            df[col] = (df[col] == 1).astype(int)
        
        # Extract image ID from filename
        df['image_id'] = df['image'].str.replace('.jpg', '', regex=False).astype(int)
        df = df.set_index('image_id')
        
        return df
    
    def _get_mask_path(self, img_id: int, class_name: str) -> Path:
        """Get the path to a specific mask file."""
        folder_num = img_id // 2000
        mask_name = f"{img_id:05d}_{class_name}.png"
        return self.mask_dir / str(folder_num) / mask_name
    
    def _load_combined_mask(self, img_id: int) -> np.ndarray:
        """
        Load and combine all mask classes into a single segmentation mask.
        Returns a single-channel mask with class indices.
        If binary_segmentation is True, returns 0 for background, 1 for foreground.
        """
        # Initialize mask with zeros (background class)
        combined_mask = np.zeros(self.img_size, dtype=np.uint8)
        
        # Load each class mask and combine
        for class_name, class_idx in self.CLASS_MAPPING.items():
            mask_path = self._get_mask_path(img_id, class_name)
            
            if mask_path.exists():
                mask = Image.open(mask_path).convert('L')
                mask = mask.resize(self.img_size, Image.NEAREST)
                mask_array = np.array(mask)
                
                if self.binary_segmentation:
                    # Binary: any non-background class becomes foreground (1)
                    combined_mask[mask_array > 0] = 1
                else:
                    # Multi-class: use original class index
                    combined_mask[mask_array > 0] = class_idx
        
        return combined_mask
    
    def get_num_classes(self) -> int:
        """Return number of classes for segmentation."""
        return 2 if self.binary_segmentation else len(self.CLASSES)
    
    def get_class_names(self) -> List[str]:
        """Return class names."""
        return self.BINARY_CLASSES if self.binary_segmentation else self.CLASSES
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary containing:
            - 'image': preprocessed image tensor (C, H, W)
            - 'mask': segmentation mask tensor (H, W) with class indices
            - 'gender': binary gender label (0 = Female, 1 = Male)
            - 'image_id': the image ID
        """
        img_id = self.image_ids[idx]
        
        # Load image
        img_path = self.img_dir / f"{img_id}.jpg"
        image = Image.open(img_path).convert('RGB')
        
        # Load combined segmentation mask
        mask = self._load_combined_mask(img_id)
        
        # Get gender attribute (Male column)
        gender = self.attributes.loc[img_id, 'Male']
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            # Convert to tensor
            mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'gender': torch.tensor(gender, dtype=torch.long),
            'image_id': torch.tensor(img_id, dtype=torch.long)
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling class imbalance.
        This can be used with CrossEntropyLoss.
        """
        # Count pixels per class (sample first 100 images for efficiency)
        class_counts = np.zeros(len(self.CLASSES))
        
        for idx in range(min(100, len(self))):
            img_id = self.image_ids[idx]
            mask = self._load_combined_mask(img_id)
            
            for class_idx in range(len(self.CLASSES)):
                class_counts[class_idx] += (mask == class_idx).sum()
        
        # Compute inverse frequency weights
        total = class_counts.sum()
        weights = total / (len(self.CLASSES) * class_counts + 1e-6)
        
        return torch.FloatTensor(weights)
    
    def get_gender_for_idx(self, idx: int) -> int:
        """Get gender for a specific index (useful for stratified splitting)."""
        img_id = self.image_ids[idx]
        return self.attributes.loc[img_id, 'Male']
    
    def get_all_genders(self) -> np.ndarray:
        """Get gender labels for all samples."""
        return np.array([self.get_gender_for_idx(i) for i in range(len(self))])


class CelebAMaskHQBiasedDataset(CelebAMaskHQDataset):
    """
    CelebAMask-HQ Dataset with label bias injection.
    
    Supports multiple bias modes:
      - erosion:  uniform morphological erosion  (under-segmentation)
      - dilation: uniform morphological dilation  (over-segmentation)
      - wave:     non-uniform sinusoidal boundary perturbation (mixed)
    """
    
    BIAS_MODES = ('erosion', 'dilation', 'wave')

    def __init__(
        self,
        root_dir: str = "/path/to/CelebAMask-HQ",
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
        img_size: Tuple[int, int] = (512, 512),
        binary_segmentation: bool = True,
        indices: Optional[List[int]] = None,
        # Bias parameters
        bias_ratio: float = 0.0,
        erosion_radius: int = 5,
        biased_gender: int = 0,
        bias_seed: int = 42,
        biased_indices: Optional[List[int]] = None,
        bias_mode: str = 'erosion',
    ):
        super().__init__(
            root_dir=root_dir,
            transform=transform,
            target_transform=target_transform,
            img_size=img_size,
            binary_segmentation=binary_segmentation,
            indices=indices
        )
        
        if bias_mode not in self.BIAS_MODES:
            raise ValueError(f"bias_mode must be one of {self.BIAS_MODES}, got '{bias_mode}'")

        self.bias_ratio = bias_ratio
        self.erosion_radius = erosion_radius
        self.biased_gender = biased_gender
        self.bias_seed = bias_seed
        self.bias_mode = bias_mode
        
        # Determine which samples should be biased
        if biased_indices is not None:
            self.biased_sample_indices = set(biased_indices)
        else:
            self.biased_sample_indices = self._select_biased_samples()
        
        # Store bias info for logging
        self.bias_info = {
            'bias_ratio': bias_ratio,
            'erosion_radius': erosion_radius,
            'bias_mode': bias_mode,
            'biased_gender': biased_gender,
            'biased_gender_name': 'Female' if biased_gender == 0 else 'Male',
            'num_biased_samples': len(self.biased_sample_indices),
            'total_target_gender': sum(1 for i in range(len(self)) if self.get_gender_for_idx(i) == biased_gender),
        }
    
    def _select_biased_samples(self) -> set:
        """Select which samples should have biased (eroded) masks."""
        if self.bias_ratio <= 0:
            return set()
        
        rng = np.random.RandomState(self.bias_seed)
        
        # Get indices of target gender samples
        target_gender_indices = [
            i for i in range(len(self)) 
            if self.get_gender_for_idx(i) == self.biased_gender
        ]
        
        # Select a fraction to bias
        n_to_bias = int(len(target_gender_indices) * self.bias_ratio)
        biased_indices = rng.choice(
            target_gender_indices, 
            size=n_to_bias, 
            replace=False
        )
        
        return set(biased_indices)
    
    def _erode_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological erosion to shrink the foreground mask."""
        if self.erosion_radius <= 0:
            return mask
        struct = ndimage.generate_binary_structure(2, 1)
        if self.binary_segmentation:
            foreground = (mask == 1)
            eroded = ndimage.binary_erosion(
                foreground, structure=struct, iterations=self.erosion_radius)
            result = np.zeros_like(mask)
            result[eroded] = 1
            return result
        else:
            result = np.zeros_like(mask)
            for class_idx in range(1, len(self.CLASSES)):
                class_mask = (mask == class_idx)
                if class_mask.any():
                    eroded = ndimage.binary_erosion(
                        class_mask, structure=struct, iterations=self.erosion_radius)
                    result[eroded] = class_idx
            return result

    def _dilate_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological dilation to expand the foreground mask."""
        if self.erosion_radius <= 0:
            return mask
        struct = ndimage.generate_binary_structure(2, 1)
        if self.binary_segmentation:
            foreground = (mask == 1)
            dilated = ndimage.binary_dilation(
                foreground, structure=struct, iterations=self.erosion_radius)
            result = np.zeros_like(mask)
            result[dilated] = 1
            return result
        else:
            result = np.zeros_like(mask)
            for class_idx in range(1, len(self.CLASSES)):
                class_mask = (mask == class_idx)
                if class_mask.any():
                    dilated = ndimage.binary_dilation(
                        class_mask, structure=struct, iterations=self.erosion_radius)
                    result[dilated] = class_idx
            return result

    def _perturb_boundary(self, mask: np.ndarray, sample_id: int) -> np.ndarray:
        """Deform the mask boundary with wave-like (sinusoidal) displacement.

        Computes a signed distance field from the boundary, then shifts the
        boundary inward/outward using a spatially-varying sinusoidal
        displacement whose amplitude is controlled by ``self.erosion_radius``.
        The result contains simultaneous local over- and under-segmentation.
        """
        amplitude = float(self.erosion_radius)
        if amplitude <= 0:
            return mask

        foreground = (mask == 1).astype(np.float32)
        if foreground.sum() == 0 or foreground.all():
            return mask

        dist_in = ndimage.distance_transform_edt(foreground)
        dist_out = ndimage.distance_transform_edt(1.0 - foreground)
        sdf = dist_in - dist_out

        h, w = mask.shape
        rng = np.random.RandomState(self.bias_seed ^ (sample_id * 2654435761 & 0xFFFFFFFF))

        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

        n_harmonics = 3
        displacement = np.zeros((h, w), dtype=np.float32)
        for _ in range(n_harmonics):
            freq = rng.uniform(0.02, 0.08)
            angle = rng.uniform(0, 2 * np.pi)
            phase = rng.uniform(0, 2 * np.pi)
            proj = xs * np.cos(angle) + ys * np.sin(angle)
            displacement += np.sin(freq * proj + phase)

        displacement *= amplitude / n_harmonics

        perturbed = (sdf > displacement).astype(mask.dtype)
        return perturbed

    def _apply_bias(self, mask: np.ndarray, sample_id: int) -> np.ndarray:
        """Dispatch to the correct bias function based on ``self.bias_mode``."""
        if self.bias_mode == 'erosion':
            return self._erode_mask(mask)
        elif self.bias_mode == 'dilation':
            return self._dilate_mask(mask)
        elif self.bias_mode == 'wave':
            return self._perturb_boundary(mask, sample_id)
        return mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item with potential bias applied."""
        img_id = self.image_ids[idx]
        
        # Load image
        img_path = self.img_dir / f"{img_id}.jpg"
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask = self._load_combined_mask(img_id)
        
        # Get gender
        gender = self.attributes.loc[img_id, 'Male']
        
        # Apply bias if this sample is selected
        is_biased = idx in self.biased_sample_indices
        if is_biased:
            mask = self._apply_bias(mask, img_id)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'gender': torch.tensor(gender, dtype=torch.long),
            'image_id': torch.tensor(img_id, dtype=torch.long),
            'is_biased': torch.tensor(is_biased, dtype=torch.bool)
        }
    
    def get_bias_summary(self) -> str:
        """Return a summary string of the bias configuration."""
        info = self.bias_info
        mode = info.get('bias_mode', 'erosion')
        radius_label = 'Amplitude' if mode == 'wave' else 'Radius'
        return (
            f"Label Bias Configuration:\n"
            f"  Bias mode: {mode}\n"
            f"  Target gender: {info['biased_gender_name']} (gender={info['biased_gender']})\n"
            f"  Bias ratio: {info['bias_ratio']*100:.1f}%\n"
            f"  {radius_label}: {info['erosion_radius']} pixels\n"
            f"  Biased samples: {info['num_biased_samples']} / {info['total_target_gender']} "
            f"({info['biased_gender_name']} samples)"
        )
    
    def save_bias_indices(self, path: str) -> None:
        """Save the indices of biased samples for reproducibility."""
        bias_data = {
            'biased_indices': [int(x) for x in sorted(self.biased_sample_indices)],
            'config': {k: int(v) if isinstance(v, (np.integer, np.int64)) else v 
                       for k, v in self.bias_info.items()}
        }
        with open(path, 'w') as f:
            json.dump(bias_data, f, indent=2)
    
    @classmethod
    def load_bias_indices(cls, path: str) -> List[int]:
        """Load biased indices from a saved file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return data['biased_indices']