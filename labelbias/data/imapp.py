import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from typing import Optional, Tuple, Dict, List

class IMAPPBiasedDataset(Dataset):
    """
    IMA++ Dataset pre-processed to pair images with two masks:
    - clean_mask: STAPLE consensus or highest-skill annotator (used for eval and unbiased groups)
    - biased_mask: Lower-skill annotator (used for biased groups)
    """
    
    def __init__(
        self,
        csv_path: str = 'configs/imapp_processed.csv',
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
        img_size: Tuple[int, int] = (256, 256),
        bias_ratio: float = 0.0,
        biased_skin_tones: List[int] = [2, 3],
        bias_seed: int = 42,
    ):
        self.csv_path = csv_path
        self.img_size = img_size
        self.transform = transform
        self.target_transform = target_transform
        
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
            
        self.df = pd.read_csv(csv_path)
        
        self.bias_ratio = bias_ratio
        self.biased_skin_tones = biased_skin_tones
        self.bias_seed = bias_seed
        self.biased_sample_indices = self._select_biased_samples()
        
    def _select_biased_samples(self) -> set:
        if self.bias_ratio <= 0:
            return set()
        rng = np.random.RandomState(self.bias_seed)
        
        target_indices = [
            i for i, gender in enumerate(self.df['gender'])
            if gender in self.biased_skin_tones
        ]
        
        n_to_bias = int(len(target_indices) * self.bias_ratio)
        biased = rng.choice(target_indices, size=n_to_bias, replace=False)
        return set(biased)
        
    def __len__(self) -> int:
        return len(self.df)
        
    def _load_mask(self, path: str) -> np.ndarray:
        mask = Image.open(path).convert('L')
        mask = mask.resize(self.img_size, Image.NEAREST)
        mask_array = np.array(mask)
        binary_mask = np.zeros(self.img_size, dtype=np.uint8)
        binary_mask[mask_array > 127] = 1
        return binary_mask
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # Load image
        image = Image.open(row['image_path']).convert('RGB')
        
        # Load masks
        clean_mask = self._load_mask(row['clean_mask_path'])
        
        is_biased = idx in self.biased_sample_indices
        if is_biased:
            train_mask = self._load_mask(row['biased_mask_path'])
        else:
            train_mask = clean_mask
            
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            train_mask_tensor = self.target_transform(train_mask)
            clean_mask_tensor = self.target_transform(clean_mask)
        else:
            train_mask_tensor = torch.from_numpy(train_mask).long()
            clean_mask_tensor = torch.from_numpy(clean_mask).long()
            
        gender = int(row['gender'])
        binary_gender = 1 if gender in self.biased_skin_tones else 0
            
        return {
            'image': image,
            'mask': train_mask_tensor,
            'clean_mask': clean_mask_tensor,
            'gender': torch.tensor(binary_gender, dtype=torch.long),
            'skin_tone': torch.tensor(gender, dtype=torch.long),
            'image_id': torch.tensor(idx, dtype=torch.long),
            'is_biased': torch.tensor(is_biased, dtype=torch.bool)
        }
