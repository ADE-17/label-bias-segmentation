#!/usr/bin/env python3
"""
Confident Learning for Segmentation Label Error Detection.

Based on: "Characterizing Label Errors: Confident Learning for Noisy-Labeled Image Segmentation"
(Zhang et al., MICCAI 2020)

This module:
1. Computes predicted probabilities for all pixels in the test set
2. Establishes confidence thresholds per class
3. Builds the joint distribution matrix Q(ỹ, y*) between noisy and true labels
4. Identifies mislabeled pixels (X̃)
5. Analyzes errors by demographic group to distinguish noise from bias

python run_cl_analysis.py \
    --experiments binary_seg_cv bias_female_r25_e5 
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import segmentation_models_pytorch as smp

from labelbias.data.celebamask import CelebAMaskHQDataset, CelebAMaskHQBiasedDataset
from labelbias.data.splits import load_splits, get_fold_indices
from labelbias.data.factory import (
    add_dataset_args, apply_dataset_defaults, create_splits,
    create_clean_eval_dataset, create_biased_eval_dataset,
    is_phc_experiment,
)


@dataclass
class ConfidentLearningResults:
    """Container for Confident Learning analysis results."""
    thresholds: np.ndarray = None
    joint_matrix: np.ndarray = None
    joint_matrix_normalized: np.ndarray = None
    joint_matrix_female: np.ndarray = None
    joint_matrix_male: np.ndarray = None
    joint_matrix_female_normalized: np.ndarray = None
    joint_matrix_male_normalized: np.ndarray = None
    
    total_pixels: int = 0
    total_errors: int = 0
    error_rate: float = 0.0
    
    female_pixels: int = 0
    female_errors: int = 0
    female_error_rate: float = 0.0
    
    male_pixels: int = 0
    male_errors: int = 0
    male_error_rate: float = 0.0
    
    # Absolute counts for direction
    total_excess_errors: int = 0
    total_deficit_errors: int = 0
    female_excess_errors: int = 0
    female_deficit_errors: int = 0
    male_excess_errors: int = 0
    male_deficit_errors: int = 0
    
    # Normalized metrics (LER and SDR)
    ler_female: float = 0.0
    ler_male: float = 0.0
    ler_gap: float = 0.0
    rr: float = 1.0
    
    sdr_female: float = 0.0
    sdr_male: float = 0.0
    sdr_gap: float = 0.0
    
    is_biased: bool = False
    bias_direction: str = ""
    bias_type: str = ""
    bias_confidence: str = ""
    
    iou_foreground: float = 0.0
    dice_foreground: float = 0.0
    female_iou_foreground: float = 0.0
    female_dice_foreground: float = 0.0
    male_iou_foreground: float = 0.0
    male_dice_foreground: float = 0.0
    iou_gap: float = 0.0
    dice_gap: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'thresholds': self.thresholds.tolist() if self.thresholds is not None else None,
            'joint_matrix': self.joint_matrix.tolist() if self.joint_matrix is not None else None,
            'total_pixels': int(self.total_pixels),
            'total_errors': int(self.total_errors),
            'error_rate': float(self.error_rate),
            'female_pixels': int(self.female_pixels),
            'female_errors': int(self.female_errors),
            'female_error_rate': float(self.female_error_rate),
            'male_pixels': int(self.male_pixels),
            'male_errors': int(self.male_errors),
            'male_error_rate': float(self.male_error_rate),
            
            'total_excess_errors': int(self.total_excess_errors),
            'total_deficit_errors': int(self.total_deficit_errors),
            'female_excess_errors': int(self.female_excess_errors),
            'female_deficit_errors': int(self.female_deficit_errors),
            'male_excess_errors': int(self.male_excess_errors),
            'male_deficit_errors': int(self.male_deficit_errors),
            
            'ler_female': float(self.ler_female),
            'ler_male': float(self.ler_male),
            'ler_gap': float(self.ler_gap),
            'rr': float(self.rr),
            
            'sdr_female': float(self.sdr_female),
            'sdr_male': float(self.sdr_male),
            'sdr_gap': float(self.sdr_gap),
            
            'is_biased': bool(self.is_biased),
            'bias_direction': str(self.bias_direction),
            'bias_type': str(self.bias_type),
            'bias_confidence': str(self.bias_confidence),
            
            'iou_foreground': float(self.iou_foreground),
            'dice_foreground': float(self.dice_foreground),
            'female_iou_foreground': float(self.female_iou_foreground),
            'female_dice_foreground': float(self.female_dice_foreground),
            'male_iou_foreground': float(self.male_iou_foreground),
            'male_dice_foreground': float(self.male_dice_foreground),
            'iou_gap': float(self.iou_gap),
            'dice_gap': float(self.dice_gap),
        }


class ConfidentLearningAnalyzer:
    """
    Confident Learning analyzer for segmentation.
    
    Implements the CL algorithm from Zhang et al. (2020) to:
    1. Estimate per-class confidence thresholds
    2. Build joint distribution matrix between observed (noisy) and predicted (true) labels
    3. Identify mislabeled pixels
    4. Analyze errors by demographic group
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        num_classes: int = 2,
        bias_threshold: float = 0.02,  # 2% difference to declare bias
        use_style: bool = False,
        eval_group: int = 0,
    ):
        """
        Args:
            model: Trained segmentation model
            dataloader: DataLoader for evaluation data
            device: torch device
            num_classes: Number of segmentation classes (2 for binary)
            bias_threshold: Minimum gap between demographics to declare bias
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.num_classes = num_classes
        self.bias_threshold = bias_threshold
        self.use_style = use_style
        self.eval_group = eval_group
        
        if self.model is not None:
            self.model.eval()
    
    @torch.no_grad()
    def compute_predictions(self) -> Tuple[List, List, List, List]:
        """
        Compute predicted probabilities for all pixels.
        
        Returns:
            all_probs: List of (H, W, C) probability arrays
            all_labels: List of (H, W) label arrays
            all_genders: List of gender labels
            all_image_ids: List of image IDs
        """
        all_probs = []
        all_labels = []
        all_genders = []
        all_image_ids = []
        
        pbar = tqdm(self.dataloader, desc='Computing predictions')
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['mask'].cpu().numpy()
            genders = batch['gender'].cpu().numpy()
            image_ids = batch['image_id'].cpu().numpy()
            
            # Get predicted probabilities
            if self.use_style:
                logits = self.model.predict(images, self.eval_group)
            else:
                logits = self.model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()  # (B, C, H, W)
            
            # Store per-sample
            for i in range(len(images)):
                # Transpose to (H, W, C)
                all_probs.append(np.transpose(probs[i], (1, 2, 0)))
                all_labels.append(labels[i])
                all_genders.append(genders[i])
                all_image_ids.append(image_ids[i])
        
        return all_probs, all_labels, all_genders, all_image_ids
    
    def compute_confidence_thresholds(
        self, 
        all_probs: List[np.ndarray], 
        all_labels: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute per-class confidence thresholds.
        
        For each class j, the threshold t_j is the average predicted probability
        for pixels labeled as class j.
        
        t_j = E[p(y=j|x) | ỹ=j]
        """
        # Accumulate probabilities for each class
        class_prob_sums = np.zeros(self.num_classes)
        class_counts = np.zeros(self.num_classes)
        
        for probs, labels in zip(all_probs, all_labels):
            for c in range(self.num_classes):
                mask = (labels == c)
                if mask.sum() > 0:
                    class_prob_sums[c] += probs[mask, c].sum()
                    class_counts[c] += mask.sum()
        
        # Compute thresholds (average probability)
        thresholds = class_prob_sums / np.maximum(class_counts, 1)
        
        return thresholds
    
    def build_joint_matrix(
        self,
        all_probs: List[np.ndarray],
        all_labels: List[np.ndarray],
        thresholds: np.ndarray,
        gender_filter: Optional[int] = None,
        all_genders: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Build the joint distribution matrix Q(ỹ, y*).
        
        Q[i,j] = count of pixels with observed label i and confident prediction j
        
        A pixel is "confidently predicted" as class j if:
        p(y=j|x) >= t_j (threshold for class j)
        
        Args:
            all_probs: List of probability arrays
            all_labels: List of label arrays
            thresholds: Per-class confidence thresholds
            gender_filter: If provided, only include samples with this gender
            all_genders: List of gender labels (required if gender_filter is set)
            
        Returns:
            joint_matrix: (num_classes, num_classes) count matrix
            total_pixels: Total number of pixels included
        """
        joint_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        total_pixels = 0
        
        for idx, (probs, labels) in enumerate(zip(all_probs, all_labels)):
            # Filter by gender if specified
            if gender_filter is not None and all_genders is not None:
                if all_genders[idx] != gender_filter:
                    continue
            
            # In semantic segmentation, high internal pixel confidence violently skews
            # E[p|y] up to 0.99+, meaning fuzzy boundary errors rarely meet the threshold.
            # To capture boundary label noise, we MUST allow non-confident pixels to fall
            # back to their raw predicted argmax, instead of aggressively discarding them
            # by reverting to the noisy label.
            
            p_norm = probs / np.maximum(thresholds, 1e-8)  # (H, W, C)
            confident_preds = np.argmax(p_norm, axis=-1)   # (H, W)
            
            # Mask of pixels that actually exceed their respective threshold
            max_p_norm = np.max(p_norm, axis=-1)           # (H, W)
            is_confident = (max_p_norm >= 1.0)
            
            # Fallback to standard argmax for naturally fuzzy boundary predictions
            raw_argmax = np.argmax(probs, axis=-1)
            final_preds = np.where(is_confident, confident_preds, raw_argmax)
            
            # Update joint matrix
            for observed in range(self.num_classes):
                for predicted in range(self.num_classes):
                    mask = (labels == observed) & (final_preds == predicted)
                    joint_matrix[observed, predicted] += mask.sum()
            
            total_pixels += labels.size
        
        return joint_matrix, total_pixels
    
    def identify_label_errors(
        self,
        all_probs: List[np.ndarray],
        all_labels: List[np.ndarray],
        thresholds: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Identify pixels that are likely mislabeled.
        
        A pixel is considered mislabeled if:
        - Its observed label is i
        - Its confident prediction is j != i
        - The prediction confidence p(y=j|x) >= t_j
        
        Returns:
            error_masks: List of boolean arrays indicating mislabeled pixels
        """
        error_masks = []
        
        for probs, labels in zip(all_probs, all_labels):
            p_norm = probs / np.maximum(thresholds, 1e-8)
            confident_preds = np.argmax(p_norm, axis=2)
            
            max_p_norm = np.max(p_norm, axis=2)
            is_confident = (max_p_norm >= 1.0)
            
            raw_argmax = np.argmax(probs, axis=2)
            final_preds = np.where(is_confident, confident_preds, raw_argmax)
            
            is_error = (final_preds != labels)
            error_masks.append(is_error)
        
        return error_masks
    
    def analyze(self) -> ConfidentLearningResults:
        """
        Run the full Confident Learning analysis.
        
        Returns:
            ConfidentLearningResults with all metrics
        """
        results = ConfidentLearningResults()
        
        print("Step 1: Computing predictions...")
        all_probs, all_labels, all_genders, all_image_ids = self.compute_predictions()
        
        print("Step 2: Computing confidence thresholds...")
        thresholds = self.compute_confidence_thresholds(all_probs, all_labels)
        results.thresholds = thresholds
        print(f"  Background threshold: {thresholds[0]:.4f}")
        print(f"  Foreground threshold: {thresholds[1]:.4f}")
        
        print("Step 2b: Computing standard raw metrics (IoU/Dice)...")
        def compute_metrics_from_raw(probs_list, labels_list):
            tp = fp = fn = tn = 0
            for p, l in zip(probs_list, labels_list):
                preds = np.argmax(p, axis=-1)
                tp += ((l == 1) & (preds == 1)).sum()
                fp += ((l == 0) & (preds == 1)).sum()
                fn += ((l == 1) & (preds == 0)).sum()
                tn += ((l == 0) & (preds == 0)).sum()
            
            iou = tp / max(tp + fp + fn, 1e-8)
            dice = 2 * tp / max(2 * tp + fp + fn, 1e-8)
            return iou, dice

        iou, dice = compute_metrics_from_raw(all_probs, all_labels)
        results.iou_foreground = float(iou)
        results.dice_foreground = float(dice)

        f_probs = [p for p, g in zip(all_probs, all_genders) if g == 0]
        f_labels = [l for l, g in zip(all_labels, all_genders) if g == 0]
        f_iou, f_dice = compute_metrics_from_raw(f_probs, f_labels)
        results.female_iou_foreground = float(f_iou)
        results.female_dice_foreground = float(f_dice)

        m_probs = [p for p, g in zip(all_probs, all_genders) if g == 1]
        m_labels = [l for l, g in zip(all_labels, all_genders) if g == 1]
        m_iou, m_dice = compute_metrics_from_raw(m_probs, m_labels)
        results.male_iou_foreground = float(m_iou)
        results.male_dice_foreground = float(m_dice)

        results.iou_gap = abs(float(f_iou - m_iou))
        results.dice_gap = abs(float(f_dice - m_dice))
        print(f"  Overall IoU: {results.iou_foreground:.4f}, Dice: {results.dice_foreground:.4f}")
        print(f"  Gap IoU: {results.iou_gap:.4f}, Gap Dice: {results.dice_gap:.4f}")
        
        print("Step 3: Building joint distribution matrices...")
        
        # Overall joint matrix
        joint_matrix, total_pixels = self.build_joint_matrix(
            all_probs, all_labels, thresholds
        )
        results.joint_matrix = joint_matrix
        results.total_pixels = total_pixels
        results.joint_matrix_normalized = joint_matrix / max(total_pixels, 1)
        
        # Female joint matrix
        joint_female, female_pixels = self.build_joint_matrix(
            all_probs, all_labels, thresholds,
            gender_filter=0, all_genders=all_genders
        )
        results.joint_matrix_female = joint_female
        results.female_pixels = female_pixels
        results.joint_matrix_female_normalized = joint_female / max(female_pixels, 1)
        
        # Male joint matrix
        joint_male, male_pixels = self.build_joint_matrix(
            all_probs, all_labels, thresholds,
            gender_filter=1, all_genders=all_genders
        )
        results.joint_matrix_male = joint_male
        results.male_pixels = male_pixels
        results.joint_matrix_male_normalized = joint_male / max(male_pixels, 1)
        
        print("Step 4: Identifying label errors...")
        error_masks = self.identify_label_errors(all_probs, all_labels, thresholds)
        
        # Compute error statistics
        total_errors = sum(mask.sum() for mask in error_masks)
        results.total_errors = total_errors
        results.error_rate = total_errors / max(total_pixels, 1)
        
        # Per-gender error rates
        female_errors = sum(
            mask.sum() for mask, g in zip(error_masks, all_genders) if g == 0
        )
        male_errors = sum(
            mask.sum() for mask, g in zip(error_masks, all_genders) if g == 1
        )
        
        results.female_errors = female_errors
        results.female_error_rate = female_errors / max(female_pixels, 1)
        
        results.male_errors = male_errors
        results.male_error_rate = male_errors / max(male_pixels, 1)
        
        print("Step 5: Computing bias metrics (LER and SDR)...")
        
        norm_metrics = self.compute_sample_normalized_metrics(all_probs, all_labels, thresholds, all_genders)
        results.ler_female = norm_metrics['ler_female']
        results.ler_male = norm_metrics['ler_male']
        results.ler_gap = abs(results.ler_female - results.ler_male)
        results.rr = (results.ler_female + 1e-8) / (results.ler_male + 1e-8)
        
        results.sdr_female = norm_metrics['sdr_female']
        results.sdr_male = norm_metrics['sdr_male']
        results.sdr_gap = results.sdr_female - results.sdr_male
        
        counts = norm_metrics['counts']
        results.total_excess_errors = counts['total_excess']
        results.total_deficit_errors = counts['total_deficit']
        results.female_excess_errors = counts['female_excess']
        results.female_deficit_errors = counts['female_deficit']
        results.male_excess_errors = counts['male_excess']
        results.male_deficit_errors = counts['male_deficit']
        
        print("Step 6: Classifying noise vs bias...")
        results = self._classify_bias(results)
        
        return results

    def compute_sample_normalized_metrics(self, all_probs, all_labels, thresholds, all_genders):
        sample_ler_female = []
        sample_sdr_female = []
        sample_ler_male = []
        sample_sdr_male = []
        
        counts = {
            'total_excess': 0, 'total_deficit': 0,
            'female_excess': 0, 'female_deficit': 0,
            'male_excess': 0, 'male_deficit': 0,
        }

        for probs, labels, gender in zip(all_probs, all_labels, all_genders):
            p_norm = probs / np.maximum(thresholds, 1e-8)
            confident_preds = np.argmax(p_norm, axis=-1)
            max_p_norm = np.max(p_norm, axis=-1)
            is_confident = (max_p_norm >= 1.0)
            raw_argmax = np.argmax(probs, axis=-1)
            final_preds = np.where(is_confident, confident_preds, raw_argmax)

            # E_exc: Label=1, CL=0 (Annotation has excess foreground -> Dilation)
            e_exc = ((labels == 1) & (final_preds == 0)).sum()
            # E_def: Label=0, CL=1 (Annotation has deficit foreground -> Erosion)
            e_def = ((labels == 0) & (final_preds == 1)).sum()
            
            # E_i is total errors
            e_i = e_exc + e_def
            
            # Union of annotation and CL prediction
            union_area = ((labels == 1) | (final_preds == 1)).sum()
            denom = union_area + 1e-8
            
            ler = e_i / denom
            sdr = (e_exc - e_def) / denom
            
            counts['total_excess'] += e_exc
            counts['total_deficit'] += e_def
            
            if gender == 0:
                sample_ler_female.append(ler)
                sample_sdr_female.append(sdr)
                counts['female_excess'] += e_exc
                counts['female_deficit'] += e_def
            else:
                sample_ler_male.append(ler)
                sample_sdr_male.append(sdr)
                counts['male_excess'] += e_exc
                counts['male_deficit'] += e_def
        
        return {
            'ler_female': float(np.mean(sample_ler_female)) if sample_ler_female else 0.0,
            'sdr_female': float(np.mean(sample_sdr_female)) if sample_sdr_female else 0.0,
            'ler_male': float(np.mean(sample_ler_male)) if sample_ler_male else 0.0,
            'sdr_male': float(np.mean(sample_sdr_male)) if sample_sdr_male else 0.0,
            'counts': counts
        }

    # ------------------------------------------------------------------
    # Cross-group Confident Learning
    # ------------------------------------------------------------------

    def compute_group_thresholds(
        self,
        all_probs: List[np.ndarray],
        all_labels: List[np.ndarray],
        all_genders: List[int],
        group_id: int,
    ) -> np.ndarray:
        """Compute CL confidence thresholds from a single demographic group.

        t_j = E[p(y=j|x) | ỹ=j, group=group_id]

        This provides a "clean reference" when computed from the unbiased group.
        """
        class_prob_sums = np.zeros(self.num_classes)
        class_counts = np.zeros(self.num_classes)

        for probs, labels, gender in zip(all_probs, all_labels, all_genders):
            if gender != group_id:
                continue
            for c in range(self.num_classes):
                mask = (labels == c)
                if mask.sum() > 0:
                    class_prob_sums[c] += probs[mask, c].sum()
                    class_counts[c] += mask.sum()

        thresholds = class_prob_sums / np.maximum(class_counts, 1)
        return thresholds

    def compute_group_prior(
        self,
        all_labels: List[np.ndarray],
        all_genders: List[int],
        group_id: int,
    ) -> np.ndarray:
        """Compute the empirical marginal prior P(y) for a demographic group."""
        class_counts = np.zeros(self.num_classes)
        total_pixels = 0
        
        for labels, gender in zip(all_labels, all_genders):
            if gender == group_id:
                for c in range(self.num_classes):
                    class_counts[c] += (labels == c).sum()
                total_pixels += labels.size
                
        return class_counts / max(total_pixels, 1)

    def analyze_cross_group(
        self,
        clean_group_id: int = 1,
        biased_group_id: int = 0,
    ) -> Dict:
        """Cross-group CL: compute thresholds from clean group, apply to biased group.

        This detects label bias even when 100% of the biased group's labels
        are systematically wrong, because the thresholds are calibrated on
        clean (unbiased) data.

        Returns a dict with:
          - thresholds: {standard, clean_group, biased_group}
          - joint matrices for biased group under each threshold set
          - error rates and gaps
          - comparison metrics
        """
        print("Cross-Group CL: computing predictions ...")
        all_probs, all_labels, all_genders, _ = self.compute_predictions()

        # --- Thresholds ---
        std_thresh = self.compute_confidence_thresholds(all_probs, all_labels)
        clean_thresh = self.compute_group_thresholds(
            all_probs, all_labels, all_genders, clean_group_id)
        biased_thresh = self.compute_group_thresholds(
            all_probs, all_labels, all_genders, biased_group_id)

        print(f"  Standard thresholds:     BG={std_thresh[0]:.4f}  FG={std_thresh[1]:.4f}")
        print(f"  Clean-group thresholds:  BG={clean_thresh[0]:.4f}  FG={clean_thresh[1]:.4f}")
        print(f"  Biased-group thresholds: BG={biased_thresh[0]:.4f}  FG={biased_thresh[1]:.4f}")

        # --- Priors ---
        clean_prior = self.compute_group_prior(all_labels, all_genders, clean_group_id)
        biased_prior = self.compute_group_prior(all_labels, all_genders, biased_group_id)
        print(f"  Clean-group Priors:      BG={clean_prior[0]:.4f}  FG={clean_prior[1]:.4f}")
        print(f"  Biased-group Priors:     BG={biased_prior[0]:.4f}  FG={biased_prior[1]:.4f}")

        results = {
            'thresholds_standard': std_thresh.tolist(),
            'thresholds_clean_group': clean_thresh.tolist(),
            'thresholds_biased_group': biased_thresh.tolist(),
            'prior_clean_group': clean_prior.tolist(),
            'prior_biased_group': biased_prior.tolist(),
            'clean_group_id': int(clean_group_id),
            'biased_group_id': int(biased_group_id),
        }

        # --- Probability Calibration ---
        calibrated_probs = []
        prior_ratio = clean_prior / np.maximum(biased_prior, 1e-8)
        
        for probs, gender in zip(all_probs, all_genders):
            if gender == biased_group_id:
                p_calib = probs * prior_ratio[None, None, :]
                p_calib = p_calib / np.maximum(p_calib.sum(axis=-1, keepdims=True), 1e-8)
                calibrated_probs.append(p_calib)
            else:
                calibrated_probs.append(probs)

        # --- Joint matrices for biased group under different thresholds ---
        evaluation_variants = [
            ('std', std_thresh, all_probs),
            ('clean_ref', clean_thresh, all_probs),
            ('own', biased_thresh, all_probs),
            ('clean_ref_calib', clean_thresh, calibrated_probs),
        ]

        for thresh_name, thresh, probs_to_use in evaluation_variants:
            joint, n_pixels = self.build_joint_matrix(
                probs_to_use, all_labels, thresh,
                gender_filter=biased_group_id, all_genders=all_genders)
            joint_norm = joint / max(n_pixels, 1)

            omission = joint_norm[0, 1]     # label=BG, pred=FG (missing FG)
            commission = joint_norm[1, 0]   # label=FG, pred=BG (excess FG)
            off_diag = omission + commission
            results[f'biased_{thresh_name}_pixels'] = int(n_pixels)
            results[f'biased_{thresh_name}_omission'] = float(omission)
            results[f'biased_{thresh_name}_commission'] = float(commission)
            results[f'biased_{thresh_name}_error_rate'] = float(off_diag)
            results[f'biased_{thresh_name}_joint'] = joint_norm.tolist()

        # --- Same for clean group (as reference) ---
        for thresh_name, thresh in [
            ('std', std_thresh),
            ('own', clean_thresh),
        ]:
            joint, n_pixels = self.build_joint_matrix(
                all_probs, all_labels, thresh,
                gender_filter=clean_group_id, all_genders=all_genders)
            joint_norm = joint / max(n_pixels, 1)
            omission = joint_norm[0, 1]
            commission = joint_norm[1, 0]
            off_diag = omission + commission
            results[f'clean_{thresh_name}_pixels'] = int(n_pixels)
            results[f'clean_{thresh_name}_omission'] = float(omission)
            results[f'clean_{thresh_name}_commission'] = float(commission)
            results[f'clean_{thresh_name}_error_rate'] = float(off_diag)

        # --- Key comparison metrics ---
        # Excess errors detected by cross-group thresholds vs own thresholds
        biased_own_err = results['biased_own_error_rate']
        biased_xref_err = results['biased_clean_ref_error_rate']
        biased_xref_calib_err = results['biased_clean_ref_calib_error_rate']
        clean_own_err = results['clean_own_error_rate']

        results['cross_group_excess_errors'] = float(biased_xref_err - biased_own_err)
        results['cross_group_calib_excess_errors'] = float(biased_xref_calib_err - biased_own_err)
        
        results['cross_group_omission_excess'] = float(
            results['biased_clean_ref_omission'] - results['biased_own_omission'])
        results['cross_group_calib_omission_excess'] = float(
            results['biased_clean_ref_calib_omission'] - results['biased_own_omission'])
            
        results['cross_group_commission_excess'] = float(
            results['biased_clean_ref_commission'] - results['biased_own_commission'])
        results['cross_group_calib_commission_excess'] = float(
            results['biased_clean_ref_calib_commission'] - results['biased_own_commission'])

        # Bias score: how much worse is biased group (with clean ref) vs clean group
        results['cross_group_bias_gap'] = float(biased_xref_err - clean_own_err)
        results['cross_group_omission_gap'] = float(
            results['biased_clean_ref_omission'] - results['clean_own_omission'])
        results['cross_group_commission_gap'] = float(
            results['biased_clean_ref_commission'] - results['clean_own_commission'])

        # Is bias detected?
        gap = results['cross_group_bias_gap']
        results['cross_group_bias_detected'] = bool(gap > self.bias_threshold)
        results['cross_group_bias_severity'] = (
            'high' if gap > 0.05 else
            'medium' if gap > 0.02 else
            'low' if gap > 0.01 else 'none'
        )

        # Print summary
        print("\n  Cross-Group CL Summary:")
        print(f"    Biased group (own thresholds):       "
              f"omit={results['biased_own_omission']*100:.2f}%  "
              f"comm={results['biased_own_commission']*100:.2f}%  "
              f"total={biased_own_err*100:.2f}%")
        print(f"    Biased group (clean-ref thresholds): "
              f"omit={results['biased_clean_ref_omission']*100:.2f}%  "
              f"comm={results['biased_clean_ref_commission']*100:.2f}%  "
              f"total={biased_xref_err*100:.2f}%")
        print(f"    Clean group  (own thresholds):       "
              f"omit={results['clean_own_omission']*100:.2f}%  "
              f"comm={results['clean_own_commission']*100:.2f}%  "
              f"total={clean_own_err*100:.2f}%")
        print(f"    Excess errors (cross-ref − own): {results['cross_group_excess_errors']*100:.2f}%")
        print(f"    Bias gap (biased_xref − clean):  {gap*100:.2f}%  "
              f"→ {'BIAS DETECTED' if results['cross_group_bias_detected'] else 'no bias'} "
              f"({results['cross_group_bias_severity']})")

        return results
    
    def _perform_statistical_tests(self, results: ConfidentLearningResults) -> ConfidentLearningResults:
        """
        Perform statistical tests to determine if errors are structured (bias) vs random (noise).
        
        Tests performed:
        1. Chi-square test: Are error patterns independent of gender?
        2. Odds ratio: How much more likely is one gender to have errors?
        3. Relative risk: Risk ratio between groups
        4. Symmetry analysis: Are errors symmetric across groups?
        """
        # Extract counts from joint matrices
        # joint_matrix[i,j] = count of pixels with label i and prediction j
        # [0,0]=TN, [0,1]=Commission, [1,0]=Omission, [1,1]=TP
        
        female_omission = int(results.joint_matrix_female[1, 0])  # Label=FG, Pred=BG
        female_commission = int(results.joint_matrix_female[0, 1])  # Label=BG, Pred=FG
        female_correct = int(results.joint_matrix_female[0, 0] + results.joint_matrix_female[1, 1])
        
        male_omission = int(results.joint_matrix_male[1, 0])
        male_commission = int(results.joint_matrix_male[0, 1])
        male_correct = int(results.joint_matrix_male[0, 0] + results.joint_matrix_male[1, 1])
        
        # 1. Chi-square test for independence
        # H0: Error distribution is independent of gender (noise)
        # H1: Error distribution depends on gender (bias)
        contingency_table = np.array([
            [female_omission, female_commission, female_correct],
            [male_omission, male_commission, male_correct]
        ])
        
        try:
            chi2, pvalue, dof, expected = stats.chi2_contingency(contingency_table)
            results.chi2_statistic = chi2
            results.chi2_pvalue = pvalue
        except:
            results.chi2_statistic = 0.0
            results.chi2_pvalue = 1.0
        
        # 2. Odds Ratio for omission errors
        # OR > 1 means female more likely to have omission errors
        # Contingency: [[female_omission, female_no_omission], [male_omission, male_no_omission]]
        female_fg_total = int(results.joint_matrix_female[1, 0] + results.joint_matrix_female[1, 1])
        male_fg_total = int(results.joint_matrix_male[1, 0] + results.joint_matrix_male[1, 1])
        
        female_no_omission = female_fg_total - female_omission
        male_no_omission = male_fg_total - male_omission
        
        # Avoid division by zero
        if male_omission > 0 and female_no_omission > 0 and male_no_omission > 0:
            results.odds_ratio_omission = (female_omission * male_no_omission) / (male_omission * female_no_omission)
        else:
            results.odds_ratio_omission = 1.0
        
        # 3. Odds Ratio for commission errors
        female_bg_total = int(results.joint_matrix_female[0, 0] + results.joint_matrix_female[0, 1])
        male_bg_total = int(results.joint_matrix_male[0, 0] + results.joint_matrix_male[0, 1])
        
        female_no_commission = female_bg_total - female_commission
        male_no_commission = male_bg_total - male_commission
        
        if male_commission > 0 and female_no_commission > 0 and male_no_commission > 0:
            results.odds_ratio_commission = (female_commission * male_no_commission) / (male_commission * female_no_commission)
        else:
            results.odds_ratio_commission = 1.0
        
        # 4. Relative Risk (easier to interpret than odds ratio)
        # RR = P(omission | female) / P(omission | male)
        if results.omission_rate_male > 0:
            results.relative_risk_omission = results.omission_rate_female / results.omission_rate_male
        else:
            results.relative_risk_omission = 1.0 if results.omission_rate_female == 0 else float('inf')
        
        if results.commission_rate_male > 0:
            results.relative_risk_commission = results.commission_rate_female / results.commission_rate_male
        else:
            results.relative_risk_commission = 1.0 if results.commission_rate_female == 0 else float('inf')
        
        # 5. Error symmetry score
        # If errors are pure noise, female and male error rates should be similar
        # Score close to 1 = symmetric (noise), close to 0 = asymmetric (bias)
        total_omission = female_omission + male_omission
        total_commission = female_commission + male_commission
        
        if total_omission > 0:
            female_omission_share = female_omission / total_omission
            # Expected share based on pixel count
            expected_female_share = female_fg_total / (female_fg_total + male_fg_total) if (female_fg_total + male_fg_total) > 0 else 0.5
            omission_symmetry = 1 - abs(female_omission_share - expected_female_share) * 2
        else:
            omission_symmetry = 1.0
        
        if total_commission > 0:
            female_commission_share = female_commission / total_commission
            expected_female_share = female_bg_total / (female_bg_total + male_bg_total) if (female_bg_total + male_bg_total) > 0 else 0.5
            commission_symmetry = 1 - abs(female_commission_share - expected_female_share) * 2
        else:
            commission_symmetry = 1.0
        
        results.error_symmetry_score = min(omission_symmetry, commission_symmetry)
        
        return results
    
    def _classify_bias(self, results: ConfidentLearningResults) -> ConfidentLearningResults:
        # Simple classification based on LER gap and SDR
        if results.ler_gap > self.bias_threshold:
            results.is_biased = True
            results.bias_direction = "female_disadvantaged" if results.ler_female > results.ler_male else "male_disadvantaged"
            
            # Use SDR to determine type
            if abs(results.sdr_female) > 0.01:
                results.bias_type = "dilation (excess)" if results.sdr_female > 0 else "erosion (deficit)"
            else:
                results.bias_type = "mixed"
                
            results.bias_confidence = "high" if results.ler_gap > 0.05 else "medium"
        else:
            results.is_biased = False
            results.bias_direction = "none"
            results.bias_type = "none"
            results.bias_confidence = "none"
        
        return results
    
    def print_results(self, results: ConfidentLearningResults):
        """Print a formatted summary of results."""
        print("\n" + "="*70)
        print("CONFIDENT LEARNING ANALYSIS RESULTS")
        print("="*70)
        
        print("\n--- Confidence Thresholds ---")
        print(f"  Background (class 0): {results.thresholds[0]:.4f}")
        print(f"  Foreground (class 1): {results.thresholds[1]:.4f}")
        
        print("\n--- Joint Distribution Matrix Q(ỹ, y*) ---")
        print("  Rows: Observed label (ỹ), Columns: Predicted true label (y*)")
        print(f"\n  Overall ({results.total_pixels:,} pixels):")
        print(f"                    Pred=BG      Pred=FG")
        print(f"    Label=BG    {results.joint_matrix_normalized[0,0]*100:8.2f}%  {results.joint_matrix_normalized[0,1]*100:8.2f}%")
        print(f"    Label=FG    {results.joint_matrix_normalized[1,0]*100:8.2f}%  {results.joint_matrix_normalized[1,1]*100:8.2f}%")
        
        print(f"\n  Female ({results.female_pixels:,} pixels):")
        print(f"                    Pred=BG      Pred=FG")
        print(f"    Label=BG    {results.joint_matrix_female_normalized[0,0]*100:8.2f}%  {results.joint_matrix_female_normalized[0,1]*100:8.2f}%")
        print(f"    Label=FG    {results.joint_matrix_female_normalized[1,0]*100:8.2f}%  {results.joint_matrix_female_normalized[1,1]*100:8.2f}%")
        
        print(f"\n  Male ({results.male_pixels:,} pixels):")
        print(f"                    Pred=BG      Pred=FG")
        print(f"    Label=BG    {results.joint_matrix_male_normalized[0,0]*100:8.2f}%  {results.joint_matrix_male_normalized[0,1]*100:8.2f}%")
        print(f"    Label=FG    {results.joint_matrix_male_normalized[1,0]*100:8.2f}%  {results.joint_matrix_male_normalized[1,1]*100:8.2f}%")
        
        print("\n--- Label Error Detection ---")
        print(f"  Total pixels analyzed: {results.total_pixels:,}")
        print(f"  Total mislabeled pixels: {results.total_errors:,} ({results.error_rate*100:.2f}%)")
        print(f"  Female mislabeled: {results.female_errors:,} ({results.female_error_rate*100:.2f}%)")
        print(f"  Male mislabeled: {results.male_errors:,} ({results.male_error_rate*100:.2f}%)")
        
        print("\n--- Bias Analysis ---")
        print(f"  Omission Rate (true FG labeled as BG):")
        print(f"    Female: {results.omission_rate_female*100:.4f}%")
        print(f"    Male:   {results.omission_rate_male*100:.4f}%")
        print(f"    Gap:    {results.omission_bias_gap*100:.4f}%")
        
        print(f"\n  Commission Rate (true BG labeled as FG):")
        print(f"    Female: {results.commission_rate_female*100:.4f}%")
        print(f"    Male:   {results.commission_rate_male*100:.4f}%")
        print(f"    Gap:    {results.commission_bias_gap*100:.4f}%")
        
        print("\n--- Statistical Tests ---")
        print(f"  Chi-square test (H0: errors independent of gender):")
        print(f"    Statistic: {results.chi2_statistic:.2f}")
        print(f"    P-value: {results.chi2_pvalue:.4f} {'***' if results.chi2_pvalue < 0.001 else '**' if results.chi2_pvalue < 0.01 else '*' if results.chi2_pvalue < 0.05 else ''}")
        print(f"    Interpretation: {'Errors DEPEND on gender' if results.chi2_pvalue < 0.05 else 'Errors are INDEPENDENT of gender'}")
        
        print(f"\n  Odds Ratio (OR > 1 means female more likely to have error):")
        print(f"    Omission OR: {results.odds_ratio_omission:.3f}")
        print(f"    Commission OR: {results.odds_ratio_commission:.3f}")
        
        print(f"\n  Relative Risk (RR = female rate / male rate):")
        print(f"    Omission RR: {results.relative_risk_omission:.3f} {'(Female {:.0f}x more likely)'.format(results.relative_risk_omission) if results.relative_risk_omission > 1 else '(Male {:.0f}x more likely)'.format(1/results.relative_risk_omission) if results.relative_risk_omission < 1 else '(Equal)'}")
        print(f"    Commission RR: {results.relative_risk_commission:.3f}")
        
        print(f"\n  Error Symmetry Score: {results.error_symmetry_score:.3f}")
        print(f"    (1.0 = perfectly symmetric/noise, 0.0 = completely asymmetric/bias)")
        
        print("\n--- Conclusion ---")
        if results.is_biased:
            print(f"  ⚠️  BIAS DETECTED (Confidence: {results.bias_confidence.upper()})")
            print(f"  Type: {results.bias_type.upper()}")
            print(f"  Direction: {results.bias_direction.replace('_', ' ').title()}")
            if results.bias_type == "omission":
                disadvantaged = "Female" if results.bias_direction == "female_disadvantaged" else "Male"
                rate = results.omission_rate_female if results.bias_direction == "female_disadvantaged" else results.omission_rate_male
                rr = results.relative_risk_omission if results.bias_direction == "female_disadvantaged" else 1/results.relative_risk_omission
                print(f"  {disadvantaged} samples have {rate*100:.2f}% of true foreground mislabeled as background")
                print(f"  {disadvantaged} is {rr:.1f}x more likely to have omission errors")
            else:
                disadvantaged = "Female" if results.bias_direction == "female_disadvantaged" else "Male"
                rate = results.commission_rate_female if results.bias_direction == "female_disadvantaged" else results.commission_rate_male
                rr = results.relative_risk_commission if results.bias_direction == "female_disadvantaged" else 1/results.relative_risk_commission
                print(f"  {disadvantaged} samples have {rate*100:.2f}% of true background mislabeled as foreground")
                print(f"  {disadvantaged} is {rr:.1f}x more likely to have commission errors")
            
            print(f"\n  Evidence:")
            print(f"    - Chi-square p-value: {results.chi2_pvalue:.4f} {'(significant)' if results.chi2_pvalue < 0.05 else '(not significant)'}")
            print(f"    - Odds ratio: {max(results.odds_ratio_omission, results.odds_ratio_commission):.2f} {'(significant)' if max(results.odds_ratio_omission, results.odds_ratio_commission) > 1.5 else '(not significant)'}")
            print(f"    - Symmetry: {results.error_symmetry_score:.2f} {'(asymmetric)' if results.error_symmetry_score < 0.8 else '(symmetric)'}")
            print(f"    - Gap: {max(results.omission_bias_gap, results.commission_bias_gap)*100:.2f}% {'(significant)' if max(results.omission_bias_gap, results.commission_bias_gap) > self.bias_threshold else '(not significant)'}")
        else:
            print(f"  ✓ No significant bias detected")
            print(f"  Label errors appear to be UNIFORM NOISE across demographics")
            print(f"\n  Evidence:")
            print(f"    - Chi-square p-value: {results.chi2_pvalue:.4f} (errors independent of gender)")
            print(f"    - Symmetry score: {results.error_symmetry_score:.2f} (errors are symmetric)")
            print(f"    - Gaps are within threshold ({self.bias_threshold*100:.1f}%)")
        
        print("="*70 + "\n")


def create_model(model_name: str, encoder: str, num_classes: int = 2):
    """Create segmentation model."""
    model_fn = {
        'unet': smp.Unet,
        'unetpp': smp.UnetPlusPlus,
        'deeplabv3': smp.DeepLabV3,
        'deeplabv3p': smp.DeepLabV3Plus,
        'fpn': smp.FPN,
        'pspnet': smp.PSPNet,
    }[model_name]
    
    return model_fn(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
    )


def plot_joint_matrices(results: ConfidentLearningResults, save_path: Path):
    """Create visualization of joint distribution matrices."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    labels = ['Background', 'Foreground']
    
    # Overall
    sns.heatmap(
        results.joint_matrix_normalized * 100,
        annot=True, fmt='.2f', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        ax=axes[0], cbar_kws={'label': '%'}
    )
    axes[0].set_title(f'Overall\n({results.total_pixels:,} pixels)')
    axes[0].set_xlabel('Predicted True Label (y*)')
    axes[0].set_ylabel('Observed Label (ỹ)')
    
    # Female
    sns.heatmap(
        results.joint_matrix_female_normalized * 100,
        annot=True, fmt='.2f', cmap='Purples',
        xticklabels=labels, yticklabels=labels,
        ax=axes[1], cbar_kws={'label': '%'}
    )
    axes[1].set_title(f'Female\n({results.female_pixels:,} pixels)')
    axes[1].set_xlabel('Predicted True Label (y*)')
    axes[1].set_ylabel('Observed Label (ỹ)')
    
    # Male
    sns.heatmap(
        results.joint_matrix_male_normalized * 100,
        annot=True, fmt='.2f', cmap='Greens',
        xticklabels=labels, yticklabels=labels,
        ax=axes[2], cbar_kws={'label': '%'}
    )
    axes[2].set_title(f'Male\n({results.male_pixels:,} pixels)')
    axes[2].set_xlabel('Predicted True Label (y*)')
    axes[2].set_ylabel('Observed Label (ỹ)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Joint matrix visualization saved to: {save_path}")


def plot_bias_comparison(results: ConfidentLearningResults, save_path: Path):
    """Create bar chart comparing error rates by gender."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Omission rates
    x = ['Female', 'Male']
    omission = [results.omission_rate_female * 100, results.omission_rate_male * 100]
    colors = ['#E57373', '#64B5F6']
    
    bars1 = axes[0].bar(x, omission, color=colors, edgecolor='black')
    axes[0].set_ylabel('Omission Rate (%)')
    axes[0].set_title(f'Omission Bias\n(True FG labeled as BG)\nGap: {results.omission_bias_gap*100:.2f}%')
    axes[0].bar_label(bars1, fmt='%.2f%%')
    
    # Commission rates
    commission = [results.commission_rate_female * 100, results.commission_rate_male * 100]
    
    bars2 = axes[1].bar(x, commission, color=colors, edgecolor='black')
    axes[1].set_ylabel('Commission Rate (%)')
    axes[1].set_title(f'Commission Bias\n(True BG labeled as FG)\nGap: {results.commission_bias_gap*100:.2f}%')
    axes[1].bar_label(bars2, fmt='%.2f%%')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Bias comparison plot saved to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Confident Learning analysis for label error detection'
    )

    # Dataset selection
    add_dataset_args(parser)

    # Experiment
    parser.add_argument('--exp_dir', type=str, required=True,
                       help='Path to experiment directory (e.g., /path/to/output/experiments/binary_seg_cv)')
    parser.add_argument('--fold', type=int, default=0,
                       help='Which fold to analyze')

    # Data
    parser.add_argument('--data_root', type=str,
                       default='/path/to/CelebAMask-HQ')
    parser.add_argument('--splits_path', type=str,
                       default='configs/splits/cv_splits.json')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)

    # Analysis
    parser.add_argument('--eval_set', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='Which set to analyze. Use "train" to detect label errors in training labels.')
    parser.add_argument('--bias_threshold', type=float, default=0.02,
                       help='Threshold for declaring bias (default 2%%)')

    # Bias injection params (needed to reconstruct biased training labels)
    parser.add_argument('--bias_ratio', type=float, default=0.0,
                       help='Bias ratio used during training (0.0 for baseline)')
    parser.add_argument('--erosion_radius', type=int, default=5,
                       help='Erosion radius used during training')
    parser.add_argument('--biased_gender', type=int, default=0,
                       help='Gender that was biased (0=Female, 1=Male)')
    parser.add_argument('--bias_seed', type=int, default=42,
                       help='Bias seed used during training')

    # Model (will try to load from config)
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--encoder', type=str, default='resnet34')

    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: exp_dir/confident_learning)')

    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    exp_dir = Path(args.exp_dir)
    fold_dir = exp_dir / f'fold_{args.fold}'

    # Try to load config from experiment (check both config.json and args.json)
    config_path = fold_dir / 'config.json'
    if not config_path.exists():
        config_path = fold_dir / 'args.json'

    if config_path.exists():
        with open(config_path, 'r') as f:
            exp_config = json.load(f)
        args.model = exp_config.get('model', args.model)
        args.encoder = exp_config.get('encoder', args.encoder)
        args.img_size = exp_config.get('img_size', args.img_size)
        # Auto-detect dataset from experiment config
        if is_phc_experiment(exp_config):
            args.dataset = 'phc'
        else:
            args.dataset = exp_config.get('dataset', 'celebamask')
        print(f"Loaded config from {config_path}")

    apply_dataset_defaults(args)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load model
    checkpoint_path = fold_dir / 'best_model.pt'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}")
    model = create_model(args.model, args.encoder, num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get('model_state_dict', checkpoint)
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
    model = model.to(device)
    model.eval()

    # Load splits
    splits = create_splits(args)
    train_indices, val_indices, test_indices = get_fold_indices(splits, args.fold)

    # Select which set to evaluate
    if args.eval_set == 'train':
        eval_indices = train_indices
    elif args.eval_set == 'val':
        eval_indices = val_indices
    else:
        eval_indices = test_indices

    # Create dataset using factory
    bias_ratio = getattr(args, 'bias_ratio', 0.0)
    if args.eval_set == 'train' and (bias_ratio > 0 or args.dataset == 'phc'):
        dataset = create_biased_eval_dataset(args, eval_indices)
        print(f"\nUsing BIASED labels for {args.dataset}")
        if hasattr(dataset, 'get_bias_summary'):
            print(dataset.get_bias_summary())
    else:
        dataset = create_clean_eval_dataset(args, eval_indices)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"\nAnalyzing {args.eval_set} set: {len(dataset)} samples")

    # Run analysis
    analyzer = ConfidentLearningAnalyzer(
        model=model,
        dataloader=dataloader,
        device=device,
        num_classes=2,
        bias_threshold=args.bias_threshold
    )

    results = analyzer.analyze()
    analyzer.print_results(results)

    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else fold_dir / 'confident_learning'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    with open(output_dir / 'cl_results.json', 'w') as f:
        json.dump(results.to_dict(), f, indent=2)
    print(f"Results saved to: {output_dir / 'cl_results.json'}")

    # Save visualizations
    plot_joint_matrices(results, output_dir / 'joint_matrices.png')
    plot_bias_comparison(results, output_dir / 'bias_comparison.png')

    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
