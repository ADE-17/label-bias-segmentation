"""
Segmentation metrics with gender-based analysis.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class SegmentationMetrics:
    """
    Accumulator for segmentation metrics.
    Tracks metrics overall and by gender group.
    """
    num_classes: int = 2
    
    # Accumulators
    total_intersection: np.ndarray = field(default=None)
    total_union: np.ndarray = field(default=None)
    total_correct: int = 0
    total_pixels: int = 0
    
    # Gender-specific accumulators
    gender_intersection: Dict[int, np.ndarray] = field(default_factory=dict)
    gender_union: Dict[int, np.ndarray] = field(default_factory=dict)
    gender_correct: Dict[int, int] = field(default_factory=dict)
    gender_pixels: Dict[int, int] = field(default_factory=dict)
    
    # Loss tracking
    total_loss: float = 0.0
    num_batches: int = 0
    gender_loss: Dict[int, float] = field(default_factory=dict)
    gender_batches: Dict[int, int] = field(default_factory=dict)
    
    def __post_init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulators."""
        self.total_intersection = np.zeros(self.num_classes)
        self.total_union = np.zeros(self.num_classes)
        self.total_correct = 0
        self.total_pixels = 0
        self.total_loss = 0.0
        self.num_batches = 0
        
        for g in [0, 1]:  # Female, Male
            self.gender_intersection[g] = np.zeros(self.num_classes)
            self.gender_union[g] = np.zeros(self.num_classes)
            self.gender_correct[g] = 0
            self.gender_pixels[g] = 0
            self.gender_loss[g] = 0.0
            self.gender_batches[g] = 0
    
    def update(
        self, 
        predictions: torch.Tensor,
        targets: torch.Tensor,
        genders: torch.Tensor,
        loss: Optional[float] = None,
        per_sample_loss: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with a batch of predictions.
        
        Args:
            predictions: (B, H, W) predicted class indices
            targets: (B, H, W) ground truth class indices
            genders: (B,) gender labels (0=Female, 1=Male)
            loss: Optional batch loss value
            per_sample_loss: Optional (B,) per-sample loss values
        """
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        genders = genders.cpu().numpy()
        
        batch_size = predictions.shape[0]
        
        # Update loss tracking
        if loss is not None:
            self.total_loss += loss
            self.num_batches += 1
        
        for i in range(batch_size):
            pred = predictions[i]
            target = targets[i]
            gender = int(genders[i])
            
            # Per-sample loss by gender
            if per_sample_loss is not None:
                sample_loss = per_sample_loss[i].item()
                self.gender_loss[gender] += sample_loss
                self.gender_batches[gender] += 1
            
            # Pixel accuracy
            correct = (pred == target).sum()
            n_pixels = pred.size
            
            self.total_correct += correct
            self.total_pixels += n_pixels
            self.gender_correct[gender] += correct
            self.gender_pixels[gender] += n_pixels
            
            # IoU per class
            for c in range(self.num_classes):
                pred_c = (pred == c)
                target_c = (target == c)
                
                intersection = (pred_c & target_c).sum()
                union = (pred_c | target_c).sum()
                
                self.total_intersection[c] += intersection
                self.total_union[c] += union
                self.gender_intersection[gender][c] += intersection
                self.gender_union[gender][c] += union
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        results = {}
        
        # Overall pixel accuracy
        results['pixel_acc'] = self.total_correct / max(self.total_pixels, 1)
        
        # Overall IoU per class
        iou_per_class = self.total_intersection / np.maximum(self.total_union, 1)
        results['mean_iou'] = np.mean(iou_per_class)
        results['iou_background'] = iou_per_class[0]
        results['iou_foreground'] = iou_per_class[1] if self.num_classes > 1 else 0.0
        
        # Dice coefficient: 2*I / (I + U)  where U = |pred ∪ target|
        dice_per_class = 2 * self.total_intersection / np.maximum(
            self.total_intersection + self.total_union, 1
        )
        results['mean_dice'] = np.mean(dice_per_class)
        results['dice_foreground'] = dice_per_class[1] if self.num_classes > 1 else 0.0
        
        # Average loss
        if self.num_batches > 0:
            results['loss'] = self.total_loss / self.num_batches
        
        # Gender-specific metrics
        for gender, gender_name in [(0, 'female'), (1, 'male')]:
            prefix = f'{gender_name}_'
            
            # Pixel accuracy
            results[f'{prefix}pixel_acc'] = (
                self.gender_correct[gender] / max(self.gender_pixels[gender], 1)
            )
            
            # IoU
            gender_iou = self.gender_intersection[gender] / np.maximum(
                self.gender_union[gender], 1
            )
            results[f'{prefix}mean_iou'] = np.mean(gender_iou)
            results[f'{prefix}iou_foreground'] = gender_iou[1] if self.num_classes > 1 else 0.0
            
            # Dice
            gender_dice = 2 * self.gender_intersection[gender] / np.maximum(
                self.gender_intersection[gender] + self.gender_union[gender], 1
            )
            results[f'{prefix}dice_foreground'] = gender_dice[1] if self.num_classes > 1 else 0.0
            
            # Loss
            if self.gender_batches[gender] > 0:
                results[f'{prefix}loss'] = (
                    self.gender_loss[gender] / self.gender_batches[gender]
                )
        
        # Fairness gap (difference between genders)
        results['iou_gap'] = abs(
            results.get('male_iou_foreground', 0) - results.get('female_iou_foreground', 0)
        )
        results['dice_gap'] = abs(
            results.get('male_dice_foreground', 0) - results.get('female_dice_foreground', 0)
        )
        
        return results


def compute_metrics_by_gender(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    genders: torch.Tensor,
    num_classes: int = 2
) -> Dict[str, Dict[str, float]]:
    """
    Compute segmentation metrics separated by gender.
    
    Args:
        predictions: (B, H, W) predicted class indices
        targets: (B, H, W) ground truth class indices  
        genders: (B,) gender labels
        num_classes: Number of segmentation classes
        
    Returns:
        Dictionary with 'overall', 'female', 'male' metrics
    """
    metrics = SegmentationMetrics(num_classes=num_classes)
    metrics.update(predictions, targets, genders)
    return metrics.compute()


def dice_loss(pred_logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Compute Dice loss for binary segmentation.
    
    Args:
        pred_logits: (B, C, H, W) raw logits from model
        targets: (B, H, W) ground truth class indices
        smooth: Smoothing factor
        
    Returns:
        Scalar dice loss
    """
    pred_probs = F.softmax(pred_logits, dim=1)
    
    # For binary, use foreground class
    pred_fg = pred_probs[:, 1]  # (B, H, W)
    target_fg = (targets == 1).float()  # (B, H, W)
    
    intersection = (pred_fg * target_fg).sum(dim=(1, 2))
    union = pred_fg.sum(dim=(1, 2)) + target_fg.sum(dim=(1, 2))
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return 1.0 - dice.mean()


def _morphological_gradient(x: torch.Tensor, width: int) -> torch.Tensor:
    width = int(width)
    if width < 1:
        raise ValueError("boundary width must be >= 1")

    k = 2 * width + 1

    dil = F.max_pool2d(x, kernel_size=k, stride=1, padding=width)
    ero = -F.max_pool2d(-x, kernel_size=k, stride=1, padding=width)

    grad = (dil - ero).clamp(min=0.0)
    return grad


def boundary_dice_loss(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    width: int = 2,
    smooth: float = 1.0,
    eps: float = 1e-6,
    detach_boundary: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:

    # --- probs ---
    if pred_logits.shape[1] == 1:
        probs = torch.sigmoid(pred_logits)
    else:
        probs = F.softmax(pred_logits, dim=1)[:, 1:2]

    tgt = (targets == 1).float().unsqueeze(1)

    # --- boundary maps ---
    if detach_boundary:
        pb = _morphological_gradient(probs.detach(), width)
    else:
        pb = _morphological_gradient(probs, width)

    tb = _morphological_gradient(tgt, width)

    # --- optional normalization (stabilizes training) ---
    # pb = pb / (pb.sum(dim=(2, 3), keepdim=True) + eps)
    # tb = tb / (tb.sum(dim=(2, 3), keepdim=True) + eps)

    # --- Dice ---
    inter = (pb * tb).sum(dim=(1, 2, 3))
    denom = pb.sum(dim=(1, 2, 3)) + tb.sum(dim=(1, 2, 3))

    dice = (2.0 * inter + smooth) / (denom + smooth + eps)
    per_sample = 1.0 - dice

    return per_sample.mean(), per_sample


def generalized_cross_entropy(
    pred_logits: torch.Tensor, 
    targets: torch.Tensor, 
    q: float = 0.7, 
    class_weights: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Cross Entropy Loss for noisy labels (Zhang et al. 2018).
    L_q = (1 - p_j^q) / q
    
    Args:
        pred_logits: (B, C, H, W) raw logits
        targets: (B, H, W) ground truth
        q: Robustness parameter (0, 1]. q->0 is CE, q->1 is MAE.
    """
    pred_probs = F.softmax(pred_logits, dim=1) # (B, C, H, W)
    targets_unsqueeze = targets.unsqueeze(1) # (B, 1, H, W)
    
    # Gather the probability of the target class
    p_j = torch.gather(pred_probs, 1, targets_unsqueeze).squeeze(1) # (B, H, W)
    
    # GCE formula: (1 - p^q)/q
    # Add epsilon for numerical stability if p_j is 0
    loss_per_pixel = (1.0 - torch.pow(p_j + 1e-6, q)) / q
    
    if class_weights is not None:
        weights = class_weights[targets]
        loss_per_pixel = loss_per_pixel * weights
        
    loss_per_sample = loss_per_pixel.mean(dim=(1, 2))
    return loss_per_sample.mean(), loss_per_sample


def bootstrapping_loss(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 0.8,
    class_weights: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Soft Bootstrapping Loss (Reed et al., ICLR 2015).
    Blends the noisy label with the model's prediction.
    L = beta * CE(pred, noisy_label) + (1 - beta) * CE(pred, pred_probs)
    
    Args:
        pred_logits: (B, C, H, W) raw logits
        targets: (B, H, W) ground truth
        beta: Weight for the noisy label (typically 0.8 to 0.95)
    """
    pred_probs = F.softmax(pred_logits, dim=1)
    log_probs = F.log_softmax(pred_logits, dim=1)
    
    # CE loss with original noisy targets
    ce_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    ce_loss_noisy = ce_loss_fn(pred_logits, targets)  # (B, H, W)
    
    # Entropy term: CE(pred_probs, pred_probs)
    if class_weights is not None:
        w = class_weights.view(1, -1, 1, 1)
        entropy = -(w * pred_probs * log_probs).sum(dim=1)
    else:
        entropy = -(pred_probs * log_probs).sum(dim=1)
        
    loss_per_pixel = beta * ce_loss_noisy + (1.0 - beta) * entropy
    loss_per_sample = loss_per_pixel.mean(dim=(1, 2))
    
    return loss_per_sample.mean(), loss_per_sample


def combined_loss(
    pred_logits: torch.Tensor, 
    targets: torch.Tensor,
    ce_weight: float = 0.5,
    dice_weight: float = 0.5,
    class_weights: Optional[torch.Tensor] = None,
    loss_mode: str = "ce_dice",
    boundary_width: int = 2,
    boundary_weight: float = 0.5,
    gce_q: float = 0.7,
    bootstrapping_beta: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combined Cross-Entropy and Dice loss.
    
    Args:
        pred_logits: (B, C, H, W) raw logits
        targets: (B, H, W) ground truth
        ce_weight: Weight for cross-entropy loss
        dice_weight: Weight for dice loss
        class_weights: Optional class weights for CE loss
        
    Returns:
        total_loss, per_sample_ce_loss (for gender tracking)
    """
    loss_mode = str(loss_mode)
    valid_modes = (
        "ce_dice", "ce", "dice", "boundary_dice", "dice_boundary", 
        "gce", "gce_dice", "bootstrapping", "bootstrapping_dice"
    )
    if loss_mode not in valid_modes:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")

    # Cross-entropy (per-pixel, then average over spatial dims per sample)
    ce_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    ce_loss_per_pixel = ce_loss_fn(pred_logits, targets)  # (B, H, W)
    ce_loss_per_sample = ce_loss_per_pixel.mean(dim=(1, 2))  # (B,)
    ce_loss = ce_loss_per_sample.mean()

    if loss_mode == "ce":
        return ce_loss, ce_loss_per_sample

    if loss_mode == "dice":
        d_loss = dice_loss(pred_logits, targets)
        # Provide a per-sample proxy for tracking (uses CE per-sample for compatibility).
        return d_loss, ce_loss_per_sample.detach()

    if loss_mode == "boundary_dice":
        b_loss, b_per_sample = boundary_dice_loss(
            pred_logits, targets, width=boundary_width
        )
        return b_loss, b_per_sample

    if loss_mode == "dice_boundary":
        # Dice per sample (binary foreground)
        pred_probs = F.softmax(pred_logits, dim=1)
        pred_fg = pred_probs[:, 1]  # (B,H,W)
        target_fg = (targets == 1).float()  # (B,H,W)

        intersection = (pred_fg * target_fg).sum(dim=(1, 2))
        union = pred_fg.sum(dim=(1, 2)) + target_fg.sum(dim=(1, 2))
        dice_per_sample = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)

        d_loss = dice_per_sample.mean()
        b_loss, b_per_sample = boundary_dice_loss(
            pred_logits, targets, width=boundary_width
        )

        total = d_loss + float(boundary_weight) * b_loss
        per_sample_total = dice_per_sample + float(boundary_weight) * b_per_sample
        return total, per_sample_total
        
    if loss_mode == "gce":
        return generalized_cross_entropy(pred_logits, targets, q=gce_q, class_weights=class_weights)
        
    if loss_mode == "gce_dice":
        gce_loss, gce_per_sample = generalized_cross_entropy(pred_logits, targets, q=gce_q, class_weights=class_weights)
        d_loss = dice_loss(pred_logits, targets)
        total_loss = ce_weight * gce_loss + dice_weight * d_loss
        return total_loss, gce_per_sample
        
    if loss_mode == "bootstrapping":
        return bootstrapping_loss(pred_logits, targets, beta=bootstrapping_beta, class_weights=class_weights)
        
    if loss_mode == "bootstrapping_dice":
        boot_loss, boot_per_sample = bootstrapping_loss(pred_logits, targets, beta=bootstrapping_beta, class_weights=class_weights)
        d_loss = dice_loss(pred_logits, targets)
        total_loss = ce_weight * boot_loss + dice_weight * d_loss
        return total_loss, boot_per_sample

    # Default: combined CE + Dice
    d_loss = dice_loss(pred_logits, targets)
    total_loss = ce_weight * ce_loss + dice_weight * d_loss
    return total_loss, ce_loss_per_sample
