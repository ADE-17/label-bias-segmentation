"""
Debiasing modules for fair segmentation.

Three strategies:
  1. Adversarial debiasing via gradient reversal (DANN-style)
  2. Fairness loss regularization (Demographic Parity / Equalized Odds)
  3. Domain-invariant learning (MMD / CORAL feature alignment)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Dict, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# 1. Gradient Reversal Layer + Gender Adversary
# ---------------------------------------------------------------------------

class GradientReversalFn(Function):
    """Reverses gradients during backprop (Ganin et al., 2016)."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def gradient_reversal(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradientReversalFn.apply(x, alpha)


class GenderAdversary(nn.Module):
    """
    Predicts gender from encoder features.
    Paired with a Gradient Reversal Layer so the encoder learns to
    produce gender-invariant representations.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Args:
            features: encoder feature map (B, C, H, W)
            alpha: GRL scaling factor (ramp up during training)
        Returns:
            gender logits (B, 1)
        """
        x = gradient_reversal(features, alpha)
        x = self.pool(x).flatten(1)  # (B, C)
        return self.head(x)


class AdversarialLoss(nn.Module):
    """Binary cross-entropy for the gender adversary."""

    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(
        self, gender_logits: torch.Tensor, gender_labels: torch.Tensor
    ) -> torch.Tensor:
        return self.criterion(
            gender_logits.squeeze(-1), gender_labels.float()
        )


# ---------------------------------------------------------------------------
# 2. Fairness Loss
# ---------------------------------------------------------------------------

class FairnessLoss(nn.Module):
    """
    Differentiable fairness regularisation penalties.

    Modes
    -----
    dp   – Demographic Parity:
           penalises |E[ŷ|A=0] − E[ŷ|A=1]|  (equalize positive prediction rates)
    eo   – Equalized Odds:
           penalises |E[ŷ|Y=y,A=0] − E[ŷ|Y=y,A=1]| for y∈{0,1}
    both – sum of dp and eo penalties
    """

    def __init__(self, mode: str = "both"):
        super().__init__()
        assert mode in ("dp", "eo", "both"), f"Unknown fairness mode: {mode}"
        self.mode = mode

    def _group_mean_pred(
        self,
        probs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean predicted FG probability over selected pixels."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=probs.device)
        return (probs * mask).sum() / mask.sum()

    def forward(
        self,
        pred_logits: torch.Tensor,
        targets: torch.Tensor,
        genders: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_logits: (B, 2, H, W) segmentation logits
            targets:     (B, H, W)    ground truth masks
            genders:     (B,)         gender labels (0/1)
        """
        fg_probs = F.softmax(pred_logits, dim=1)[:, 1]  # (B, H, W)

        gender_mask_0 = (genders == 0).float()  # (B,)
        gender_mask_1 = (genders == 1).float()

        # Expand to spatial dims
        g0 = gender_mask_0[:, None, None].expand_as(fg_probs)  # (B,H,W)
        g1 = gender_mask_1[:, None, None].expand_as(fg_probs)

        loss = torch.tensor(0.0, device=pred_logits.device)

        if self.mode in ("dp", "both"):
            rate_0 = self._group_mean_pred(fg_probs, g0)
            rate_1 = self._group_mean_pred(fg_probs, g1)
            loss = loss + (rate_0 - rate_1).abs()

        if self.mode in ("eo", "both"):
            fg_gt = (targets == 1).float()
            bg_gt = (targets == 0).float()

            # TPR gap: E[ŷ=1 | Y=1, A] across genders
            tpr_0 = self._group_mean_pred(fg_probs, g0 * fg_gt)
            tpr_1 = self._group_mean_pred(fg_probs, g1 * fg_gt)
            loss = loss + (tpr_0 - tpr_1).abs()

            # FPR gap: E[ŷ=1 | Y=0, A] across genders
            fpr_0 = self._group_mean_pred(fg_probs, g0 * bg_gt)
            fpr_1 = self._group_mean_pred(fg_probs, g1 * bg_gt)
            loss = loss + (fpr_0 - fpr_1).abs()

        return loss


# ---------------------------------------------------------------------------
# 3. Domain-Invariant Learning (Feature Alignment)
# ---------------------------------------------------------------------------

class _GaussianKernelMMD(nn.Module):
    """Shared multi-scale Gaussian kernel MMD computation."""

    def __init__(
        self,
        kernel_bandwidth: float = 1.0,
        n_kernels: int = 5,
        normalize_mmd: bool = False,
    ):
        super().__init__()
        bw = kernel_bandwidth
        self.normalize_mmd = normalize_mmd
        self.register_buffer(
            "bandwidths",
            torch.tensor([bw * (2 ** i) for i in range(-n_kernels // 2, n_kernels // 2 + 1)]),
        )

    def _mmd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xx = torch.cdist(x, x).pow(2)
        yy = torch.cdist(y, y).pow(2)
        xy = torch.cdist(x, y).pow(2)

        k_xx = sum(torch.exp(-xx / (2 * bw ** 2)) for bw in self.bandwidths)
        k_yy = sum(torch.exp(-yy / (2 * bw ** 2)) for bw in self.bandwidths)
        k_xy = sum(torch.exp(-xy / (2 * bw ** 2)) for bw in self.bandwidths)

        n, m = x.size(0), y.size(0)
        return k_xx.sum() / (n * n) + k_yy.sum() / (m * m) - 2 * k_xy.sum() / (n * m)


class MMDLoss(_GaussianKernelMMD):
    """
    MMD on GAP-pooled encoder features (original implementation).
    Minimises distance between feature distributions of demographic groups.
    """

    def forward(
        self,
        features: torch.Tensor,
        genders: torch.Tensor,
    ) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)  # (B, C)
        if self.normalize_mmd:
            # L2-normalize so bandwidth is less sensitive to feature scale.
            pooled = F.normalize(pooled, p=2, dim=1)

        idx_0 = (genders == 0).nonzero(as_tuple=True)[0]
        idx_1 = (genders == 1).nonzero(as_tuple=True)[0]

        if len(idx_0) < 2 or len(idx_1) < 2:
            return torch.tensor(0.0, device=features.device)

        return self._mmd(pooled[idx_0], pooled[idx_1])


class LogitMMDLoss(_GaussianKernelMMD):
    """
    Class-conditional MMD on output logits (per the paper).

    For each segmentation class c, aligns the logit distribution
    of Group A with Group A' by sampling pixel logits per class.
    More stable than feature-level MMD and directly targets output fairness.
    """

    def __init__(
        self,
        kernel_bandwidth: float = 1.0,
        n_kernels: int = 5,
        n_pixels: int = 256,
        normalize_mmd: bool = False,
    ):
        super().__init__(
            kernel_bandwidth=kernel_bandwidth,
            n_kernels=n_kernels,
            normalize_mmd=normalize_mmd,
        )
        self.n_pixels = n_pixels

    def forward(
        self,
        seg_logits: torch.Tensor,
        genders: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            seg_logits: (B, C, H, W) raw logits
            genders:    (B,) gender labels
            targets:    (B, H, W) ground truth masks
        """
        B, C, H, W = seg_logits.shape

        idx_0 = (genders == 0).nonzero(as_tuple=True)[0]
        idx_1 = (genders == 1).nonzero(as_tuple=True)[0]
        if len(idx_0) < 1 or len(idx_1) < 1:
            return torch.tensor(0.0, device=seg_logits.device)

        loss = torch.tensor(0.0, device=seg_logits.device)

        for cls in range(C):
            # Gather pixel logits for this class from each group
            pixels_0 = self._sample_class_logits(seg_logits[idx_0], targets[idx_0], cls)
            pixels_1 = self._sample_class_logits(seg_logits[idx_1], targets[idx_1], cls)

            if pixels_0 is None or pixels_1 is None:
                continue

            if self.normalize_mmd:
                # L2-normalize logit vectors per pixel.
                pixels_0 = F.normalize(pixels_0, p=2, dim=1)
                pixels_1 = F.normalize(pixels_1, p=2, dim=1)

            loss = loss + self._mmd(pixels_0, pixels_1)

        return loss / max(C, 1)

    def _sample_class_logits(self, logits, targets, cls):
        """Sample pixel logits belonging to class cls, return (N, C) vectors."""
        mask = (targets == cls)  # (B, H, W)
        if mask.sum() == 0:
            return None

        # Flatten spatial dims: (B, C, H*W) and mask: (B, H*W)
        B, C, H, W = logits.shape
        flat_logits = logits.reshape(B, C, -1)  # (B, C, H*W)
        flat_mask = mask.reshape(B, -1)  # (B, H*W)

        # Collect all pixels of this class across batch
        all_pixels = []
        for i in range(B):
            px = flat_logits[i, :, flat_mask[i]]  # (C, n_pixels_i)
            if px.shape[1] > 0:
                all_pixels.append(px.T)  # (n_pixels_i, C)

        if not all_pixels:
            return None

        pooled = torch.cat(all_pixels, dim=0)  # (total, C)

        # Subsample for memory/speed
        if pooled.size(0) > self.n_pixels:
            idx = torch.randperm(pooled.size(0), device=pooled.device)[:self.n_pixels]
            pooled = pooled[idx]

        return pooled


class CORALLoss(nn.Module):
    """
    Correlation Alignment (CORAL) loss.
    Aligns the second-order statistics (covariance) of feature
    distributions across demographic groups.
    """

    def forward(
        self,
        features: torch.Tensor,
        genders: torch.Tensor,
    ) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)  # (B, C)
        d = pooled.size(1)

        idx_0 = (genders == 0).nonzero(as_tuple=True)[0]
        idx_1 = (genders == 1).nonzero(as_tuple=True)[0]

        if len(idx_0) < 2 or len(idx_1) < 2:
            return torch.tensor(0.0, device=features.device)

        f0 = pooled[idx_0]
        f1 = pooled[idx_1]

        cov_0 = self._cov(f0)
        cov_1 = self._cov(f1)

        return (cov_0 - cov_1).pow(2).sum() / (4 * d * d)

    @staticmethod
    def _cov(x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        x_centered = x - x.mean(dim=0, keepdim=True)
        return (x_centered.T @ x_centered) / (n - 1)


# ---------------------------------------------------------------------------
# GRL alpha schedule
# ---------------------------------------------------------------------------

def grl_alpha_schedule(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int = 5,
    max_alpha: float = 1.0,
) -> float:
    """
    Ramp GRL strength from 0 → max_alpha following Ganin et al.
    Uses p = (epoch - warmup) / (total - warmup), alpha = 2/(1+exp(-10p)) - 1
    """
    if epoch < warmup_epochs:
        return 0.0
    p = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return float(max_alpha * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0))


# ---------------------------------------------------------------------------
# Wrapper model that exposes encoder features
# ---------------------------------------------------------------------------

class SegmentationWithDebiasing(nn.Module):
    """
    Wraps an SMP segmentation model and attaches debiasing heads.
    Exposes encoder features for adversarial / domain-invariant branches.
    """

    def __init__(
        self,
        segmentation_model: nn.Module,
        method: str = "adversarial",
        adversary_hidden: int = 256,
        fairness_mode: str = "both",
        domain_method: str = "mmd",
        kernel_bandwidth: float = 1.0,
        mmd_normalize: bool = False,
    ):
        """
        Args:
            segmentation_model: an SMP model (e.g. smp.Unet)
            method: 'adversarial' | 'fairness' | 'domain_invariant'
            adversary_hidden: hidden dim of gender adversary
            fairness_mode: 'dp' | 'eo' | 'both'
            domain_method: 'mmd' | 'coral'
            kernel_bandwidth: bandwidth for MMD kernels
        """
        super().__init__()
        self.seg_model = segmentation_model
        self.method = method

        encoder_out_channels = segmentation_model.encoder.out_channels[-1]

        if method == "adversarial":
            self.adversary = GenderAdversary(encoder_out_channels, adversary_hidden)
            self.adv_loss_fn = AdversarialLoss()
        elif method == "fairness":
            self.fairness_loss_fn = FairnessLoss(mode=fairness_mode)
        elif method == "domain_invariant":
            self.domain_method = domain_method
            if domain_method == "mmd":
                self.domain_loss_fn = MMDLoss(
                    kernel_bandwidth=kernel_bandwidth,
                    normalize_mmd=mmd_normalize,
                )
            elif domain_method == "mmd_logit":
                self.domain_loss_fn = LogitMMDLoss(
                    kernel_bandwidth=kernel_bandwidth,
                    normalize_mmd=mmd_normalize,
                )
            else:
                self.domain_loss_fn = CORALLoss()
        else:
            raise ValueError(f"Unknown method: {method}")

    def forward(
        self,
        images: torch.Tensor,
        genders: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
              'seg_logits': (B, C, H, W) segmentation logits
              'encoder_features': last encoder feature map
              'debiasing_loss': scalar debiasing loss (0 during eval if no labels)
              'adv_accuracy': adversary accuracy (adversarial only)
        """
        features = self.seg_model.encoder(images)
        decoder_output = self.seg_model.decoder(features)
        seg_logits = self.seg_model.segmentation_head(decoder_output)

        result = {
            "seg_logits": seg_logits,
            "encoder_features": features[-1],
        }

        debiasing_loss = torch.tensor(0.0, device=images.device)

        if genders is not None and self.training:
            if self.method == "adversarial":
                gender_logits = self.adversary(features[-1], alpha=alpha)
                debiasing_loss = self.adv_loss_fn(gender_logits, genders)
                with torch.no_grad():
                    preds = (gender_logits.squeeze(-1) > 0).long()
                    result["adv_accuracy"] = (preds == genders).float().mean()

            elif self.method == "fairness":
                debiasing_loss = self.fairness_loss_fn(seg_logits, targets, genders)

            elif self.method == "domain_invariant":
                if getattr(self, 'domain_method', '') == 'mmd_logit':
                    debiasing_loss = self.domain_loss_fn(seg_logits, genders, targets)
                else:
                    debiasing_loss = self.domain_loss_fn(features[-1], genders)

        result["debiasing_loss"] = debiasing_loss
        return result

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Inference: just segmentation logits, no debiasing overhead."""
        return self.seg_model(images)
