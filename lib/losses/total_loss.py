"""
Total Training Objective
=========================

Implements the composite loss from Section 3.5, Eq. 23:

    L = λ_cls · L_cls + λ_reg · L_reg + λ_iou · L_iou + λ_eq · L_eq

The four terms supervise complementary aspects:
    - L_cls (Focal Loss): Foreground/background discrimination
    - L_reg (L1 Loss): Bounding box coordinate accuracy
    - L_iou (GIoU Loss): Bounding box overlap quality
    - L_eq  (Equivariance Loss): Transformation consistency (Eq. 24-26)

Default weights from Section 5.3 (Implementation Details):
    λ_cls = 2.0, λ_reg = 5.0, λ_iou = 2.0, λ_eq = 1.0

Sensitivity analysis (Appendix D.6, Table D.7):
    - λ_iou has the largest impact: halving causes 1.1% AUC drop
    - λ_cls reduction: 0.9% drop
    - λ_eq too low (0.5): 0.8% drop; too high (2.0): 0.3% drop
    - Overall AUC fluctuates within 1.1% across tested configurations

References:
    - Eq. 23: Total loss
    - Table D.7: Loss weight sensitivity analysis
    - Algorithm 1 (line 10): Update Θ ← Θ - η∇_Θ L
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .detection_loss import DetectionLoss
from .equivariance_loss import TransformationConsistencyLoss, sample_scale_factor


class TotalLoss(nn.Module):
    """
    Total training loss combining detection and equivariance objectives.

    L = L_det + λ_eq · L_eq
      = (λ_cls·L_cls + λ_reg·L_reg + λ_iou·L_iou) + λ_eq·(L_eq^inv + L_eq^eq + L_eq^cp)

    Args:
        lambda_cls: Classification loss weight (default 2.0).
        lambda_reg: Regression loss weight (default 5.0).
        lambda_iou: IoU loss weight (default 2.0).
        lambda_eq: Equivariance loss weight (default 1.0).
        focal_alpha: Focal loss alpha parameter.
        focal_gamma: Focal loss gamma parameter.
        eq_weight_inv: Weight for invariant equivariance term.
        eq_weight_eq: Weight for equivariant equivariance term.
        eq_weight_cp: Weight for coupled equivariance term.
    """

    def __init__(
        self,
        lambda_cls: float = 2.0,
        lambda_reg: float = 5.0,
        lambda_iou: float = 2.0,
        lambda_eq: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        eq_weight_inv: float = 1.0,
        eq_weight_eq: float = 1.0,
        eq_weight_cp: float = 1.0,
    ):
        super().__init__()
        self.lambda_eq = lambda_eq

        self.detection_loss = DetectionLoss(
            lambda_cls=lambda_cls,
            lambda_reg=lambda_reg,
            lambda_iou=lambda_iou,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )

        self.equivariance_loss = TransformationConsistencyLoss(
            weight_inv=eq_weight_inv,
            weight_eq=eq_weight_eq,
            weight_cp=eq_weight_cp,
        )

    def forward(
        self,
        pred_cls: torch.Tensor,
        pred_boxes: torch.Tensor,
        target_cls: torch.Tensor,
        target_boxes: torch.Tensor,
        features_original: Dict[str, torch.Tensor],
        features_transformed: Dict[str, torch.Tensor],
        fg_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss (Eq. 23).

        Args:
            pred_cls: Predicted classification [B, N].
            pred_boxes: Predicted boxes [B, N, 4].
            target_cls: GT classification [B, N].
            target_boxes: GT boxes [B, N, 4].
            features_original: Dict 'inv'/'eq'/'cp' → [B, C, H, W] from φ(I).
            features_transformed: Dict 'inv'/'eq'/'cp' → [B, C, H, W] from φ(T_s I).
            fg_mask: Foreground mask [B, N].

        Returns:
            Tuple of (total_loss, loss_dict with all individual terms).
        """
        # Detection losses: L_cls + L_reg + L_iou
        l_det, det_dict = self.detection_loss(
            pred_cls, pred_boxes, target_cls, target_boxes, fg_mask
        )

        # Equivariance loss: L_eq = L_eq^inv + L_eq^eq + L_eq^cp
        l_eq, eq_dict = self.equivariance_loss(
            features_original, features_transformed
        )

        # Total: L = L_det + λ_eq · L_eq (Eq. 23)
        total = l_det + self.lambda_eq * l_eq

        # Merge all loss terms for logging
        loss_dict = {**det_dict, **eq_dict, "loss_total": total}

        return total, loss_dict


def build_loss(cfg) -> TotalLoss:
    """Factory function to build TotalLoss from config."""
    return TotalLoss(
        lambda_cls=cfg.LOSS.LAMBDA_CLS,
        lambda_reg=cfg.LOSS.LAMBDA_REG,
        lambda_iou=cfg.LOSS.LAMBDA_IOU,
        lambda_eq=cfg.LOSS.LAMBDA_EQ,
    )
