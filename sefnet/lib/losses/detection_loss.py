"""
Detection Losses for Tracking
===============================

Implements the standard tracking detection losses used in Eq. 23:
    L_cls: Focal Loss for foreground/background classification
    L_reg: L1 Loss for bounding box coordinate regression
    L_iou: Generalized IoU Loss for box quality

These three losses supervise the tracking head output and are combined
with the equivariance loss L_eq to form the total training objective.

The focal loss addresses class imbalance between foreground (target) and
background tokens in the search region. L1 and GIoU losses provide
complementary regression supervision: L1 penalizes absolute coordinate
errors while GIoU captures overlap quality including non-overlapping cases.

Default weights from Section 5.3:
    λ_cls = 2.0, λ_reg = 5.0, λ_iou = 2.0

References:
    - Eq. 23: L = λ_cls·L_cls + λ_reg·L_reg + λ_iou·L_iou + λ_eq·L_eq
    - Lin et al. (2017): Focal Loss for Dense Object Detection
    - Rezatofighi et al. (2019): Generalized Intersection over Union
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for classification (L_cls in Eq. 23).

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    The modulating factor (1 - p_t)^γ down-weights easy examples and
    focuses training on hard negatives, addressing the severe class
    imbalance in tracking where most search tokens are background.

    Args:
        alpha: Balancing factor for positive/negative classes (default 0.25).
        gamma: Focusing parameter, γ > 0 reduces loss for well-classified
               examples (default 2.0).
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            pred: Predicted logits [B, N] or [B, N, 1].
            target: Ground truth labels [B, N], values in {0, 1}.

        Returns:
            loss: Focal loss scalar.
        """
        pred = pred.squeeze(-1) if pred.dim() == 3 else pred
        target = target.float()

        # Sigmoid activation
        p = torch.sigmoid(pred)
        # p_t = p for positive, (1-p) for negative
        p_t = p * target + (1 - p) * (1 - target)

        # α_t = α for positive, (1-α) for negative
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        # Focal modulating factor
        focal_weight = (1 - p_t) ** self.gamma

        # Binary cross-entropy (numerically stable)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        # Focal loss
        loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class L1Loss(nn.Module):
    """
    L1 Loss for bounding box regression (L_reg in Eq. 23).

    L_reg = (1/N_pos) · Σ_i ||b_pred_i - b_gt_i||_1

    Penalizes absolute coordinate errors for the predicted bounding box.
    Only computed over positive (foreground) predictions.

    Args:
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute L1 regression loss.

        Args:
            pred_boxes: Predicted boxes [B, N, 4] as (cx, cy, w, h) normalized.
            target_boxes: Ground truth boxes [B, N, 4].
            mask: Optional foreground mask [B, N], 1 for positive tokens.

        Returns:
            loss: L1 loss scalar.
        """
        l1 = F.l1_loss(pred_boxes, target_boxes, reduction="none")  # [B, N, 4]

        if mask is not None:
            # Only compute loss for positive tokens
            mask = mask.unsqueeze(-1).expand_as(l1)  # [B, N, 4]
            l1 = l1 * mask
            n_pos = mask.sum().clamp(min=1.0)
            return l1.sum() / n_pos
        else:
            return l1.mean()


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss for bounding box quality (L_iou in Eq. 23).

    GIoU = IoU - |C \ (A ∪ B)| / |C|
    L_iou = 1 - GIoU

    GIoU handles non-overlapping boxes (IoU=0) by measuring the ratio
    of the enclosing area not covered by the union. This provides
    meaningful gradients even when predicted and ground truth boxes
    do not overlap, which is critical during early training.

    Args:
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute GIoU loss.

        Args:
            pred_boxes: Predicted boxes [B, N, 4] as (cx, cy, w, h) normalized.
            target_boxes: Ground truth boxes [B, N, 4].
            mask: Optional foreground mask [B, N].

        Returns:
            loss: GIoU loss scalar.
        """
        # Convert (cx, cy, w, h) → (x1, y1, x2, y2)
        pred_xyxy = self._cxcywh_to_xyxy(pred_boxes)
        tgt_xyxy = self._cxcywh_to_xyxy(target_boxes)

        giou = self._compute_giou(pred_xyxy, tgt_xyxy)  # [B, N]
        loss = 1.0 - giou

        if mask is not None:
            loss = loss * mask
            n_pos = mask.sum().clamp(min=1.0)
            return loss.sum() / n_pos
        else:
            return loss.mean()

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def _compute_giou(
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GIoU between two sets of boxes.

        Args:
            boxes1, boxes2: [*, 4] as (x1, y1, x2, y2).

        Returns:
            giou: [*] GIoU values in range [-1, 1].
        """
        # Intersection
        inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
        inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
        inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
        inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # Areas
        area1 = (boxes1[..., 2] - boxes1[..., 0]).clamp(min=0) * \
                (boxes1[..., 3] - boxes1[..., 1]).clamp(min=0)
        area2 = (boxes2[..., 2] - boxes2[..., 0]).clamp(min=0) * \
                (boxes2[..., 3] - boxes2[..., 1]).clamp(min=0)

        union_area = area1 + area2 - inter_area

        # IoU
        iou = inter_area / union_area.clamp(min=1e-8)

        # Enclosing box
        enc_x1 = torch.min(boxes1[..., 0], boxes2[..., 0])
        enc_y1 = torch.min(boxes1[..., 1], boxes2[..., 1])
        enc_x2 = torch.max(boxes1[..., 2], boxes2[..., 2])
        enc_y2 = torch.max(boxes1[..., 3], boxes2[..., 3])

        enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0)

        # GIoU = IoU - |C \ (A ∪ B)| / |C|
        giou = iou - (enc_area - union_area) / enc_area.clamp(min=1e-8)

        return giou


class DetectionLoss(nn.Module):
    """
    Combined detection loss for tracking.

    L_det = λ_cls · L_cls + λ_reg · L_reg + λ_iou · L_iou

    Args:
        lambda_cls: Weight for focal classification loss (default 2.0).
        lambda_reg: Weight for L1 regression loss (default 5.0).
        lambda_iou: Weight for GIoU loss (default 2.0).
        focal_alpha: Focal loss balancing factor.
        focal_gamma: Focal loss focusing parameter.
    """

    def __init__(
        self,
        lambda_cls: float = 2.0,
        lambda_reg: float = 5.0,
        lambda_iou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.lambda_iou = lambda_iou

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.l1_loss = L1Loss()
        self.giou_loss = GIoULoss()

    def forward(
        self,
        pred_cls: torch.Tensor,
        pred_boxes: torch.Tensor,
        target_cls: torch.Tensor,
        target_boxes: torch.Tensor,
        fg_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined detection loss.

        Args:
            pred_cls: Predicted classification logits [B, N].
            pred_boxes: Predicted boxes [B, N, 4].
            target_cls: GT classification labels [B, N].
            target_boxes: GT boxes [B, N, 4].
            fg_mask: Foreground mask [B, N] for regression losses.

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        l_cls = self.focal_loss(pred_cls, target_cls)
        l_reg = self.l1_loss(pred_boxes, target_boxes, mask=fg_mask)
        l_iou = self.giou_loss(pred_boxes, target_boxes, mask=fg_mask)

        total = (self.lambda_cls * l_cls +
                 self.lambda_reg * l_reg +
                 self.lambda_iou * l_iou)

        loss_dict = {
            "loss_cls": l_cls,
            "loss_reg": l_reg,
            "loss_iou": l_iou,
            "loss_det": total,
        }

        return total, loss_dict
