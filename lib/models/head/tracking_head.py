"""
Tracking Prediction Head
=========================

Implements the classification and bounding box regression head that
operates on the fused output from GAB (Section 3.4).

The head receives F^out ∈ R^{C × H_s × W_s} from GAB's adaptive fusion
and produces:
    - Classification score map: p̂ ∈ R^{H_s × W_s} (foreground probability)
    - Bounding box prediction: b̂ ∈ R^{H_s × W_s × 4} (normalized cxcywh)

Architecture:
    Fused tokens [B, N_s, C]
        ├── Classification branch → [B, N_s, 1] (sigmoid → probability)
        └── Regression branch     → [B, N_s, 4] (sigmoid → normalized coords)

The classification branch predicts whether each search token corresponds
to the target center. The regression branch predicts normalized bounding
box coordinates (cx, cy, w, h) relative to the search region.

During inference (Algorithm 2, line 10-11):
    1. Predict (p̂, b̂) ← Head(F^out)
    2. Select box: b_t = b̂[argmax(p̂)]
    3. Smooth: b̂_t ← 0.9 · b̂_{t-1} + 0.1 · b_t  (EMA, momentum 0.9)

Complexity: O(C · N_s), negligible compared to backbone and GAB.

References:
    - Algorithm 1, line 9: Predict (p̂, b̂) ← Head(F^out)
    - Algorithm 2, line 10-11: Predict and smooth
    - Section 4: Prediction head complexity O(C · N_s)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class CornerHead(nn.Module):
    """
    Corner-based prediction head.

    Predicts bounding box as (left, top, right, bottom) distances from
    each token position, then converts to (cx, cy, w, h) format.

    This is more numerically stable than direct center prediction for
    ViT-based trackers where tokens correspond to spatial grid positions.

    Args:
        in_dim: Input token dimension C.
        hidden_dim: Hidden dimension of prediction MLPs.
        n_layers: Number of MLP layers (default 3).
    """

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()

        # Classification branch: token → foreground probability
        cls_layers = []
        for i in range(n_layers):
            d_in = in_dim if i == 0 else hidden_dim
            cls_layers.append(nn.Linear(d_in, hidden_dim))
            cls_layers.append(nn.ReLU(inplace=True))
        cls_layers.append(nn.Linear(hidden_dim, 1))
        self.cls_head = nn.Sequential(*cls_layers)

        # Regression branch: token → box coordinates (ltrb or cxcywh)
        reg_layers = []
        for i in range(n_layers):
            d_in = in_dim if i == 0 else hidden_dim
            reg_layers.append(nn.Linear(d_in, hidden_dim))
            reg_layers.append(nn.ReLU(inplace=True))
        reg_layers.append(nn.Linear(hidden_dim, 4))
        self.reg_head = nn.Sequential(*reg_layers)

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict classification scores and bounding boxes.

        Args:
            tokens: Fused search tokens [B, N_s, C] from GAB output.

        Returns:
            Tuple of:
                - cls_score: Classification logits [B, N_s, 1].
                - pred_boxes: Predicted boxes [B, N_s, 4] as normalized (cx,cy,w,h).
        """
        cls_score = self.cls_head(tokens)             # [B, N_s, 1]
        pred_boxes = self.reg_head(tokens).sigmoid()   # [B, N_s, 4] in [0, 1]

        return cls_score, pred_boxes


class CenterHead(nn.Module):
    """
    Center-based prediction head with separate cls and offset branches.

    Predicts:
        - Center heatmap: which token is the target center
        - Size prediction: (w, h) of the target
        - Offset prediction: sub-grid center offset (cx_off, cy_off)

    Args:
        in_dim: Input token dimension C.
        hidden_dim: Hidden dimension.
    """

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Center heatmap: [B, N_s, 1]
        self.cls_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # Size prediction: [B, N_s, 2] → (w, h)
        self.size_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

        # Offset prediction: [B, N_s, 2] → (cx_off, cy_off)
        self.offset_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: [B, N_s, C].

        Returns:
            cls_score: [B, N_s, 1] center heatmap logits.
            pred_boxes: [B, N_s, 4] as (cx, cy, w, h) normalized.
        """
        cls_score = self.cls_head(tokens)          # [B, N_s, 1]
        size = self.size_head(tokens).sigmoid()     # [B, N_s, 2]
        offset = self.offset_head(tokens).sigmoid() # [B, N_s, 2]

        # Combine offset and size into full box prediction
        pred_boxes = torch.cat([offset, size], dim=-1)  # [B, N_s, 4]

        return cls_score, pred_boxes


class TrackingHead(nn.Module):
    """
    SEFNet tracking prediction head.

    Wraps the prediction head with post-processing utilities for
    converting token-level predictions to image-level bounding boxes.

    Supports both corner-based and center-based head variants.

    Args:
        in_dim: Input token dimension C (default 256).
        hidden_dim: Hidden dimension (default 256).
        head_type: 'corner' or 'center' (default 'corner').
        search_size: Search region size in pixels (default 384).
        search_feat_size: Search feature map size (default 24, i.e. 384/16).
    """

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 256,
        head_type: str = "corner",
        search_size: int = 384,
        search_feat_size: int = 24,
    ):
        super().__init__()
        self.search_size = search_size
        self.search_feat_size = search_feat_size

        if head_type == "corner":
            self.head = CornerHead(in_dim, hidden_dim)
        elif head_type == "center":
            self.head = CenterHead(in_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown head type: {head_type}")

    def forward(
        self,
        fused_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: predict classification and boxes.

        Args:
            fused_tokens: [B, N_s, C] from GAB adaptive fusion.

        Returns:
            Dict with:
                - 'cls': Classification logits [B, N_s, 1].
                - 'boxes': Predicted boxes [B, N_s, 4] normalized.
                - 'cls_prob': Classification probabilities [B, N_s, 1].
        """
        cls_score, pred_boxes = self.head(fused_tokens)

        return {
            "cls": cls_score,
            "boxes": pred_boxes,
            "cls_prob": cls_score.sigmoid(),
        }

    def get_best_prediction(
        self,
        head_output: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract the best bounding box prediction per batch element.

        Selects the token with the highest classification score and
        returns its predicted box. Used during inference.

        Algorithm 2, line 11: b̂_t = b_t(argmax p̂)

        Args:
            head_output: Dict from forward() with 'cls_prob' and 'boxes'.

        Returns:
            Tuple of:
                - best_box: [B, 4] best predicted box per sample.
                - best_score: [B, 1] classification score.
        """
        cls_prob = head_output["cls_prob"].squeeze(-1)  # [B, N_s]
        boxes = head_output["boxes"]                     # [B, N_s, 4]

        # argmax over tokens
        best_idx = cls_prob.argmax(dim=1)  # [B]

        # Gather best boxes
        B = best_idx.shape[0]
        best_box = boxes[torch.arange(B, device=boxes.device), best_idx]  # [B, 4]
        best_score = cls_prob[torch.arange(B, device=cls_prob.device), best_idx]  # [B]

        return best_box, best_score.unsqueeze(1)

    @staticmethod
    def apply_ema_smoothing(
        prev_box: torch.Tensor,
        curr_box: torch.Tensor,
        momentum: float = 0.9,
    ) -> torch.Tensor:
        """
        Exponential moving average smoothing for temporal consistency.

        b̂_t ← momentum · b̂_{t-1} + (1 - momentum) · b_t

        Algorithm 2, line 11: momentum = 0.9 for temporal consistency.

        Args:
            prev_box: Previous smoothed box [B, 4].
            curr_box: Current predicted box [B, 4].
            momentum: EMA momentum (default 0.9).

        Returns:
            smoothed_box: [B, 4].
        """
        return momentum * prev_box + (1 - momentum) * curr_box

    def map_box_to_image(
        self,
        box_norm: torch.Tensor,
        search_center: torch.Tensor,
        search_scale: float = 4.0,
        original_target_size: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Map normalized box prediction back to image coordinates.

        The search region is cropped at the previous predicted location
        with a context factor of 4.0 (Sec 3.5). The normalized box
        coordinates must be mapped back to the full image frame.

        Args:
            box_norm: Normalized box [B, 4] as (cx, cy, w, h) in [0, 1].
            search_center: Center of the search region in image coords [B, 2].
            search_scale: Context factor (default 4.0).
            original_target_size: Original target (w, h) [B, 2] for scale reference.

        Returns:
            box_image: Box in image coordinates [B, 4] as (cx, cy, w, h).
        """
        # Scale box from normalized [0,1] to search region pixels
        cx = box_norm[:, 0] * self.search_size
        cy = box_norm[:, 1] * self.search_size
        w = box_norm[:, 2] * self.search_size
        h = box_norm[:, 3] * self.search_size

        # Map from search region center to image coordinates
        # Search region spans search_scale * target_size centered at search_center
        if original_target_size is not None:
            region_w = original_target_size[:, 0] * search_scale
            region_h = original_target_size[:, 1] * search_scale
        else:
            region_w = torch.full_like(cx, self.search_size)
            region_h = torch.full_like(cy, self.search_size)

        # Convert from search-region-relative to image-absolute
        scale_x = region_w / self.search_size
        scale_y = region_h / self.search_size

        img_cx = search_center[:, 0] + (cx - self.search_size / 2) * scale_x
        img_cy = search_center[:, 1] + (cy - self.search_size / 2) * scale_y
        img_w = w * scale_x
        img_h = h * scale_y

        return torch.stack([img_cx, img_cy, img_w, img_h], dim=1)


def build_tracking_head(cfg) -> TrackingHead:
    """Factory function to build TrackingHead from config."""
    return TrackingHead(
        in_dim=cfg.MODEL.CHANNELS,
        hidden_dim=cfg.HEAD.HIDDEN_DIM,
        head_type=cfg.HEAD.TYPE,
        search_size=cfg.DATA.SEARCH_SIZE,
        search_feat_size=cfg.DATA.SEARCH_FEAT_SIZE,
    )
