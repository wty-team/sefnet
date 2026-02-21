"""
SEFNet: Selective Equivariant Features for Robust Scale-Adaptive UAV Tracking
================================================================================

Main model class integrating all components (Figure 2).

Architecture overview:
    Input: Template z (192×192), Search x (384×384)
        │
        ├── ViT Backbone (12 layers) ──── Multi-layer features {F^(l)}_{l=1}^{12}
        │
        ├── EDM (Equivariant Decomposition Module, Sec 3.3)
        │       For selected layers:
        │       F^(l) ──→ [F_inv^(l), F_eq^(l), F_cp^(l)]  (Eq. 11-12)
        │
        ├── GAB (Geometry-Aware Bridging, Sec 3.4)
        │       ├── Layer Selection: l_t* = argmax q_t^(l)(s)  (Eq. 17)
        │       ├── Cross-Attention: bidirectional matching     (Eq. 21)
        │       └── Adaptive Fusion: X^fused = αX̂_eq + βX̂_inv + γX̂_cp  (Eq. 22)
        │
        └── Head (Classification + Regression)
                ├── p̂ ∈ R^{H×W}   (foreground score map)
                └── b̂ ∈ R^{H×W×4} (bounding box prediction)

Training (Algorithm 1):
    1. Sample frame pair (I_z, I_x) with GT box b_gt
    2. Apply scale augmentation T_s to create transformation pair
    3. Extract multi-layer features from backbone
    4. EDM decomposition on all candidate layers
    5. GAB: select, match, fuse
    6. Head prediction
    7. Compute L = λ_cls·L_cls + λ_reg·L_reg + λ_iou·L_iou + λ_eq·L_eq

Inference (Algorithm 2):
    1. Initialize: extract and cache template features
    2. Per frame: extract search features, EDM decompose, GAB fuse
    3. Predict and smooth: b̂_t ← 0.9·b̂_{t-1} + 0.1·b_t

Computational overhead: 0.34% beyond ViT-Base (Appendix C, Eq. C.1):
    C_total = C_ViT + C_EDM + C_GAB + C_head
    Overhead = (C_EDM + C_GAB + C_head) / C_ViT ≈ 0.34%

References:
    - Figure 2: Overall architecture diagram
    - Algorithm 1: Training procedure
    - Algorithm 2: Inference procedure
    - Appendix C: Computational complexity analysis
    - Section 5.3: Implementation details
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List

from .backbone.vit import build_vit_backbone
from .edm import EquivariantDecompositionModule, build_edm
from .gab import GeometryAwareBridgingBlock, build_gab
from .head import TrackingHead, build_tracking_head


class SEFNet(nn.Module):
    """
    SEFNet: Selective Equivariant Features for UAV Tracking.

    End-to-end model combining ViT backbone, EDM, GAB, and tracking head.

    Args:
        backbone: ViT backbone for multi-layer feature extraction.
        edm: Equivariant Decomposition Module.
        gab: Geometry-Aware Bridging Block.
        head: Tracking prediction head.
        candidate_layers: Backbone layers to consider for decomposition.
            Default [1,2,3,4,5,6,7,8,9,10,11,12] (1-indexed).
    """

    def __init__(
        self,
        backbone: nn.Module,
        edm: EquivariantDecompositionModule,
        gab: GeometryAwareBridgingBlock,
        head: TrackingHead,
        candidate_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.edm = edm
        self.gab = gab
        self.head = head

        # Default: all 12 ViT layers as candidates
        if candidate_layers is None:
            candidate_layers = list(range(1, 13))
        self.candidate_layers = candidate_layers

        # Cached template features for inference efficiency
        self._template_backbone_feats = None
        self._template_decomposed = None

    def extract_backbone_features(
        self,
        image: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract multi-layer features from ViT backbone.

        Args:
            image: Input image [B, 3, H, W].

        Returns:
            Dict layer_idx → features [B, C, H_feat, W_feat].
        """
        return self.backbone(image, return_layers=self.candidate_layers)

    def decompose_all_layers(
        self,
        backbone_features: Dict[int, torch.Tensor],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Apply EDM decomposition to all candidate layers.

        For each layer l in candidate_layers:
            F^(l) → {F_inv^(l), F_eq^(l), F_cp^(l)}  (Eq. 11-12)

        Args:
            backbone_features: Dict layer → [B, C, H, W].

        Returns:
            Dict layer → Dict type → [B, C, H, W].
        """
        decomposed = {}
        for layer_idx, features in backbone_features.items():
            decomposed[layer_idx] = self.edm(features)
        return decomposed

    def forward_train(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
        template_aug: Optional[torch.Tensor] = None,
        search_aug: Optional[torch.Tensor] = None,
        target_cls: Optional[torch.Tensor] = None,
        target_boxes: Optional[torch.Tensor] = None,
        fg_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass (Algorithm 1).

        Steps:
            1. Extract multi-layer features for template and search
            2. EDM decomposition on all candidate layers
            3. GAB: layer selection + cross-attention + fusion
            4. Head prediction
            5. Return predictions and intermediate features for loss

        For equivariance loss computation (Eq. 24-26), scale-augmented
        versions of the images are also processed through the backbone
        and EDM to create transformation pairs.

        Args:
            template: Template image [B, 3, 192, 192].
            search: Search region [B, 3, 384, 384].
            template_aug: Scale-augmented template [B, 3, 192, 192] (optional).
            search_aug: Scale-augmented search [B, 3, 384, 384] (optional).
            target_cls: GT classification labels [B, N_s].
            target_boxes: GT boxes [B, N_s, 4].
            fg_mask: Foreground mask [B, N_s].

        Returns:
            Dict with predictions and features for loss computation.
        """
        # ---- Step 1: Backbone feature extraction ----
        template_feats = self.extract_backbone_features(template)
        search_feats = self.extract_backbone_features(search)

        # ---- Step 2: EDM decomposition on all candidate layers ----
        template_decomposed = self.decompose_all_layers(template_feats)
        search_decomposed = self.decompose_all_layers(search_feats)

        # ---- Step 3: GAB processing ----
        fused_output, gab_info = self.gab(
            decomposed_search=search_decomposed,
            decomposed_template=template_decomposed,
            training=True,
        )

        # ---- Step 4: Head prediction ----
        head_output = self.head(fused_output)

        # ---- Step 5: Process augmented pair for equivariance loss ----
        features_original = None
        features_transformed = None

        if search_aug is not None:
            # Original features: from selected layers of non-augmented search
            selected_info = gab_info.get("selected_layers", {})

            # Get decomposed features from augmented search
            search_aug_feats = self.extract_backbone_features(search_aug)
            search_aug_decomposed = self.decompose_all_layers(search_aug_feats)

            # Use the features from GAB's selected layers for equivariance loss
            # This ensures the loss aligns with the layers actually used
            features_original = self._extract_selected_features(
                search_decomposed, selected_info, gab_info
            )
            features_transformed = self._extract_selected_features(
                search_aug_decomposed, selected_info, gab_info
            )

        output = {
            "cls": head_output["cls"],
            "boxes": head_output["boxes"],
            "cls_prob": head_output["cls_prob"],
            "features_original": features_original,
            "features_transformed": features_transformed,
            "gab_info": gab_info,
        }

        return output

    def _extract_selected_features(
        self,
        decomposed: Dict[int, Dict[str, torch.Tensor]],
        selected_info: Dict,
        gab_info: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from the layers selected by GAB.

        Uses the same layer selection as the main forward pass to
        ensure equivariance loss (Eq. 24-26) trains the same features
        that are used for prediction.

        Args:
            decomposed: Full decomposed features for all layers.
            selected_info: Layer selection information from GAB.
            gab_info: Full GAB info dict.

        Returns:
            Dict 'inv'/'eq'/'cp' → features [B, C, H, W].
        """
        # Use GAB's gather function for consistency
        return self.gab._gather_selected_features(
            decomposed, selected_info, soft_mode=True,
        )

    def forward_inference(
        self,
        search: torch.Tensor,
        current_box: Optional[torch.Tensor] = None,
        initial_box: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference forward pass (Algorithm 2, line 5-11).

        Uses cached template features for efficiency.
        Only processes search region through backbone → EDM → GAB → Head.

        Args:
            search: Search region [B, 3, 384, 384].
            current_box: Current predicted box [B, 4] for scale estimation.
            initial_box: Initial template box [B, 4].

        Returns:
            Dict with predictions.
        """
        assert self._template_decomposed is not None, \
            "Call initialize_template() before inference"

        # ---- Extract and decompose search features ----
        search_feats = self.extract_backbone_features(search)
        search_decomposed = self.decompose_all_layers(search_feats)

        # ---- GAB with cached template ----
        fused_output, gab_info = self.gab(
            decomposed_search=search_decomposed,
            decomposed_template=self._template_decomposed,
            current_box=current_box,
            initial_box=initial_box,
            training=False,
        )

        # ---- Head prediction ----
        head_output = self.head(fused_output)

        # ---- Extract best prediction ----
        best_box, best_score = self.head.get_best_prediction(head_output)

        return {
            "cls": head_output["cls"],
            "boxes": head_output["boxes"],
            "cls_prob": head_output["cls_prob"],
            "best_box": best_box,
            "best_score": best_score,
            "gab_info": gab_info,
        }

    def initialize_template(self, template: torch.Tensor):
        """
        Cache template features for inference (Algorithm 2, line 3-4).

        Called once at the start of each tracking sequence.
        Stores decomposed template features to avoid redundant computation.

        Args:
            template: Template image [B, 3, 192, 192].
        """
        with torch.no_grad():
            self._template_backbone_feats = self.extract_backbone_features(template)
            self._template_decomposed = self.decompose_all_layers(
                self._template_backbone_feats
            )

    def reset_template(self):
        """Clear cached template features."""
        self._template_backbone_feats = None
        self._template_decomposed = None

    def forward(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass dispatching to train/inference mode.

        Args:
            template: Template image [B, 3, 192, 192].
            search: Search region [B, 3, 384, 384].
            **kwargs: Additional arguments passed to the respective method.

        Returns:
            Dict with model outputs.
        """
        if self.training:
            return self.forward_train(template, search, **kwargs)
        else:
            return self.forward_inference(search, **kwargs)


def build_sefnet(cfg) -> SEFNet:
    """
    Factory function to build SEFNet from configuration.

    Constructs all submodules and assembles the full model.

    Args:
        cfg: Configuration object.

    Returns:
        Configured SEFNet instance.
    """
    backbone = build_vit_backbone(cfg)
    edm = build_edm(cfg)
    gab = build_gab(cfg)
    head = build_tracking_head(cfg)

    model = SEFNet(
        backbone=backbone,
        edm=edm,
        gab=gab,
        head=head,
        candidate_layers=cfg.MODEL.CANDIDATE_LAYERS,
    )

    return model
