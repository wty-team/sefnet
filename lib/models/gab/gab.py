"""
Geometry-Aware Bridging Block (GAB)
====================================

Main module integrating all GAB components (Section 3.4).

GAB addresses the heterogeneity across backbone layers: shallow layers
preserve geometric structure suitable for equivariant features, while
deep layers capture semantic abstractions suitable for invariant features.

Processing pipeline:
    1. Compute activation statistics for all L layers → O(L·C·H·W)
    2. Compute reliability R_t^(l)(s) for all (type, layer) pairs
    3. Compute selection scores q_t^(l)(s) = β × R × conf (Eq. 17)
    4. Select optimal layer per type: l_t* = argmax_l q_t^(l)(s)
    5. Extract decomposed features from selected layers
    6. Apply bidirectional cross-attention per subspace (Eq. 21)
    7. Compute fusion weights [α, β, γ] from reliability (Eq. 22)
    8. Fuse: X^fused = α·X̂_eq + β·X̂_inv + γ·X̂_cp

The selective processing reduces computation from 3L full passes
to exactly 3, achieving layer utilization ratio 1/L ≈ 8.3% for L=12
(Appendix C.3, Eq. C.2).

References:
    - Section 3.4: Full GAB description
    - Eq. 17: Selection score
    - Eq. 18-20: Reliability measures
    - Eq. 21: Cross-attention
    - Eq. 22: Adaptive fusion
    - Algorithm 1 (line 7-8): Training integration
    - Algorithm 2 (line 8-9): Inference integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

from .layer_selection import LayerSelection
from .reliability import ReliabilityModule, estimate_scale
from .cross_attention import SubspaceCrossAttentionPipeline


class GeometryAwareBridgingBlock(nn.Module):
    """
    Geometry-Aware Bridging Block (GAB).

    Integrates confidence-based layer selection, reliability scoring,
    bidirectional cross-attention, and adaptive fusion into a unified
    module for producing the final tracking representation.

    Args:
        dim: Feature channel dimension C (default 256).
        n_layers: Number of backbone layers L (default 12).
        n_heads: Number of attention heads (default 8).
        mlp_ratio: Feed-forward expansion ratio (default 4.0).
        tau_inv: Temperature for invariant reliability (Eq. 18).
        tau_fusion: Temperature for fusion softmax τ (Eq. 22).
        selection_temperature: Temperature for soft layer selection.
        drop: Dropout rate.
    """

    def __init__(
        self,
        dim: int = 256,
        n_layers: int = 12,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        tau_inv: float = 0.5,
        tau_fusion: float = 1.0,
        selection_temperature: float = 0.1,
        drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.selection_temperature = selection_temperature

        # ---- Component modules ----

        # Layer selection: evaluates all L layers, picks best per type (Eq. 17)
        self.layer_selection = LayerSelection(
            n_layers=n_layers,
            n_types=3,
            stat_dim=3,
            hidden_dim=16,
        )

        # Reliability: computes R_inv, R_eq, R_cp (Eq. 18-20)
        self.reliability = ReliabilityModule(
            tau_inv=tau_inv,
            tau_fusion=tau_fusion,
        )

        # Cross-attention + fusion pipeline (Eq. 21-22)
        self.cross_attn_fusion = SubspaceCrossAttentionPipeline(
            dim=dim,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            drop=drop,
        )

    def _compute_activation_stats(
        self,
        decomposed_features: Dict[int, Dict[str, torch.Tensor]],
    ) -> Dict[int, torch.Tensor]:
        """
        Compute lightweight activation statistics for all layers.

        These feed into the confidence MLP for layer selection.
        Cost: O(C·H·W) per layer, much cheaper than full EDM convolution.

        Stats per layer: [mean, std, gradient_magnitude] aggregated
        across all three subspace types.

        Args:
            decomposed_features: Dict layer → Dict type → [B, C, H, W].

        Returns:
            Dict layer → stats [B, 3].
        """
        stats = {}
        for layer_idx, feat_dict in decomposed_features.items():
            # Aggregate statistics across all three subspaces
            all_feats = torch.cat([
                feat_dict["eq"], feat_dict["inv"], feat_dict["cp"]
            ], dim=1)  # [B, 3C, H, W]

            B = all_feats.shape[0]
            mu = all_feats.mean(dim=(1, 2, 3))   # [B]
            sigma = all_feats.std(dim=(1, 2, 3))  # [B]

            # Gradient magnitude via finite differences
            gx = all_feats[:, :, :, 1:] - all_feats[:, :, :, :-1]
            gy = all_feats[:, :, 1:, :] - all_feats[:, :, :-1, :]
            grad_mag = (gx.pow(2).mean(dim=(1, 2, 3)) +
                        gy.pow(2).mean(dim=(1, 2, 3))).sqrt()  # [B]

            stats[layer_idx] = torch.stack([mu, sigma, grad_mag], dim=1)  # [B, 3]

        return stats

    def _gather_selected_features(
        self,
        decomposed_features: Dict[int, Dict[str, torch.Tensor]],
        selected_layers: Dict[str, torch.Tensor],
        soft_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Gather features from selected layers for each feature type.

        Hard mode (inference): directly index the selected layer.
        Soft mode (training): weighted average across all layers.

        Args:
            decomposed_features: Dict layer → Dict type → [B, C, H, W].
            selected_layers: Dict type → indices [B] or soft weights [B, L].
            soft_mode: Use soft (differentiable) gathering.

        Returns:
            Dict type → selected features [B, C, H, W].
        """
        feature_types = ["eq", "inv", "cp"]
        layer_indices = sorted(decomposed_features.keys())
        selected = {}

        for t in feature_types:
            if soft_mode:
                # Soft selection: weighted average across layers
                weights = selected_layers[t]  # [B, L]
                B = weights.shape[0]

                # Stack features from all layers: [B, L, C, H, W]
                feat_stack = torch.stack(
                    [decomposed_features[l][t] for l in layer_indices],
                    dim=1,
                )
                # Weighted sum: [B, L, 1, 1, 1] * [B, L, C, H, W] → [B, C, H, W]
                w = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                selected[t] = (w * feat_stack).sum(dim=1)
            else:
                # Hard selection: gather from specific layer per batch element
                B = selected_layers[t].shape[0]
                layer_idx_list = selected_layers[t].tolist()

                # Map layer indices back to dict keys
                feats = []
                for b in range(B):
                    l = int(layer_idx_list[b])
                    feats.append(decomposed_features[l][t][b:b+1])
                selected[t] = torch.cat(feats, dim=0)  # [B, C, H, W]

        return selected

    def forward(
        self,
        decomposed_search: Dict[int, Dict[str, torch.Tensor]],
        decomposed_template: Dict[int, Dict[str, torch.Tensor]],
        current_box: Optional[torch.Tensor] = None,
        initial_box: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Full GAB forward pass.

        Implements the complete pipeline from Algorithm 1 (line 7-8)
        and Algorithm 2 (line 8-9):

        1. Evaluate all layers via activation stats + reliability
        2. Select optimal layer per feature type
        3. Cross-attention matching between template and search
        4. Reliability-based adaptive fusion

        Args:
            decomposed_search: Dict layer → Dict type → [B, C, H_s, W_s].
                Decomposed search region features from EDM.
            decomposed_template: Dict layer → Dict type → [B, C, H_t, W_t].
                Decomposed template features from EDM (cached during inference).
            current_box: Current predicted box [B, 4] (cx, cy, w, h).
                Used for scale estimation during inference.
            initial_box: Initial template box [B, 4] (cx, cy, w, h).
            training: Whether in training mode (soft selection).

        Returns:
            Tuple of:
                - fused_output: [B, N_s, C] fused search representation
                - info_dict: Dict with intermediate results for loss/debug
        """
        info = {}

        # ---- Step 1: Compute activation statistics for all layers ----
        # Cost: O(L · C · H · W), lightweight (Appendix C.1)
        search_stats = self._compute_activation_stats(decomposed_search)

        # ---- Step 2: Compute reliability for all (type, layer) pairs ----
        # Uses template as "original" and search as "transformed"
        # During training, scale augmentation creates the transformation pair
        # During inference, the natural scale change between frames serves
        reliability_all = self.reliability.compute_for_all_layers(
            decomposed_template, decomposed_search
        )
        info["reliability_all"] = reliability_all

        # ---- Step 3: Layer selection (Eq. 17) ----
        # q_t^(l)(s) = β_t^(l) × R_t^(l)(s) × conf_t^(l)
        selected, scores = self.layer_selection(
            search_stats, reliability_all,
            training=training,
            temperature=self.selection_temperature,
        )
        info["selection_scores"] = scores
        info["selected_layers"] = selected

        # ---- Step 4: Gather features from selected layers ----
        soft_mode = training
        search_selected = self._gather_selected_features(
            decomposed_search, selected, soft_mode=soft_mode,
        )
        template_selected = self._gather_selected_features(
            decomposed_template, selected, soft_mode=soft_mode,
        )

        # ---- Step 5: Compute fusion weights (Eq. 22) ----
        # Use reliability from the selected layers for fusion weights
        # For soft mode, use average reliability across weighted layers
        if soft_mode:
            # Approximate: use reliability of the template-search pair
            # at the highest-weighted layer
            rel_selected = self.reliability(template_selected, search_selected)
        else:
            rel_selected = self.reliability(template_selected, search_selected)

        alpha, beta, gamma = self.reliability.compute_fusion_weights(rel_selected)
        info["fusion_weights"] = {"alpha": alpha, "beta": beta, "gamma": gamma}
        info["reliability_selected"] = rel_selected

        # ---- Step 6: Cross-attention + fusion (Eq. 21-22) ----
        fused_output, enhanced = self.cross_attn_fusion(
            search_selected, template_selected,
            alpha, beta, gamma,
        )
        info["enhanced_features"] = enhanced

        return fused_output, info

    def get_layer_preferences(self) -> Dict[str, torch.Tensor]:
        """
        Return the learned layer affinity parameters β_t^(l).

        Useful for visualization and analysis (Table 7 in paper).
        Expected patterns after training:
            eq  → concentrate on L1-4 (78.7%)
            inv → concentrate on L7-12 (80.7%)
            cp  → distribute across L3-8 (71.1%)

        Returns:
            Dict type → affinity weights [L], summing to 1.
        """
        return self.layer_selection.get_layer_affinity()


def build_gab(cfg) -> GeometryAwareBridgingBlock:
    """
    Factory function to build GAB from config.

    Args:
        cfg: Configuration object with GAB parameters.

    Returns:
        Configured GeometryAwareBridgingBlock instance.
    """
    return GeometryAwareBridgingBlock(
        dim=cfg.MODEL.CHANNELS,
        n_layers=cfg.MODEL.N_LAYERS,
        n_heads=cfg.GAB.N_HEADS,
        mlp_ratio=cfg.GAB.MLP_RATIO,
        tau_inv=cfg.GAB.TAU_INV,
        tau_fusion=cfg.GAB.TAU_FUSION,
        selection_temperature=cfg.GAB.SELECTION_TEMPERATURE,
        drop=cfg.GAB.DROPOUT,
    )
