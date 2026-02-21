"""
Confidence-Based Layer Selection
=================================

Implements the layer selection mechanism described in Section 3.4 of the paper.

For each feature type t ∈ {eq, inv, cp} and backbone layer l ∈ {1, ..., L},
a selection score combines three multiplicative factors (Eq. 17):

    q_t^(l)(s) = β_t^(l) × R_t^(l)(s) × conf_t^(l)

where:
    - β_t^(l): Learned layer affinity, softmax-normalized so Σ_l β_t^(l) = 1
    - R_t^(l)(s): Scale-dependent reliability (see reliability.py, Eq. 18-20)
    - conf_t^(l): Feature quality from activation statistics via lightweight MLP

Layer selection identifies the optimal layer per feature type:
    l_t* = argmax_l q_t^(l)(s)

This dynamically favors:
    - Shallow layers for large targets with prominent geometric boundaries
    - Deeper layers for small targets that require semantic robustness

The learned preferences (Table 7 in paper):
    - Equivariant features → L1-4 (78.7% frequency)
    - Invariant features   → L7-12 (80.7% frequency)
    - Coupled features     → L3-8 (71.1% frequency)

Complexity: O(L·C·H·W) for evaluating all layers (Appendix C.1, Eq. C.1).
This is negligible compared to the O(C²·H·W·K²) per-layer convolution.

References:
    - Eq. 17: Selection score q_t^(l)(s) = β × R × conf
    - Table 7: Learned layer preferences
    - Appendix C.1: Layer selection complexity O(LCHW)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class LayerSelection(nn.Module):
    """
    Confidence-based layer selection for the Geometry-Aware Bridging Block.

    Evaluates all L backbone layers and selects the optimal layer for each
    of the three feature types based on a composite score (Eq. 17).

    Args:
        n_layers: Number of backbone layers L (default 12 for ViT-Base).
        n_types: Number of feature types (default 3: eq, inv, cp).
        stat_dim: Dimension of activation statistics vector (default 3).
        hidden_dim: Hidden dimension of the confidence MLP (default 16).
    """

    FEATURE_TYPES = ("eq", "inv", "cp")

    def __init__(
        self,
        n_layers: int = 12,
        n_types: int = 3,
        stat_dim: int = 3,
        hidden_dim: int = 16,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_types = n_types

        # ----------------------------------------------------------------
        # Layer affinity parameters β_t^(l).
        #
        # Learned via softmax-normalized parameters satisfying:
        #   Σ_l β_t^(l) = 1  for each feature type t
        #
        # These are initialized uniformly and learn to concentrate on
        # specific layer groups during training (Table 7):
        #   eq  → shallow layers (L1-4)
        #   inv → deep layers (L7-12)
        #   cp  → mid layers (L3-8)
        # ----------------------------------------------------------------
        # Shape: [n_types, n_layers] — raw logits before softmax
        self.affinity_logits = nn.Parameter(
            torch.zeros(n_types, n_layers)
        )

        # ----------------------------------------------------------------
        # Feature quality MLP: conf_t^(l) = MLP([μ, σ, g]).
        #
        # Maps activation statistics (mean, std, gradient magnitude) to
        # a scalar confidence score. Lightweight: O(d_h) per evaluation,
        # dominated by the O(CHW) cost of computing the statistics.
        #
        # One MLP per feature type to allow type-specific quality criteria.
        # ----------------------------------------------------------------
        self.conf_mlps = nn.ModuleDict()
        for t in self.FEATURE_TYPES:
            self.conf_mlps[t] = nn.Sequential(
                nn.Linear(stat_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),  # Confidence in [0, 1]
            )

    def get_layer_affinity(self) -> Dict[str, torch.Tensor]:
        """
        Compute softmax-normalized layer affinity β_t^(l) for each type.

        The softmax ensures Σ_l β_t^(l) = 1, so affinities form a
        probability distribution over layers for each feature type.

        Returns:
            Dict mapping feature type → affinity tensor [L].
        """
        # Apply softmax over the layer dimension for each type
        affinities = F.softmax(self.affinity_logits, dim=1)  # [n_types, L]

        return {
            t: affinities[i]
            for i, t in enumerate(self.FEATURE_TYPES)
        }

    def compute_confidence(
        self,
        activation_stats: Dict[int, torch.Tensor],
        feature_type: str,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute feature quality confidence scores for all layers.

        conf_t^(l) maps activation statistics [μ^(l), σ^(l), g^(l)]
        through a lightweight MLP to a scalar in [0, 1].

        Args:
            activation_stats: Dict mapping layer index → stats tensor [B, 3].
                             Stats = [mean, std, gradient_magnitude].
            feature_type: One of 'eq', 'inv', 'cp'.

        Returns:
            Dict mapping layer index → confidence score [B, 1].

        Complexity: O(d_h) per layer, negligible vs O(CHW) for stats.
        """
        mlp = self.conf_mlps[feature_type]
        confidences = {}
        for layer_idx, stats in activation_stats.items():
            confidences[layer_idx] = mlp(stats)  # [B, 1]
        return confidences

    def compute_selection_scores(
        self,
        activation_stats: Dict[int, torch.Tensor],
        reliability_scores: Dict[str, Dict[int, torch.Tensor]],
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        Compute the full selection score for each (type, layer) pair.

        Implements Eq. 17:
            q_t^(l)(s) = β_t^(l) × R_t^(l)(s) × conf_t^(l)

        The three factors capture complementary aspects:
            - β (affinity): Which layers are generally best for this type
            - R (reliability): How well this layer satisfies the type's
              transformation property at the current scale s
            - conf (quality): How informative this layer's features are

        Args:
            activation_stats: Dict layer → stats [B, 3] for all layers.
            reliability_scores: Dict type → Dict layer → R_t^(l)(s) [B, 1].

        Returns:
            Dict type → Dict layer → selection score [B, 1].
        """
        affinities = self.get_layer_affinity()  # type → [L]
        scores = {}

        for t_idx, t in enumerate(self.FEATURE_TYPES):
            # Confidence scores for this type across all layers
            conf = self.compute_confidence(activation_stats, t)

            scores[t] = {}
            for layer_idx in activation_stats.keys():
                # β_t^(l): scalar, layer affinity
                beta = affinities[t][layer_idx]

                # R_t^(l)(s): [B, 1], scale-dependent reliability
                R = reliability_scores[t][layer_idx]

                # conf_t^(l): [B, 1], feature quality
                c = conf[layer_idx]

                # q_t^(l)(s) = β × R × conf (Eq. 17)
                # β is scalar, broadcast to batch dimension
                scores[t][layer_idx] = beta * R * c  # [B, 1]

        return scores

    def select_layers(
        self,
        scores: Dict[str, Dict[int, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Select the optimal layer for each feature type.

        l_t* = argmax_l q_t^(l)(s)

        For training, we use soft selection (weighted average) to maintain
        gradient flow. For inference, we use hard selection (argmax).

        Args:
            scores: Dict type → Dict layer → score [B, 1].

        Returns:
            Dict mapping feature type → selected layer indices [B].
        """
        selected = {}

        for t in self.FEATURE_TYPES:
            # Stack scores across layers: [B, L]
            layer_indices = sorted(scores[t].keys())
            score_stack = torch.cat(
                [scores[t][l] for l in layer_indices], dim=1
            )  # [B, L]

            # Hard selection: argmax
            best_idx = score_stack.argmax(dim=1)  # [B]
            # Map back to actual layer indices
            idx_tensor = torch.tensor(layer_indices, device=best_idx.device)
            selected[t] = idx_tensor[best_idx]  # [B]

        return selected

    def select_layers_soft(
        self,
        scores: Dict[str, Dict[int, torch.Tensor]],
        temperature: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Soft layer selection using Gumbel-Softmax for differentiable training.

        During training, we need gradients to flow through the layer selection
        to update the affinity parameters β_t^(l). Gumbel-Softmax provides
        a differentiable relaxation of the argmax operation.

        Args:
            scores: Dict type → Dict layer → score [B, 1].
            temperature: Softmax temperature (lower → more peaked).

        Returns:
            Dict mapping feature type → soft selection weights [B, L].
        """
        soft_weights = {}

        for t in self.FEATURE_TYPES:
            layer_indices = sorted(scores[t].keys())
            score_stack = torch.cat(
                [scores[t][l] for l in layer_indices], dim=1
            )  # [B, L]

            # Temperature-scaled softmax for soft selection
            weights = F.softmax(score_stack / temperature, dim=1)  # [B, L]
            soft_weights[t] = weights

        return soft_weights

    def forward(
        self,
        activation_stats: Dict[int, torch.Tensor],
        reliability_scores: Dict[str, Dict[int, torch.Tensor]],
        training: bool = True,
        temperature: float = 0.1,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[int, torch.Tensor]]]:
        """
        Full layer selection pipeline.

        1. Compute selection scores q_t^(l)(s) for all (type, layer) pairs
        2. Select optimal layers (soft during training, hard during inference)

        Args:
            activation_stats: Dict layer → stats [B, 3].
            reliability_scores: Dict type → Dict layer → R [B, 1].
            training: Whether to use soft (differentiable) selection.
            temperature: Softmax temperature for soft selection.

        Returns:
            Tuple of:
                - selected: Dict type → layer indices [B] or soft weights [B, L]
                - scores: Dict type → Dict layer → score [B, 1]
        """
        # Compute composite selection scores (Eq. 17)
        scores = self.compute_selection_scores(
            activation_stats, reliability_scores
        )

        # Select layers
        if training:
            selected = self.select_layers_soft(scores, temperature)
        else:
            selected = self.select_layers(scores)

        return selected, scores
