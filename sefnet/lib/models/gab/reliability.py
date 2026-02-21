"""
Scale-Dependent Reliability Measures
======================================

Implements the three reliability measures described in Section 3.4, Eq. 18-20.

Each measure evaluates whether features from a given layer satisfy their
designated transformation property at the current scale factor s.

The three measures correspond directly to the theoretical properties
of each subspace (Theorems 2-4 in the main paper):

1. R_inv^(l)(s): Invariant reliability (Eq. 18)
   - Measures approximate stability under scale transformation
   - Based on Theorem 2 (Quantitative Invariance): ||L_g F - F|| ≤ ε|τ₀|·||F||
   - High when features change little despite scale change

2. R_eq^(l)(s): Equivariant reliability (Eq. 19)
   - Measures directional consistency under scale transformation
   - Based on Theorem 3 (Rotation Invariance and Scale Sensitivity)
   - High when features maintain their direction while magnitude scales

3. R_cp^(l)(s): Coupled reliability (Eq. 20)
   - Measures structural consistency accommodating phase variations
   - Based on Theorem 4 (Scale-Rotation Coupling): phase shift e^{-in₀θ₀}
   - Uses absolute cosine similarity to accommodate phase modulation

These reliability scores serve dual purposes:
    (a) As factors in the layer selection score (Eq. 17)
    (b) As fusion weights via temperature-scaled softmax (Eq. 22)

The training losses (Eq. 24-26) are designed to maximize these reliability
measures, creating direct alignment between training and inference.

References:
    - Eq. 18: R_inv = exp(-||φ_inv(T_s I) - φ_inv(I)||² / (2τ²||φ_inv(I)||²))
    - Eq. 19: R_eq = <φ_eq(T_s I), φ_eq(I)> / (||·||·||·||)
    - Eq. 20: R_cp = |<φ_cp(T_s I), φ_cp(I)>| / (||·||·||·||)
    - Eq. 22: [α, β, γ] = Softmax(1/τ · [R_eq, R_inv, R_cp])
    - Theorems 2-4: Theoretical properties of each subspace
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class InvariantReliability(nn.Module):
    """
    Invariant reliability measure R_inv^(l)(s) (Eq. 18).

    Evaluates whether invariant features remain stable under scale
    transformation, as guaranteed by Theorem 2:

        R_inv^(l)(s) = exp(-||φ_inv(T_s I) - φ_inv(I)||² / (2τ_inv² · ||φ_inv(I)||²))

    Interpretation:
        - R_inv → 1: Features are highly stable (good invariance)
        - R_inv → 0: Features deviate significantly (poor invariance)

    The temperature τ_inv controls sensitivity:
        - Small τ_inv → strict invariance requirement
        - Large τ_inv → more tolerant of deviations

    Args:
        tau: Temperature parameter τ_inv (default 0.5).
    """

    def __init__(self, tau: float = 0.5):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        features_original: torch.Tensor,
        features_transformed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute invariant reliability score.

        Args:
            features_original: φ_inv(I) [B, C, H, W] — features from original input.
            features_transformed: φ_inv(T_s I) [B, C, H, W] — features from scaled input.

        Returns:
            R_inv: Reliability score [B, 1] in range (0, 1].
        """
        B = features_original.shape[0]

        # Flatten spatial dimensions for norm computation
        f_orig = features_original.reshape(B, -1)     # [B, C*H*W]
        f_trans = features_transformed.reshape(B, -1)  # [B, C*H*W]

        # Numerator: ||φ_inv(T_s I) - φ_inv(I)||²
        diff_sq = (f_trans - f_orig).pow(2).sum(dim=1)  # [B]

        # Denominator: 2τ² · ||φ_inv(I)||²
        orig_sq = f_orig.pow(2).sum(dim=1).clamp(min=1e-8)  # [B]
        denom = 2.0 * self.tau ** 2 * orig_sq

        # R_inv = exp(-numerator / denominator) (Eq. 18)
        R_inv = torch.exp(-diff_sq / denom)  # [B]

        return R_inv.unsqueeze(1)  # [B, 1]


class EquivariantReliability(nn.Module):
    """
    Equivariant reliability measure R_eq^(l)(s) (Eq. 19).

    Evaluates whether equivariant features preserve their directional
    consistency under scale transformation, based on Theorem 3:

        R_eq^(l)(s) = <φ_eq(T_s I), φ_eq(I)> / (||φ_eq(T_s I)|| · ||φ_eq(I)||)

    This is the cosine similarity between original and transformed features.
    Since Theorem 3 establishes that functions in F_eq satisfy exact rotation
    invariance and respond to scale as 1D shifts in log-scale domain,
    directional consistency (high cosine similarity) indicates proper
    equivariant behavior.

    For the power-law scaling constraint φ(T_s I) = s^α · φ(I)
    (Appendix B.6), perfect equivariance yields R_eq = 1 since scaling
    by s^α preserves the direction of the feature vector.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        features_original: torch.Tensor,
        features_transformed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute equivariant reliability score.

        Args:
            features_original: φ_eq(I) [B, C, H, W].
            features_transformed: φ_eq(T_s I) [B, C, H, W].

        Returns:
            R_eq: Reliability score [B, 1] in range [-1, 1].
                  Values near 1 indicate good equivariance.
        """
        B = features_original.shape[0]

        f_orig = features_original.reshape(B, -1)     # [B, C*H*W]
        f_trans = features_transformed.reshape(B, -1)  # [B, C*H*W]

        # Cosine similarity: <φ_eq(T_s I), φ_eq(I)> / (||·|| · ||·||)  (Eq. 19)
        dot_product = (f_trans * f_orig).sum(dim=1)  # [B]
        norm_orig = f_orig.norm(dim=1).clamp(min=1e-8)  # [B]
        norm_trans = f_trans.norm(dim=1).clamp(min=1e-8)  # [B]

        R_eq = dot_product / (norm_orig * norm_trans)  # [B]

        return R_eq.unsqueeze(1)  # [B, 1]


class CoupledReliability(nn.Module):
    """
    Coupled reliability measure R_cp^(l)(s) (Eq. 20).

    Evaluates structural consistency of coupled features while
    accommodating phase variations predicted by Theorem 4:

        R_cp^(l)(s) = |<φ_cp(T_s I), φ_cp(I)>| / (||φ_cp(T_s I)|| · ||φ_cp(I)||)

    The absolute value is crucial: Theorem 4 states that rotation by θ₀
    induces a phase shift e^{-in₀θ₀} on coupled features. When scale
    and rotation interact, the feature vector may flip sign (phase shift
    of π), but the structural pattern is preserved. The absolute cosine
    similarity captures this structural consistency despite phase modulation.

    Interpretation:
        - |R_cp| → 1: Structural pattern preserved (with possible phase flip)
        - |R_cp| → 0: Structural pattern lost
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        features_original: torch.Tensor,
        features_transformed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute coupled reliability score.

        Args:
            features_original: φ_cp(I) [B, C, H, W].
            features_transformed: φ_cp(T_s I) [B, C, H, W].

        Returns:
            R_cp: Reliability score [B, 1] in range [0, 1].
        """
        B = features_original.shape[0]

        f_orig = features_original.reshape(B, -1)     # [B, C*H*W]
        f_trans = features_transformed.reshape(B, -1)  # [B, C*H*W]

        # Absolute cosine similarity (Eq. 20)
        # The |·| accommodates the phase modulation from Theorem 4
        dot_product = (f_trans * f_orig).sum(dim=1)  # [B]
        norm_orig = f_orig.norm(dim=1).clamp(min=1e-8)
        norm_trans = f_trans.norm(dim=1).clamp(min=1e-8)

        R_cp = torch.abs(dot_product) / (norm_orig * norm_trans)  # [B]

        return R_cp.unsqueeze(1)  # [B, 1]


class ReliabilityModule(nn.Module):
    """
    Unified module for computing all three reliability measures.

    Combines R_inv, R_eq, R_cp into a single module for convenience.
    Also handles the scale estimation and feature transformation
    needed to compute reliability during inference.

    During inference (Algorithm 2):
        1. Scale parameter s_t = sqrt(w_t * h_t / w_0 * h_0)
        2. Template features serve as φ(I) (cached from frame 1)
        3. Search features serve as φ(T_s I) (extracted each frame)
        4. Reliability scores determine layer selection and fusion weights

    During training (Algorithm 1):
        1. Scale s is sampled from the log-uniform distribution
        2. Input image I is augmented with T_s to create the pair
        3. Both φ(I) and φ(T_s I) are computed through the full pipeline

    Args:
        tau_inv: Temperature for invariant reliability (default 0.5).
        tau_fusion: Temperature for reliability-based fusion softmax (Eq. 22).
    """

    def __init__(
        self,
        tau_inv: float = 0.5,
        tau_fusion: float = 1.0,
    ):
        super().__init__()

        self.R_inv = InvariantReliability(tau=tau_inv)
        self.R_eq = EquivariantReliability()
        self.R_cp = CoupledReliability()
        self.tau_fusion = tau_fusion

    def forward(
        self,
        features_original: Dict[str, torch.Tensor],
        features_transformed: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reliability scores for all three subspaces.

        Args:
            features_original: Dict with keys 'inv', 'eq', 'cp',
                each [B, C, H, W] from the original/template image.
            features_transformed: Dict with keys 'inv', 'eq', 'cp',
                each [B, C, H, W] from the scaled/search image.

        Returns:
            Dict with keys 'inv', 'eq', 'cp', each [B, 1] reliability score.
        """
        return {
            "inv": self.R_inv(
                features_original["inv"],
                features_transformed["inv"],
            ),
            "eq": self.R_eq(
                features_original["eq"],
                features_transformed["eq"],
            ),
            "cp": self.R_cp(
                features_original["cp"],
                features_transformed["cp"],
            ),
        }

    def compute_fusion_weights(
        self,
        reliability: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute adaptive fusion weights from reliability scores (Eq. 22).

        [α, β, γ] = Softmax(1/τ · [R_eq(s), R_inv(s), R_cp(s)])

        The temperature τ controls the sharpness of the fusion:
            - Small τ → winner-take-all (one subspace dominates)
            - Large τ → equal weighting (all subspaces contribute equally)

        The ordering follows the paper convention (Eq. 22):
            α → equivariant weight
            β → invariant weight
            γ → coupled weight

        As the target scale increases, geometric boundaries become prominent
        and the equivariant weight α rises. As the target scale decreases,
        invariant features provide more stable cues and β dominates.

        Args:
            reliability: Dict 'inv'/'eq'/'cp' → [B, 1].

        Returns:
            Tuple (α, β, γ), each [B, 1], summing to 1.
        """
        # Stack: [B, 3] — order: eq, inv, cp (matching Eq. 22)
        R_stack = torch.cat([
            reliability["eq"],
            reliability["inv"],
            reliability["cp"],
        ], dim=1)  # [B, 3]

        # Temperature-scaled softmax (Eq. 22)
        weights = F.softmax(R_stack / self.tau_fusion, dim=1)  # [B, 3]

        alpha = weights[:, 0:1]  # equivariant weight [B, 1]
        beta = weights[:, 1:2]   # invariant weight [B, 1]
        gamma = weights[:, 2:3]  # coupled weight [B, 1]

        return alpha, beta, gamma

    def compute_for_all_layers(
        self,
        decomposed_original: Dict[int, Dict[str, torch.Tensor]],
        decomposed_transformed: Dict[int, Dict[str, torch.Tensor]],
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        Compute reliability scores for all layers and feature types.

        Used by the layer selection mechanism to evaluate all L layers.
        Each (type, layer) pair gets an independent reliability score.

        Args:
            decomposed_original: Dict layer → Dict type → [B, C, H, W].
            decomposed_transformed: Dict layer → Dict type → [B, C, H, W].

        Returns:
            Dict type → Dict layer → reliability score [B, 1].

        Complexity: O(L · C · H · W) total for all layers and types.
        """
        reliability_all = {"eq": {}, "inv": {}, "cp": {}}

        for layer_idx in decomposed_original.keys():
            orig = decomposed_original[layer_idx]
            trans = decomposed_transformed[layer_idx]

            # Compute reliability for each type at this layer
            reliability_all["inv"][layer_idx] = self.R_inv(
                orig["inv"], trans["inv"]
            )
            reliability_all["eq"][layer_idx] = self.R_eq(
                orig["eq"], trans["eq"]
            )
            reliability_all["cp"][layer_idx] = self.R_cp(
                orig["cp"], trans["cp"]
            )

        return reliability_all


def estimate_scale(
    current_box: torch.Tensor,
    initial_box: torch.Tensor,
) -> torch.Tensor:
    """
    Recursive scale estimation from bounding box dimensions.

    s_t = sqrt(w_t · h_t / w_0 · h_0)

    This provides the scale parameter needed for reliability computation
    during inference (Algorithm 2, line 7).

    Args:
        current_box: Current predicted box [B, 4] as (cx, cy, w, h).
        initial_box: Initial template box [B, 4] as (cx, cy, w, h).

    Returns:
        scale: Estimated scale factor [B, 1].
    """
    # Extract width and height (last two elements)
    w_t, h_t = current_box[:, 2], current_box[:, 3]
    w_0, h_0 = initial_box[:, 2], initial_box[:, 3]

    # s_t = sqrt(w_t * h_t / (w_0 * h_0))
    area_ratio = (w_t * h_t) / (w_0 * h_0).clamp(min=1e-8)
    scale = torch.sqrt(area_ratio.clamp(min=1e-8))

    return scale.unsqueeze(1)  # [B, 1]
