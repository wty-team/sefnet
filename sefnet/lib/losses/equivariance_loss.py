"""
Transformation Consistency Loss (L_eq)
=======================================

Implements the equivariance loss described in Section 3.5, Eq. 24-26.

The transformation consistency loss L_eq = L_eq^inv + L_eq^eq + L_eq^cp
aligns training with the reliability measures used for layer selection
and fusion during inference (Eq. 18-20).

The three terms correspond to the theoretical properties of each subspace:

1. L_eq^inv (Eq. 24): Minimizes normalized feature deviation
   - Directly maximizes R_inv(s) in Eq. 18
   - Based on Theorem 2: ||L_g F - F|| ≤ ε|τ₀| · ||F||

2. L_eq^eq (Eq. 25): Maximizes directional consistency
   - Corresponds to R_eq(s) in Eq. 19
   - Based on Theorem 3: exact rotation invariance, scale sensitivity
   - Includes power-law scaling constraint φ(T_s I) = s^α · φ(I)

3. L_eq^cp (Eq. 26): Measures structural consistency with phase tolerance
   - Corresponds to R_cp(s) in Eq. 20
   - Based on Theorem 4: phase shift e^{-in₀θ₀} under rotation
   - Uses absolute value to accommodate phase modulation

Scale factors s are sampled from log-uniform distribution on [0.5, 2.0]
(Sec 3.5). The gradient scaling under rescaling follows Proposition A.6.

References:
    - Eq. 23: Total loss L = λ_cls·L_cls + λ_reg·L_reg + λ_iou·L_iou + λ_eq·L_eq
    - Eq. 24: L_eq^inv = E[||φ_inv(T_s I) - φ_inv(I)||² / ||φ_inv(I)||²]
    - Eq. 25: L_eq^eq = E[1 - <φ_eq(T_s I), φ_eq(I)> / (||·||·||·||)]
    - Eq. 26: L_eq^cp = E[1 - |<φ_cp(T_s I), φ_cp(I)>| / (||·||·||·||)]
    - Eq. 27-29: Equivariance error metrics E_eq, E_inv, E_cp
    - Appendix B.6: Power-law scaling constraint
    - Proposition A.6: Gradient scaling ∇φ(T_s I) = s^{-1} ∇φ(I)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class InvariantLoss(nn.Module):
    """
    Invariant transformation consistency loss L_eq^inv (Eq. 24).

    Minimizes the normalized feature deviation under scale transformation:

        L_eq^inv = E_{s,I} [||φ_inv(T_s I) - φ_inv(I)||² / ||φ_inv(I)||²]

    This directly maximizes the invariant reliability R_inv(s) (Eq. 18),
    because minimizing the numerator and normalizing by the denominator
    correspond exactly to maximizing the Gaussian kernel in R_inv.

    The loss encourages features in F_inv to satisfy the approximate
    invariance bound from Theorem 2:
        ||L_g F - F|| ≤ ε|τ₀| · ||F||
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        features_original: torch.Tensor,
        features_transformed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute invariant consistency loss.

        Args:
            features_original: φ_inv(I) [B, C, H, W].
            features_transformed: φ_inv(T_s I) [B, C, H, W].

        Returns:
            loss: Scalar, mean normalized squared deviation.
        """
        B = features_original.shape[0]
        f_orig = features_original.reshape(B, -1)
        f_trans = features_transformed.reshape(B, -1)

        # ||φ_inv(T_s I) - φ_inv(I)||²
        diff_sq = (f_trans - f_orig).pow(2).sum(dim=1)  # [B]

        # ||φ_inv(I)||² (normalize to make loss scale-independent)
        norm_sq = f_orig.pow(2).sum(dim=1).clamp(min=1e-8)  # [B]

        # L_eq^inv = E[diff² / norm²] (Eq. 24)
        loss = (diff_sq / norm_sq).mean()

        return loss


class EquivariantLoss(nn.Module):
    """
    Equivariant transformation consistency loss L_eq^eq (Eq. 25).

    Maximizes directional consistency under scale transformation:

        L_eq^eq = E_{s,I} [1 - <φ_eq(T_s I), φ_eq(I)> / (||·|| · ||·||)]

    This corresponds to R_eq(s) in Eq. 19. The loss equals 1 - cosine_similarity,
    so minimizing it pushes features toward perfect directional alignment.

    For the power-law constraint φ(T_s I) = s^α · φ(I) (Appendix B.6),
    perfect equivariance yields cosine similarity = 1 since scaling by
    s^α preserves direction.

    The scale exponent α is learned per layer (Table D.8):
        L1-2: α ≈ 0.92 (near-linear, shallow geometric features)
        L7-8: α ≈ 0.58 (attenuated, deeper semantic features)
        L11-12: α ≈ 0.31 (near-invariant, highest abstraction)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        features_original: torch.Tensor,
        features_transformed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute equivariant consistency loss.

        Args:
            features_original: φ_eq(I) [B, C, H, W].
            features_transformed: φ_eq(T_s I) [B, C, H, W].

        Returns:
            loss: Scalar, mean (1 - cosine_similarity).
        """
        B = features_original.shape[0]
        f_orig = features_original.reshape(B, -1)
        f_trans = features_transformed.reshape(B, -1)

        # Cosine similarity
        dot = (f_trans * f_orig).sum(dim=1)  # [B]
        norm_orig = f_orig.norm(dim=1).clamp(min=1e-8)
        norm_trans = f_trans.norm(dim=1).clamp(min=1e-8)
        cos_sim = dot / (norm_orig * norm_trans)  # [B]

        # L_eq^eq = E[1 - cos_sim] (Eq. 25)
        loss = (1.0 - cos_sim).mean()

        return loss


class CoupledLoss(nn.Module):
    """
    Coupled transformation consistency loss L_eq^cp (Eq. 26).

    Measures structural consistency while accommodating phase variations:

        L_eq^cp = E_{s,I} [1 - |<φ_cp(T_s I), φ_cp(I)>| / (||·|| · ||·||)]

    The absolute value is critical: Theorem 4 predicts that rotation
    by θ₀ induces phase shift e^{-in₀θ₀} on coupled features. When
    scale and rotation interact (as they do in coupled subspace),
    the feature vector may undergo sign flips. The absolute cosine
    similarity captures structural preservation despite such phase
    modulation.

    Corresponds to R_cp(s) in Eq. 20.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        features_original: torch.Tensor,
        features_transformed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute coupled consistency loss.

        Args:
            features_original: φ_cp(I) [B, C, H, W].
            features_transformed: φ_cp(T_s I) [B, C, H, W].

        Returns:
            loss: Scalar, mean (1 - |cosine_similarity|).
        """
        B = features_original.shape[0]
        f_orig = features_original.reshape(B, -1)
        f_trans = features_transformed.reshape(B, -1)

        # Absolute cosine similarity (accommodates phase modulation)
        dot = (f_trans * f_orig).sum(dim=1)
        norm_orig = f_orig.norm(dim=1).clamp(min=1e-8)
        norm_trans = f_trans.norm(dim=1).clamp(min=1e-8)
        abs_cos_sim = torch.abs(dot) / (norm_orig * norm_trans)  # [B]

        # L_eq^cp = E[1 - |cos_sim|] (Eq. 26)
        loss = (1.0 - abs_cos_sim).mean()

        return loss


class TransformationConsistencyLoss(nn.Module):
    """
    Combined transformation consistency loss L_eq (Sec 3.5).

    L_eq = L_eq^inv + L_eq^eq + L_eq^cp

    This loss aligns training with inference by penalizing features
    that violate their designated transformation properties from
    Theorems 2-4. The three terms directly correspond to the three
    reliability measures (Eq. 18-20) used for layer selection and
    fusion during inference.

    Scale augmentation:
        During training, scale factors s are sampled from a log-uniform
        distribution on [0.5, 2.0] (Sec 3.5). The input image I is
        augmented with T_s to create transformation pairs.

    Args:
        weight_inv: Weight for invariant loss term (default 1.0).
        weight_eq: Weight for equivariant loss term (default 1.0).
        weight_cp: Weight for coupled loss term (default 1.0).
    """

    def __init__(
        self,
        weight_inv: float = 1.0,
        weight_eq: float = 1.0,
        weight_cp: float = 1.0,
    ):
        super().__init__()
        self.weight_inv = weight_inv
        self.weight_eq = weight_eq
        self.weight_cp = weight_cp

        self.loss_inv = InvariantLoss()
        self.loss_eq = EquivariantLoss()
        self.loss_cp = CoupledLoss()

    def forward(
        self,
        features_original: Dict[str, torch.Tensor],
        features_transformed: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the full transformation consistency loss.

        Args:
            features_original: Dict 'inv'/'eq'/'cp' → [B, C, H, W].
                Features from the original input image φ(I).
            features_transformed: Dict 'inv'/'eq'/'cp' → [B, C, H, W].
                Features from the scale-augmented image φ(T_s I).

        Returns:
            Tuple of:
                - total_loss: Scalar, weighted sum of three terms.
                - loss_dict: Dict with individual loss values for logging.
        """
        l_inv = self.loss_inv(
            features_original["inv"], features_transformed["inv"]
        )
        l_eq = self.loss_eq(
            features_original["eq"], features_transformed["eq"]
        )
        l_cp = self.loss_cp(
            features_original["cp"], features_transformed["cp"]
        )

        total = (self.weight_inv * l_inv +
                 self.weight_eq * l_eq +
                 self.weight_cp * l_cp)

        loss_dict = {
            "loss_eq_inv": l_inv,
            "loss_eq_eq": l_eq,
            "loss_eq_cp": l_cp,
            "loss_eq_total": total,
        }

        return total, loss_dict


def sample_scale_factor(
    batch_size: int,
    scale_range: Tuple[float, float] = (0.5, 2.0),
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Sample scale factors from log-uniform distribution (Sec 3.5).

    The log-uniform distribution ensures equal probability density
    in the logarithmic scale domain, so s=0.5 and s=2.0 are
    equally likely in log-space (symmetric around s=1.0).

    Args:
        batch_size: Number of scale factors to sample.
        scale_range: (s_min, s_max), default (0.5, 2.0).
        device: Target device.

    Returns:
        scales: Sampled scale factors [B].
    """
    s_min, s_max = scale_range
    log_min = torch.log(torch.tensor(s_min))
    log_max = torch.log(torch.tensor(s_max))

    # Uniform in log-space → log-uniform in linear space
    log_s = torch.rand(batch_size, device=device) * (log_max - log_min) + log_min
    scales = torch.exp(log_s)

    return scales


class EquivarianceErrorMetrics(nn.Module):
    """
    Equivariance error metrics for evaluation (Eq. 27-29).

    These metrics verify that learned features satisfy the transformation
    laws in Theorems 2-4. They are averaged over s ∈ {0.5, 0.75, 1.5, 2.0}
    as described in Section 5.2.

    E_eq(s) = ||φ_eq(T_s I) - s^α · φ_eq(I)|| / ||φ_eq(I)||  (Eq. 27)
    E_inv(s) = ||φ_inv(T_s I) - φ_inv(I)|| / ||φ_inv(I)||     (Eq. 28)
    E_cp(s) = ||∇φ_cp(T_s I) - s^{-1} ∇φ_cp(I)|| / ||∇φ_cp(I)||  (Eq. 29)

    The gradient-based E_cp follows from Proposition A.6 which establishes
    that spatial gradients scale as s^{-1} under rescaling.
    """

    def __init__(self, eval_scales: Tuple[float, ...] = (0.5, 0.75, 1.5, 2.0)):
        super().__init__()
        self.eval_scales = eval_scales

    def compute_E_eq(
        self,
        features_original: torch.Tensor,
        features_transformed: torch.Tensor,
        scale: float,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Equivariance error E_eq (Eq. 27).

        E_eq = ||φ_eq(T_s I) - s^α · φ_eq(I)|| / ||φ_eq(I)||

        Tests the power-law scaling constraint from Appendix B.6.
        """
        B = features_original.shape[0]
        f_orig = features_original.reshape(B, -1)
        f_trans = features_transformed.reshape(B, -1)

        expected = (scale ** alpha) * f_orig
        error = (f_trans - expected).norm(dim=1)
        norm = f_orig.norm(dim=1).clamp(min=1e-8)

        return (error / norm).mean()

    def compute_E_inv(
        self,
        features_original: torch.Tensor,
        features_transformed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Invariance error E_inv (Eq. 28).

        E_inv = ||φ_inv(T_s I) - φ_inv(I)|| / ||φ_inv(I)||

        Tests the approximate invariance from Theorem 2.
        """
        B = features_original.shape[0]
        f_orig = features_original.reshape(B, -1)
        f_trans = features_transformed.reshape(B, -1)

        error = (f_trans - f_orig).norm(dim=1)
        norm = f_orig.norm(dim=1).clamp(min=1e-8)

        return (error / norm).mean()

    def compute_E_cp(
        self,
        features_original: torch.Tensor,
        features_transformed: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """
        Coupled error E_cp (Eq. 29).

        E_cp = ||∇φ_cp(T_s I) - s^{-1} ∇φ_cp(I)|| / ||∇φ_cp(I)||

        Tests the gradient scaling from Proposition A.6:
            ∇φ(T_s I)(x) = s^{-1} · ∇φ(I)(s^{-1}x)
        """
        # Compute spatial gradients via finite differences
        grad_orig_x = features_original[:, :, :, 1:] - features_original[:, :, :, :-1]
        grad_orig_y = features_original[:, :, 1:, :] - features_original[:, :, :-1, :]

        grad_trans_x = features_transformed[:, :, :, 1:] - features_transformed[:, :, :, :-1]
        grad_trans_y = features_transformed[:, :, 1:, :] - features_transformed[:, :, :-1, :]

        # Expected: s^{-1} · ∇φ(I) (Proposition A.6)
        expected_x = (1.0 / scale) * grad_orig_x
        expected_y = (1.0 / scale) * grad_orig_y

        # Truncate to matching sizes
        min_w = min(grad_trans_x.shape[3], expected_x.shape[3])
        min_h = min(grad_trans_y.shape[2], expected_y.shape[2])

        error_x = (grad_trans_x[:, :, :, :min_w] - expected_x[:, :, :, :min_w]).pow(2)
        error_y = (grad_trans_y[:, :, :min_h, :] - expected_y[:, :, :min_h, :]).pow(2)

        error = (error_x.sum() + error_y.sum()).sqrt()
        norm = (grad_orig_x.pow(2).sum() + grad_orig_y.pow(2).sum()).sqrt().clamp(min=1e-8)

        return error / norm
