"""
Coefficient Generator for EDM Parameterized Convolution.

Implements the input-adaptive kernel weight generation described in Sec 3.3:

    "A global context vector z = GAP(F) ∈ R^C is computed through global
     average pooling. For each subspace type t ∈ {inv, eq, cp}, a coefficient
     generator g_t parameterized as a two-layer MLP produces the basis weights
     w_t = g_t(z) ∈ R^{N_t}."

This design separates two concerns (Sec 3.3):
    1. Basis functions {ψ_k^(t)} GUARANTEE transformation properties
       (Theorems 2-4)
    2. Coefficients {w_t^(k)} ADAPT to task-specific patterns for
       the current input

Rotation Invariance (Proposition A.7, Appendix B.6):
    "The pathway F → z = GAP(F) → w_t = g_t(z) preserves rotation
     invariance, i.e., g_t(GAP(R_θ F)) = g_t(GAP(F)) for any rotation R_θ."

    This holds because GAP sums over ALL spatial positions, which is
    invariant under permutation (rotation permutes positions but not
    the set of values). Since the MLP input is identical for F and
    R_θ F, the output coefficients are also identical.

Computational cost (Appendix C.2):
    "The coefficient generator contributes O(C·D_h + D_h·N·C_out)"
    For C=256, D_h=64, N=40: ~6.7×10^5 operations (< 0.065% of EDM cost)
"""

import torch
import torch.nn as nn


class CoefficientGenerator(nn.Module):
    """
    Generates input-adaptive basis function coefficients via GAP + MLP.

    Architecture:
        Input feature F [B, C, H, W]
          → GAP: spatial average → z [B, C]
          → MLP: Linear(C, D_h) → GELU → Linear(D_h, N_basis × C_out)
          → reshape → w [B, C_out, N_basis]

    The generated coefficients w_t^(k) weight each basis function ψ_k^(t)
    to construct the final convolution kernel (Eq.10):
        Ψ_t(r, θ) = Σ_k w_t^(k) · ψ_k^(t)(r, θ)

    Args:
        in_channels: Input feature channels C (default 256).
        num_bases: Number of basis functions N_t for this subspace.
        out_channels: Output feature channels C_out (default 256).
        hidden_dim: MLP hidden dimension D_h (default 64, Sec 5.1.3).
    """

    def __init__(self, in_channels, num_bases, out_channels=None, hidden_dim=64):
        super().__init__()
        self.in_channels = in_channels
        self.num_bases = num_bases
        self.out_channels = out_channels or in_channels
        self.hidden_dim = hidden_dim

        # =====================================================================
        # Two-layer MLP coefficient generator (Sec 5.1.3):
        #   "Coefficient generators are two-layer MLPs with hidden dimension 64"
        #
        # Input:  z ∈ R^C (global context vector from GAP)
        # Output: w ∈ R^{N_basis × C_out} (basis weights per output channel)
        #
        # Complexity (Appendix C.2):
        #   Layer 1: C × D_h = 256 × 64 = 16,384 ops
        #   Layer 2: D_h × (N × C_out) = 64 × (N × 256) ops
        # =====================================================================
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_bases * self.out_channels),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize MLP weights for stable training."""
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features):
        """
        Generate basis coefficients from input features.

        Step 1 (GAP): Compute rotation-invariant global context
            z = GAP(F) = (1/HW) Σ_{h,w} F(h,w)  ∈ R^C

            This is rotation-invariant because spatial averaging is
            invariant under any permutation of positions (Prop A.7).

        Step 2 (MLP): Map context to basis weights
            w_t = g_t(z)  ∈ R^{N_t × C_out}

        Args:
            features: Input feature map [B, C, H, W] from a backbone layer.

        Returns:
            coefficients: Basis weights [B, C_out, N_basis].
                Each entry w[b, c, k] weights basis function k for
                output channel c at batch element b.
        """
        B = features.shape[0]

        # Step 1: Global Average Pooling → rotation-invariant context
        # [B, C, H, W] → [B, C]
        z = features.mean(dim=[2, 3])  # GAP over spatial dimensions

        # Step 2: MLP maps context to per-channel basis weights
        # [B, C] → [B, N_basis × C_out]
        w = self.mlp(z)

        # Reshape: [B, N_basis × C_out] → [B, C_out, N_basis]
        w = w.reshape(B, self.out_channels, self.num_bases)

        return w


class SubspaceCoefficientGenerators(nn.Module):
    """
    Collection of three coefficient generators, one per subspace.

    Creates generators for:
        - Invariant subspace:   g_inv(z) → w_inv ∈ R^{N_inv}
        - Equivariant subspace: g_eq(z)  → w_eq  ∈ R^{N_eq}
        - Coupled subspace:     g_cp(z)  → w_cp  ∈ R^{N_cp}

    From Sec 3.3:
        "The basis functions encode the geometric constraints from the
         theory, while the learnable coefficients adapt to task-specific
         patterns."

    Args:
        cfg: Configuration dict with MODEL.EDM fields.
        in_channels: Input feature channels C (default from cfg).
    """

    def __init__(self, cfg, in_channels=None):
        super().__init__()
        C = in_channels or cfg.MODEL.BACKBONE.PROJECT_DIM  # 256
        D_h = cfg.MODEL.EDM.COEFF_HIDDEN_DIM                # 64

        # One generator per subspace type t ∈ {inv, eq, cp}
        self.gen_inv = CoefficientGenerator(
            in_channels=C,
            num_bases=cfg.MODEL.EDM.N_INV,   # 8
            out_channels=C,
            hidden_dim=D_h,
        )
        self.gen_eq = CoefficientGenerator(
            in_channels=C,
            num_bases=cfg.MODEL.EDM.N_EQ,    # 16
            out_channels=C,
            hidden_dim=D_h,
        )
        self.gen_cp = CoefficientGenerator(
            in_channels=C,
            num_bases=cfg.MODEL.EDM.N_CP,    # 16
            out_channels=C,
            hidden_dim=D_h,
        )

    def forward(self, features):
        """
        Generate coefficients for all three subspaces.

        Args:
            features: Input feature map [B, C, H, W].

        Returns:
            w_inv: Invariant basis weights   [B, C, N_inv].
            w_eq:  Equivariant basis weights  [B, C, N_eq].
            w_cp:  Coupled basis weights      [B, C, N_cp].
        """
        w_inv = self.gen_inv(features)
        w_eq = self.gen_eq(features)
        w_cp = self.gen_cp(features)

        return w_inv, w_eq, w_cp
