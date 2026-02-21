"""
Basis Functions for Equivariant Decomposition Module (EDM).

Implements three types of basis functions prescribed by Theorem 5
(Kernel-Subspace Correspondence, Sec 3.2.4):

    "The angular structure of the kernel Ψ determines the subspace
     membership of the transform W_Ψ f."

1. Invariant Basis (Eq.13):  Radially symmetric, large-σ Gaussian derivatives
   → Projects features into F_inv (approximate scale invariance)

2. Equivariant Basis (Eq.14): Radially symmetric, small-σ Gaussian derivatives
   → Projects features into F_eq (scale-sensitive, rotation-invariant)

3. Coupled Basis (Eq.15-16): Circular harmonic functions with n ≠ 0
   → Projects features into F_cp (scale-rotation coupling)

The key principle from Theorem 5:
  - Radially symmetric kernels (ψ_n(r) = 0 for n ≠ 0)
    → output ∈ F_inv ⊕ F_eq
  - Kernels with angular frequency n₀ ≠ 0
    → output ∈ F_cp

Separation of F_inv from F_eq within the isotropic subspace uses
the Mellin relation (Eq.7): large spatial extent → low |ω| (invariant),
small spatial extent → high |ω| (equivariant).
"""

import torch
import torch.nn as nn
import numpy as np
import math


class GaussianDerivativeBasis(nn.Module):
    """
    Isotropic (radially symmetric) Gaussian derivative basis functions.

    Used for BOTH invariant and equivariant subspaces. The distinction
    is controlled by the scale parameter σ:
      - Large σ (σ^L ∈ {4,6,8}) → F_inv  (Eq.13)
      - Small σ (σ^S ∈ {1,2,3}) → F_eq   (Eq.14)

    From Section 3.3:
        "ψ_k^(iso)(r) = ∂^{m_k}/∂r^{m_k} G_{σ_k}(r)"    (Eq.11)

    where G_σ(r) = exp(-r²/2σ²) is the Gaussian function.

    These basis functions depend ONLY on radial distance r and have
    NO angular component, which guarantees features belong to the
    isotropic subspace F_inv ⊕ F_eq (Theorem 5, case 1).

    The Mellin relation (Eq.7) establishes the correspondence:
        "Kernels with large spatial extent concentrate their Mellin
         spectrum at small |ω|"  → invariant region Ω_inv
        "Spatially compact kernels spread their spectrum to larger |ω|"
         → equivariant region Ω_eq

    Args:
        sigmas: List of σ values, e.g. [4,6,8] for invariant or [1,2,3] for equivariant.
        derivative_orders: List of derivative orders m_k, e.g. [0,1,2].
        kernel_size: Spatial grid size K (default 7 → 7×7 kernel).
    """

    def __init__(self, sigmas, derivative_orders, kernel_size=7):
        super().__init__()
        self.sigmas = sigmas
        self.derivative_orders = derivative_orders
        self.kernel_size = kernel_size

        # Pre-compute all basis functions on the K×K discrete grid
        # and register as buffer (not learnable, but moves with device)
        bases = self._build_bases()
        self.register_buffer("bases", bases)  # [N_basis, K, K]
        self.num_bases = bases.shape[0]

    def _build_bases(self):
        """
        Construct Gaussian derivative basis functions on a discrete grid.

        For each (σ, m) pair, compute:
            ψ(r) = d^m/dr^m exp(-r² / 2σ²)

        Derivative formulas (closed-form):
            m=0: G(r)     = exp(-r²/2σ²)
            m=1: G'(r)    = -(r/σ²) · exp(-r²/2σ²)
            m=2: G''(r)   = (r²/σ⁴ - 1/σ²) · exp(-r²/2σ²)   (Laplacian of Gaussian)

        Returns:
            Tensor [N_basis, K, K] of basis functions.
        """
        K = self.kernel_size
        half = (K - 1) / 2.0

        # Centered coordinates: spatial positions relative to kernel center
        # Following Definition A.8 (Appendix B.6):
        #   "h̃ = h - (H+1)/2,  w̃ = w - (W+1)/2"
        coords = torch.arange(K, dtype=torch.float32) - half  # [-3,-2,-1,0,1,2,3] for K=7
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")

        # Radial distance from center: r = sqrt(x² + y²)
        r_squared = xx ** 2 + yy ** 2  # [K, K]

        bases_list = []
        for sigma in self.sigmas:
            s2 = sigma ** 2
            # Gaussian: G_σ(r) = exp(-r²/2σ²)
            gaussian = torch.exp(-r_squared / (2.0 * s2))  # [K, K]

            for m in self.derivative_orders:
                if m == 0:
                    # Zeroth derivative: smoothed intensity
                    # Captures average structure at scale σ
                    basis = gaussian

                elif m == 1:
                    # First radial derivative: G'(r) = -(r/σ²)·G(r)
                    # Responds to edges at scale σ
                    # Note: for 2D, we use the radial gradient magnitude
                    # ∂G/∂x = -(x/σ²)·G,  ∂G/∂y = -(y/σ²)·G
                    # We take the x-component (radially symmetric → equivalent)
                    r = torch.sqrt(r_squared + 1e-8)
                    basis = -(r / s2) * gaussian

                elif m == 2:
                    # Second radial derivative (Laplacian of Gaussian):
                    # G''(r) = (r²/σ⁴ - 1/σ²)·G(r)
                    # Detects curvature and blob-like structures
                    basis = (r_squared / (s2 ** 2) - 1.0 / s2) * gaussian

                else:
                    raise ValueError(f"Derivative order {m} not supported (use 0, 1, or 2).")

                # L2-normalize each basis to ensure comparable magnitudes
                basis = basis / (basis.norm() + 1e-8)
                bases_list.append(basis)

        return torch.stack(bases_list, dim=0)  # [N_basis, K, K]

    def forward(self):
        """
        Return the pre-computed basis functions.

        Returns:
            Tensor [N_basis, K, K] of radially symmetric basis functions.
        """
        return self.bases


class CircularHarmonicBasis(nn.Module):
    """
    Circular harmonic basis functions for the coupled subspace F_cp.

    From Section 3.3, Eq.15-16:
        "ψ_{k,n}^(cp)(r, θ) = R_k(r) · cos(nθ)"
        "ψ̃_{k,n}^(cp)(r, θ) = R_k(r) · sin(nθ)"

    where:
        - n ∈ {1,2,3,4} are NON-ZERO angular frequencies
        - R_k(r) is a learnable radial profile (two-layer MLP)

    These basis functions have EXPLICIT angular dependence through
    cos(nθ) and sin(nθ), which places extracted features in F_cp
    (Theorem 5, case 2).

    From Theorem 4 (Scale-Rotation Coupling):
        "Rotation by angle θ₀ induces a phase shift e^{-in₀θ₀}.
         This property enables the capture of orientation-selective
         patterns such as directional edges and textures."

    Physical interpretation (Appendix B.5):
        n=1: Simple oriented edges
        n=2: Twofold symmetric patterns (e.g., elongated structures)
        n=3: Threefold patterns
        n=4: Fourfold symmetric patterns (e.g., cross-like textures)

    Args:
        angular_freqs: List of angular frequencies n, e.g. [1,2,3,4].
        radial_hidden_dim: Hidden dimension for radial MLP R_k(r).
        kernel_size: Spatial grid size K (default 7).
    """

    def __init__(self, angular_freqs, radial_hidden_dim=32, kernel_size=7):
        super().__init__()
        self.angular_freqs = angular_freqs
        self.kernel_size = kernel_size

        # Pre-compute angular components cos(nθ) and sin(nθ) on the grid
        K = kernel_size
        half = (K - 1) / 2.0
        coords = torch.arange(K, dtype=torch.float32) - half
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")

        # Polar angle: θ = atan2(y, x) (Definition A.8)
        theta = torch.atan2(yy, xx)  # [K, K], range [-π, π]

        # Radial distance with ε_r offset (Definition A.8):
        #   "ε_r = 10^{-6} that prevents the logarithm from diverging"
        r = torch.sqrt(xx ** 2 + yy ** 2 + 1e-6)  # [K, K]

        self.register_buffer("theta", theta)
        self.register_buffer("r", r)

        # Pre-compute angular harmonics: cos(nθ) and sin(nθ) for each n
        # These are FIXED — they encode the angular structure required by theory
        cos_harmonics = []
        sin_harmonics = []
        for n in angular_freqs:
            cos_harmonics.append(torch.cos(n * theta))  # [K, K]
            sin_harmonics.append(torch.sin(n * theta))  # [K, K]

        # [N_freqs, K, K]
        self.register_buffer("cos_harmonics", torch.stack(cos_harmonics, dim=0))
        self.register_buffer("sin_harmonics", torch.stack(sin_harmonics, dim=0))

        # Learnable radial profiles R_k(r): two-layer MLP with hidden dim 32
        # (Appendix B.6): "R_k(r) is a two-layer MLP with hidden dimension 32
        #  and GELU activation"
        # One radial MLP per angular frequency
        self.radial_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, radial_hidden_dim),
                nn.GELU(),
                nn.Linear(radial_hidden_dim, 1),
            )
            for _ in angular_freqs
        ])

        # Total number of basis functions: 2 × N_freqs (cos + sin for each n)
        self.num_bases = 2 * len(angular_freqs)

    def forward(self):
        """
        Construct circular harmonic basis functions with learned radial profiles.

        For each angular frequency n, produces two basis functions:
            ψ_{cos}(r, θ) = R_k(r) · cos(nθ)
            ψ_{sin}(r, θ) = R_k(r) · sin(nθ)

        Returns:
            Tensor [N_cp, K, K] of coupled basis functions.
        """
        K = self.kernel_size
        bases_list = []

        # Flatten radial distances for MLP input: [K*K, 1]
        r_flat = self.r.reshape(-1, 1)

        for i, n in enumerate(self.angular_freqs):
            # Compute learnable radial profile R_k(r) via MLP
            # Input: radial distance r → Output: radial weight
            R_k = self.radial_mlps[i](r_flat)  # [K*K, 1]
            R_k = R_k.reshape(K, K)             # [K, K]

            # Compose: R_k(r) · cos(nθ)  and  R_k(r) · sin(nθ)
            cos_basis = R_k * self.cos_harmonics[i]  # [K, K]
            sin_basis = R_k * self.sin_harmonics[i]  # [K, K]

            # Normalize each basis function
            cos_basis = cos_basis / (cos_basis.norm() + 1e-8)
            sin_basis = sin_basis / (sin_basis.norm() + 1e-8)

            bases_list.append(cos_basis)
            bases_list.append(sin_basis)

        return torch.stack(bases_list, dim=0)  # [N_cp, K, K]


def build_basis_functions(cfg):
    """
    Factory function to create all three types of basis functions.

    Following Section 5.1.3 (Implementation Details):
        "EDM employs N_eq=16 harmonic bases (σ^S ∈ {1,2,3} pixels),
         N_inv=8 radially symmetric Gaussian derivatives (σ^L ∈ {4,6,8} pixels),
         and N_cp=16 directional derivatives with angular orders n ∈ {1,2,3,4}"

    Args:
        cfg: Configuration dict.

    Returns:
        inv_basis: GaussianDerivativeBasis for invariant subspace.
        eq_basis:  GaussianDerivativeBasis for equivariant subspace.
        cp_basis:  CircularHarmonicBasis for coupled subspace.
    """
    K = cfg.MODEL.EDM.KERNEL_SIZE  # 7

    # Invariant basis: large-σ radially symmetric (Eq.13)
    inv_basis = GaussianDerivativeBasis(
        sigmas=cfg.MODEL.EDM.INV_SIGMAS,               # [4, 6, 8]
        derivative_orders=cfg.MODEL.EDM.INV_DERIVATIVE_ORDERS,  # [0, 1, 2]
        kernel_size=K,
    )

    # Equivariant basis: small-σ radially symmetric (Eq.14)
    eq_basis = GaussianDerivativeBasis(
        sigmas=cfg.MODEL.EDM.EQ_SIGMAS,                 # [1, 2, 3]
        derivative_orders=cfg.MODEL.EDM.EQ_DERIVATIVE_ORDERS,  # [0, 1, 2]
        kernel_size=K,
    )

    # Coupled basis: circular harmonics with n ≠ 0 (Eq.15-16)
    cp_basis = CircularHarmonicBasis(
        angular_freqs=cfg.MODEL.EDM.CP_ANGULAR_FREQS,   # [1, 2, 3, 4]
        radial_hidden_dim=cfg.MODEL.EDM.CP_RADIAL_HIDDEN_DIM,  # 32
        kernel_size=K,
    )

    return inv_basis, eq_basis, cp_basis
