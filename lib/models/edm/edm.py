"""
Equivariant Decomposition Module (EDM)
=======================================

Implements the core feature decomposition described in Section 3.3 of the paper.

Given input feature map F^(l) at backbone layer l, EDM produces three output
streams (F_inv, F_eq, F_cp) by convolving with parameterized kernels whose
angular structure is prescribed by the Kernel-Subspace Correspondence theorem
(Theorem 5 / Theorem A.6 in Appendix).

Architecture (per layer):
    F^(l) ──┬── Ψ_inv * F^(l) ──> F_inv^(l)   (invariant subspace)
             ├── Ψ_eq  * F^(l) ──> F_eq^(l)    (equivariant subspace)
             └── Ψ_cp  * F^(l) ──> F_cp^(l)    (coupled subspace)

Each kernel Ψ_t is constructed via basis function expansion (Eq. 12):
    Ψ_t(r, θ) = Σ_k w_t^(k) · ψ_k^(t)(r, θ)

where:
    - {ψ_k^(t)} are basis functions with prescribed angular structure
    - {w_t^(k)} are learnable coefficients from the coefficient generator

References:
    - Eq. 11: F_t^(l) = Ψ_t * F^(l), t ∈ {inv, eq, cp}
    - Eq. 12: Kernel basis expansion
    - Eq. 13-14: Gaussian derivative bases for inv/eq (large/small σ)
    - Eq. 15-16: Circular harmonic bases for cp
    - Theorem 5 (Appendix Theorem A.6): Kernel-Subspace Correspondence
    - Proposition A.7: Coefficient generator rotation invariance
    - Appendix C.2: Computational complexity O(C²HWK²) per selected layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional

from .basis_functions import GaussianDerivativeBasis, CircularHarmonicBasis
from .coefficient_generator import CoefficientGenerator


class EquivariantDecompositionModule(nn.Module):
    """
    Equivariant Decomposition Module (EDM).

    Decomposes input feature maps into three orthogonal subspaces via
    group-theoretic parameterized convolution (Sec 3.3, Fig. 3).

    The three subspaces correspond to the spectral regions defined in
    Eqs. 3-5 of the main paper:
        Ω_inv(ε) = {(ω, n) ∈ Ĝ : |ω| ≤ ε, n = 0}  → invariant
        Ω_eq(ε)  = {(ω, n) ∈ Ĝ : |ω| > ε, n = 0}  → equivariant
        Ω_cp     = {(ω, n) ∈ Ĝ : n ≠ 0}             → coupled

    Args:
        in_channels: Number of input feature channels (C).
        out_channels: Number of output channels per subspace.
        kernel_size: Spatial size of the discretized kernel (default 7, Sec 5.3).
        n_inv_basis: Number of invariant basis functions (N_inv=8, Sec 5.3).
        n_eq_basis: Number of equivariant basis functions (N_eq=16, Sec 5.3).
        n_cp_basis: Number of coupled basis functions (N_cp=16, Sec 5.3).
        sigma_large: Scale parameters for invariant kernels (Eq. 13).
        sigma_small: Scale parameters for equivariant kernels (Eq. 14).
        angular_orders: Angular frequency orders for coupled kernels (Eq. 15-16).
        hidden_dim: Hidden dimension of the coefficient generator MLP.
        radial_hidden_dim: Hidden dimension of learnable radial MLP in coupled basis.
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        kernel_size: int = 7,
        n_inv_basis: int = 8,
        n_eq_basis: int = 16,
        n_cp_basis: int = 16,
        sigma_large: Tuple[float, ...] = (4.0, 6.0, 8.0),
        sigma_small: Tuple[float, ...] = (1.0, 2.0, 3.0),
        angular_orders: Tuple[int, ...] = (1, 2, 3, 4),
        hidden_dim: int = 64,
        radial_hidden_dim: int = 32,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_inv_basis = n_inv_basis
        self.n_eq_basis = n_eq_basis
        self.n_cp_basis = n_cp_basis

        # ----------------------------------------------------------------
        # Step 1: Construct basis functions for each subspace.
        #
        # Per Theorem 5 (Kernel-Subspace Correspondence):
        #   - Radially symmetric kernels (n=0) → F_inv ⊕ F_eq
        #   - Kernels with angular freq n≠0  → F_cp
        #
        # We further separate F_inv from F_eq using spatial scale (Eq. 10):
        #   - Large σ → low |ω| → Ω_inv  (Eq. 13)
        #   - Small σ → high |ω| → Ω_eq  (Eq. 14)
        # ----------------------------------------------------------------

        # Invariant basis: large-scale Gaussian derivatives (Eq. 13)
        # These are radially symmetric with large spatial extent,
        # concentrating Mellin spectrum at small |ω| → Ω_inv
        self.inv_basis = GaussianDerivativeBasis(
            n_basis=n_inv_basis,
            kernel_size=kernel_size,
            sigmas=sigma_large,
            derivative_orders=(0, 1, 2),
        )

        # Equivariant basis: small-scale Gaussian derivatives (Eq. 14)
        # Radially symmetric with small spatial extent,
        # spreading Mellin spectrum to large |ω| → Ω_eq
        self.eq_basis = GaussianDerivativeBasis(
            n_basis=n_eq_basis,
            kernel_size=kernel_size,
            sigmas=sigma_small,
            derivative_orders=(0, 1, 2),
        )

        # Coupled basis: circular harmonic functions (Eq. 15-16)
        # These have non-zero angular frequency n ∈ {1,2,3,4},
        # placing output features in F_cp by Theorem 5
        self.cp_basis = CircularHarmonicBasis(
            n_basis=n_cp_basis,
            kernel_size=kernel_size,
            angular_orders=angular_orders,
            radial_hidden_dim=radial_hidden_dim,
        )

        # ----------------------------------------------------------------
        # Step 2: Coefficient generators (Proposition A.7).
        #
        # For each subspace type t, a coefficient generator g_t produces
        # basis weights: w_t = g_t(GAP(F)) ∈ R^{N_t}
        #
        # The GAP → MLP pathway preserves rotation invariance because
        # GAP is invariant under spatial permutations (Proposition A.7).
        # ----------------------------------------------------------------

        self.coeff_gen_inv = CoefficientGenerator(
            in_channels=in_channels,
            n_basis=n_inv_basis,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
        )
        self.coeff_gen_eq = CoefficientGenerator(
            in_channels=in_channels,
            n_basis=n_eq_basis,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
        )
        self.coeff_gen_cp = CoefficientGenerator(
            in_channels=in_channels,
            n_basis=n_cp_basis,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
        )

        # ----------------------------------------------------------------
        # Step 3: Post-convolution normalization and projection.
        #
        # Each subspace output is layer-normalized and projected to
        # the target channel dimension for downstream processing.
        # ----------------------------------------------------------------

        self.norm_inv = nn.LayerNorm(out_channels)
        self.norm_eq = nn.LayerNorm(out_channels)
        self.norm_cp = nn.LayerNorm(out_channels)

        # 1x1 convolution for channel alignment if in/out differ
        self.proj_inv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.proj_eq = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.proj_cp = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        # Precompute the polar coordinate grid for the kernel
        # Used by basis functions during kernel construction
        self._register_coordinate_grid()

    def _register_coordinate_grid(self):
        """
        Precompute polar coordinate grid (r, θ) for the K×K kernel.

        Following Definition A.8 in the appendix:
            r_hw = sqrt(h̃² + w̃²)
            θ_hw = atan2(h̃, w̃)
        where h̃ = h - (H+1)/2, w̃ = w - (W+1)/2 are centered coordinates.

        The grid is registered as a buffer (non-learnable, moves with device).
        """
        K = self.kernel_size
        # Create centered coordinates
        coords = torch.arange(K, dtype=torch.float32) - (K - 1) / 2.0
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")

        # Polar coordinates
        r = torch.sqrt(grid_x ** 2 + grid_y ** 2 + 1e-8)  # ε_r = 1e-8 (Def A.8)
        theta = torch.atan2(grid_y, grid_x)

        # Register as buffers: shape [K, K]
        self.register_buffer("grid_r", r)
        self.register_buffer("grid_theta", theta)

    def _build_kernel(
        self,
        basis_module,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        """
        Construct a parameterized convolution kernel from basis functions
        and learnable coefficients (Eq. 12).

        Ψ_t(r, θ) = Σ_k w_t^(k) · ψ_k^(t)(r, θ)

        Args:
            basis_module: Basis function module (inv/eq/cp).
            coefficients: Basis weights [B, N_t, C_out] from coefficient generator.

        Returns:
            kernel: Convolution kernel [B, C_out, K, K].

        Complexity: O(N · K² · C_out), independent of spatial resolution H×W.
        This is negligible compared to the convolution itself (Appendix C.2).
        """
        # Get basis function values on the grid: [N_t, K, K]
        basis_values = basis_module(self.grid_r, self.grid_theta)

        # Weighted combination: [B, C_out, K, K]
        # coefficients: [B, N_t, C_out] → [B, C_out, N_t]
        # basis_values: [N_t, K, K] → [1, N_t, K*K]
        B = coefficients.shape[0]
        N_t = basis_values.shape[0]
        K = self.kernel_size

        coeff = coefficients.permute(0, 2, 1)  # [B, C_out, N_t]
        basis_flat = basis_values.reshape(N_t, K * K).unsqueeze(0)  # [1, N_t, K*K]

        # Batched matrix multiply: [B, C_out, N_t] @ [1, N_t, K*K] → [B, C_out, K*K]
        kernel = torch.bmm(coeff, basis_flat.expand(B, -1, -1))
        kernel = kernel.reshape(B, -1, K, K)  # [B, C_out, K, K]

        return kernel

    def _apply_parameterized_conv(
        self,
        feature: torch.Tensor,
        kernel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the parameterized convolution kernel to input features.

        This implements the core operation F_t^(l) = Ψ_t * F^(l) (Eq. 11).

        For batch processing, we use grouped convolution where each sample
        in the batch has its own dynamically generated kernel.

        Args:
            feature: Input feature map [B, C_in, H, W].
            kernel: Parameterized kernel [B, C_out, K, K].

        Returns:
            output: Convolved feature map [B, C_out, H, W].

        Complexity: O(C_in · C_out · H · W · K²) per sample (Appendix C.2).
        """
        B, C_in, H, W = feature.shape
        C_out = kernel.shape[1]
        K = self.kernel_size
        pad = K // 2

        # Strategy: Use grouped convolution for batch-wise dynamic kernels
        # Reshape feature: [B, C_in, H, W] → [1, B*C_in, H, W]
        feature_grouped = feature.reshape(1, B * C_in, H, W)

        # Kernel: [B, C_out, K, K] → need [B*C_out, C_in, K, K] for depthwise
        # Actually, we use a simpler approach: iterate over batch if B is small,
        # or use einsum-based convolution for larger batches.

        if B == 1:
            # Single sample: standard convolution
            # kernel: [1, C_out, K, K] → [C_out, 1, K, K] for depthwise
            # But we need full conv: kernel should be [C_out, C_in, K, K]
            # Since our kernel is per-output-channel, we broadcast across C_in
            weight = kernel.squeeze(0).unsqueeze(1).expand(-1, C_in, -1, -1)
            output = F.conv2d(feature, weight, padding=pad)
        else:
            # Batch processing: iterate (B is typically small during inference)
            outputs = []
            for b in range(B):
                weight = kernel[b].unsqueeze(1).expand(-1, C_in, -1, -1)
                out_b = F.conv2d(feature[b:b+1], weight, padding=pad)
                outputs.append(out_b)
            output = torch.cat(outputs, dim=0)

        return output

    def decompose_single_layer(
        self,
        feature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Decompose a single backbone layer's features into three subspaces.

        This is the core EDM operation for one layer (Eq. 11-12):
            F_inv^(l) = Ψ_inv * F^(l)   (invariant features)
            F_eq^(l)  = Ψ_eq  * F^(l)   (equivariant features)
            F_cp^(l)  = Ψ_cp  * F^(l)   (coupled features)

        Args:
            feature: Input feature map [B, C, H, W] from backbone layer l.

        Returns:
            Dict with keys 'inv', 'eq', 'cp', each [B, C_out, H, W].
        """
        B, C, H, W = feature.shape

        # ---- Step 1: Generate coefficients via GAP → MLP (Proposition A.7) ----
        # Global context vector z = GAP(F) ∈ R^C
        # This is rotation-invariant by construction (Proposition A.7)
        z = feature.mean(dim=(-2, -1))  # [B, C] — Global Average Pooling

        # Coefficient generators produce basis weights for each subspace
        coeff_inv = self.coeff_gen_inv(z)  # [B, N_inv, C_out]
        coeff_eq = self.coeff_gen_eq(z)    # [B, N_eq, C_out]
        coeff_cp = self.coeff_gen_cp(z)    # [B, N_cp, C_out]

        # ---- Step 2: Build parameterized kernels (Eq. 12) ----
        kernel_inv = self._build_kernel(self.inv_basis, coeff_inv)  # [B, C_out, K, K]
        kernel_eq = self._build_kernel(self.eq_basis, coeff_eq)
        kernel_cp = self._build_kernel(self.cp_basis, coeff_cp)

        # ---- Step 3: Apply convolution (Eq. 11) ----
        f_inv = self._apply_parameterized_conv(feature, kernel_inv)  # [B, C_out, H, W]
        f_eq = self._apply_parameterized_conv(feature, kernel_eq)
        f_cp = self._apply_parameterized_conv(feature, kernel_cp)

        # ---- Step 4: Project and normalize ----
        # Channel projection if needed
        f_inv = self.proj_inv(f_inv) if not isinstance(self.proj_inv, nn.Identity) else f_inv
        f_eq = self.proj_eq(f_eq) if not isinstance(self.proj_eq, nn.Identity) else f_eq
        f_cp = self.proj_cp(f_cp) if not isinstance(self.proj_cp, nn.Identity) else f_cp

        # Layer normalization: reshape for LayerNorm (operates on last dim)
        f_inv = self.norm_inv(f_inv.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        f_eq = self.norm_eq(f_eq.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        f_cp = self.norm_cp(f_cp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return {"inv": f_inv, "eq": f_eq, "cp": f_cp}

    def forward(
        self,
        layer_features: Dict[int, torch.Tensor],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Decompose features from multiple backbone layers.

        Processes each layer independently through the three-way decomposition.
        In practice, only selected layers (from GAB's confidence scoring)
        undergo full convolution; others need only lightweight statistics
        for confidence evaluation (see GAB module).

        Args:
            layer_features: Dict mapping layer index l → feature tensor [B, C, H, W].
                           Typically all 12 ViT layers: {0: F^(0), 1: F^(1), ..., 11: F^(11)}.

        Returns:
            Dict mapping layer index l → Dict with keys 'inv', 'eq', 'cp',
            each containing the decomposed features [B, C_out, H, W].

        Complexity per layer: O(C² · H · W · K²) dominated by convolution.
        Total for 3 selected layers: O(3 · C² · H · W · K²) (Sec 4, Eq. 23).
        """
        decomposed = {}
        for layer_idx, feat in layer_features.items():
            decomposed[layer_idx] = self.decompose_single_layer(feat)
        return decomposed

    def get_activation_stats(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute lightweight activation statistics for confidence scoring.

        This is used by GAB's layer selection mechanism to evaluate all
        layers without performing full EDM convolution. The statistics
        cost O(CHW) per layer vs O(C²HWK²) for full decomposition.

        Stats: [mean, std, gradient_magnitude] (Appendix C.1)

        Args:
            feature: Input feature map [B, C, H, W].

        Returns:
            stats: Activation statistics [B, 3].
        """
        B = feature.shape[0]

        # Mean activation: μ^(l)
        mu = feature.mean(dim=(1, 2, 3))  # [B]

        # Standard deviation: σ^(l)
        sigma = feature.std(dim=(1, 2, 3))  # [B]

        # Gradient magnitude: g^(l) = C^{-1} Σ_c ||∇F_t^(l)(c,:,:)||_F
        # Approximate with finite differences
        grad_x = feature[:, :, :, 1:] - feature[:, :, :, :-1]  # horizontal
        grad_y = feature[:, :, 1:, :] - feature[:, :, :-1, :]  # vertical
        grad_mag = (grad_x.pow(2).mean(dim=(1, 2, 3)) +
                    grad_y.pow(2).mean(dim=(1, 2, 3))).sqrt()  # [B]

        return torch.stack([mu, sigma, grad_mag], dim=1)  # [B, 3]


def build_edm(cfg) -> EquivariantDecompositionModule:
    """
    Factory function to build EDM from config.

    Args:
        cfg: Configuration object with EDM parameters.

    Returns:
        Configured EquivariantDecompositionModule instance.
    """
    return EquivariantDecompositionModule(
        in_channels=cfg.MODEL.CHANNELS,
        out_channels=cfg.MODEL.CHANNELS,
        kernel_size=cfg.EDM.KERNEL_SIZE,
        n_inv_basis=cfg.EDM.N_INV_BASIS,
        n_eq_basis=cfg.EDM.N_EQ_BASIS,
        n_cp_basis=cfg.EDM.N_CP_BASIS,
        sigma_large=tuple(cfg.EDM.SIGMA_LARGE),
        sigma_small=tuple(cfg.EDM.SIGMA_SMALL),
        angular_orders=tuple(cfg.EDM.ANGULAR_ORDERS),
        hidden_dim=cfg.EDM.HIDDEN_DIM,
        radial_hidden_dim=cfg.EDM.RADIAL_HIDDEN_DIM,
    )
