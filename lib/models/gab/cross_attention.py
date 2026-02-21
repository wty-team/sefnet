"""
Cross-Attention Matching and Adaptive Fusion
==============================================

Implements the cross-attention and fusion described in Section 3.4, Eq. 21-22.

Since the three subspaces satisfy orthogonality (Theorem 1), GAB processes
them through separate cross-attention pathways before fusion.

For each feature type t, template tokens X_t^temp and search tokens X_t^search
undergo bidirectional enhancement (Eq. 21):

    X̂_t^(i) = Attn(X_t^(i), X_t^(j)) + X_t^(i),  i ≠ j

where Attn takes queries from the first argument and keys-values from the second.

The enhanced features are combined through reliability-based fusion (Eq. 22):

    [α, β, γ] = Softmax(1/τ · [R_eq(s), R_inv(s), R_cp(s)])
    X^fused = α · X̂_eq^search + β · X̂_inv^search + γ · X̂_cp^search

This connects to the orthogonal decomposition (Eq. 6), where any function
admits unique representation as a sum of components from three subspaces.

Dynamic behavior:
    - Large target scale → α rises (equivariant features dominate)
    - Small target scale → β rises (invariant features dominate)

References:
    - Eq. 21: Bidirectional cross-attention with residual
    - Eq. 22: Reliability-based fusion weights via temperature-scaled softmax
    - Eq. 6: Orthogonal decomposition F = F_inv + F_eq + F_cp
    - Appendix C.4: Cross-attention complexity O(3 · C · N_s · N_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional


class CrossAttentionBlock(nn.Module):
    """
    Single-direction cross-attention block.

    Computes Attn(Q_source, KV_target) where:
        - Queries come from source tokens
        - Keys and Values come from target tokens

    This implements standard multi-head attention with the query/key/value
    split across source and target token sets.

    Args:
        dim: Token embedding dimension (C).
        n_heads: Number of attention heads (default 8).
        qkv_bias: Whether to use bias in Q/K/V projections.
        attn_drop: Dropout rate on attention weights.
        proj_drop: Dropout rate on output projection.
    """

    def __init__(
        self,
        dim: int = 256,
        n_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        # Separate projections for Q (from source) and K,V (from target)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-attention: source queries attend to target keys/values.

        Args:
            source: Query tokens [B, N_s, C] — provides Q.
            target: Key-Value tokens [B, N_t, C] — provides K, V.

        Returns:
            output: Attended tokens [B, N_s, C].
        """
        B, N_s, C = source.shape
        N_t = target.shape[1]

        # Project queries from source, keys/values from target
        Q = self.q_proj(source).reshape(B, N_s, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(target).reshape(B, N_t, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(target).reshape(B, N_t, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q: [B, heads, N_s, head_dim], K/V: [B, heads, N_t, head_dim]

        # Scaled dot-product attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # [B, heads, N_s, N_t]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum of values
        out = (attn @ V).transpose(1, 2).reshape(B, N_s, C)  # [B, N_s, C]
        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out


class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention for template-search matching (Eq. 21).

    Each stream queries the other while retaining its own representation
    through residual addition:

        X̂_t^search = Attn(X_t^search, X_t^temp) + X_t^search
        X̂_t^temp   = Attn(X_t^temp, X_t^search) + X_t^temp

    This allows bidirectional information flow: search tokens attend to
    template tokens to find the target, and template tokens attend to
    search tokens to refine the target model.

    Args:
        dim: Token embedding dimension.
        n_heads: Number of attention heads.
        mlp_ratio: Expansion ratio for the feed-forward MLP.
        drop: Dropout rate.
    """

    def __init__(
        self,
        dim: int = 256,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()

        # Cross-attention: search → template (search queries, template KV)
        self.cross_attn_s2t = CrossAttentionBlock(
            dim=dim, n_heads=n_heads, attn_drop=drop, proj_drop=drop,
        )
        # Cross-attention: template → search (template queries, search KV)
        self.cross_attn_t2s = CrossAttentionBlock(
            dim=dim, n_heads=n_heads, attn_drop=drop, proj_drop=drop,
        )

        # Layer norms for pre-normalization
        self.norm_s = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)
        self.norm_s2 = nn.LayerNorm(dim)
        self.norm_t2 = nn.LayerNorm(dim)

        # Feed-forward MLPs after cross-attention
        mlp_hidden = int(dim * mlp_ratio)
        self.ffn_s = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )
        self.ffn_t = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(
        self,
        search_tokens: torch.Tensor,
        template_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional cross-attention (Eq. 21).

        Args:
            search_tokens: X_t^search [B, N_s, C].
            template_tokens: X_t^temp [B, N_t, C].

        Returns:
            Tuple of (enhanced_search [B, N_s, C], enhanced_template [B, N_t, C]).
        """
        # ---- Cross-attention with residual (Eq. 21) ----
        # Search attends to template
        s_norm = self.norm_s(search_tokens)
        t_norm = self.norm_t(template_tokens)

        # X̂_search = Attn(X_search, X_temp) + X_search
        search_out = self.cross_attn_s2t(s_norm, t_norm) + search_tokens
        # X̂_temp = Attn(X_temp, X_search) + X_temp
        template_out = self.cross_attn_t2s(t_norm, s_norm) + template_tokens

        # ---- Feed-forward with residual ----
        search_out = self.ffn_s(self.norm_s2(search_out)) + search_out
        template_out = self.ffn_t(self.norm_t2(template_out)) + template_out

        return search_out, template_out


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion of three subspace features using reliability weights (Eq. 22).

    Fused representation:
        X^fused = α · X̂_eq^search + β · X̂_inv^search + γ · X̂_cp^search

    where [α, β, γ] = Softmax(1/τ · [R_eq(s), R_inv(s), R_cp(s)])

    This connects directly to the orthogonal decomposition (Theorem 1, Eq. 6):
    any function admits unique representation F = F_inv + F_eq + F_cp.
    The fusion weights determine the relative contribution based on which
    subspace is most reliable at the current scale condition.

    Args:
        dim: Token embedding dimension.
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim

        # Final projection after fusion
        self.out_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        feat_eq: torch.Tensor,
        feat_inv: torch.Tensor,
        feat_cp: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reliability-based adaptive fusion (Eq. 22).

        X^fused = α · X̂_eq + β · X̂_inv + γ · X̂_cp

        Args:
            feat_eq: Enhanced equivariant features [B, N, C].
            feat_inv: Enhanced invariant features [B, N, C].
            feat_cp: Enhanced coupled features [B, N, C].
            alpha: Equivariant weight [B, 1] from reliability softmax.
            beta: Invariant weight [B, 1].
            gamma: Coupled weight [B, 1].

        Returns:
            fused: Fused feature representation [B, N, C].
        """
        # Expand weights for broadcasting: [B, 1] → [B, 1, 1]
        alpha = alpha.unsqueeze(-1)  # [B, 1, 1]
        beta = beta.unsqueeze(-1)
        gamma = gamma.unsqueeze(-1)

        # Weighted sum (Eq. 22)
        fused = alpha * feat_eq + beta * feat_inv + gamma * feat_cp

        # Final projection
        fused = self.out_proj(fused)

        return fused


class SubspaceCrossAttentionPipeline(nn.Module):
    """
    Complete cross-attention pipeline for all three subspaces.

    Creates three independent BidirectionalCrossAttention blocks,
    one per subspace, respecting the orthogonality of the decomposition.

    Processing each subspace independently preserves the theoretical
    guarantee from Theorem 1 that the subspaces are orthogonal —
    mixing would violate this property.

    Args:
        dim: Token embedding dimension.
        n_heads: Number of attention heads.
        mlp_ratio: Feed-forward expansion ratio.
        drop: Dropout rate.
    """

    def __init__(
        self,
        dim: int = 256,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()

        # Independent cross-attention for each subspace
        # Orthogonality (Theorem 1) requires separate processing
        self.cross_attn_eq = BidirectionalCrossAttention(
            dim=dim, n_heads=n_heads, mlp_ratio=mlp_ratio, drop=drop,
        )
        self.cross_attn_inv = BidirectionalCrossAttention(
            dim=dim, n_heads=n_heads, mlp_ratio=mlp_ratio, drop=drop,
        )
        self.cross_attn_cp = BidirectionalCrossAttention(
            dim=dim, n_heads=n_heads, mlp_ratio=mlp_ratio, drop=drop,
        )

        # Adaptive fusion module
        self.fusion = AdaptiveFusion(dim=dim)

    def forward(
        self,
        search_features: Dict[str, torch.Tensor],
        template_features: Dict[str, torch.Tensor],
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Full cross-attention + fusion pipeline.

        1. Flatten spatial features to token sequences
        2. Apply bidirectional cross-attention per subspace (Eq. 21)
        3. Fuse via reliability weights (Eq. 22)

        Args:
            search_features: Dict 'eq'/'inv'/'cp' → [B, C, H_s, W_s].
            template_features: Dict 'eq'/'inv'/'cp' → [B, C, H_t, W_t].
            alpha, beta, gamma: Fusion weights [B, 1] from reliability.

        Returns:
            Tuple of:
                - fused_search: [B, N_s, C] fused search representation
                - enhanced: Dict with all enhanced features for debugging
        """
        B = search_features["eq"].shape[0]
        C = search_features["eq"].shape[1]

        # ---- Step 1: Spatial → Token sequences ----
        # [B, C, H, W] → [B, N, C] where N = H*W
        def to_tokens(feat):
            return feat.flatten(2).transpose(1, 2)  # [B, C, H*W] → [B, H*W, C]

        search_eq = to_tokens(search_features["eq"])    # [B, N_s, C]
        search_inv = to_tokens(search_features["inv"])
        search_cp = to_tokens(search_features["cp"])

        template_eq = to_tokens(template_features["eq"])  # [B, N_t, C]
        template_inv = to_tokens(template_features["inv"])
        template_cp = to_tokens(template_features["cp"])

        # ---- Step 2: Bidirectional cross-attention per subspace (Eq. 21) ----
        # Each subspace processed independently to preserve orthogonality
        enhanced_search_eq, enhanced_temp_eq = self.cross_attn_eq(
            search_eq, template_eq
        )
        enhanced_search_inv, enhanced_temp_inv = self.cross_attn_inv(
            search_inv, template_inv
        )
        enhanced_search_cp, enhanced_temp_cp = self.cross_attn_cp(
            search_cp, template_cp
        )

        # ---- Step 3: Adaptive fusion (Eq. 22) ----
        # X^fused = α · X̂_eq + β · X̂_inv + γ · X̂_cp
        fused_search = self.fusion(
            enhanced_search_eq, enhanced_search_inv, enhanced_search_cp,
            alpha, beta, gamma,
        )

        # Also fuse template for consistency (same weights)
        fused_template = self.fusion(
            enhanced_temp_eq, enhanced_temp_inv, enhanced_temp_cp,
            alpha, beta, gamma,
        )

        enhanced = {
            "search": {
                "eq": enhanced_search_eq,
                "inv": enhanced_search_inv,
                "cp": enhanced_search_cp,
                "fused": fused_search,
            },
            "template": {
                "eq": enhanced_temp_eq,
                "inv": enhanced_temp_inv,
                "cp": enhanced_temp_cp,
                "fused": fused_template,
            },
        }

        return fused_search, enhanced
