"""
ViT Backbone for Multi-Layer Feature Extraction
==================================================

Wraps a Vision Transformer (ViT-Base/16) to extract intermediate
layer features for EDM processing.

The backbone provides {F^(l)}_{l=1}^{12} where each F^(l) ∈ R^{C×H×W}.
ViT-Base/16: patch_size=16, embed_dim=768, depth=12, num_heads=12.

A channel projection maps 768 → C (default 256) for efficiency.

References:
    - Section 5.3: ViT-Base pretrained on MAE
    - Appendix C: Backbone complexity dominates total compute
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class ViTBackbone(nn.Module):
    """
    ViT backbone with multi-layer feature extraction.

    Args:
        model_name: timm model name (default 'vit_base_patch16_224').
        pretrained: Load pretrained weights.
        out_channels: Output channel dimension C after projection.
        img_size: Expected input image size.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        out_channels: int = 256,
        img_size: int = 384,
    ):
        super().__init__()
        self.out_channels = out_channels

        if HAS_TIMM:
            self.vit = timm.create_model(
                model_name,
                pretrained=pretrained,
                img_size=img_size,
                num_classes=0,
            )
            self.embed_dim = self.vit.embed_dim
        else:
            # Fallback: minimal ViT stub for testing
            self.embed_dim = 768
            self.vit = None

        # Channel projection: embed_dim → out_channels
        self.channel_proj = nn.Linear(self.embed_dim, out_channels)

        # Patch size for spatial reshaping
        self.patch_size = 16

    def _extract_intermediate(
        self,
        x: torch.Tensor,
        return_layers: List[int],
    ) -> Dict[int, torch.Tensor]:
        """
        Extract intermediate features from specified layers.

        Args:
            x: Input image [B, 3, H, W].
            return_layers: 1-indexed layer indices to extract.

        Returns:
            Dict layer_idx → features [B, C, H_feat, W_feat].
        """
        B, _, H, W = x.shape
        H_feat = H // self.patch_size
        W_feat = W // self.patch_size

        if self.vit is not None:
            # Patch embedding
            x = self.vit.patch_embed(x)

            # Add CLS token and position embedding
            cls_token = self.vit.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            x = x + self.vit.pos_embed
            x = self.vit.pos_drop(x)

            features = {}
            for i, block in enumerate(self.vit.blocks):
                x = block(x)
                layer_idx = i + 1  # 1-indexed
                if layer_idx in return_layers:
                    # Remove CLS token, project, reshape to spatial
                    tokens = x[:, 1:, :]  # [B, N, embed_dim]
                    projected = self.channel_proj(tokens)  # [B, N, C]
                    spatial = projected.transpose(1, 2).reshape(
                        B, self.out_channels, H_feat, W_feat
                    )
                    features[layer_idx] = spatial

            return features
        else:
            # Stub: return random features for testing
            features = {}
            for l in return_layers:
                features[l] = torch.randn(
                    B, self.out_channels, H_feat, W_feat, device=x.device
                )
            return features

    def forward(
        self,
        x: torch.Tensor,
        return_layers: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Forward pass extracting multi-layer features.

        Args:
            x: Input image [B, 3, H, W].
            return_layers: Which layers to return (1-indexed).

        Returns:
            Dict layer_idx → [B, C, H_feat, W_feat].
        """
        if return_layers is None:
            return_layers = list(range(1, 13))

        return self._extract_intermediate(x, return_layers)


def build_vit_backbone(cfg) -> ViTBackbone:
    """Factory function to build ViT backbone from config."""
    backbone_map = {
        "vit_base_patch16": "vit_base_patch16_224",
        "vit_large_patch16": "vit_large_patch16_224",
    }
    model_name = backbone_map.get(cfg.MODEL.BACKBONE, cfg.MODEL.BACKBONE)

    return ViTBackbone(
        model_name=model_name,
        pretrained=True,
        out_channels=cfg.MODEL.CHANNELS,
        img_size=cfg.DATA.SEARCH_SIZE,
    )
