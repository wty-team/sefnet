"""
Geometry-Aware Bridging Block (GAB) Package
============================================

Implements the GAB module described in Section 3.4 of the paper.

GAB performs confidence-based layer selection and adaptive fusion
to produce the final tracking representation from decomposed features.

Modules:
    - layer_selection: Confidence-based layer selection (Eq. 17)
    - reliability: Scale-dependent reliability measures (Eq. 18-20)
    - cross_attention: Bidirectional cross-attention and fusion (Eq. 21-22)
    - gab: Main GAB module integrating all components
"""

from .reliability import (
    ReliabilityModule,
    InvariantReliability,
    EquivariantReliability,
    CoupledReliability,
    estimate_scale,
)
from .layer_selection import LayerSelection
# cross_attention and gab will be imported after creation

__all__ = [
    "ReliabilityModule",
    "InvariantReliability",
    "EquivariantReliability",
    "CoupledReliability",
    "LayerSelection",
    "estimate_scale",
]
