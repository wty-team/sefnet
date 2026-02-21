"""
SEFNet Configuration Module.

All hyperparameters follow the paper specification:
- Section 5.1.3 (Implementation Details)
- Appendix D (Sensitivity Analysis)
"""

from .sefnet_config import get_default_config, merge_from_yaml

__all__ = ["get_default_config", "merge_from_yaml"]
