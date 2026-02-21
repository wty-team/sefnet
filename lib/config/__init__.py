"""
SEFNet Configuration System
=============================

YAML-based configuration with hierarchical defaults.
All hyperparameters referenced in Section 5.3.
"""

import os
import yaml
from types import SimpleNamespace


# Default configuration matching Section 5.3
_DEFAULTS = {
    "MODEL": {
        "NAME": "SEFNet",
        "BACKBONE": "vit_base_patch16",
        "CHANNELS": 256,
        "N_LAYERS": 12,
        "CANDIDATE_LAYERS": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    },
    "EDM": {
        "KERNEL_SIZE": 7,
        "N_INV_BASIS": 8,
        "N_EQ_BASIS": 16,
        "N_CP_BASIS": 16,
        "SIGMA_LARGE": [4.0, 6.0, 8.0],
        "SIGMA_SMALL": [1.0, 2.0, 3.0],
        "ANGULAR_ORDERS": [1, 2, 3, 4],
        "HIDDEN_DIM": 64,
        "RADIAL_HIDDEN_DIM": 32,
        "EPSILON": 0.10,
    },
    "GAB": {
        "N_HEADS": 8,
        "MLP_RATIO": 4.0,
        "DROPOUT": 0.0,
        "TAU_INV": 0.5,
        "TAU_FUSION": 1.0,
        "SELECTION_TEMPERATURE": 0.1,
    },
    "HEAD": {
        "TYPE": "corner",
        "HIDDEN_DIM": 256,
    },
    "LOSS": {
        "LAMBDA_CLS": 2.0,
        "LAMBDA_REG": 5.0,
        "LAMBDA_IOU": 2.0,
        "LAMBDA_EQ": 1.0,
    },
    "DATA": {
        "ROOT": "data",
        "DATASET": "uav123",
        "TEMPLATE_SIZE": 192,
        "SEARCH_SIZE": 384,
        "SEARCH_FEAT_SIZE": 24,
        "CONTEXT_FACTOR": 4.0,
        "NUM_WORKERS": 8,
    },
    "TRAIN": {
        "EPOCHS": 500,
        "BATCH_SIZE": 32,
        "LR": 4e-4,
        "WEIGHT_DECAY": 1e-4,
        "WARMUP_EPOCHS": 5,
        "LOG_INTERVAL": 50,
        "SAVE_INTERVAL": 20,
    },
}


def _dict_to_namespace(d):
    """Recursively convert dict to SimpleNamespace."""
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _dict_to_namespace(v))
        else:
            setattr(ns, k, v)
    return ns


def _merge_dicts(base, override):
    """Deep merge override into base dict."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def get_config(yaml_path=None):
    """
    Load configuration from YAML file merged with defaults.

    Args:
        yaml_path: Optional path to YAML config file.

    Returns:
        SimpleNamespace config object.
    """
    cfg = _DEFAULTS.copy()

    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            override = yaml.safe_load(f)
        if override:
            cfg = _merge_dicts(cfg, override)

    return _dict_to_namespace(cfg)
