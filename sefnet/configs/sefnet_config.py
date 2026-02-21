"""
SEFNet Configuration — All Hyperparameters from the Paper.

Every parameter is annotated with its source location in the paper:
  - Sec X.X  = main paper section
  - Eq. (N)  = equation number
  - Tab. N   = table number
  - App. X.X = appendix section
  - Algo. N  = algorithm number

Reference: "SEFNet: Selective Equivariant Features for Robust Scale-Adaptive UAV Tracking"
"""

from easydict import EasyDict as edict
import yaml
import os


def get_default_config():
    """
    Build the full default configuration dictionary.
    Returns an EasyDict for dot-access (cfg.MODEL.BACKBONE.TYPE).
    """
    cfg = edict()

    # =========================================================================
    # MODEL
    # =========================================================================
    cfg.MODEL = edict()

    # --- Backbone (Sec 3.1, Sec 5.1.3) ---
    # "SEFNet adopts a pretrained ViT-Base backbone with 12 transformer layers."
    cfg.MODEL.BACKBONE = edict()
    cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"  # ViT-Base from timm
    cfg.MODEL.BACKBONE.PRETRAINED = True
    cfg.MODEL.BACKBONE.NUM_LAYERS = 12       # L=12 transformer layers
    cfg.MODEL.BACKBONE.EMBED_DIM = 768       # ViT-Base hidden dimension
    cfg.MODEL.BACKBONE.PROJECT_DIM = 256     # C=256, projected channel dim (Sec 4, Eq.30)
    cfg.MODEL.BACKBONE.PATCH_SIZE = 16       # ViT patch size
    cfg.MODEL.BACKBONE.DROP_PATH_RATE = 0.1

    # --- EDM: Equivariant Decomposition Module (Sec 3.3, App. B.6) ---
    # "EDM employs N_eq=16 harmonic bases, N_inv=8 Gaussian derivatives,
    #  N_cp=16 directional derivatives" (Sec 5.1.3)
    cfg.MODEL.EDM = edict()
    cfg.MODEL.EDM.KERNEL_SIZE = 7            # K=7, "discretized on a 7×7 spatial grid" (Sec 3.3)
    cfg.MODEL.EDM.COEFF_HIDDEN_DIM = 64      # "Coefficient generators are two-layer MLPs with hidden dim 64"

    # Equivariant subspace basis functions (Eq.14)
    # "σ^S ∈ {1,2,3} pixels" for small-scale Gaussian derivatives
    cfg.MODEL.EDM.N_EQ = 16                  # N_eq = 16 harmonic bases
    cfg.MODEL.EDM.EQ_SIGMAS = [1.0, 2.0, 3.0]       # σ^S: small scale parameters (Eq.14)
    cfg.MODEL.EDM.EQ_DERIVATIVE_ORDERS = [0, 1, 2]   # m_k ∈ {0,1,2} (Eq.11)

    # Invariant subspace basis functions (Eq.13)
    # "σ^L ∈ {4,6,8} pixels" for large-scale Gaussian derivatives
    cfg.MODEL.EDM.N_INV = 8                  # N_inv = 8 radially symmetric bases
    cfg.MODEL.EDM.INV_SIGMAS = [4.0, 6.0, 8.0]      # σ^L: large scale parameters (Eq.13)
    cfg.MODEL.EDM.INV_DERIVATIVE_ORDERS = [0, 1, 2]  # m_k ∈ {0,1,2} (Eq.11)

    # Coupled subspace basis functions (Eq.15-16)
    # "n ∈ {1,2,3,4} are non-zero angular frequencies"
    cfg.MODEL.EDM.N_CP = 16                  # N_cp = 16 directional derivatives
    cfg.MODEL.EDM.CP_ANGULAR_FREQS = [1, 2, 3, 4]   # n ∈ {1,2,3,4} (Eq.15-16)
    cfg.MODEL.EDM.CP_RADIAL_HIDDEN_DIM = 32          # "R_k(r) is a two-layer MLP with hidden dim 32"

    # Spectral boundary parameter (Eq.3-5, App. D.3 Tab.8)
    # "The bandwidth parameter ε=0.10 defines the spectral boundary
    #  between Ω_inv and Ω_eq" (Sec 5.1.3)
    cfg.MODEL.EDM.EPSILON = 0.10             # ε = 0.10

    # --- GAB: Geometry-Aware Bridging Block (Sec 3.4) ---
    cfg.MODEL.GAB = edict()
    cfg.MODEL.GAB.FUSION_TEMPERATURE = 0.5   # τ in softmax fusion (Eq.22)
    cfg.MODEL.GAB.INV_TAU = 0.5              # τ_inv in R_inv reliability (Eq.18)
    cfg.MODEL.GAB.NUM_HEADS = 8              # Cross-attention heads
    cfg.MODEL.GAB.ATTN_DROP = 0.0
    cfg.MODEL.GAB.PROJ_DROP = 0.0

    # --- Tracking Head (Sec 3.5) ---
    cfg.MODEL.HEAD = edict()
    cfg.MODEL.HEAD.TYPE = "corner"           # Corner-based prediction
    cfg.MODEL.HEAD.HIDDEN_DIM = 256
    cfg.MODEL.HEAD.NUM_LAYERS = 3

    # --- Scale Exponent (App. D.7, Tab.13-14) ---
    # "The scale exponent α is learned per layer during training"
    cfg.MODEL.ALPHA = edict()
    cfg.MODEL.ALPHA.INIT = 1.0               # Initial value
    cfg.MODEL.ALPHA.MIN = 0.1                # "α ∈ [0.1, 2.0] through projected gradient descent"
    cfg.MODEL.ALPHA.MAX = 2.0
    cfg.MODEL.ALPHA.REGULARIZATION = 0.01    # "λ_α = 0.01" (App. B.7)

    # =========================================================================
    # DATA (Sec 5.1.3)
    # =========================================================================
    cfg.DATA = edict()

    # "Search and template regions are cropped to 384×384 and 192×192 pixels"
    cfg.DATA.SEARCH_SIZE = 384               # Search region size
    cfg.DATA.TEMPLATE_SIZE = 192             # Template region size
    cfg.DATA.SEARCH_FACTOR = 4.0             # "context factor of 4.0" (Sec 3.5.2)
    cfg.DATA.TEMPLATE_FACTOR = 2.0           # Template context factor
    cfg.DATA.CENTER_JITTER_SEARCH = 3.0      # Search region jitter
    cfg.DATA.CENTER_JITTER_TEMPLATE = 0.0    # Template jitter
    cfg.DATA.SCALE_JITTER_SEARCH = 0.25      # Scale augmentation factor
    cfg.DATA.SCALE_JITTER_TEMPLATE = 0.0

    # Patch token counts derived from crop sizes & patch_size=16:
    #   N_s = (384/16)^2 = 576,  N_t = (192/16)^2 = 144   (Sec 4, Eq.30)
    cfg.DATA.NUM_SEARCH_TOKENS = 576         # N_s = 576
    cfg.DATA.NUM_TEMPLATE_TOKENS = 144       # N_t = 144

    # =========================================================================
    # TRAIN (Sec 5.1.3, Algo 1)
    # =========================================================================
    cfg.TRAIN = edict()
    cfg.TRAIN.EPOCHS = 300                   # "We train for 300 epochs"
    cfg.TRAIN.BATCH_SIZE = 16                # "batch size 16"
    cfg.TRAIN.OPTIMIZER = "AdamW"            # "AdamW"
    cfg.TRAIN.LR = 1e-4                      # "learning rate 10^{-4}"
    cfg.TRAIN.WEIGHT_DECAY = 1e-4            # "weight decay 10^{-4}"
    cfg.TRAIN.LR_SCHEDULER = "cosine"
    cfg.TRAIN.WARMUP_EPOCHS = 10
    cfg.TRAIN.NUM_WORKERS = 8
    cfg.TRAIN.SEED = 42                      # "mean of three runs with different seeds"
    cfg.TRAIN.GRAD_CLIP_NORM = 1.0

    # Loss weights (Eq.23, App. D.6 Tab.12)
    # "λ_cls=2.0, λ_reg=5.0, λ_iou=2.0, λ_eq=1.0"
    cfg.TRAIN.LOSS = edict()
    cfg.TRAIN.LOSS.LAMBDA_CLS = 2.0          # λ_cls: focal loss weight
    cfg.TRAIN.LOSS.LAMBDA_REG = 5.0          # λ_reg: L1 regression weight
    cfg.TRAIN.LOSS.LAMBDA_IOU = 2.0          # λ_iou: GIoU loss weight
    cfg.TRAIN.LOSS.LAMBDA_EQ = 1.0           # λ_eq:  equivariance loss weight

    # Equivariance loss scale sampling (Sec 3.5.1)
    # "Scale factors s are sampled from a log-uniform distribution on [0.5, 2.0]"
    cfg.TRAIN.LOSS.SCALE_RANGE = [0.5, 2.0]

    # Datasets used for training
    cfg.TRAIN.DATASETS = ["lasot", "got10k", "coco", "trackingnet"]

    # =========================================================================
    # TEST / INFERENCE (Sec 3.5.2, Algo 2)
    # =========================================================================
    cfg.TEST = edict()
    cfg.TEST.SEARCH_FACTOR = 4.0             # "context factor of 4.0" (Sec 3.5.2)
    cfg.TEST.EMA_MOMENTUM = 0.9              # "EMA smoothing at momentum 0.9" (Algo 2, line 11)
    cfg.TEST.WINDOW_PENALTY = 0.0            # Hanning window penalty (optional)
    cfg.TEST.GPU_ID = 0

    # Evaluation scale factors for equivariance error (Eq.27-29, Sec 5.1.2)
    # "averaged over s ∈ {0.5, 0.75, 1.5, 2.0}"
    cfg.TEST.EQUIV_EVAL_SCALES = [0.5, 0.75, 1.5, 2.0]

    # =========================================================================
    # DATASET PATHS
    # =========================================================================
    cfg.DATASET = edict()
    cfg.DATASET.UAV123 = edict()
    cfg.DATASET.UAV123.ROOT = "data/UAV123"
    cfg.DATASET.LASOT = edict()
    cfg.DATASET.LASOT.ROOT = "data/LaSOT"
    cfg.DATASET.VISDRONE = edict()
    cfg.DATASET.VISDRONE.ROOT = "data/VisDrone-SOT"
    cfg.DATASET.ARDMAV = edict()
    cfg.DATASET.ARDMAV.ROOT = "data/ARD-MAV"
    cfg.DATASET.VOTS2024 = edict()
    cfg.DATASET.VOTS2024.ROOT = "data/VOTS2024"

    return cfg


def merge_from_yaml(cfg, yaml_path):
    """
    Merge configuration from a YAML file into the default config.

    Args:
        cfg: EasyDict default configuration.
        yaml_path: Path to YAML config file.

    Returns:
        Updated EasyDict configuration.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        yaml_cfg = yaml.safe_load(f)

    def _recursive_update(base, override):
        """Recursively update nested dict."""
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                _recursive_update(base[key], value)
            else:
                base[key] = value

    _recursive_update(cfg, yaml_cfg)
    return cfg
