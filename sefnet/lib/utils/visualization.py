"""
SEFNet Visualization Utilities
================================

Tools for visualizing learned behaviors and analysis results,
corresponding to key figures and tables in the paper:

    - Figure 5: Layer selection frequency distribution
    - Figure 6: Reliability curves vs scale factor
    - Figure 7: Fusion weight evolution during tracking
    - Table 7: Learned layer affinity preferences
    - Figure 8: Equivariance error curves (Eq. 27-29)

These visualizations help validate that the model learns the expected
behaviors described in the theoretical analysis.
"""

import os
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

import torch
from typing import Dict, List, Optional, Tuple


def plot_layer_preferences(
    affinity: Dict[str, np.ndarray],
    save_path: str = "layer_preferences.pdf",
):
    """
    Visualize learned layer affinity β_t^(l) (Table 7, Figure 5).

    Expected patterns after training:
        eq  → L1-4 (78.7%): shallow geometric features
        inv → L7-12 (80.7%): deep semantic features
        cp  → L3-8 (71.1%): mid-level contextual patterns

    Args:
        affinity: Dict type → weights [L].
        save_path: Output file path.
    """
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    type_names = {"eq": "Equivariant", "inv": "Invariant", "cp": "Coupled"}
    colors = {"eq": "#2196F3", "inv": "#4CAF50", "cp": "#FF9800"}

    for ax, (t, name) in zip(axes, type_names.items()):
        weights = affinity[t]
        L = len(weights)
        layers = np.arange(1, L + 1)

        ax.bar(layers, weights, color=colors[t], alpha=0.8, edgecolor="white")
        ax.set_xlabel("Layer Index", fontsize=11)
        ax.set_ylabel("Affinity Weight β", fontsize=11)
        ax.set_title(f"{name} Features", fontsize=12, fontweight="bold")
        ax.set_xticks(layers)
        ax.set_ylim(0, max(weights) * 1.2)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_reliability_curves(
    scales: np.ndarray,
    R_inv: np.ndarray,
    R_eq: np.ndarray,
    R_cp: np.ndarray,
    save_path: str = "reliability_curves.pdf",
):
    """
    Plot reliability measures vs scale factor (Figure 6).

    Shows how each subspace's reliability varies with target scale.
    Expected behavior:
        R_inv: Peaks near s=1 (identity), decays symmetrically
        R_eq:  Relatively stable (direction preserved under scaling)
        R_cp:  Moderate with slight asymmetry (phase effects)

    Args:
        scales: Scale factor values [N].
        R_inv, R_eq, R_cp: Reliability values [N].
        save_path: Output file path.
    """
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(scales, R_eq, "b-", linewidth=2, label=r"$R_{eq}(s)$ (Eq. 19)")
    ax.plot(scales, R_inv, "g--", linewidth=2, label=r"$R_{inv}(s)$ (Eq. 18)")
    ax.plot(scales, R_cp, "r-.", linewidth=2, label=r"$R_{cp}(s)$ (Eq. 20)")

    ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5, label="s = 1 (no change)")
    ax.set_xlabel("Scale Factor s", fontsize=12)
    ax.set_ylabel("Reliability Score", fontsize=12)
    ax.set_title("Reliability Measures vs Scale", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower left")
    ax.set_xlim(scales.min(), scales.max())
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_fusion_weights(
    frames: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    save_path: str = "fusion_weights.pdf",
):
    """
    Plot fusion weight evolution during tracking (Figure 7).

    Shows how [α, β, γ] adapt as the target's scale changes.
    Expected: α rises when target enlarges, β rises when target shrinks.

    Args:
        frames: Frame indices [T].
        alphas, betas, gammas: Fusion weights [T].
        save_path: Output file path.
    """
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.fill_between(frames, 0, alphas, alpha=0.3, color="#2196F3")
    ax.fill_between(frames, alphas, alphas + betas, alpha=0.3, color="#4CAF50")
    ax.fill_between(frames, alphas + betas, 1.0, alpha=0.3, color="#FF9800")

    ax.plot(frames, alphas, "b-", linewidth=1.5, label=r"$\alpha$ (equivariant)")
    ax.plot(frames, alphas + betas, "g-", linewidth=1.5, label=r"$\beta$ (invariant)")

    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel("Cumulative Weight", fontsize=11)
    ax.set_title("Adaptive Fusion Weight Evolution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_equivariance_errors(
    scales: np.ndarray,
    E_eq: np.ndarray,
    E_inv: np.ndarray,
    E_cp: np.ndarray,
    save_path: str = "equivariance_errors.pdf",
):
    """
    Plot equivariance error metrics (Eq. 27-29, Figure 8).

    Evaluates how well learned features satisfy transformation laws.
    Lower is better. Evaluated at s ∈ {0.5, 0.75, 1.5, 2.0}.

    Args:
        scales: Scale factors [N].
        E_eq, E_inv, E_cp: Error values [N].
        save_path: Output file path.
    """
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(scales, E_eq, "bo-", linewidth=2, markersize=8,
            label=r"$E_{eq}(s)$ (Eq. 27)")
    ax.plot(scales, E_inv, "gs-", linewidth=2, markersize=8,
            label=r"$E_{inv}(s)$ (Eq. 28)")
    ax.plot(scales, E_cp, "r^-", linewidth=2, markersize=8,
            label=r"$E_{cp}(s)$ (Eq. 29)")

    ax.set_xlabel("Scale Factor s", fontsize=12)
    ax.set_ylabel("Equivariance Error", fontsize=12)
    ax.set_title("Transformation Error Metrics", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_tracking_result(
    frame: np.ndarray,
    gt_box: np.ndarray,
    pred_box: np.ndarray,
    frame_idx: int = 0,
    save_path: Optional[str] = None,
):
    """
    Visualize tracking result on a single frame.

    Args:
        frame: Image [H, W, 3] RGB.
        gt_box: Ground truth [x, y, w, h].
        pred_box: Predicted [x, y, w, h].
        frame_idx: Frame number for title.
        save_path: Optional save path.
    """
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(frame)

    # GT box (green)
    from matplotlib.patches import Rectangle
    rect_gt = Rectangle(
        (gt_box[0], gt_box[1]), gt_box[2], gt_box[3],
        linewidth=2, edgecolor="lime", facecolor="none", label="GT",
    )
    ax.add_patch(rect_gt)

    # Predicted box (red)
    rect_pred = Rectangle(
        (pred_box[0], pred_box[1]), pred_box[2], pred_box[3],
        linewidth=2, edgecolor="red", facecolor="none", label="Pred",
    )
    ax.add_patch(rect_pred)

    iou = _compute_iou_np(gt_box, pred_box)
    ax.set_title(f"Frame {frame_idx} | IoU: {iou:.3f}", fontsize=12)
    ax.legend(fontsize=10)
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _compute_iou_np(box1, box2):
    """Compute IoU for numpy boxes [x,y,w,h]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    return inter / max(union, 1e-8)


def generate_analysis_report(
    model,
    output_dir: str = "analysis",
):
    """
    Generate comprehensive analysis visualizations.

    Produces all key figures from the paper for a trained model:
        - Layer preference distribution (Table 7)
        - Reliability curves (Figure 6)

    Args:
        model: Trained SEFNet model.
        output_dir: Output directory for figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Layer preferences
    with torch.no_grad():
        affinity = model.gab.get_layer_preferences()
        aff_np = {t: w.cpu().numpy() for t, w in affinity.items()}
    plot_layer_preferences(aff_np, os.path.join(output_dir, "layer_preferences.pdf"))

    print(f"Analysis report saved to {output_dir}/")
