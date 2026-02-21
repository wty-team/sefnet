"""
SEFNet Evaluation Script
==========================

Implements the One-Pass Evaluation (OPE) protocol (Section 5.1).

Evaluation metrics:
    - AUC (Area Under Curve): Success plot threshold sweep [0, 1]
    - Precision: Center Location Error at threshold 20 pixels
    - Normalized Precision: Normalized by target size

Supported benchmarks:
    - UAV123: 123 sequences (Section 5.2, Table 1)
    - DTB70: 70 sequences
    - VisDrone2019-SOT: 96 test sequences
    - ARD-MAV: Air-to-air tracking (Table 2)

Scale-stratified analysis (Section 5.2):
    Sequences grouped by scale variation magnitude into
    Small/Medium/Large categories to evaluate scale robustness.

References:
    - Algorithm 2: Inference procedure
    - Section 5.1: Evaluation protocol
    - Section 5.2: Main results and ablation
    - Table 1-3: Benchmark results
"""

import os
import sys
import json
import time
import argparse
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import get_config
from lib.models.sefnet import build_sefnet
from lib.models.head.tracking_head import TrackingHead

logger = logging.getLogger("sefnet.eval")


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes in [x, y, w, h] format.

    Args:
        box1, box2: Bounding boxes [x, y, w, h].

    Returns:
        IoU value.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter

    return inter / max(union, 1e-8)


def compute_center_error(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute center location error between two boxes [x,y,w,h]."""
    c1 = np.array([box1[0] + box1[2] / 2, box1[1] + box1[3] / 2])
    c2 = np.array([box2[0] + box2[2] / 2, box2[1] + box2[3] / 2])
    return np.linalg.norm(c1 - c2)


def compute_success_auc(ious: np.ndarray) -> float:
    """
    Compute AUC of the success plot.

    Sweeps IoU threshold from 0 to 1 and computes the fraction
    of frames above each threshold. AUC is the area under this curve.
    """
    thresholds = np.linspace(0, 1, 101)
    success = np.array([np.mean(ious >= t) for t in thresholds])
    return np.trapz(success, thresholds)


def compute_precision(errors: np.ndarray, threshold: float = 20.0) -> float:
    """Compute precision at given pixel threshold."""
    return np.mean(errors <= threshold)


def compute_normalized_precision(
    errors: np.ndarray, gt_boxes: np.ndarray,
) -> float:
    """Compute normalized precision (error / target_size)."""
    target_sizes = np.sqrt(gt_boxes[:, 2] * gt_boxes[:, 3]).clip(min=1)
    norm_errors = errors / target_sizes
    thresholds = np.linspace(0, 0.5, 51)
    prec = np.array([np.mean(norm_errors <= t) for t in thresholds])
    return np.trapz(prec, thresholds) / 0.5


class Tracker:
    """
    SEFNet tracker wrapper for evaluation.

    Implements Algorithm 2 with template caching and EMA smoothing.

    Args:
        model: SEFNet model (eval mode).
        search_size: Search region size (default 384).
        template_size: Template size (default 192).
        context_factor: Context factor (default 4.0).
        ema_momentum: Box smoothing momentum (default 0.9).
    """

    def __init__(
        self,
        model,
        search_size: int = 384,
        template_size: int = 192,
        context_factor: float = 4.0,
        ema_momentum: float = 0.9,
    ):
        self.model = model
        self.search_size = search_size
        self.template_size = template_size
        self.context_factor = context_factor
        self.ema_momentum = ema_momentum

        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

        self._state = {}

    def initialize(self, image: np.ndarray, box: np.ndarray):
        """
        Initialize tracker with first frame (Algorithm 2, line 1-4).

        Args:
            image: First frame [H, W, 3] RGB.
            box: Initial target box [x, y, w, h].
        """
        self._state = {
            "target_box": box.copy(),
            "initial_box": box.copy(),
        }

        # Crop and encode template
        template_crop = self._crop_region(
            image, box, self.template_size, context=2.0
        )
        template_tensor = self._preprocess(template_crop)

        with torch.no_grad():
            self.model.initialize_template(template_tensor)

    def track(self, image: np.ndarray) -> np.ndarray:
        """
        Track target in new frame (Algorithm 2, line 5-11).

        Args:
            image: Current frame [H, W, 3] RGB.

        Returns:
            Predicted box [x, y, w, h].
        """
        prev_box = self._state["target_box"]

        # Crop search region centered on previous prediction
        search_crop = self._crop_region(
            image, prev_box, self.search_size, context=self.context_factor
        )
        search_tensor = self._preprocess(search_crop)

        # Build box tensors for scale estimation
        current_box = torch.tensor(
            self._xywh_to_cxcywh(prev_box), dtype=torch.float32
        ).unsqueeze(0).cuda()
        initial_box = torch.tensor(
            self._xywh_to_cxcywh(self._state["initial_box"]), dtype=torch.float32
        ).unsqueeze(0).cuda()

        # Forward inference
        with torch.no_grad():
            output = self.model.forward_inference(
                search=search_tensor,
                current_box=current_box,
                initial_box=initial_box,
            )

        # Extract best prediction
        best_box = output["best_box"][0].cpu().numpy()  # [4] normalized cxcywh

        # Map back to image coordinates
        pred_box = self._map_box_back(best_box, prev_box)

        # EMA smoothing (Algorithm 2, line 11)
        smoothed = self.ema_momentum * prev_box + (1 - self.ema_momentum) * pred_box
        self._state["target_box"] = smoothed

        return smoothed

    def _crop_region(
        self, image: np.ndarray, box: np.ndarray,
        output_size: int, context: float,
    ) -> np.ndarray:
        """Crop and resize region centered on box."""
        import math
        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        target_size = math.sqrt(w * h)
        crop_size = target_size * context
        half = crop_size / 2

        img_h, img_w = image.shape[:2]
        x1, y1 = int(cx - half), int(cy - half)
        x2, y2 = int(cx + half), int(cy + half)

        # Pad boundaries
        pad = [max(0, -x1), max(0, -y1), max(0, x2 - img_w), max(0, y2 - img_h)]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        crop = image[y1:y2, x1:x2]

        if any(p > 0 for p in pad):
            crop = np.pad(crop, ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
                          mode="constant", constant_values=128)

        # Resize using PIL
        pil_crop = Image.fromarray(crop)
        pil_crop = pil_crop.resize((output_size, output_size), Image.BILINEAR)
        return np.array(pil_crop)

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        """Normalize and convert to tensor [1, 3, H, W]."""
        img = crop.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        return tensor.cuda()

    def _map_box_back(self, box_norm: np.ndarray, prev_box: np.ndarray) -> np.ndarray:
        """Map normalized prediction back to image coordinates."""
        import math
        x, y, w, h = prev_box
        cx_prev, cy_prev = x + w / 2, y + h / 2
        target_size = math.sqrt(w * h)
        crop_size = target_size * self.context_factor

        # Denormalize from [0,1] to crop pixels, then to image pixels
        scale = crop_size / self.search_size
        pred_cx = (box_norm[0] * self.search_size - self.search_size / 2) * scale + cx_prev
        pred_cy = (box_norm[1] * self.search_size - self.search_size / 2) * scale + cy_prev
        pred_w = box_norm[2] * self.search_size * scale
        pred_h = box_norm[3] * self.search_size * scale

        return np.array([pred_cx - pred_w / 2, pred_cy - pred_h / 2, pred_w, pred_h])

    @staticmethod
    def _xywh_to_cxcywh(box):
        return [box[0] + box[2] / 2, box[1] + box[3] / 2, box[2], box[3]]


def evaluate_sequence(tracker: Tracker, frames: list, gt_boxes: np.ndarray):
    """
    Evaluate tracker on a single sequence (OPE protocol).

    Args:
        tracker: Tracker instance.
        frames: List of frame file paths.
        gt_boxes: GT boxes [N_frames, 4] as [x, y, w, h].

    Returns:
        Dict with per-frame IoUs, center errors, and timing.
    """
    n_frames = len(frames)
    pred_boxes = np.zeros((n_frames, 4))
    ious = np.zeros(n_frames)
    center_errors = np.zeros(n_frames)
    times = []

    # Initialize with first frame
    first_frame = np.array(Image.open(frames[0]).convert("RGB"))
    tracker.initialize(first_frame, gt_boxes[0])
    pred_boxes[0] = gt_boxes[0]
    ious[0] = 1.0
    center_errors[0] = 0.0

    # Track remaining frames
    for i in range(1, n_frames):
        frame = np.array(Image.open(frames[i]).convert("RGB"))

        t_start = time.time()
        pred_box = tracker.track(frame)
        t_elapsed = time.time() - t_start
        times.append(t_elapsed)

        pred_boxes[i] = pred_box
        ious[i] = compute_iou(pred_box, gt_boxes[i])
        center_errors[i] = compute_center_error(pred_box, gt_boxes[i])

    fps = len(times) / sum(times) if times else 0

    return {
        "pred_boxes": pred_boxes,
        "ious": ious,
        "center_errors": center_errors,
        "fps": fps,
    }


def evaluate_dataset(
    tracker: Tracker,
    data_root: str,
    dataset_name: str,
    output_dir: str,
):
    """
    Evaluate on full dataset and compute aggregate metrics.

    Args:
        tracker: Tracker instance.
        data_root: Dataset root directory.
        dataset_name: Dataset name.
        output_dir: Directory for saving results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset annotations
    ann_file = os.path.join(data_root, dataset_name, "test.json")
    with open(ann_file, "r") as f:
        data = json.load(f)

    all_ious = []
    all_errors = []
    all_gt_boxes = []
    fps_list = []
    results_per_seq = {}

    for seq in data["sequences"]:
        seq_name = seq["name"]
        frames = seq["frames"]
        gt_boxes = np.array(seq["boxes"], dtype=np.float64)

        logger.info(f"Evaluating: {seq_name} ({len(frames)} frames)")

        result = evaluate_sequence(tracker, frames, gt_boxes)
        results_per_seq[seq_name] = {
            "auc": compute_success_auc(result["ious"]),
            "precision": compute_precision(result["center_errors"]),
            "fps": result["fps"],
        }

        all_ious.append(result["ious"])
        all_errors.append(result["center_errors"])
        all_gt_boxes.append(gt_boxes)
        fps_list.append(result["fps"])

        logger.info(
            f"  AUC: {results_per_seq[seq_name]['auc']:.3f}  "
            f"Prec: {results_per_seq[seq_name]['precision']:.3f}  "
            f"FPS: {result['fps']:.1f}"
        )

    # Aggregate metrics
    all_ious = np.concatenate(all_ious)
    all_errors = np.concatenate(all_errors)
    all_gt = np.concatenate(all_gt_boxes)

    overall = {
        "AUC": compute_success_auc(all_ious),
        "Precision": compute_precision(all_errors),
        "NormPrecision": compute_normalized_precision(all_errors, all_gt),
        "AvgFPS": np.mean(fps_list),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"AUC:            {overall['AUC']:.3f}")
    logger.info(f"Precision:      {overall['Precision']:.3f}")
    logger.info(f"Norm Precision: {overall['NormPrecision']:.3f}")
    logger.info(f"Avg FPS:        {overall['AvgFPS']:.1f}")
    logger.info(f"{'='*50}")

    # Save results
    results = {"overall": overall, "per_sequence": results_per_seq}
    results_file = os.path.join(output_dir, f"{dataset_name}_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    return overall


def main():
    parser = argparse.ArgumentParser(description="SEFNet Evaluation")
    parser.add_argument("--config", type=str, default="experiments/sefnet_vit_base.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="uav123")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="output/eval")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cfg = get_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build and load model
    model = build_sefnet(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Build tracker
    tracker = Tracker(
        model,
        search_size=cfg.DATA.SEARCH_SIZE,
        template_size=cfg.DATA.TEMPLATE_SIZE,
    )

    # Evaluate
    evaluate_dataset(tracker, args.data_root, args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
