"""
UAV Tracking Dataset
======================

Implements the dataset pipeline for SEFNet training and evaluation.

Supports UAV-specific tracking benchmarks (Section 5.1):
    - UAV123: 123 sequences, various UAV scenarios
    - DTB70: 70 drone-captured sequences with fast motion
    - VisDrone2019-SOT: 96 test sequences from drone platform
    - ARD-MAV: Air-to-air drone tracking (most challenging)

Data loading pipeline:
    1. Sample frame pair (template frame, search frame)
    2. Crop template (192×192) and search (384×384) centered on target
    3. Apply data augmentation (color jitter, horizontal flip)
    4. Apply scale augmentation T_s for equivariance loss (Sec 3.5)
    5. Generate classification labels and regression targets

Scale augmentation (Sec 3.5):
    Scale factors sampled from log-uniform on [0.5, 2.0].
    The augmented search region creates the transformation pair
    (I, T_s I) used to compute L_eq (Eq. 24-26).

Context factor: 4.0× target size for search region cropping.

References:
    - Section 5.1: Datasets and evaluation metrics
    - Section 5.3: Training details (augmentation, cropping)
    - Algorithm 1, line 1-3: Sample pair and augment
"""

import os
import json
import random
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, List
from PIL import Image

try:
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
except ImportError:
    T = None
    TF = None


class UAVTrackingDataset(Dataset):
    """
    UAV tracking dataset for SEFNet training.

    Each sample is a (template, search) frame pair from the same sequence.
    The template is cropped from the reference frame using the GT box,
    and the search region is cropped from a nearby frame.

    Args:
        data_root: Root directory of the dataset.
        dataset_name: Name of the dataset ('uav123', 'dtb70', 'visdrone', 'ard_mav').
        split: 'train' or 'test'.
        template_size: Template crop size (default 192).
        search_size: Search region crop size (default 384).
        context_factor: Search region context factor (default 4.0).
        max_gap: Maximum frame gap for sampling pairs (default 100).
        scale_aug: Whether to apply scale augmentation for L_eq (default True).
        scale_range: Scale factor range for augmentation (default (0.5, 2.0)).
        color_jitter: Color jitter probability (default 0.3).
    """

    def __init__(
        self,
        data_root: str,
        dataset_name: str = "uav123",
        split: str = "train",
        template_size: int = 192,
        search_size: int = 384,
        context_factor: float = 4.0,
        max_gap: int = 100,
        scale_aug: bool = True,
        scale_range: Tuple[float, float] = (0.5, 2.0),
        color_jitter: float = 0.3,
    ):
        super().__init__()
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.split = split
        self.template_size = template_size
        self.search_size = search_size
        self.context_factor = context_factor
        self.max_gap = max_gap
        self.scale_aug = scale_aug
        self.scale_range = scale_range

        # Load sequence annotations
        self.sequences = self._load_annotations()

        # Data augmentation transforms
        if T is not None:
            self.color_transform = T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ) if color_jitter > 0 else None
            self.color_jitter_prob = color_jitter

            self.normalize = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.color_transform = None
            self.color_jitter_prob = 0.0
            self.normalize = None

    def _load_annotations(self) -> List[Dict]:
        """
        Load dataset annotations.

        Expected format per sequence:
            {
                'name': 'sequence_name',
                'frames': ['path/to/frame001.jpg', ...],
                'boxes': [[x, y, w, h], ...]  # per-frame GT boxes
            }

        Returns:
            List of sequence dicts.
        """
        ann_file = os.path.join(
            self.data_root, self.dataset_name, f"{self.split}.json"
        )
        if os.path.exists(ann_file):
            with open(ann_file, "r") as f:
                data = json.load(f)
            return data["sequences"]

        # Fallback: try to load from individual sequence directories
        sequences = []
        seq_dir = os.path.join(self.data_root, self.dataset_name, self.split)
        if os.path.isdir(seq_dir):
            for seq_name in sorted(os.listdir(seq_dir)):
                seq_path = os.path.join(seq_dir, seq_name)
                gt_file = os.path.join(seq_path, "groundtruth_rect.txt")
                if os.path.exists(gt_file):
                    boxes = np.loadtxt(gt_file, delimiter=",").tolist()
                    frames = sorted([
                        os.path.join(seq_path, "img", f)
                        for f in os.listdir(os.path.join(seq_path, "img"))
                        if f.endswith((".jpg", ".png"))
                    ])
                    sequences.append({
                        "name": seq_name,
                        "frames": frames,
                        "boxes": boxes,
                    })

        return sequences

    def __len__(self) -> int:
        return sum(len(seq["boxes"]) for seq in self.sequences)

    def _sample_pair(self) -> Tuple[Dict, int, int]:
        """
        Sample a (template_frame, search_frame) pair.

        The template frame is sampled uniformly, and the search frame
        is sampled within max_gap frames of the template.

        Returns:
            Tuple of (sequence_dict, template_idx, search_idx).
        """
        seq = random.choice(self.sequences)
        n_frames = len(seq["boxes"])

        template_idx = random.randint(0, n_frames - 1)

        # Sample search frame within max_gap
        min_idx = max(0, template_idx - self.max_gap)
        max_idx = min(n_frames - 1, template_idx + self.max_gap)
        search_idx = random.randint(min_idx, max_idx)

        return seq, template_idx, search_idx

    def _crop_and_resize(
        self,
        image: Image.Image,
        box: List[float],
        output_size: int,
        context: float = 1.0,
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Crop region centered on box with context padding and resize.

        The context factor determines how much area around the target
        is included. A factor of 4.0 means the crop is 4× the target size.

        Args:
            image: PIL Image.
            box: [x, y, w, h] ground truth box.
            output_size: Target crop size in pixels.
            context: Context factor multiplier.

        Returns:
            Tuple of (cropped_resized_image, box_in_crop_normalized).
        """
        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2

        # Context region
        target_size = math.sqrt(w * h)
        crop_size = target_size * context
        half = crop_size / 2

        # Crop coordinates
        x1 = int(cx - half)
        y1 = int(cy - half)
        x2 = int(cx + half)
        y2 = int(cy + half)

        # Handle boundary with padding
        img_w, img_h = image.size
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - img_w)
        pad_bottom = max(0, y2 - img_h)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        # Crop
        crop = image.crop((x1, y1, x2, y2))

        # Pad if necessary (mean padding)
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            padded = Image.new(crop.mode, (
                crop.width + pad_left + pad_right,
                crop.height + pad_top + pad_bottom
            ), (128, 128, 128))
            padded.paste(crop, (pad_left, pad_top))
            crop = padded

        # Resize
        crop_resized = crop.resize((output_size, output_size), Image.BILINEAR)

        # Compute box coordinates in the resized crop (normalized to [0, 1])
        scale = output_size / crop_size if crop_size > 0 else 1.0
        box_cx = (cx - (cx - half)) * scale / output_size
        box_cy = (cy - (cy - half)) * scale / output_size
        box_w = w * scale / output_size
        box_h = h * scale / output_size
        box_norm = np.array([box_cx, box_cy, box_w, box_h], dtype=np.float32)
        box_norm = np.clip(box_norm, 0, 1)

        return crop_resized, box_norm

    def _apply_scale_augmentation(
        self,
        image: Image.Image,
        box: List[float],
        scale_factor: float,
        output_size: int,
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Apply scale augmentation T_s to create transformation pair (Sec 3.5).

        Rescales the crop region by factor s, so the target appears at
        a different scale within the same output resolution.

        Args:
            image: Original PIL Image.
            box: [x, y, w, h] box.
            scale_factor: Scale factor s from log-uniform [0.5, 2.0].
            output_size: Target output size.

        Returns:
            Tuple of (augmented_crop, box_in_augmented_normalized).
        """
        # Modified context: scale the crop region inversely
        # Larger s → target appears larger → crop smaller region
        aug_context = self.context_factor / scale_factor
        return self._crop_and_resize(image, box, output_size, context=aug_context)

    def _sample_scale_factor(self) -> float:
        """
        Sample scale factor from log-uniform distribution on [s_min, s_max].

        Returns:
            Scale factor s.
        """
        s_min, s_max = self.scale_range
        log_s = random.uniform(math.log(s_min), math.log(s_max))
        return math.exp(log_s)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            Dict with:
                - 'template': Template image [3, 192, 192].
                - 'search': Search image [3, 384, 384].
                - 'search_aug': Scale-augmented search [3, 384, 384] (if scale_aug).
                - 'template_box': GT box in template (normalized).
                - 'search_box': GT box in search (normalized).
                - 'scale_factor': Applied scale factor.
                - 'cls_label': Classification label map [N_s].
                - 'fg_mask': Foreground mask [N_s].
        """
        seq, t_idx, s_idx = self._sample_pair()

        # Load images
        t_img = Image.open(seq["frames"][t_idx]).convert("RGB")
        s_img = Image.open(seq["frames"][s_idx]).convert("RGB")

        t_box = seq["boxes"][t_idx]
        s_box = seq["boxes"][s_idx]

        # Crop template and search
        template_crop, t_box_norm = self._crop_and_resize(
            t_img, t_box, self.template_size, context=2.0
        )
        search_crop, s_box_norm = self._crop_and_resize(
            s_img, s_box, self.search_size, context=self.context_factor
        )

        # Color jitter (applied consistently to both)
        if self.color_transform and random.random() < self.color_jitter_prob:
            template_crop = self.color_transform(template_crop)
            search_crop = self.color_transform(search_crop)

        # Scale augmentation for equivariance loss
        scale_factor = 1.0
        search_aug_crop = None
        if self.scale_aug:
            scale_factor = self._sample_scale_factor()
            search_aug_crop, _ = self._apply_scale_augmentation(
                s_img, s_box, scale_factor, self.search_size
            )
            if self.color_transform and random.random() < self.color_jitter_prob:
                search_aug_crop = self.color_transform(search_aug_crop)

        # Convert to tensors
        if self.normalize:
            template = self.normalize(template_crop)
            search = self.normalize(search_crop)
            search_aug = self.normalize(search_aug_crop) if search_aug_crop else search
        else:
            template = torch.from_numpy(
                np.array(template_crop).transpose(2, 0, 1).astype(np.float32) / 255.0
            )
            search = torch.from_numpy(
                np.array(search_crop).transpose(2, 0, 1).astype(np.float32) / 255.0
            )
            search_aug = search

        # Generate labels for the search region
        feat_size = self.search_size // 16  # 384/16 = 24
        cls_label, fg_mask = self._generate_labels(s_box_norm, feat_size)

        # Expand search box to N_s tokens for regression target
        N_s = feat_size * feat_size
        search_box_expanded = torch.tensor(s_box_norm).unsqueeze(0).expand(N_s, -1)

        sample = {
            "template": template,
            "search": search,
            "search_aug": search_aug,
            "template_box": torch.tensor(t_box_norm, dtype=torch.float32),
            "search_box": torch.tensor(s_box_norm, dtype=torch.float32),
            "search_box_expanded": search_box_expanded,
            "scale_factor": torch.tensor(scale_factor, dtype=torch.float32),
            "cls_label": cls_label,
            "fg_mask": fg_mask,
        }

        return sample

    def _generate_labels(
        self,
        box_norm: np.ndarray,
        feat_size: int,
        pos_radius: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate classification labels and foreground mask.

        Tokens within pos_radius of the target center are labeled
        as foreground. Uses Gaussian-like weighting centered on the
        target center position.

        Args:
            box_norm: Normalized box [cx, cy, w, h] in [0, 1].
            feat_size: Feature map size (e.g. 24).
            pos_radius: Positive radius as fraction of feature map.

        Returns:
            Tuple of (cls_label [N], fg_mask [N]).
        """
        cx, cy = box_norm[0], box_norm[1]

        # Create grid of token positions
        N = feat_size * feat_size
        ys = torch.arange(feat_size).float() / feat_size + 0.5 / feat_size
        xs = torch.arange(feat_size).float() / feat_size + 0.5 / feat_size
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_x = grid_x.reshape(-1)  # [N]
        grid_y = grid_y.reshape(-1)

        # Distance from each token to target center
        dist = torch.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)

        # Foreground mask: tokens within radius
        fg_mask = (dist < pos_radius).float()

        # Classification label: Gaussian weighting
        sigma = pos_radius / 2
        cls_label = torch.exp(-dist ** 2 / (2 * sigma ** 2))

        # Ensure at least one positive
        if fg_mask.sum() == 0:
            nearest = dist.argmin()
            fg_mask[nearest] = 1.0
            cls_label[nearest] = 1.0

        return cls_label, fg_mask


def build_dataloader(
    cfg,
    split: str = "train",
    distributed: bool = False,
) -> DataLoader:
    """
    Build dataloader from config.

    Args:
        cfg: Configuration object.
        split: 'train' or 'test'.
        distributed: Whether to use DistributedSampler.

    Returns:
        Configured DataLoader.
    """
    dataset = UAVTrackingDataset(
        data_root=cfg.DATA.ROOT,
        dataset_name=cfg.DATA.DATASET,
        split=split,
        template_size=cfg.DATA.TEMPLATE_SIZE,
        search_size=cfg.DATA.SEARCH_SIZE,
        context_factor=cfg.DATA.CONTEXT_FACTOR,
        scale_aug=(split == "train"),
    )

    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=(split == "train"))

    loader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE if split == "train" else 1,
        shuffle=(split == "train" and sampler is None),
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=True,
        drop_last=(split == "train"),
        sampler=sampler,
    )

    return loader
