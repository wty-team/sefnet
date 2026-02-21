"""
SEFNet Demo Script
====================

Quick demo for running SEFNet on a video or image sequence.

Usage:
    python tools/demo.py --config experiments/sefnet_vit_base.yaml \
                         --checkpoint output/best_model.pth \
                         --video input.mp4 \
                         --output output/demo/

    python tools/demo.py --config experiments/sefnet_vit_base.yaml \
                         --checkpoint output/best_model.pth \
                         --image_dir data/sequence/img/ \
                         --init_box 100,50,80,120 \
                         --output output/demo/
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import get_config
from lib.models.sefnet import build_sefnet
from lib.utils import load_checkpoint


def load_video_frames(video_path: str):
    """Load frames from video file using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_image_sequence(image_dir: str):
    """Load frames from directory of images."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted([
        f for f in os.listdir(image_dir)
        if Path(f).suffix.lower() in exts
    ])
    frames = [np.array(Image.open(os.path.join(image_dir, f)).convert("RGB")) for f in files]
    return frames


def select_roi(frame: np.ndarray):
    """Let user select ROI via OpenCV."""
    import cv2
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    roi = cv2.selectROI("Select Target", bgr, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    return np.array(roi, dtype=np.float64)  # [x, y, w, h]


def save_results(
    frames: list,
    pred_boxes: np.ndarray,
    output_dir: str,
):
    """Save tracking results as annotated frames."""
    os.makedirs(output_dir, exist_ok=True)

    try:
        from lib.utils.visualization import plot_tracking_result
        for i, (frame, box) in enumerate(zip(frames, pred_boxes)):
            save_path = os.path.join(output_dir, f"frame_{i:06d}.jpg")
            plot_tracking_result(frame, box, box, frame_idx=i, save_path=save_path)
    except Exception:
        # Fallback: save with PIL
        from PIL import ImageDraw
        for i, (frame, box) in enumerate(zip(frames, pred_boxes)):
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            x, y, w, h = box
            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
            img.save(os.path.join(output_dir, f"frame_{i:06d}.jpg"))

    # Save boxes
    np.savetxt(
        os.path.join(output_dir, "predictions.txt"),
        pred_boxes, delimiter=",", fmt="%.2f"
    )
    print(f"Results saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="SEFNet Demo")
    parser.add_argument("--config", type=str, default="experiments/sefnet_vit_base.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--video", type=str, default=None, help="Input video path")
    parser.add_argument("--image_dir", type=str, default=None, help="Image sequence directory")
    parser.add_argument("--init_box", type=str, default=None,
                        help="Initial box as x,y,w,h (if not provided, select via GUI)")
    parser.add_argument("--output", type=str, default="output/demo")
    args = parser.parse_args()

    # Load config and model
    cfg = get_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_sefnet(cfg).to(device)
    load_checkpoint(model, args.checkpoint, strict=False)
    model.eval()
    print(f"Model loaded from {args.checkpoint}")

    # Load frames
    if args.video:
        frames = load_video_frames(args.video)
    elif args.image_dir:
        frames = load_image_sequence(args.image_dir)
    else:
        print("Error: Provide --video or --image_dir")
        return

    print(f"Loaded {len(frames)} frames")

    # Get initial box
    if args.init_box:
        init_box = np.array([float(x) for x in args.init_box.split(",")], dtype=np.float64)
    else:
        init_box = select_roi(frames[0])

    print(f"Initial box: {init_box}")

    # Import tracker
    from tools.eval import Tracker
    tracker = Tracker(model, search_size=cfg.DATA.SEARCH_SIZE, template_size=cfg.DATA.TEMPLATE_SIZE)

    # Run tracking
    pred_boxes = np.zeros((len(frames), 4))
    tracker.initialize(frames[0], init_box)
    pred_boxes[0] = init_box

    for i in range(1, len(frames)):
        pred_boxes[i] = tracker.track(frames[i])
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(frames)} frames")

    # Save results
    save_results(frames, pred_boxes, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
