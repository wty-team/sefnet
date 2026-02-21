"""
SEFNet Training Script
========================

Implements the training procedure from Algorithm 1 and Section 5.3.

Training details:
    - Optimizer: AdamW with weight decay 1e-4
    - Learning rate: 4e-5 for backbone, 4e-4 for other modules
    - Schedule: Cosine annealing over 500 epochs (300 for GOT-10k)
    - Batch size: 32 (across 4 GPUs)
    - Warm-up: 5 epochs linear warm-up
    - Mixed precision: FP16 for efficiency

Loss weights (Section 5.3):
    λ_cls = 2.0, λ_reg = 5.0, λ_iou = 2.0, λ_eq = 1.0

References:
    - Algorithm 1: Training procedure
    - Section 5.3: Implementation details
    - Table D.7: Loss weight sensitivity
"""

import os
import sys
import time
import math
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import get_config
from lib.models.sefnet import build_sefnet
from lib.losses.total_loss import build_loss
from lib.datasets import build_dataloader


logger = logging.getLogger("sefnet.train")


def setup_logging(output_dir: str):
    """Configure logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "train.log")),
            logging.StreamHandler(),
        ],
    )


def build_optimizer(model: nn.Module, cfg) -> optim.Optimizer:
    """
    Build AdamW optimizer with differential learning rates.

    Backbone uses 10× lower LR than other modules (Sec 5.3):
        backbone: 4e-5, others: 4e-4

    Args:
        model: SEFNet model.
        cfg: Configuration.

    Returns:
        Configured optimizer.
    """
    backbone_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": cfg.TRAIN.LR * 0.1},
        {"params": other_params, "lr": cfg.TRAIN.LR},
    ]

    optimizer = optim.AdamW(
        param_groups,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
    )

    return optimizer


def build_scheduler(optimizer: optim.Optimizer, cfg) -> optim.lr_scheduler._LRScheduler:
    """
    Build cosine annealing LR scheduler with linear warm-up.

    Args:
        optimizer: Optimizer.
        cfg: Configuration.

    Returns:
        LR scheduler.
    """
    warmup_epochs = cfg.TRAIN.WARMUP_EPOCHS
    total_epochs = cfg.TRAIN.EPOCHS

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warm-up
            return epoch / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    dataloader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    cfg,
    writer: SummaryWriter = None,
):
    """
    Train for one epoch (Algorithm 1).

    Steps per iteration:
        1. Sample frame pair with scale augmentation
        2. Forward pass: backbone → EDM → GAB → Head
        3. Compute L = L_det + λ_eq · L_eq
        4. Backward + optimizer step

    Args:
        model: SEFNet model.
        criterion: TotalLoss.
        dataloader: Training dataloader.
        optimizer: Optimizer.
        scaler: AMP gradient scaler.
        epoch: Current epoch number.
        cfg: Configuration.
        writer: TensorBoard writer.
    """
    model.train()
    total_loss_avg = 0.0
    n_iter = 0
    t_start = time.time()

    for batch_idx, batch in enumerate(dataloader):
        template = batch["template"].cuda(non_blocking=True)
        search = batch["search"].cuda(non_blocking=True)
        search_aug = batch["search_aug"].cuda(non_blocking=True)
        cls_label = batch["cls_label"].cuda(non_blocking=True)
        search_box = batch["search_box_expanded"].cuda(non_blocking=True)
        fg_mask = batch["fg_mask"].cuda(non_blocking=True)

        optimizer.zero_grad()

        # ---- Forward pass with mixed precision ----
        with autocast():
            output = model.forward_train(
                template=template,
                search=search,
                search_aug=search_aug,
            )

            # Reshape predictions: [B, N_s, D] → match label shapes
            pred_cls = output["cls"].squeeze(-1)   # [B, N_s]
            pred_boxes = output["boxes"]            # [B, N_s, 4]

            # Compute total loss (Eq. 23)
            loss, loss_dict = criterion(
                pred_cls=pred_cls,
                pred_boxes=pred_boxes,
                target_cls=cls_label,
                target_boxes=search_box,
                features_original=output["features_original"],
                features_transformed=output["features_transformed"],
                fg_mask=fg_mask,
            )

        # ---- Backward pass with gradient scaling ----
        scaler.scale(loss).backward()

        # Gradient clipping (max norm 0.1, Sec 5.3)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

        scaler.step(optimizer)
        scaler.update()

        # ---- Logging ----
        total_loss_avg += loss.item()
        n_iter += 1
        global_step = epoch * len(dataloader) + batch_idx

        if batch_idx % cfg.TRAIN.LOG_INTERVAL == 0:
            lr = optimizer.param_groups[1]["lr"]
            elapsed = time.time() - t_start
            logger.info(
                f"Epoch [{epoch}][{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} "
                f"(cls={loss_dict['loss_cls']:.4f}, "
                f"reg={loss_dict['loss_reg']:.4f}, "
                f"iou={loss_dict['loss_iou']:.4f}, "
                f"eq={loss_dict['loss_eq_total']:.4f}) "
                f"LR: {lr:.6f} "
                f"Time: {elapsed:.1f}s"
            )

        if writer is not None:
            for k, v in loss_dict.items():
                writer.add_scalar(f"train/{k}", v.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[1]["lr"], global_step)

    avg_loss = total_loss_avg / max(n_iter, 1)
    logger.info(f"Epoch [{epoch}] Average Loss: {avg_loss:.4f}")
    return avg_loss


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    loss: float,
    output_dir: str,
    is_best: bool = False,
):
    """Save model checkpoint."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
    }
    ckpt_path = os.path.join(output_dir, f"checkpoint_ep{epoch:04d}.pth")
    torch.save(state, ckpt_path)

    if is_best:
        best_path = os.path.join(output_dir, "best_model.pth")
        torch.save(state, best_path)

    logger.info(f"Saved checkpoint: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="SEFNet Training")
    parser.add_argument("--config", type=str, default="experiments/sefnet_vit_base.yaml")
    parser.add_argument("--output_dir", type=str, default="output/train")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    # ---- Setup ----
    cfg = get_config(args.config)
    setup_logging(args.output_dir)
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {args.output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Build model, loss, optimizer ----
    model = build_sefnet(cfg).to(device)
    criterion = build_loss(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = GradScaler()

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params / 1e6:.2f}M")

    # ---- Resume from checkpoint ----
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    # ---- Build dataloader ----
    train_loader = build_dataloader(cfg, split="train")
    logger.info(f"Training samples: {len(train_loader.dataset)}")

    # ---- TensorBoard ----
    writer = SummaryWriter(os.path.join(args.output_dir, "tb_logs"))

    # ---- Training loop ----
    best_loss = float("inf")

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        avg_loss = train_one_epoch(
            model, criterion, train_loader, optimizer, scaler,
            epoch, cfg, writer,
        )
        scheduler.step()

        # Save checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        if (epoch + 1) % cfg.TRAIN.SAVE_INTERVAL == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch, avg_loss,
                args.output_dir, is_best=is_best,
            )

    writer.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
