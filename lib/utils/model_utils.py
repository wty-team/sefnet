"""
Model Utilities
=================

Helper functions for model management, analysis, and benchmarking.

Includes:
    - Checkpoint loading with key matching
    - Parameter counting and module analysis
    - FPS benchmarking (Table 4: 147 FPS)
    - Overhead analysis (Appendix C, Eq. C.1: 0.34%)
"""

import os
import time
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from collections import OrderedDict


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = True,
    map_location: str = "cpu",
) -> Dict:
    """
    Load model checkpoint with key matching.

    Handles common mismatches:
        - 'module.' prefix from DataParallel
        - Missing/unexpected keys with warning

    Args:
        model: Target model.
        checkpoint_path: Path to .pth file.
        strict: Whether to enforce exact key matching.
        map_location: Device for loading.

    Returns:
        Checkpoint dict (for extracting epoch, optimizer, etc.).
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location)

    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    # Remove 'module.' prefix if present (from DataParallel)
    new_state = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state[name] = v

    missing, unexpected = model.load_state_dict(new_state, strict=strict)
    if missing:
        print(f"[WARN] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    return ckpt


def count_parameters(model: nn.Module, verbose: bool = False) -> Dict[str, int]:
    """
    Count model parameters by module.

    Args:
        model: Model to analyze.
        verbose: Print per-module breakdown.

    Returns:
        Dict with parameter counts.
    """
    total = 0
    trainable = 0
    per_module = {}

    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
        per_module[name] = {"total": n_params, "trainable": n_train}
        total += n_params
        trainable += n_train

    if verbose:
        print(f"\n{'Module':<20} {'Total':>12} {'Trainable':>12} {'%':>8}")
        print("-" * 54)
        for name, counts in per_module.items():
            pct = 100.0 * counts["total"] / max(total, 1)
            print(f"{name:<20} {counts['total']:>12,} {counts['trainable']:>12,} {pct:>7.1f}%")
        print("-" * 54)
        print(f"{'TOTAL':<20} {total:>12,} {trainable:>12,}")

    return {"total": total, "trainable": trainable, "per_module": per_module}


def benchmark_fps(
    model: nn.Module,
    template_size: Tuple[int, int] = (192, 192),
    search_size: Tuple[int, int] = (384, 384),
    n_warmup: int = 50,
    n_runs: int = 200,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Benchmark model inference speed.

    Expected: ~147 FPS on RTX 3090 (Table 4).

    Measures:
        - Template initialization time (one-time)
        - Per-frame tracking time
        - Overall FPS

    Args:
        model: SEFNet model in eval mode.
        template_size: Template dimensions.
        search_size: Search region dimensions.
        n_warmup: Warmup iterations.
        n_runs: Benchmark iterations.
        device: Target device.

    Returns:
        Dict with timing results.
    """
    model.eval()
    model = model.to(device)

    template = torch.randn(1, 3, *template_size, device=device)
    search = torch.randn(1, 3, *search_size, device=device)

    # Warmup
    with torch.no_grad():
        model.initialize_template(template)
        for _ in range(n_warmup):
            model.forward_inference(search)

    # Benchmark template initialization
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        model.initialize_template(template)
    torch.cuda.synchronize()
    init_time = (time.time() - t0) / n_runs

    # Benchmark per-frame tracking
    model.initialize_template(template)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        model.forward_inference(search)
    torch.cuda.synchronize()
    track_time = (time.time() - t0) / n_runs

    fps = 1.0 / track_time

    results = {
        "init_time_ms": init_time * 1000,
        "track_time_ms": track_time * 1000,
        "fps": fps,
    }

    print(f"\nFPS Benchmark ({device}):")
    print(f"  Template init: {results['init_time_ms']:.2f} ms")
    print(f"  Per-frame:     {results['track_time_ms']:.2f} ms")
    print(f"  FPS:           {results['fps']:.1f}")

    return results


def compute_overhead(
    model: nn.Module,
    template_size: Tuple[int, int] = (192, 192),
    search_size: Tuple[int, int] = (384, 384),
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute computational overhead vs backbone (Appendix C, Eq. C.1).

    Expected: 0.34% overhead beyond ViT-Base backbone.

    C_overhead = (C_total - C_backbone) / C_backbone

    Args:
        model: SEFNet model.
        template_size: Template dimensions.
        search_size: Search dimensions.
        device: Target device.

    Returns:
        Dict with FLOP estimates and overhead percentage.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        print("[WARN] fvcore not installed, skipping FLOP analysis")
        return {}

    model.eval().to(device)
    template = torch.randn(1, 3, *template_size, device=device)
    search = torch.randn(1, 3, *search_size, device=device)

    # Total model FLOPs
    flops_total = FlopCountAnalysis(model, (template, search))
    total = flops_total.total()

    # Backbone-only FLOPs
    flops_backbone = FlopCountAnalysis(model.backbone, (search,))
    backbone = flops_backbone.total()

    overhead = (total - backbone) / backbone * 100

    results = {
        "total_gflops": total / 1e9,
        "backbone_gflops": backbone / 1e9,
        "overhead_gflops": (total - backbone) / 1e9,
        "overhead_pct": overhead,
    }

    print(f"\nOverhead Analysis (Eq. C.1):")
    print(f"  Total:    {results['total_gflops']:.2f} GFLOPs")
    print(f"  Backbone: {results['backbone_gflops']:.2f} GFLOPs")
    print(f"  Overhead: {results['overhead_pct']:.2f}%")

    return results
