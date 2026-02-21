# SEFNet: Selective Equivariant Features for Robust Scale-Adaptive UAV Tracking

[![Paper](https://img.shields.io/badge/Paper-IEEE-blue)](https://ieeexplore.ieee.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Official implementation of **SEFNet**, a selective equivariant feature network that addresses heterogeneous transformation behavior of visual features under scale changes for UAV tracking.

## Highlights

- **72.4% AUC** on UAV123 &nbsp;|&nbsp; **54.2% AUC** on ARD-MAV &nbsp;|&nbsp; **147 FPS** real-time
- Monotonically increasing margins over SOTA as scale difficulty intensifies: **+0.9%** → **+1.8%** → **+3.1%**
- Only **0.34%** computational overhead beyond the ViT backbone

## Core Idea

Scale variation is not merely one challenge among many — it is a **root cause** that triggers or amplifies occlusion, motion blur, and low resolution. Existing methods either learn scale-handling implicitly from data or enforce uniform equivariant constraints across all features.

**Our key observation:** visual features exhibit *heterogeneous* transformation behaviors:

| Subspace | Behavior | Basis Functions |
|----------|----------|-----------------|
| **Equivariant** (𝓕_eq) | Edge features follow predictable geometric patterns | Small-σ Gaussian derivatives |
| **Invariant** (𝓕_inv) | Texture features maintain semantic stability | Large-σ Gaussian derivatives |
| **Coupled** (𝓕_cp) | Contextual features display scale-rotation coupling | Circular harmonics |

Through **Pontryagin duality analysis** of the similarity transformation group, we prove this three-way decomposition is *mathematically unique* (Theorem 1), not an arbitrary design choice.

## Architecture

<p align="center"><b>ViT Backbone → EDM (Equivariant Decomposition Module) → GAB (Geometry-Aware Bridging Block) → Tracking Head</b></p>

- **EDM** decomposes each backbone layer into three orthogonal subspaces via group-theoretic parameterized convolution with type-specific basis functions.
- **GAB** performs confidence-based layer selection and adaptive fusion, dynamically favoring shallow layers for large targets and deeper layers for small targets.

## Main Results

### UAV Benchmarks (ViT-B backbone)

| Method | UAV123 AUC | VisDrone AUC | ARD-MAV AUC | FPS |
|--------|:----------:|:------------:|:-----------:|:---:|
| SPMTrack | 71.5 | 61.7 | 51.1 | 155 |
| ARTrackV2 | 70.5 | 59.2 | 49.2 | 180 |
| ODTrack | 70.8 | 59.8 | 49.8 | 95 |
| DreamTrack | 71.2 | 61.2 | 50.5 | 18 |
| **SEFNet (Ours)** | **72.4** | **63.5** | **54.2** | **147** |

### Scale-Specific Metrics

| Method | SCR (ARD-MAV) | STR (ARD-MAV) |
|--------|:-------------:|:-------------:|
| SPMTrack | [0.25, 3.8] (15×) | 0.25 |
| **SEFNet** | **[0.18, 5.2] (29×)** | **0.35** |

### Transformation Consistency

| Type | E_overall | E_eq | E_inv | E_cp |
|------|:---------:|:----:|:-----:|:----:|
| Implicit (best) | 0.285 | — | — | — |
| Uniform Equiv. | 0.148 | — | — | — |
| **SEFNet (Selective)** | **0.137** | **0.089** | **0.124** | **0.198** |


## Data Preparation

Download and organize datasets:

```
data/
├── UAV123/
├── LaSOT/
├── VisDrone-SOT/
└── ARD-MAV/
```


## Ablation Summary

| Configuration | LaSOT AUC | UAV123 AUC | GFLOPs | FPS |
|--------------|:---------:|:----------:|:------:|:---:|
| Baseline (ViT) | 71.0 | 68.2 | 52 | 165 |
| +EDM | 73.1 | 70.1 | 55 | 155 |
| +GAB | 72.4 | 69.3 | 54 | 158 |
| **+EDM+GAB (Full)** | **75.2** | **72.4** | **58** | **147** |



