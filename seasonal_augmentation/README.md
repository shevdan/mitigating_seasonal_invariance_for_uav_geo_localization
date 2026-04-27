# Seasonal Augmentation Pipeline

Generate synthetic seasonal variations of UAV images for training season-invariant geo-localization models.

## Goal

Given UAV images captured in summer, generate realistic versions in other seasons (autumn, winter, spring) while preserving scene geometry and structure. A quality gate (DINOv2 + Edge F1) rejects hallucinated outputs automatically.

## Pipeline

```
Original UAV image (summer)
    ├── MiDaS depth estimation → depth map
    ├── Canny edge detection → edge map
    └── (img2img mode: original as starting point)
            ↓
    Multi-ControlNet (depth + canny) + Stable Diffusion 1.5
            ↓
    Quality Gate: DINOv2 similarity + Edge F1 + SSIM-edge
            ↓
    Accept → save  /  Reject → retry (up to 3x) or skip
```

## Quick Start

```bash
# From project root (diploma/)

# 1. Depth estimation (one-time, ~75min for DenseUAV)
conda run -n diploma_controlnet python seasonal_augmentation/scripts/generate_for_dataset.py \
    --dataset denseuav --transformation summer_to_autumn \
    --method multicontrolnet --skip-generation

# 2. Generate with quality gate
conda run -n diploma_controlnet python seasonal_augmentation/scripts/generate_for_dataset.py \
    --dataset denseuav --transformation summer_to_autumn \
    --method multicontrolnet --skip-depth \
    --quality-gate --qg-dino-cls 0.75 --qg-dino-patch 0.72 --qg-edge-f1 0.74
```

## Generation Methods

### Multi-ControlNet (recommended)
Dual conditioning (depth + canny edges). Two modes:

- **img2img** (autumn, spring): starts from original, `img2img_strength` 0.3-0.45. Better structure preservation.
- **txt2img** (winter): generates from noise. More dramatic but higher hallucination risk.

### InstructPix2Pix (alternative)
No depth estimation needed. Uses natural language instructions. Produces subtler changes.

## Quality Gate

Integrated into generation — rejects hallucinated images before saving.

| Metric | What it measures | Color-invariant? |
|--------|-----------------|-----------------|
| DINOv2 CLS cosine sim | Global structure | Yes |
| DINOv2 patch cosine sim | Local structure | Yes |
| Edge F1 (Canny, 2px tolerance) | Boundary preservation | Yes |
| SSIM on edge maps | Edge structural similarity | Yes |

**Sparse edges**: When original has <1% edge pixels (water/river), edge_f1 is skipped.

### Calibrated Thresholds

| Mode | dino_cls | dino_patch | edge_f1 |
|------|----------|------------|---------|
| img2img (autumn/spring) | 0.75 | 0.72 | 0.74 |
| txt2img (winter) | 0.35 | 0.43 | 0.35 |

## Season Configs

| Config | Mode | Strength | Notes |
|--------|------|----------|-------|
| `summer_to_autumn.yaml` | img2img | 0.3 | Subtle warm color shift |
| `summer_to_winter.yaml` | txt2img | — | Snow/frost, controlnet_scale=0.95 |
| `summer_to_spring.yaml` | img2img | 0.45 | Fresh greens, blooming |

Only forward transforms from summer source images.

## Evaluation

```bash
conda run -n diploma_controlnet python seasonal_augmentation/scripts/evaluate_consistency.py \
    --original-dir data/DenseUAV/train/drone \
    --generated-dir seasonal_augmentation/outputs/denseuav/generated_summer_to_autumn/train/drone \
    --transformation summer_to_autumn \
    --output output/eval_autumn \
    --save-heatmaps
```

## Folder Structure

```
seasonal_augmentation/
├── configs/
│   ├── datasets.yaml               # Local dataset paths
│   ├── datasets_vastai.yaml         # Vast.ai absolute paths
│   ├── summer_to_autumn.yaml        # Season transform configs
│   ├── summer_to_winter.yaml
│   └── summer_to_spring.yaml
├── scripts/
│   ├── generate_for_dataset.py      # Main entry point
│   ├── generate_multicontrolnet.py  # Multi-ControlNet (depth + canny)
│   ├── generate_controlnet.py       # Single ControlNet (depth)
│   ├── generate_instructpix2pix.py  # InstructPix2Pix
│   ├── estimate_depth.py            # MiDaS/DPT depth estimation
│   ├── consistency.py               # Quality gate metrics
│   ├── evaluate_consistency.py      # Batch evaluation
│   ├── vastai_setup.sh              # Remote instance setup
│   └── pack_for_vastai.sh           # Create deployment archive
└── outputs/
    ├── denseuav/
    │   ├── depth_maps/
    │   ├── generated_summer_to_autumn/
    │   ├── generated_summer_to_winter/
    │   └── generated_summer_to_spring/
    └── uavvisloc/
        ├── depth_maps/
        └── generated_summer_to_*/

## Vast.ai Deployment

See `scripts/pack_for_vastai.sh` and `scripts/vastai_setup.sh`. One season per GPU instance for parallelism. Use `--dataset-config configs/datasets_vastai.yaml` on remote.

## Hardware

| GPU | ~Time/image | DenseUAV full (9099 imgs) |
|-----|-------------|--------------------------|
| RTX 4070 Ti (12GB) | ~35-40s | ~70h |
| RTX 4090 (24GB) | ~12-15s | ~25h |
| A100 (40/80GB) | ~6-8s | ~12h |
```
