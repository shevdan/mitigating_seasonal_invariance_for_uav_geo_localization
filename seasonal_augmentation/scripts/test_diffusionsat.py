#!/usr/bin/env python3
"""
Test DiffusionSat for UAV-height synthetic view generation.

Compares DiffusionSat (satellite-pretrained) vs vanilla SD 2.1 for generating
overhead imagery at different GSD values (satellite vs UAV altitude).

Setup:
    1. Clone DiffusionSat:
       git clone https://github.com/samar-khanna/DiffusionSat.git /tmp/DiffusionSat

    2. Download checkpoint (512x512 single-image model):
       wget -O /tmp/diffusionsat_512.zip \
         https://zenodo.org/records/13751498/files/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64.zip
       unzip /tmp/diffusionsat_512.zip -d /tmp/diffusionsat_ckpt/

    3. Run:
       conda run -n seasonalgeo python test_diffusionsat.py \
         --ckpt /tmp/diffusionsat_ckpt/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64

Usage:
    python test_diffusionsat.py --ckpt <path_to_diffusionsat_checkpoint>
    python test_diffusionsat.py --ckpt <path> --image <uav_image.jpg>  # side-by-side comparison
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# ── DiffusionSat imports ──
# We need DiffusionSat repo on the path
DIFFUSIONSAT_REPO = None  # Set via --repo arg or auto-detect


def setup_diffusionsat_path(repo_path: str = None):
    """Add DiffusionSat repo to sys.path."""
    candidates = [
        repo_path,
        str(Path(__file__).parent.parent.parent / "DiffusionSat"),
        "/tmp/DiffusionSat",
    ]
    for p in candidates:
        if p and Path(p).exists() and (Path(p) / "diffusionsat").exists():
            if p not in sys.path:
                sys.path.insert(0, p)
            print(f"Using DiffusionSat from: {p}")
            return p
    raise FileNotFoundError(
        "DiffusionSat repo not found. Clone it first:\n"
        "  git clone https://github.com/samar-khanna/DiffusionSat.git /tmp/DiffusionSat\n"
        "Then install: cd /tmp/DiffusionSat && pip install -e '.[torch]'"
    )


def metadata_normalize(metadata, base_lon=180, base_lat=90, base_year=1980, max_gsd=1.0, scale=1000):
    """Normalize metadata to DiffusionSat expected format.

    Args:
        metadata: [longitude, latitude, gsd_meters, cloud_cover, year, month, day]
    """
    lon, lat, gsd, cloud_cover, year, month, day = metadata
    lon = lon / (180 + base_lon) * scale
    lat = lat / (90 + base_lat) * scale
    gsd = gsd / max_gsd * scale
    cloud_cover = cloud_cover * scale
    year = year / (2100 - base_year) * scale
    month = month / 12 * scale
    day = day / 31 * scale
    return torch.tensor([lon, lat, gsd, cloud_cover, year, month, day])


# ── Test locations (from our datasets) ──
TEST_LOCATIONS = {
    "hangzhou": {
        "description": "DenseUAV - Hangzhou, China",
        "lon": 120.38,
        "lat": 30.32,
        "caption": "aerial photograph of urban area with buildings and roads in Hangzhou China",
    },
    "hong_kong": {
        "description": "UAV-VisLoc - Hong Kong",
        "lon": 114.17,
        "lat": 22.32,
        "caption": "aerial photograph of urban area with buildings and vegetation in Hong Kong",
    },
    "generic_urban": {
        "description": "Generic urban scene",
        "lon": -73.98,
        "lat": 40.75,
        "caption": "aerial photograph of dense urban area with buildings and streets",
    },
}

# GSD values to test (meters per pixel)
GSD_LEVELS = {
    "satellite_10m": 10.0,       # Sentinel-2 level
    "satellite_3m": 3.0,         # Planet Labs level
    "satellite_1m": 1.0,         # High-res satellite
    "satellite_0.5m": 0.5,       # Very high-res satellite
    "uav_0.1m": 0.1,             # High-altitude UAV (~300m)
    "uav_0.05m": 0.05,           # Medium UAV (~150m)
    "uav_0.02m": 0.02,           # Low UAV (~60m)
}

# Seasonal prompts for comparison with our current pipeline
SEASONAL_PROMPTS = {
    "summer": "in summer with lush green foliage and bright sunlight",
    "winter": "in winter with snow covered ground and bare trees",
    "autumn": "in autumn with orange and red fall foliage",
}


def generate_diffusionsat_samples(
    ckpt_path: str,
    output_dir: Path,
    location: str = "hangzhou",
    seasons: list = None,
    gsd_levels: dict = None,
    num_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 42,
):
    """Generate images with DiffusionSat at various GSD levels.

    This tests whether GSD conditioning produces visually different results
    at satellite vs UAV altitudes.
    """
    from diffusionsat import SatUNet, DiffusionSatPipeline

    if seasons is None:
        seasons = ["summer"]
    if gsd_levels is None:
        gsd_levels = GSD_LEVELS

    loc = TEST_LOCATIONS[location]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load DiffusionSat model
    print(f"\nLoading DiffusionSat from: {ckpt_path}")
    ckpt_path = Path(ckpt_path)

    # Find the checkpoint subfolder (e.g., checkpoint-150000)
    checkpoint_dirs = sorted(ckpt_path.glob("checkpoint-*"))
    if checkpoint_dirs:
        unet_path = str(checkpoint_dirs[-1])  # Use latest checkpoint
        print(f"  Using UNet checkpoint: {unet_path}")
    else:
        unet_path = str(ckpt_path)
        print(f"  Using UNet from: {unet_path}")

    unet = SatUNet.from_pretrained(
        unet_path,
        subfolder="unet",
        torch_dtype=torch.float16,
    )
    pipe = DiffusionSatPipeline.from_pretrained(
        str(ckpt_path),
        unet=unet,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")

    print(f"\nGenerating for: {loc['description']}")
    print(f"  Location: ({loc['lat']}, {loc['lon']})")
    print(f"  GSD levels: {list(gsd_levels.keys())}")
    print(f"  Seasons: {seasons}")

    results = {}

    for season in seasons:
        season_prompt = SEASONAL_PROMPTS.get(season, "")
        for gsd_name, gsd_val in gsd_levels.items():
            caption = f"{loc['caption']} {season_prompt}"

            # Normalize metadata: [lon, lat, gsd, cloud_cover, year, month, day]
            # Pick a date matching the season
            month_map = {"summer": 7, "winter": 1, "autumn": 10}
            month = month_map.get(season, 7)

            raw_metadata = [loc["lon"], loc["lat"], gsd_val, 0.0, 2023, month, 15]
            metadata = metadata_normalize(raw_metadata).tolist()

            generator = torch.Generator(device="cuda").manual_seed(seed)

            print(f"  Generating: {season} @ {gsd_name} (GSD={gsd_val}m)...")
            image = pipe(
                caption,
                metadata=metadata,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512,
                generator=generator,
            ).images[0]

            fname = f"diffsat_{location}_{season}_{gsd_name}.png"
            image.save(output_dir / fname)
            results[(season, gsd_name)] = image
            print(f"    Saved: {fname}")

    return results


def generate_sd21_baseline(
    output_dir: Path,
    location: str = "hangzhou",
    seasons: list = None,
    num_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 42,
):
    """Generate baseline images with vanilla SD 2.1 (no satellite pretraining).

    Uses DiffusionSat's SatUNet with use_metadata=False to get a vanilla SD 2.1
    baseline without needing to download SD 2.1 separately (it's gated on HF).
    """
    from diffusionsat import SatUNet, DiffusionSatPipeline

    if seasons is None:
        seasons = ["summer"]

    loc = TEST_LOCATIONS[location]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading SD 2.1 baseline (SatUNet with use_metadata=False)...")
    # Load SatUNet without metadata — effectively vanilla SD 2.1 UNet
    unet = SatUNet.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        subfolder="unet",
        use_metadata=False,
        torch_dtype=torch.float16,
    )
    pipe = DiffusionSatPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        unet=unet,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")

    results = {}
    for season in seasons:
        season_prompt = SEASONAL_PROMPTS.get(season, "")
        caption = f"{loc['caption']} {season_prompt}, satellite view, high resolution"

        generator = torch.Generator(device="cuda").manual_seed(seed)

        print(f"  Generating SD2.1 baseline: {season}...")
        image = pipe(
            caption,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=512,
            width=512,
            generator=generator,
        ).images[0]

        fname = f"sd21_baseline_{location}_{season}.png"
        image.save(output_dir / fname)
        results[season] = image
        print(f"    Saved: {fname}")

    return results


def create_comparison_grid(
    diffsat_results: dict,
    baseline_results: dict,
    output_path: Path,
    location: str = "hangzhou",
    real_uav_image: Image.Image = None,
):
    """Create a visual comparison grid."""
    import matplotlib.pyplot as plt

    seasons = sorted(set(s for s, _ in diffsat_results.keys()))
    gsd_names = sorted(set(g for _, g in diffsat_results.keys()),
                       key=lambda x: GSD_LEVELS.get(x, 0), reverse=True)

    n_cols = len(gsd_names) + 1  # +1 for baseline
    if real_uav_image is not None:
        n_cols += 1
    n_rows = len(seasons)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    loc = TEST_LOCATIONS[location]

    for row, season in enumerate(seasons):
        col = 0

        # Real UAV image (if provided)
        if real_uav_image is not None:
            axes[row, col].imshow(real_uav_image)
            axes[row, col].set_title("Real UAV", fontsize=10)
            axes[row, col].axis("off")
            col += 1

        # SD 2.1 baseline
        if season in baseline_results:
            axes[row, col].imshow(baseline_results[season])
            axes[row, col].set_title(f"SD 2.1 baseline\n({season})", fontsize=10)
        axes[row, col].axis("off")
        col += 1

        # DiffusionSat at various GSD levels
        for gsd_name in gsd_names:
            key = (season, gsd_name)
            if key in diffsat_results:
                axes[row, col].imshow(diffsat_results[key])
                gsd_val = GSD_LEVELS[gsd_name]
                axes[row, col].set_title(
                    f"DiffSat {gsd_name}\n(GSD={gsd_val}m)", fontsize=9
                )
            axes[row, col].axis("off")
            col += 1

    plt.suptitle(
        f"DiffusionSat vs SD 2.1 — {loc['description']}\n"
        f"Does GSD conditioning produce UAV-like detail?",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nComparison grid saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test DiffusionSat for UAV-height synthetic view generation"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to DiffusionSat checkpoint directory",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Path to DiffusionSat repo (auto-detected if not specified)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/diffusionsat_test",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--location",
        type=str,
        default="hangzhou",
        choices=list(TEST_LOCATIONS.keys()),
        help="Test location",
    )
    parser.add_argument(
        "--seasons",
        type=str,
        nargs="+",
        default=["summer"],
        choices=list(SEASONAL_PROMPTS.keys()),
        help="Seasons to test",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional real UAV image for side-by-side comparison",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip SD 2.1 baseline generation",
    )
    parser.add_argument(
        "--gsd-only",
        type=str,
        nargs="+",
        default=None,
        help="Only test specific GSD levels (e.g., satellite_1m uav_0.05m)",
    )

    args = parser.parse_args()

    # Setup DiffusionSat
    setup_diffusionsat_path(args.repo)

    output_dir = Path(args.output)

    # Select GSD levels
    if args.gsd_only:
        gsd_levels = {k: GSD_LEVELS[k] for k in args.gsd_only if k in GSD_LEVELS}
    else:
        # Default: test a few representative levels
        gsd_levels = {
            "satellite_3m": GSD_LEVELS["satellite_3m"],
            "satellite_0.5m": GSD_LEVELS["satellite_0.5m"],
            "uav_0.05m": GSD_LEVELS["uav_0.05m"],
        }

    # Generate DiffusionSat samples
    diffsat_results = generate_diffusionsat_samples(
        ckpt_path=args.ckpt,
        output_dir=output_dir / "diffusionsat",
        location=args.location,
        seasons=args.seasons,
        gsd_levels=gsd_levels,
        num_steps=args.steps,
        seed=args.seed,
    )

    # Generate SD 2.1 baseline
    baseline_results = {}
    if not args.skip_baseline:
        baseline_results = generate_sd21_baseline(
            output_dir=output_dir / "sd21_baseline",
            location=args.location,
            seasons=args.seasons,
            num_steps=args.steps,
            seed=args.seed,
        )

    # Load real UAV image if provided
    real_uav = None
    if args.image:
        real_uav = Image.open(args.image).convert("RGB").resize((512, 512))

    # Create comparison grid
    create_comparison_grid(
        diffsat_results,
        baseline_results,
        output_dir / "comparison_grid.png",
        location=args.location,
        real_uav_image=real_uav,
    )

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"  - DiffusionSat images: {output_dir / 'diffusionsat'}/")
    if not args.skip_baseline:
        print(f"  - SD 2.1 baseline: {output_dir / 'sd21_baseline'}/")
    print(f"  - Comparison grid: {output_dir / 'comparison_grid.png'}")
    print()
    print("Key things to evaluate:")
    print("  1. Does lower GSD (UAV-like) produce more detailed/realistic overhead views?")
    print("  2. Does DiffusionSat produce more realistic aerial imagery than SD 2.1?")
    print("  3. Are seasonal variations more natural for overhead imagery?")
    print("  4. How does it compare to a real UAV image (if --image provided)?")


if __name__ == "__main__":
    main()
