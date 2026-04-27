#!/usr/bin/env python3
"""
Generate seasonal variations for any supported dataset.

Supports: UAV-VisLoc, DenseUAV, SUES-200, University-1652

Loads the diffusion model ONCE and processes all splits (train/test),
preserving the directory structure. Optionally runs a quality gate
(DINOv2 + Edge F1 + SSIM-edge) to reject hallucinated outputs.

Usage:
    # Basic
    python generate_for_dataset.py --dataset denseuav --transformation summer_to_winter

    # With quality gate
    python generate_for_dataset.py --dataset denseuav --transformation summer_to_winter \
        --method multicontrolnet --skip-depth --quality-gate

    # Custom thresholds
    python generate_for_dataset.py --dataset denseuav --transformation summer_to_winter \
        --method multicontrolnet --skip-depth --quality-gate \
        --qg-dino-cls 0.6 --qg-max-retries 5
"""

import argparse
import subprocess
from pathlib import Path

import yaml

from consistency import add_quality_gate_args, create_quality_gate_from_args


def load_dataset_config(config_path: str, dataset_name: str) -> dict:
    """Load dataset configuration from YAML."""
    with open(config_path) as f:
        configs = yaml.safe_load(f)

    if dataset_name not in configs:
        available = list(configs.keys())
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")

    return configs[dataset_name]


def find_all_images(base_path: Path, drone_paths: list, extensions: list) -> list:
    """Find all drone images in the dataset."""
    images = []
    base_path = Path(base_path)

    for drone_path in drone_paths:
        full_path = base_path / drone_path
        if not full_path.exists():
            print(f"Warning: Path does not exist: {full_path}")
            continue

        for ext in extensions:
            images.extend(full_path.rglob(f"*{ext}"))

    return sorted(images)


def run_depth_estimation(input_paths: list, rel_paths: list, depth_dir: Path, model: str = "DPT_Large"):
    """Run depth estimation on input images."""
    print("\n" + "=" * 60)
    print("Step 1: Depth Estimation")
    print("=" * 60)

    script_path = Path(__file__).parent / "estimate_depth.py"

    for input_path, rel_path in zip(input_paths, rel_paths):
        if not input_path.exists():
            continue

        depth_output = depth_dir / rel_path

        print(f"\nProcessing: {input_path} -> {depth_output}")

        cmd = [
            "python",
            str(script_path),
            "--input",
            str(input_path),
            "--output",
            str(depth_output),
            "--model",
            model,
        ]

        subprocess.run(cmd, check=True)


# ─── Direct generation (loads model once, supports quality gate) ──────────────


def run_multicontrolnet_direct(
    input_paths: list,
    rel_paths: list,
    depth_dir: Path,
    output_dir: Path,
    transformation: str,
    depth_scale: float = 0.8,
    canny_scale: float = 0.5,
    guidance_scale: float = 7.5,
    use_img2img: bool = False,
    img2img_strength: float = 0.4,
    config_path: str = None,
    quality_gate=None,
    max_retries: int = 3,
):
    """Run Multi-ControlNet generation directly (single model load)."""
    from generate_multicontrolnet import load_pipeline, process_directory

    # Load YAML config first, then overlay CLI args
    config = {}
    if config_path:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    # CLI args override YAML values
    config.setdefault("depth_scale", depth_scale)
    config.setdefault("canny_scale", canny_scale)
    config.setdefault("guidance_scale", guidance_scale)
    # img2img: CLI flag wins if set, otherwise fall back to YAML
    if use_img2img:
        config["use_img2img"] = True
    config.setdefault("use_img2img", False)
    if img2img_strength != 0.4:  # CLI explicitly set
        config["img2img_strength"] = img2img_strength
    config.setdefault("img2img_strength", 0.4)

    use_img2img = config["use_img2img"]
    mode = "img2img" if use_img2img else "txt2img"
    print("\n" + "=" * 60)
    print(f"Generate {transformation} (Multi-ControlNet: depth + canny, {mode})")
    if use_img2img:
        print(f"  img2img_strength: {config['img2img_strength']}")
    print("=" * 60)

    pipe = None

    for input_path, rel_path in zip(input_paths, rel_paths):
        if not input_path.exists():
            continue

        depth_path = depth_dir / rel_path
        gen_output = output_dir / rel_path

        if not depth_path.exists():
            print(f"Warning: No depth maps for {input_path}")
            continue

        print(f"\nProcessing: {input_path} -> {gen_output}")

        # Load pipeline once on first iteration
        if pipe is None:
            pipe = load_pipeline(use_img2img=use_img2img)

        process_directory(
            input_dir=input_path,
            depth_dir=depth_path,
            output_dir=gen_output,
            transformation=transformation,
            config=config,
            quality_gate=quality_gate,
            max_retries=max_retries,
            _pipe=pipe,
        )


def run_controlnet_direct(
    input_paths: list,
    rel_paths: list,
    depth_dir: Path,
    output_dir: Path,
    transformation: str,
    config_path: str = None,
    use_img2img: bool = False,
    img2img_strength: float = 0.4,
    quality_gate=None,
    max_retries: int = 3,
):
    """Run ControlNet generation directly (single model load)."""
    from generate_controlnet import load_pipeline, process_directory

    print("\n" + "=" * 60)
    print(f"Generate {transformation} (ControlNet)")
    print("=" * 60)

    config = None
    if config_path:
        with open(config_path) as f:
            config = yaml.safe_load(f)

    pipe = None

    for input_path, rel_path in zip(input_paths, rel_paths):
        if not input_path.exists():
            continue

        depth_path = depth_dir / rel_path
        gen_output = output_dir / rel_path

        if not depth_path.exists():
            print(f"Warning: No depth maps for {input_path}")
            continue

        print(f"\nProcessing: {input_path} -> {gen_output}")

        if pipe is None:
            pipe = load_pipeline(use_img2img=use_img2img)

        process_directory(
            input_dir=input_path,
            depth_dir=depth_path,
            output_dir=gen_output,
            transformation=transformation,
            config=config,
            use_img2img=use_img2img,
            quality_gate=quality_gate,
            max_retries=max_retries,
            _pipe=pipe,
        )


def run_instructpix2pix_direct(
    input_paths: list,
    rel_paths: list,
    output_dir: Path,
    transformation: str,
    image_guidance_scale: float = 1.5,
    guidance_scale: float = 7.5,
    quality_gate=None,
    max_retries: int = 3,
):
    """Run InstructPix2Pix generation directly (single model load)."""
    from generate_instructpix2pix import load_pipeline, process_directory

    print("\n" + "=" * 60)
    print(f"Generate {transformation} (InstructPix2Pix)")
    print("=" * 60)

    config = {
        "image_guidance_scale": image_guidance_scale,
        "guidance_scale": guidance_scale,
    }

    pipe = None

    for input_path, rel_path in zip(input_paths, rel_paths):
        if not input_path.exists():
            continue

        gen_output = output_dir / rel_path

        print(f"\nProcessing: {input_path} -> {gen_output}")

        if pipe is None:
            pipe = load_pipeline()

        process_directory(
            input_dir=input_path,
            output_dir=gen_output,
            transformation=transformation,
            config=config,
            quality_gate=quality_gate,
            max_retries=max_retries,
            _pipe=pipe,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate seasonal variations for UAV datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["uavvisloc", "denseuav", "sues200", "university1652"],
        help="Dataset to process",
    )
    parser.add_argument(
        "--transformation",
        type=str,
        default="summer_to_autumn",
        help="Seasonal transformation (e.g., summer_to_autumn, summer_to_winter)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: ./outputs/<dataset>)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to transformation config YAML",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Path to datasets.yaml config",
    )
    parser.add_argument(
        "--skip-depth",
        action="store_true",
        help="Skip depth estimation (use existing)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip image generation (use existing)",
    )
    parser.add_argument(
        "--depth-model",
        type=str,
        default="DPT_Large",
        choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
        help="Depth estimation model",
    )
    parser.add_argument(
        "--use-img2img",
        action="store_true",
        help="Use img2img mode for better structure preservation.",
    )
    parser.add_argument(
        "--img2img-strength",
        type=float,
        default=0.4,
        help="Strength for img2img mode (0.0=no change, 1.0=complete change).",
    )

    # Method selection
    parser.add_argument(
        "--method",
        type=str,
        default="controlnet",
        choices=["controlnet", "multicontrolnet", "instructpix2pix"],
        help="Generation method",
    )

    # InstructPix2Pix specific
    parser.add_argument(
        "--image-guidance-scale",
        type=float,
        default=1.5,
        help="[InstructPix2Pix] Image guidance scale",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale",
    )

    # Multi-ControlNet specific
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=0.8,
        help="[MultiControlNet] Depth conditioning scale (0.0-1.0)",
    )
    parser.add_argument(
        "--canny-scale",
        type=float,
        default=0.5,
        help="[MultiControlNet] Canny edge conditioning scale (0.0-1.0)",
    )

    # Quality gate
    add_quality_gate_args(parser)

    args = parser.parse_args()

    # Find configs directory
    script_dir = Path(__file__).parent.parent
    configs_dir = script_dir / "configs"

    # Load dataset config
    dataset_config_path = args.dataset_config or (configs_dir / "datasets.yaml")
    dataset_config = load_dataset_config(dataset_config_path, args.dataset)

    print("\n" + "=" * 60)
    print(f"Seasonal Augmentation: {dataset_config['name']}")
    print("=" * 60)
    print(f"Description: {dataset_config['description']}")
    print(f"Transformation: {args.transformation}")
    print(f"Method: {args.method}")

    # Setup paths
    base_path = Path(script_dir) / dataset_config["base_path"]
    base_path = base_path.resolve()

    if not base_path.exists():
        raise FileNotFoundError(f"Dataset not found: {base_path}")

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = script_dir / "outputs" / args.dataset

    output_dir.mkdir(parents=True, exist_ok=True)

    depth_dir = output_dir / "depth_maps"
    gen_dir = output_dir / f"generated_{args.transformation}"

    # Get full paths to drone folders — use the FULL relative path (train/drone, test/query_drone)
    # so that output preserves the train/test split
    drone_paths = dataset_config["drone_paths"]  # e.g. ["train/drone", "test/query_drone"]
    input_paths = [base_path / p for p in drone_paths]
    rel_paths = drone_paths  # used as output subdirectory names

    print(f"\nBase path: {base_path}")
    print(f"Input paths ({len(input_paths)} splits):")
    for p, rp in zip(input_paths, rel_paths):
        print(f"  - {p}  ->  {rp}")
    print(f"Output directory: {output_dir}")

    # Count images
    total_images = len(find_all_images(
        base_path, dataset_config["drone_paths"], dataset_config["extensions"]
    ))
    print(f"Total drone images found: {total_images}")

    # Setup quality gate
    quality_gate, max_retries = create_quality_gate_from_args(args)

    # ── Depth estimation (shared by controlnet and multicontrolnet) ──
    if args.method in ("controlnet", "multicontrolnet"):
        if not args.skip_depth:
            run_depth_estimation(input_paths, rel_paths, depth_dir, args.depth_model)
        else:
            print("\nSkipping depth estimation (--skip-depth)")

    # ── Generation ──
    if args.skip_generation:
        print("\nSkipping generation (--skip-generation)")
    elif args.method == "multicontrolnet":
        # Find transformation config YAML
        transform_config = args.config
        if not transform_config:
            default_config = configs_dir / f"{args.transformation}.yaml"
            if default_config.exists():
                transform_config = str(default_config)

        run_multicontrolnet_direct(
            input_paths,
            rel_paths,
            depth_dir,
            gen_dir,
            args.transformation,
            depth_scale=args.depth_scale,
            canny_scale=args.canny_scale,
            guidance_scale=args.guidance_scale,
            use_img2img=args.use_img2img,
            img2img_strength=args.img2img_strength,
            config_path=transform_config,
            quality_gate=quality_gate,
            max_retries=max_retries,
        )
    elif args.method == "controlnet":
        transform_config = args.config
        if not transform_config:
            default_config = configs_dir / f"{args.transformation}.yaml"
            if default_config.exists():
                transform_config = str(default_config)

        run_controlnet_direct(
            input_paths,
            rel_paths,
            depth_dir,
            gen_dir,
            args.transformation,
            config_path=transform_config,
            use_img2img=args.use_img2img,
            img2img_strength=args.img2img_strength,
            quality_gate=quality_gate,
            max_retries=max_retries,
        )
    elif args.method == "instructpix2pix":
        run_instructpix2pix_direct(
            input_paths,
            rel_paths,
            gen_dir,
            args.transformation,
            image_guidance_scale=args.image_guidance_scale,
            guidance_scale=args.guidance_scale,
            quality_gate=quality_gate,
            max_retries=max_retries,
        )

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}")
    if args.method != "instructpix2pix":
        print(f"  - Depth maps: {depth_dir}")
    print(f"  - Generated images: {gen_dir}")
    if quality_gate:
        print(f"  - {quality_gate.summary()}")


if __name__ == "__main__":
    main()
