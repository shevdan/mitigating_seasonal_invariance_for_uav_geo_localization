#!/usr/bin/env python3
"""
Batch processing pipeline for seasonal augmentation.

This script orchestrates the full pipeline:
1. Estimate depth maps
2. Generate seasonal variations
3. Organize outputs for Sample4Geo evaluation
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def run_depth_estimation(input_dir: str, output_dir: str, model: str = "DPT_Large"):
    """Run depth estimation on input images."""
    print("\n" + "=" * 60)
    print("Step 1: Depth Estimation")
    print("=" * 60)

    script_path = Path(__file__).parent / "estimate_depth.py"

    cmd = [
        "python",
        str(script_path),
        "--input",
        input_dir,
        "--output",
        output_dir,
        "--model",
        model,
    ]

    subprocess.run(cmd, check=True)


def run_controlnet_generation(
    input_dir: str,
    depth_dir: str,
    output_dir: str,
    transformation: str,
    config: str = None,
):
    """Run ControlNet seasonal generation."""
    print("\n" + "=" * 60)
    print(f"Step 2: Generate {transformation}")
    print("=" * 60)

    script_path = Path(__file__).parent / "generate_controlnet.py"

    cmd = [
        "python",
        str(script_path),
        "--input",
        input_dir,
        "--depth",
        depth_dir,
        "--output",
        output_dir,
        "--transformation",
        transformation,
    ]

    if config:
        cmd.extend(["--config", config])

    subprocess.run(cmd, check=True)


def prepare_for_sample4geo(
    original_data_dir: str,
    generated_dir: str,
    output_dir: str,
    transformation: str,
):
    """
    Organize generated images for Sample4Geo evaluation.

    Creates a structure compatible with UAVVisLocDatasetEval:
    - Copies original satellite images
    - Uses generated images as drone queries
    """
    print("\n" + "=" * 60)
    print("Step 3: Prepare for Sample4Geo Evaluation")
    print("=" * 60)

    original_dir = Path(original_data_dir)
    gen_dir = Path(generated_dir)
    out_dir = Path(output_dir)

    # Read original pairs.csv
    pairs_csv = original_dir / "pairs.csv"
    if not pairs_csv.exists():
        print(f"Error: {pairs_csv} not found")
        return

    df = pd.read_csv(pairs_csv)

    # Create output structure
    drone_out = out_dir / "drone"
    sat_out = out_dir / "satellite"
    drone_out.mkdir(parents=True, exist_ok=True)
    sat_out.mkdir(parents=True, exist_ok=True)

    new_pairs = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Organizing files"):
        orig_drone_path = Path(row["drone_path"])
        orig_sat_path = Path(row["sat_path"])

        # Find generated drone image
        loc_id = row["loc_id"]
        gen_drone_name = f"{orig_drone_path.stem}_{transformation}.jpg"
        gen_drone_path = gen_dir / str(loc_id).zfill(2) / gen_drone_name

        if not gen_drone_path.exists():
            # Try without zero-padding
            gen_drone_path = gen_dir / str(loc_id) / gen_drone_name

        if not gen_drone_path.exists():
            continue

        # Copy/link files
        new_drone_path = drone_out / str(loc_id).zfill(2) / gen_drone_name
        new_sat_path = sat_out / str(loc_id).zfill(2) / orig_sat_path.name

        new_drone_path.parent.mkdir(parents=True, exist_ok=True)
        new_sat_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy generated drone image
        if not new_drone_path.exists():
            shutil.copy(gen_drone_path, new_drone_path)

        # Link/copy original satellite image
        if not new_sat_path.exists():
            if orig_sat_path.exists():
                shutil.copy(orig_sat_path, new_sat_path)
            else:
                continue

        new_pairs.append(
            {
                "loc_id": loc_id,
                "drone_path": str(new_drone_path),
                "sat_path": str(new_sat_path),
                "lat": row["lat"],
                "lon": row["lon"],
                "date": row["date"],
                "height": row["height"],
                "original_drone": str(orig_drone_path),
                "transformation": transformation,
            }
        )

    # Save new pairs.csv
    new_df = pd.DataFrame(new_pairs)
    new_df.to_csv(out_dir / "pairs.csv", index=False)

    print(f"Prepared {len(new_pairs)} pairs for evaluation")
    print(f"Output saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Full seasonal augmentation pipeline"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory (UAV_VisLoc_processed)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for all results",
    )
    parser.add_argument(
        "--transformation",
        type=str,
        default="summer_to_autumn",
        help="Seasonal transformation to apply",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
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
        "--locations",
        type=str,
        nargs="+",
        default=None,
        help="Only process specific locations (e.g., 01 02 05)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    drone_input = input_dir / "drone"
    depth_output = output_dir / "depth_maps"
    gen_output = output_dir / f"generated_{args.transformation}"
    eval_output = output_dir / f"eval_{args.transformation}"

    # Filter by locations if specified
    if args.locations:
        # Process only specified locations
        for loc in args.locations:
            loc_input = drone_input / loc.zfill(2)
            loc_depth = depth_output / loc.zfill(2)
            loc_gen = gen_output / loc.zfill(2)

            if not args.skip_depth:
                run_depth_estimation(str(loc_input), str(loc_depth))

            if not args.skip_generation:
                run_controlnet_generation(
                    str(loc_input),
                    str(loc_depth),
                    str(loc_gen),
                    args.transformation,
                    args.config,
                )
    else:
        # Process all
        if not args.skip_depth:
            run_depth_estimation(str(drone_input), str(depth_output))

        if not args.skip_generation:
            run_controlnet_generation(
                str(drone_input),
                str(depth_output),
                str(gen_output),
                args.transformation,
                args.config,
            )

    # Prepare for evaluation
    prepare_for_sample4geo(
        str(input_dir),
        str(gen_output),
        str(eval_output),
        args.transformation,
    )

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  Depth maps: {depth_output}")
    print(f"  Generated images: {gen_output}")
    print(f"  Evaluation data: {eval_output}")
    print(f"\nTo evaluate with Sample4Geo:")
    print(f"  Use {eval_output} as test data path")


if __name__ == "__main__":
    main()
