#!/usr/bin/env python3
"""
Prepare generated seasonal images for Sample4Geo training.

Creates a dataset directory that mirrors the original DenseUAV/UAV-VisLoc structure
but with generated seasonal drone images. Satellite images are symlinked from the
original dataset (they don't change with seasons).

The result can be used directly as `data_folder` in Sample4Geo training scripts.

Usage:
    # DenseUAV — all seasons
    python prepare_for_sample4geo.py --dataset denseuav --seasons autumn winter spring

    # UAV-VisLoc — single season
    python prepare_for_sample4geo.py --dataset uavvisloc --seasons winter
"""

import argparse
import os
import shutil
from pathlib import Path
from tqdm import tqdm


DATASET_CONFIGS = {
    "denseuav": {
        "original": "data/DenseUAV",
        "generated_base": "seasonal_augmentation/outputs/denseuav",
        "scraped_satellite_base": "output/formatted/denseuav",
        "splits": {
            "train": {
                "drone": "train/drone",
                "satellite": "train/satellite",
            },
            "test": {
                "drone": "test/query_drone",
                "satellite": "test/gallery_satellite",
            },
        },
        "output_base": "data/DenseUAV_multiseasonal",
    },
    "uavvisloc": {
        "original": "data/UAV_VisLoc_dataset",
        "generated_base": "seasonal_augmentation/outputs/uavvisloc",
        "scraped_satellite_base": "output/formatted/uavvisloc",
        "splits": {
            # UAV-VisLoc doesn't have separate train/test dirs for drone
            # All flights are in {NN}/drone/
            "all": {
                "drone_paths": [f"{i:02d}/drone" for i in range(1, 12)],
                "satellite_paths": [f"{i:02d}" for i in range(1, 12)],
            },
        },
        "output_base": "data/UAV_VisLoc_multiseasonal",
    },
}


def rename_generated_file(filename: str, transformation: str) -> str:
    """Remove transformation suffix from generated filename.

    H80_summer_to_autumn.jpg -> H80.JPG
    01_0001_summer_to_winter.jpg -> 01_0001.JPG
    """
    suffix = f"_{transformation}"
    stem = Path(filename).stem
    if stem.endswith(suffix):
        original_stem = stem[:-len(suffix)]
    else:
        original_stem = stem
    # Use .JPG to match original DenseUAV naming
    return f"{original_stem}.JPG"


def prepare_denseuav(
    project_root: Path,
    season: str,
    use_symlinks: bool = True,
    satellite_source: str = "scraped",
    satellite_provider: str = "arcgis",
    satellite_year: int = 2023,
):
    """Prepare DenseUAV seasonal dataset for Sample4Geo.

    Args:
        satellite_source: "scraped" = use ArcGIS/Planet seasonal satellite,
                          "original" = symlink original (summer) satellite.
        satellite_provider: Provider name for scraped satellite (e.g. "arcgis").
        satellite_year: Year of scraped satellite imagery.
    """
    cfg = DATASET_CONFIGS["denseuav"]
    transformation = f"summer_to_{season}"
    generated_dir = project_root / cfg["generated_base"] / f"generated_{transformation}"
    original_dir = project_root / cfg["original"]
    season_dir_name = season if satellite_source == "scraped" else f"{season}_orig_sat"
    output_dir = project_root / cfg["output_base"] / season_dir_name

    # Scraped seasonal satellite directory
    sat_combo = f"{season}_{satellite_year}_{satellite_provider}"
    scraped_sat_dir = project_root / cfg["scraped_satellite_base"] / sat_combo

    print(f"\n{'='*60}")
    print(f"Preparing DenseUAV — {season}")
    print(f"{'='*60}")
    print(f"  Generated drone: {generated_dir}")
    print(f"  Original:        {original_dir}")
    print(f"  Output:          {output_dir}")

    if satellite_source == "scraped" and scraped_sat_dir.exists():
        print(f"  Satellite:       {scraped_sat_dir} (seasonal {sat_combo})")
    else:
        if satellite_source == "scraped":
            print(f"  WARNING: Scraped satellite not found: {scraped_sat_dir}")
            print(f"           Falling back to original satellite")
        satellite_source = "original"
        print(f"  Satellite:       {original_dir} (original/summer)")

    if not generated_dir.exists():
        print(f"  ERROR: Generated directory not found: {generated_dir}")
        return

    link_fn = os.symlink if use_symlinks else shutil.copy2

    for split_name, split_cfg in cfg["splits"].items():
        drone_rel = split_cfg["drone"]
        sat_rel = split_cfg["satellite"]

        gen_drone_dir = generated_dir / drone_rel
        out_drone_dir = output_dir / drone_rel
        out_sat_dir = output_dir / sat_rel

        if not gen_drone_dir.exists():
            print(f"  WARNING: {gen_drone_dir} not found, skipping {split_name}")
            continue

        # Process drone images — rename generated files
        gen_files = list(gen_drone_dir.rglob("*.jpg"))
        print(f"  {split_name} drone: {len(gen_files)} generated images")

        for gen_file in tqdm(gen_files, desc=f"  {split_name} drone"):
            rel_path = gen_file.relative_to(gen_drone_dir)
            new_name = rename_generated_file(rel_path.name, transformation)
            out_path = out_drone_dir / rel_path.parent / new_name

            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() or out_path.is_symlink():
                continue

            if use_symlinks:
                link_fn(gen_file.resolve(), out_path)
            else:
                link_fn(str(gen_file), str(out_path))

        # Satellite images — use scraped seasonal or original
        if satellite_source == "scraped":
            src_sat_dir = scraped_sat_dir / sat_rel
        else:
            src_sat_dir = original_dir / sat_rel

        if src_sat_dir.exists():
            sat_files = list(src_sat_dir.rglob("*"))
            sat_files = [f for f in sat_files if f.is_file()
                         and not f.name.endswith("_old.tif")]
            label = f"seasonal {sat_combo}" if satellite_source == "scraped" else "original"
            print(f"  {split_name} satellite: {len(sat_files)} files ({label})")

            for sat_file in tqdm(sat_files, desc=f"  {split_name} satellite"):
                rel_path = sat_file.relative_to(src_sat_dir)
                out_path = out_sat_dir / rel_path

                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.exists() or out_path.is_symlink():
                    continue

                if use_symlinks:
                    os.symlink(sat_file.resolve(), out_path)
                else:
                    shutil.copy2(str(sat_file), str(out_path))

    print(f"\n  Done! Dataset at: {output_dir}")
    print(f"  Use in Sample4Geo: data_folder = '{output_dir}'")


def prepare_uavvisloc(
    project_root: Path,
    season: str,
    use_symlinks: bool = True,
    satellite_source: str = "scraped",
    satellite_provider: str = "arcgis",
    satellite_year: int = 2023,
):
    """Prepare UAV-VisLoc seasonal dataset for Sample4Geo."""
    cfg = DATASET_CONFIGS["uavvisloc"]
    transformation = f"summer_to_{season}"
    generated_dir = project_root / cfg["generated_base"] / f"generated_{transformation}"
    original_dir = project_root / cfg["original"]

    season_dir_name = season if satellite_source == "scraped" else f"{season}_orig_sat"
    output_dir = project_root / cfg["output_base"] / season_dir_name

    # Scraped seasonal satellite directory (formatted patches)
    sat_combo = f"{season}_{satellite_year}_{satellite_provider}"
    scraped_sat_dir = project_root / cfg["scraped_satellite_base"] / sat_combo

    print(f"\n{'='*60}")
    print(f"Preparing UAV-VisLoc — {season}")
    print(f"{'='*60}")
    print(f"  Generated drone: {generated_dir}")
    print(f"  Original:        {original_dir}")
    print(f"  Output:          {output_dir}")

    if satellite_source == "scraped" and scraped_sat_dir.exists():
        print(f"  Satellite:       {scraped_sat_dir} (seasonal {sat_combo})")
    else:
        if satellite_source == "scraped":
            print(f"  WARNING: Scraped satellite not found: {scraped_sat_dir}")
            print(f"           Falling back to original satellite")
        satellite_source = "original"
        print(f"  Satellite:       {original_dir} (original)")

    if not generated_dir.exists():
        print(f"  ERROR: Generated directory not found: {generated_dir}")
        return

    link_fn = os.symlink if use_symlinks else shutil.copy2

    # Process each flight's drone images
    total_drone = 0
    for flight_drone in sorted(generated_dir.glob("*/drone")):
        flight_id = flight_drone.parent.name  # e.g., "01"
        gen_files = list(flight_drone.glob("*.jpg"))
        total_drone += len(gen_files)

        out_drone_dir = output_dir / flight_id / "drone"

        for gen_file in gen_files:
            new_name = rename_generated_file(gen_file.name, transformation)
            out_path = out_drone_dir / new_name

            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() or out_path.is_symlink():
                continue

            if use_symlinks:
                os.symlink(gen_file.resolve(), out_path)
            else:
                shutil.copy2(str(gen_file), str(out_path))

    print(f"  Drone images: {total_drone}")

    # Satellite data
    total_sat = 0
    if satellite_source == "scraped":
        # Symlink scraped seasonal satellite patches
        sat_dir = scraped_sat_dir / "satellite"
        if sat_dir.exists():
            for flight_dir in sorted(sat_dir.glob("[0-9][0-9]")):
                flight_id = flight_dir.name
                for sat_file in flight_dir.glob("*_sat.jpg"):
                    out_path = output_dir / flight_id / "satellite" / sat_file.name
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    if out_path.exists() or out_path.is_symlink():
                        continue
                    os.symlink(sat_file.resolve(), out_path)
                    total_sat += 1

        # Also symlink flight CSVs and other non-drone files from original
        for flight_dir in sorted(original_dir.glob("[0-9][0-9]")):
            flight_id = flight_dir.name
            for item in flight_dir.iterdir():
                if item.name == "drone" or item.name == "satellite":
                    continue
                out_path = output_dir / flight_id / item.name
                if out_path.exists() or out_path.is_symlink():
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(item.resolve(), out_path)

        # Symlink the original satellite TIF for each flight
        for flight_dir in sorted(original_dir.glob("[0-9][0-9]")):
            flight_id = flight_dir.name
            for tif in flight_dir.glob("satellite*.tif"):
                out_path = output_dir / flight_id / tif.name
                if out_path.exists() or out_path.is_symlink():
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(tif.resolve(), out_path)

        label = f"seasonal {sat_combo}"
    else:
        # Symlink everything except /drone from original
        for flight_dir in sorted(original_dir.glob("[0-9][0-9]")):
            flight_id = flight_dir.name
            for item in flight_dir.iterdir():
                if item.name == "drone":
                    continue
                out_path = output_dir / flight_id / item.name
                if out_path.exists() or out_path.is_symlink():
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if item.is_dir():
                    os.symlink(item.resolve(), out_path)
                else:
                    os.symlink(item.resolve(), out_path)
                    total_sat += 1
        label = "original"

    print(f"  Satellite files: {total_sat} ({label})")
    print(f"\n  Done! Dataset at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare generated seasonal images for Sample4Geo training"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["denseuav", "uavvisloc"],
        help="Dataset to prepare",
    )
    parser.add_argument(
        "--seasons",
        type=str,
        nargs="+",
        default=["autumn", "winter", "spring"],
        help="Seasons to prepare",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root directory",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinking (uses more disk space)",
    )
    parser.add_argument(
        "--satellite-source",
        type=str,
        choices=["scraped", "original"],
        default="scraped",
        help="Satellite source: 'scraped' = seasonal ArcGIS/Planet, 'original' = summer only (default: scraped)",
    )
    parser.add_argument(
        "--satellite-provider",
        type=str,
        default="arcgis",
        help="Provider for scraped satellite (default: arcgis)",
    )
    parser.add_argument(
        "--satellite-year",
        type=int,
        default=2023,
        help="Year of scraped satellite imagery (default: 2023)",
    )

    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()

    for season in args.seasons:
        if args.dataset == "denseuav":
            prepare_denseuav(
                project_root, season,
                use_symlinks=not args.copy,
                satellite_source=args.satellite_source,
                satellite_provider=args.satellite_provider,
                satellite_year=args.satellite_year,
            )
        elif args.dataset == "uavvisloc":
            prepare_uavvisloc(
                project_root, season,
                use_symlinks=not args.copy,
                satellite_source=args.satellite_source,
                satellite_provider=args.satellite_provider,
                satellite_year=args.satellite_year,
            )

    print("\n" + "="*60)
    print("All done!")
    print("="*60)
    print("\nTo train Sample4Geo with seasonal data:")
    print(f"  data_folder = 'data/{args.dataset.upper()}_seasonal/<season>'")


if __name__ == "__main__":
    main()
