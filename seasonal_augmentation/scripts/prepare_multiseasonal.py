#!/usr/bin/env python3
"""
Prepare unified multiseasonal datasets for Sample4Geo experiments.

Creates a single directory per dataset where each location has all seasonal
variants side by side (season suffix in filename), enabling flexible mixing
of drone/satellite seasons via dataloader config.

Structure:
    DenseUAV_multiseasonal/
      train/
        drone/000000/H80.JPG  H80_autumn.JPG  H80_winter.JPG  H80_spring.JPG
        satellite/000000/H80.tif  H80_autumn.tif  H80_winter.tif  H80_spring.tif
      test/
        query_drone/...
        gallery_satellite/...

    UAV_VisLoc_multiseasonal/
      {flight_id}/
        drone/01_0001.JPG  01_0001_autumn.JPG  01_0001_winter.JPG  ...
        satellite/01_0001_sat.jpg  01_0001_sat_autumn.jpg  ...
      satellite{flight_id}.tif  (symlinked from original)
      {flight_id}.csv            (symlinked from original)

    University1652_multiseasonal/
      train/satellite/0001/0001.jpg  0001_autumn.jpg  0001_winter.jpg  ...
      test/gallery_satellite/0001/...
      train/drone/...   (symlinked from original — no generated views)
      test/query_drone/...

Usage:
    python prepare_multiseasonal.py --dataset denseuav
    python prepare_multiseasonal.py --dataset uavvisloc
    python prepare_multiseasonal.py --dataset university1652
    python prepare_multiseasonal.py --dataset all
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    import rasterio
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


SEASONS = ["autumn", "winter", "spring"]


def symlink(src: Path, dst: Path):
    """Create symlink, skip if already exists."""
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src.resolve(), dst)


def prepare_denseuav(project_root: Path):
    """DenseUAV: drone (original + generated) + satellite (original + ArcGIS)."""
    original = project_root / "data" / "DenseUAV"
    generated_base = project_root / "seasonal_augmentation" / "outputs" / "denseuav"
    scraped_base = project_root / "output" / "formatted" / "denseuav"
    output = project_root / "data" / "DenseUAV_multiseasonal"

    splits = {
        "train": {"drone": "train/drone", "satellite": "train/satellite"},
        "test": {"drone": "test/query_drone", "satellite": "test/gallery_satellite"},
    }

    print(f"\n{'='*60}")
    print("Preparing DenseUAV multiseasonal")
    print(f"{'='*60}")

    for split_name, paths in splits.items():
        drone_rel = paths["drone"]
        sat_rel = paths["satellite"]

        orig_drone = original / drone_rel
        orig_sat = original / sat_rel

        # 1) Original drone + satellite (summer)
        if orig_drone.exists():
            files = [f for f in orig_drone.rglob("*") if f.is_file()
                     and not f.name.endswith("_old.tif")]
            print(f"  {split_name} drone original: {len(files)} files")
            for f in tqdm(files, desc=f"  {split_name} drone original"):
                rel = f.relative_to(orig_drone)
                symlink(f, output / drone_rel / rel)

        if orig_sat.exists():
            files = [f for f in orig_sat.rglob("*") if f.is_file()
                     and "_old" not in f.name]
            print(f"  {split_name} satellite original: {len(files)} files")
            for f in tqdm(files, desc=f"  {split_name} satellite original"):
                rel = f.relative_to(orig_sat)
                symlink(f, output / sat_rel / rel)

        # 2) Generated drone per season
        for season in SEASONS:
            gen_dir = generated_base / f"generated_summer_to_{season}" / drone_rel
            if not gen_dir.exists():
                print(f"  WARNING: {gen_dir} not found, skipping")
                continue
            gen_files = list(gen_dir.rglob("*.jpg"))
            print(f"  {split_name} drone {season}: {len(gen_files)} files")
            for f in tqdm(gen_files, desc=f"  {split_name} drone {season}"):
                rel = f.relative_to(gen_dir)
                # H80_summer_to_autumn.jpg → H80_autumn.JPG
                stem = rel.name.replace(f"_summer_to_{season}", f"_{season}")
                stem = Path(stem).stem + ".JPG"
                symlink(f, output / drone_rel / rel.parent / stem)

        # 3) Scraped satellite per season
        for season in SEASONS:
            scraped_dir = scraped_base / f"{season}_2023_arcgis" / sat_rel
            if not scraped_dir.exists():
                print(f"  WARNING: {scraped_dir} not found, skipping")
                continue
            scraped_files = [f for f in scraped_dir.rglob("*") if f.is_file()]
            print(f"  {split_name} satellite {season}: {len(scraped_files)} files")
            for f in tqdm(scraped_files, desc=f"  {split_name} satellite {season}"):
                rel = f.relative_to(scraped_dir)
                # H80.tif → H80_autumn.tif
                stem = Path(rel.name).stem
                ext = Path(rel.name).suffix
                new_name = f"{stem}_{season}{ext}"
                symlink(f, output / sat_rel / rel.parent / new_name)

    print(f"\n  Done! {output}")


def _get_satellite_bounds(csv_path: Path) -> dict:
    """Read satellite coordinate bounds from CSV."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    bounds = {}
    for _, row in df.iterrows():
        fid = row['mapname'].replace('satellite', '').replace('.tif', '')
        bounds[fid] = {
            'lt_lat': row['LT_lat_map'],
            'lt_lon': row['LT_lon_map'],
            'rb_lat': row['RB_lat_map'],
            'rb_lon': row['RB_lon_map'],
        }
    return bounds


def _extract_sat_patch(tif_path: Path, lat: float, lon: float,
                       bounds: dict, patch_size: int = 512) -> np.ndarray | None:
    """Extract a 512x512 patch from original satellite TIF at GPS position."""
    if not RASTERIO_AVAILABLE:
        return None

    with rasterio.open(tif_path) as src:
        h, w = src.height, src.width
        lat_range = bounds['lt_lat'] - bounds['rb_lat']
        lon_range = bounds['rb_lon'] - bounds['lt_lon']
        y_norm = (bounds['lt_lat'] - lat) / lat_range
        x_norm = (lon - bounds['lt_lon']) / lon_range
        px_x, px_y = int(x_norm * w), int(y_norm * h)

        half = patch_size // 2
        col_off = max(0, px_x - half)
        row_off = max(0, px_y - half)
        width = min(patch_size, w - col_off)
        height = min(patch_size, h - row_off)
        if width <= 0 or height <= 0:
            return None

        window = Window(col_off, row_off, width, height)
        patch = src.read([1, 2, 3], window=window)
        patch = np.transpose(patch, (1, 2, 0))  # CHW → HWC

        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            padded = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
            padded[:patch.shape[0], :patch.shape[1], :] = patch
            patch = padded

    return patch


def prepare_uavvisloc(project_root: Path):
    """UAV-VisLoc: drone (original + generated) + satellite (original TIF + ArcGIS seasonal)."""
    original = project_root / "data" / "UAV_VisLoc_dataset"
    generated_base = project_root / "seasonal_augmentation" / "outputs" / "uavvisloc"
    scraped_base = project_root / "output" / "formatted" / "uavvisloc"
    output = project_root / "data" / "UAV_VisLoc_multiseasonal"

    print(f"\n{'='*60}")
    print("Preparing UAV-VisLoc multiseasonal")
    print(f"{'='*60}")

    # Load satellite bounds for patch extraction
    bounds_csv = original / "satellite_ coordinates_range.csv"
    sat_bounds = _get_satellite_bounds(bounds_csv)

    for flight_dir in sorted(original.glob("[0-9][0-9]")):
        fid = flight_dir.name  # "01"

        # Symlink non-drone files (satellite TIF, CSV, etc.)
        for item in flight_dir.iterdir():
            if item.name == "drone":
                continue
            symlink(item, output / fid / item.name)

        # 1) Original drone (summer)
        orig_drone = flight_dir / "drone"
        if orig_drone.exists():
            drone_files = list(orig_drone.glob("*.JPG"))
            for f in drone_files:
                symlink(f, output / fid / "drone" / f.name)
            print(f"  Flight {fid}: {len(drone_files)} original drone")

        # 2) Generated drone per season
        for season in SEASONS:
            gen_drone = generated_base / f"generated_summer_to_{season}" / fid / "drone"
            if not gen_drone.exists():
                continue
            gen_files = list(gen_drone.glob("*.jpg"))
            for f in gen_files:
                stem = f.stem.replace(f"_summer_to_{season}", f"_{season}")
                symlink(f, output / fid / "drone" / f"{stem}.JPG")
            print(f"  Flight {fid}: {len(gen_files)} {season} drone")

        # 3) Summer satellite: extract patches from ORIGINAL high-res TIF
        #    This preserves the native resolution (~0.26 m/px) and scale
        import pandas as pd
        tif_path = flight_dir / f"satellite{fid}.tif"
        drone_csv = flight_dir / f"{fid}.csv"

        if tif_path.exists() and drone_csv.exists() and fid in sat_bounds:
            sat_out = output / fid / "satellite"
            sat_out.mkdir(parents=True, exist_ok=True)

            df = pd.read_csv(drone_csv)
            extracted = 0
            for _, row in df.iterrows():
                stem = Path(str(row['filename'])).stem  # "01_0001"
                out_path = sat_out / f"{stem}_sat.jpg"
                if out_path.exists():
                    extracted += 1
                    continue

                patch = _extract_sat_patch(
                    tif_path, row['lat'], row['lon'], sat_bounds[fid],
                )
                if patch is not None:
                    cv2.imwrite(
                        str(out_path),
                        cv2.cvtColor(patch, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95],
                    )
                    extracted += 1

            print(f"  Flight {fid}: {extracted} summer satellite (from original TIF)")
        else:
            # Fallback: try split TIF files (e.g. flight 09)
            split_tifs = list(flight_dir.glob(f"satellite{fid}_*.tif"))
            if split_tifs and drone_csv.exists() and fid in sat_bounds:
                print(f"  Flight {fid}: split TIF found, skipping summer sat (TODO)")
            else:
                print(f"  Flight {fid}: WARNING — no original TIF, no summer satellite")

        # 4) Seasonal satellite: ArcGIS patches (different resolution but
        #    only source for seasonal variants)
        for season in SEASONS:
            scraped_sat = scraped_base / f"{season}_2023_arcgis" / "satellite" / fid
            if not scraped_sat.exists():
                continue
            sat_files = list(scraped_sat.glob("*_sat.jpg"))
            for f in sat_files:
                stem = f.stem  # "01_0001_sat"
                symlink(f, output / fid / "satellite" / f"{stem}_{season}.jpg")
            print(f"  Flight {fid}: {len(sat_files)} {season} satellite (ArcGIS)")

    print(f"\n  Done! {output}")


def prepare_university1652(project_root: Path):
    """University-1652: satellite (original + ArcGIS) + drone (original only, symlinked)."""
    original = project_root / "data" / "University-Release"
    scraped_base = project_root / "output" / "formatted" / "university1652"
    output = project_root / "data" / "University1652_multiseasonal"

    print(f"\n{'='*60}")
    print("Preparing University-1652 multiseasonal")
    print(f"{'='*60}")

    # Satellite splits
    sat_splits = {
        "train": "train/satellite",
        "test": "test/gallery_satellite",
    }
    # Drone splits (symlink as-is, no generated views)
    drone_splits = {
        "train/drone": "train/drone",
        "train/google": "train/google",
        "train/street": "train/street",
        "test/query_drone": "test/query_drone",
        "test/query_satellite": "test/query_satellite",
        "test/query_street": "test/query_street",
        "test/gallery_drone": "test/gallery_drone",
        "test/gallery_street": "test/gallery_street",
        "test/4K_drone": "test/4K_drone",
    }

    # 1) Symlink all drone/other views from original
    for src_rel, dst_rel in drone_splits.items():
        src_dir = original / src_rel
        if not src_dir.exists():
            continue
        files = [f for f in src_dir.rglob("*") if f.is_file()]
        print(f"  {dst_rel}: {len(files)} files (original)")
        for f in tqdm(files, desc=f"  {dst_rel}"):
            rel = f.relative_to(src_dir)
            symlink(f, output / dst_rel / rel)

    # 2) Original satellite
    for split_name, sat_rel in sat_splits.items():
        orig_sat = original / sat_rel
        if not orig_sat.exists():
            continue
        files = [f for f in orig_sat.rglob("*") if f.is_file()]
        print(f"  {sat_rel} original: {len(files)} files")
        for f in tqdm(files, desc=f"  {sat_rel} original"):
            rel = f.relative_to(orig_sat)
            symlink(f, output / sat_rel / rel)

    # 3) Scraped satellite per season
    for season in SEASONS:
        for split_name, sat_rel in sat_splits.items():
            scraped_dir = scraped_base / f"{season}_2023_arcgis" / sat_rel
            if not scraped_dir.exists():
                print(f"  WARNING: {scraped_dir} not found")
                continue
            scraped_files = [f for f in scraped_dir.rglob("*") if f.is_file()]
            print(f"  {sat_rel} {season}: {len(scraped_files)} files")
            for f in tqdm(scraped_files, desc=f"  {sat_rel} {season}"):
                rel = f.relative_to(scraped_dir)
                # 0001.jpg → 0001_autumn.jpg
                stem = Path(rel.name).stem
                ext = Path(rel.name).suffix
                new_name = f"{stem}_{season}{ext}"
                symlink(f, output / sat_rel / rel.parent / new_name)

    print(f"\n  Done! {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare unified multiseasonal datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["denseuav", "uavvisloc", "university1652", "all"],
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
    )
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()

    datasets = (
        ["denseuav", "uavvisloc", "university1652"]
        if args.dataset == "all"
        else [args.dataset]
    )

    for ds in datasets:
        if ds == "denseuav":
            prepare_denseuav(project_root)
        elif ds == "uavvisloc":
            prepare_uavvisloc(project_root)
        elif ds == "university1652":
            prepare_university1652(project_root)

    print(f"\n{'='*60}")
    print("All done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
