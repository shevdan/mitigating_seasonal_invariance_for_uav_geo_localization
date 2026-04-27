"""
GPS-based hard negative sampling for geo-localization datasets.

Pre-computes a dictionary mapping each location to its geographically
nearest neighbors, sorted by distance. Used during training to fill
batches with nearby (hard negative) locations.

Usage:
    python -m sample4geo.dataset.gps_sampling \
        --georecords data/georecords/denseuav.csv \
        --output data/DenseUAV_multiseasonal/gps_dict.pkl
"""

import csv
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km between two GPS points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_gps_dict(
    georecords_csv: str | Path,
    dataset_filter: str | None = None,
    split_filter: str | None = None,
    id_transform: str = "auto",
    neighbour_range: int = 128,
) -> dict:
    """Build GPS neighbor dict from georecords CSV.

    Args:
        georecords_csv: Path to georecords CSV (from seasonalgeo-parse).
        dataset_filter: Filter to specific dataset (e.g. "denseuav").
        split_filter: Filter to specific split (e.g. "train").
        id_transform: How to convert location_id to the integer key used
            by the dataset. "auto" detects from dataset name:
            - denseuav: "denseuav_000123" → 123 (index in sorted list)
            - university1652: "university1652_0123" → 123 (folder name as int)
            - uavvisloc: "uavvisloc_01_0001" → position index
        neighbour_range: Number of nearest neighbors to store.

    Returns:
        dict: {location_idx: [neighbor_idx_1, neighbor_idx_2, ...]}
            sorted by GPS distance (nearest first).
    """
    # Load records
    records = []
    with open(georecords_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if dataset_filter and row["dataset"] != dataset_filter:
                continue
            if split_filter and row.get("split") != split_filter:
                continue
            records.append({
                "location_id": row["location_id"],
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "split": row.get("split", ""),
            })

    # Sort to get deterministic index mapping
    records.sort(key=lambda r: r["location_id"])

    n = len(records)
    lats = np.array([r["lat"] for r in records])
    lons = np.array([r["lon"] for r in records])

    print(f"Computing GPS distances for {n} locations...")

    # Compute pairwise distances
    gps_dict = {}
    for i in tqdm(range(n)):
        distances = []
        for j in range(n):
            if i == j:
                continue
            d = haversine_km(lats[i], lons[i], lats[j], lons[j])
            distances.append((j, d))

        # Sort by distance, keep nearest
        distances.sort(key=lambda x: x[1])
        gps_dict[i] = [idx for idx, _ in distances[:neighbour_range]]

    return gps_dict


def build_gps_dict_uavvisloc(
    data_root: str | Path,
    neighbour_range: int = 128,
) -> dict:
    """Build GPS neighbor dict for UAV-VisLoc from per-flight CSVs.

    Uses the same position ID indexing as UAVVisLocMultiseasonalTrain.
    """
    import pandas as pd

    data_root = Path(data_root)
    positions = []

    for fid_dir in sorted(data_root.glob("[0-9][0-9]")):
        fid = fid_dir.name
        csv_path = fid_dir / f"{fid}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            stem = Path(str(row["filename"])).stem
            positions.append({
                "pos_id": stem,
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
            })

    positions.sort(key=lambda p: p["pos_id"])
    pos_id_to_idx = {p["pos_id"]: i for i, p in enumerate(positions)}

    n = len(positions)
    lats = np.array([p["lat"] for p in positions])
    lons = np.array([p["lon"] for p in positions])

    print(f"Computing GPS distances for {n} UAV-VisLoc positions...")

    gps_dict = {}
    for i in tqdm(range(n)):
        distances = []
        for j in range(n):
            if i == j:
                continue
            d = haversine_km(lats[i], lons[i], lats[j], lons[j])
            distances.append((j, d))

        distances.sort(key=lambda x: x[1])
        gps_dict[i] = [idx for idx, _ in distances[:neighbour_range]]

    return gps_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build GPS neighbor dict")
    parser.add_argument("--georecords", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--neighbour-range", type=int, default=128)
    args = parser.parse_args()

    gps_dict = build_gps_dict(
        args.georecords,
        dataset_filter=args.dataset,
        split_filter=args.split,
        neighbour_range=args.neighbour_range,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(gps_dict, f)

    print(f"Saved GPS dict ({len(gps_dict)} locations) to {args.output}")
