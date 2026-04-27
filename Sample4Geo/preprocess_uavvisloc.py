#!/usr/bin/env python3
"""
Preprocess UAV-VisLoc dataset for Sample4Geo training.

This script extracts satellite patches at drone GPS locations.
Run this before training on UAV-VisLoc.

Usage:
    python preprocess_uavvisloc.py --data_root /path/to/UAV_VisLoc_dataset --output /path/to/output
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

from sample4geo.dataset.uavvisloc import (
    UAVVisLocPreprocessor,
    SUMMER_LOCATIONS,
    AUTUMN_LOCATIONS,
    ALL_LOCATIONS
)


def main():
    parser = argparse.ArgumentParser(description="Preprocess UAV-VisLoc dataset")

    parser.add_argument(
        '--data_root',
        type=str,
        default='./data/UAV_VisLoc_dataset',
        help='Path to UAV-VisLoc dataset root'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./data/UAV_VisLoc_processed',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=512,
        help='Size of satellite patches to extract'
    )
    parser.add_argument(
        '--locations',
        nargs='+',
        default=None,
        help='Specific locations to process (e.g., 01 02 03). Default: all'
    )
    parser.add_argument(
        '--season',
        type=str,
        choices=['summer', 'autumn', 'all'],
        default='all',
        help='Process only specific season locations'
    )

    args = parser.parse_args()

    # Determine locations to process
    if args.locations:
        locations = args.locations
    elif args.season == 'summer':
        locations = SUMMER_LOCATIONS
    elif args.season == 'autumn':
        locations = AUTUMN_LOCATIONS
    else:
        locations = ALL_LOCATIONS

    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output}")
    print(f"Patch size: {args.patch_size}")
    print(f"Locations: {locations}")
    print()

    # Create preprocessor
    preprocessor = UAVVisLocPreprocessor(
        data_root=args.data_root,
        output_dir=args.output,
        patch_size=args.patch_size
    )

    # Process all locations
    pairs = preprocessor.process_all(locations=locations)

    print(f"\nDone! Processed {len(pairs)} pairs total.")
    print(f"Output saved to: {args.output}")

    # Print summary by season
    summer_count = sum(1 for p in pairs if p['loc_id'] in SUMMER_LOCATIONS)
    autumn_count = sum(1 for p in pairs if p['loc_id'] in AUTUMN_LOCATIONS)

    print(f"\nSeasonal distribution:")
    print(f"  Summer (June): {summer_count} pairs")
    print(f"  Autumn (Sept-Oct): {autumn_count} pairs")


if __name__ == "__main__":
    main()
