#!/usr/bin/env python3
"""Format raw retrieval output into training-ready dataset structure.

Converts raw satellite imagery into 512x512 RGB images organized to match
each dataset's original folder structure for drop-in use with Sample4Geo.

Usage:
    # Format DenseUAV winter Planet images
    seasonalgeo-format --dataset denseuav --provider planet \
        --seasons winter --years 2023

    # Format University-1652 with custom paths
    seasonalgeo-format --dataset university1652 --provider sentinel2 \
        --raw-output-dir output/ --formatted-output-dir output/formatted/ \
        --seasons spring autumn winter --years 2023

    # Or run as module:
    python -m seasonalgeo.scripts.s03_format_dataset --dataset denseuav ...
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from seasonalgeo.models.schema import BBox, Dataset, GeoRecord
from seasonalgeo.output.formatter import FORMATTER_REGISTRY, UAVVisLocFormatter

console = Console()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def load_georecords(csv_path: Path, dataset: str | None = None) -> list[GeoRecord]:
    """Load GeoRecords from CSV."""
    records = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if dataset and row["dataset"] != dataset:
                continue
            records.append(GeoRecord(
                location_id=row["location_id"],
                dataset=Dataset(row["dataset"]),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                bbox=BBox(
                    min_lat=float(row["bbox_min_lat"]),
                    min_lon=float(row["bbox_min_lon"]),
                    max_lat=float(row["bbox_max_lat"]),
                    max_lon=float(row["bbox_max_lon"]),
                ),
                original_tile_path=row["original_tile_path"],
                original_tile_width=int(row["original_tile_width"]),
                original_tile_height=int(row["original_tile_height"]),
                split=row.get("split") or None,
            ))
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Format raw satellite imagery into training-ready dataset structure"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=list(FORMATTER_REGISTRY.keys()),
        help="Target dataset to format for",
    )
    parser.add_argument(
        "--provider", type=str, required=True,
        choices=["sentinel2", "planet", "arcgis"],
        help="Provider whose output to format",
    )
    parser.add_argument(
        "--seasons", nargs="+", required=True,
        choices=["spring", "summer", "autumn", "winter"],
        help="Seasons to format",
    )
    parser.add_argument(
        "--years", nargs="+", type=int, required=True,
        help="Years to format",
    )
    parser.add_argument(
        "--raw-output-dir", type=Path, default=Path("output"),
        help="Directory with raw retrieval output (default: output/)",
    )
    parser.add_argument(
        "--formatted-output-dir", type=Path, default=None,
        help="Directory for formatted output (default: output/formatted/)",
    )
    parser.add_argument(
        "--georecords-csv", type=Path, default=None,
        help="GeoRecords CSV (default: data/georecords/{dataset}.csv)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Root data directory (for UAV-VisLoc drone CSVs; default: data/)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    raw_dir = args.raw_output_dir.resolve()
    fmt_dir = (args.formatted_output_dir or raw_dir / "formatted").resolve()
    csv_path = args.georecords_csv or Path("data/georecords") / f"{args.dataset}.csv"
    csv_path = csv_path.resolve()

    if not csv_path.exists():
        console.print(f"[red]GeoRecords CSV not found: {csv_path}[/]")
        console.print("Run [bold]seasonalgeo-parse[/] first.")
        sys.exit(1)

    records = load_georecords(csv_path, args.dataset)

    console.print(f"[bold]SeasonalGeo Dataset Formatter[/]")
    console.print(f"Dataset: {args.dataset}")
    console.print(f"Provider: {args.provider}")
    console.print(f"Seasons: {', '.join(args.seasons)}")
    console.print(f"Years: {args.years}")
    console.print(f"Locations: {len(records)}")
    console.print(f"Raw output: {raw_dir}")
    console.print(f"Formatted output: {fmt_dir}")
    console.print()

    formatter = FORMATTER_REGISTRY[args.dataset]()

    # Summary table
    summary_table = Table(title="Formatting Results")
    summary_table.add_column("Season/Year", style="cyan")
    summary_table.add_column("Formatted", justify="right", style="green")
    summary_table.add_column("Missing", justify="right", style="yellow")
    summary_table.add_column("Errors", justify="right", style="red")

    total_formatted = 0
    total_missing = 0
    total_errors = 0

    # UAV-VisLoc flight-level: need drone positions and satellite bounds
    uavvisloc_flight_mode = (
        args.dataset == "uavvisloc"
        and all(r.location_id.startswith("uavvisloc_flight") for r in records)
    )
    if uavvisloc_flight_mode:
        visloc_dir = args.data_dir.resolve() / "UAV_VisLoc_dataset"
        sat_bounds_csv = visloc_dir / "satellite_ coordinates_range.csv"
        if not sat_bounds_csv.exists():
            console.print(f"[red]satellite_coordinates_range.csv not found: {sat_bounds_csv}[/]")
            sys.exit(1)
        sat_bounds_df = pd.read_csv(sat_bounds_csv)
        sat_bounds_map = {}
        for _, row in sat_bounds_df.iterrows():
            fid = row["mapname"].replace("satellite", "").replace(".tif", "")
            sat_bounds_map[fid] = {
                "lt_lat": row["LT_lat_map"],
                "lt_lon": row["LT_lon_map"],
                "rb_lat": row["RB_lat_map"],
                "rb_lon": row["RB_lon_map"],
            }
        console.print("[dim]UAV-VisLoc flight-level mode: extracting patches from GeoTIFs[/]")

    for year in args.years:
        for season in args.seasons:
            combo = f"{season}_{year}_{args.provider}"
            output_root = fmt_dir / args.dataset / combo

            formatted = 0
            missing = 0
            errors = 0

            if uavvisloc_flight_mode:
                # Extract patches from flight-level GeoTIFs
                visloc_formatter = UAVVisLocFormatter()
                for record in records:
                    flight_id = record.location_id.removeprefix("uavvisloc_flight")
                    raw_tif = (
                        raw_dir / "uavvisloc" / record.location_id
                        / f"{season}_{year}_{args.provider}.tif"
                    )
                    if not raw_tif.exists():
                        missing += 1
                        continue

                    if flight_id not in sat_bounds_map:
                        logging.error("No sat bounds for flight %s", flight_id)
                        errors += 1
                        continue

                    # Load drone positions for this flight
                    drone_csv = visloc_dir / flight_id / f"{flight_id}.csv"
                    if not drone_csv.exists():
                        logging.error("Drone CSV not found: %s", drone_csv)
                        errors += 1
                        continue

                    drone_df = pd.read_csv(drone_csv)
                    drone_positions = []
                    for _, drow in drone_df.iterrows():
                        # filename is e.g. "01_0001.JPG" → stem "01_0001"
                        img_stem = Path(str(drow["filename"])).stem
                        drone_positions.append({
                            "img_stem": img_stem,
                            "lat": float(drow["lat"]),
                            "lon": float(drow["lon"]),
                        })

                    try:
                        created = visloc_formatter.format_flight_geotiff(
                            raw_tif, flight_id, drone_positions,
                            sat_bounds_map[flight_id], output_root,
                        )
                        formatted += len(created)
                    except Exception as e:
                        errors += 1
                        logging.error(
                            "Failed to format flight %s: %s", flight_id, e,
                            exc_info=True,
                        )
            else:
                for record in records:
                    # Find raw image
                    raw_jpg = (
                        raw_dir / record.dataset.value / record.location_id
                        / f"{season}_{year}_{args.provider}.jpg"
                    )

                    if not raw_jpg.exists():
                        missing += 1
                        continue

                    try:
                        formatter.format_location(raw_jpg, record, output_root)
                        formatted += 1
                    except Exception as e:
                        errors += 1
                        logging.error(
                            "Failed to format %s: %s", record.location_id, e
                        )

            summary_table.add_row(
                combo, str(formatted), str(missing), str(errors)
            )
            total_formatted += formatted
            total_missing += missing
            total_errors += errors

            if formatted > 0:
                console.print(
                    f"  [green]{combo}[/]: {formatted} formatted, "
                    f"{missing} missing, {errors} errors"
                )

    summary_table.add_section()
    summary_table.add_row(
        "[bold]Total[/]",
        f"[bold]{total_formatted}[/]",
        f"[bold]{total_missing}[/]",
        f"[bold]{total_errors}[/]",
    )

    console.print()
    console.print(summary_table)
    console.print(f"\n[bold green]Done! Formatted output saved to {fmt_dir}[/]")


if __name__ == "__main__":
    main()
