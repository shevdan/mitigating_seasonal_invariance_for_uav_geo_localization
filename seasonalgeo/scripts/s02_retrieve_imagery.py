#!/usr/bin/env python3
"""Retrieve seasonal satellite imagery for parsed GeoRecords.

Usage:
    # Sentinel-2 (default): 3 locations, summer 2023
    seasonalgeo-retrieve --data-dir data/ --num-locations 3 --seasons summer --years 2023

    # Planet Labs PlanetScope: 3 locations, summer 2023
    seasonalgeo-retrieve --provider planet --data-dir data/ --num-locations 3 --seasons summer --years 2023

    # All seasons, single year
    seasonalgeo-retrieve --data-dir data/ --num-locations 5 --years 2023

    # Specific dataset
    seasonalgeo-retrieve --data-dir data/ --dataset denseuav --num-locations 10

    # Or run as module:
    python -m seasonalgeo.scripts.s02_retrieve_imagery --provider planet --data-dir data/ --num-locations 3
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

# Force unbuffered output so conda run doesn't swallow progress
os.environ["PYTHONUNBUFFERED"] = "1"

import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from seasonalgeo.models.schema import BBox, Dataset, GeoRecord, Season
from seasonalgeo.providers.retriever import SeasonalRetriever

console = Console()

SEASON_MAP = {
    "spring": Season.SPRING,
    "summer": Season.SUMMER,
    "autumn": Season.AUTUMN,
    "winter": Season.WINTER,
}


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    # Suppress noisy HTTP logging from Planet SDK and httpx
    for noisy in ("httpx", "httpcore", "planet.http", "planet_auth"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_georecords(csv_path: Path, dataset: str | None = None) -> list[GeoRecord]:
    """Load GeoRecords from CSV exported by Phase 1."""
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


def create_provider(provider_name: str, config: dict):
    """Instantiate the appropriate provider based on name."""
    if provider_name == "planet":
        from seasonalgeo.providers.planet import PlanetProvider
        return PlanetProvider(config)
    elif provider_name == "arcgis":
        from seasonalgeo.providers.arcgis import ArcGISProvider
        return ArcGISProvider(config)
    else:
        from seasonalgeo.providers.gee_sentinel2 import Sentinel2Provider
        return Sentinel2Provider(config)


def main():
    parser = argparse.ArgumentParser(description="Retrieve seasonal satellite imagery")
    parser.add_argument(
        "--provider", type=str, default="sentinel2",
        choices=["sentinel2", "planet", "arcgis"],
        help="Satellite imagery provider (default: sentinel2)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Root data directory (default: data/)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Provider config file (default: configs/{provider}.yaml)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=["denseuav", "uavvisloc", "university1652"],
        help="Filter to specific dataset",
    )
    parser.add_argument(
        "--num-locations", type=int, default=3,
        help="Number of locations to process (default: 3)",
    )
    parser.add_argument(
        "--seasons", nargs="+", default=None,
        choices=["spring", "summer", "autumn", "winter"],
        help="Seasons to retrieve (default: all)",
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=None,
        help="Years to retrieve (default: from config)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Resolve config path (defaults to configs/{provider}.yaml in package)
    if args.config is not None:
        config_path = args.config.resolve()
    else:
        config_path = (
            Path(__file__).resolve().parent.parent
            / "configs" / f"{args.provider}.yaml"
        )

    if not config_path.exists():
        console.print(f"[red]Config not found: {config_path}[/]")
        sys.exit(1)
    config = load_config(config_path)

    # Output directory
    output_dir = (args.output_dir or Path("output")).resolve()

    # Parse season args
    seasons = None
    if args.seasons:
        seasons = [SEASON_MAP[s] for s in args.seasons]

    # Parse year args
    years = args.years
    if years is None:
        yr = config.get("year_range", [2023, 2023])
        years = list(range(yr[0], yr[1] + 1))

    # Load GeoRecords
    csv_path = args.data_dir / "georecords" / "all_georecords.csv"
    if not csv_path.exists():
        console.print(f"[red]GeoRecords CSV not found: {csv_path}[/]")
        console.print("Run [bold]seasonalgeo-parse[/] first.")
        sys.exit(1)

    all_records = load_georecords(csv_path, args.dataset)
    records = all_records[: args.num_locations]

    provider_labels = {
        "planet": "Planet PlanetScope",
        "arcgis": "ArcGIS World Imagery",
        "sentinel2": "Sentinel-2",
    }
    provider_label = provider_labels.get(args.provider, args.provider)
    console.print(f"[bold]SeasonalGeo {provider_label} Retrieval[/]")
    console.print(f"Config: {config_path}")
    console.print(f"Output: {output_dir}")
    console.print(f"Locations: {len(records)} / {len(all_records)}")
    console.print(f"Seasons: {', '.join(s.value for s in (seasons or list(Season)))}")
    console.print(f"Years: {years}")
    console.print()

    # Authenticate
    provider = create_provider(args.provider, config)
    console.print(f"[bold cyan]Authenticating with {provider_label}...[/]")
    provider.authenticate()

    # Show quota info for Planet
    if args.provider == "planet":
        quota = provider.quota
        method = provider.download_method
        if method == "tiles":
            console.print(
                f"[yellow]Planet tiles quota: {quota.remaining_tiles:,} tiles remaining "
                f"(limit: {quota.monthly_limit_tiles:,}/month)[/]"
            )
        else:
            console.print(
                f"[yellow]Planet scene quota: {quota.remaining_km2:.0f} km² remaining "
                f"(limit: {quota.monthly_limit_km2:.0f} km²/month)[/]"
            )
        console.print(f"[dim]Download method: {method}[/]")
        console.print()

    # Create retriever
    retriever = SeasonalRetriever(provider, config, output_dir)

    # Process locations
    total_images = 0
    results_summary = []

    for i, record in enumerate(records, 1):
        console.print(
            f"\n[bold cyan][{i}/{len(records)}] {record.location_id}[/] "
            f"({record.lat:.4f}, {record.lon:.4f})"
        )

        images = retriever.retrieve_location(record, seasons=seasons, years=years)
        total_images += len(images)

        results_summary.append({
            "location_id": record.location_id,
            "dataset": record.dataset.value,
            "images": len(images),
            "seasons": [img.season.value for img in images],
        })

        label = "seasonal scenes" if args.provider == "planet" else "seasonal composites"
        console.print(f"  Retrieved {len(images)} {label}")

    # Print summary
    table = Table(title="Retrieval Summary")
    table.add_column("Location", style="cyan")
    table.add_column("Dataset", style="white")
    table.add_column("Images", justify="right", style="green")
    table.add_column("Seasons", style="white")

    for r in results_summary:
        table.add_row(
            r["location_id"],
            r["dataset"],
            str(r["images"]),
            ", ".join(r["seasons"]) or "-",
        )

    table.add_section()
    table.add_row("[bold]Total[/]", "", f"[bold]{total_images}[/]", "")

    console.print()
    console.print(table)

    # Show final quota for Planet
    if args.provider == "planet":
        quota = provider.quota
        method = provider.download_method
        if method == "tiles":
            console.print(
                f"\n[yellow]Planet tiles used this run: {quota.used_tiles} "
                f"(remaining: {quota.remaining_tiles:,})[/]"
            )
        else:
            console.print(
                f"\n[yellow]Planet scene quota used this run: {quota.used_km2:.2f} km² "
                f"(remaining: {quota.remaining_km2:.0f} km²)[/]"
            )

    console.print(f"\n[bold green]Done! Output saved to {output_dir}[/]")


if __name__ == "__main__":
    main()
