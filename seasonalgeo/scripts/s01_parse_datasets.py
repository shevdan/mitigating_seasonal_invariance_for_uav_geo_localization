#!/usr/bin/env python3
"""Parse all available datasets and export unified GeoRecords.

Usage:
    seasonalgeo-parse --data-dir data/
    seasonalgeo-parse --data-dir data/ --output-dir output/georecords/
    seasonalgeo-parse --data-dir data/ --datasets denseuav uavvisloc

    # Or run as module:
    python -m seasonalgeo.scripts.s01_parse_datasets --data-dir data/
"""

import argparse
import logging
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

from seasonalgeo.models.schema import GeoRecord
from seasonalgeo.parsers.denseuav import DenseUAVParser
from seasonalgeo.parsers.uavvisloc import UAVVisLocParser
from seasonalgeo.parsers.university1652 import University1652Parser
from seasonalgeo.parsers.sues200 import SUES200Parser

console = Console()

# Dataset name -> (directory name in data/, parser class)
DATASET_REGISTRY = {
    "denseuav": ("DenseUAV", DenseUAVParser),
    "uavvisloc": ("UAV_VisLoc_dataset", UAVVisLocParser),
    "university1652": ("University-Release", University1652Parser),
    "sues200": ("SUES-200-512x512-V2", SUES200Parser),
}


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def parse_dataset(
    name: str, data_dir: Path, **kwargs
) -> list[GeoRecord]:
    """Parse a single dataset and return GeoRecords."""
    dir_name, parser_cls = DATASET_REGISTRY[name]
    dataset_path = data_dir / dir_name

    if not dataset_path.exists():
        console.print(f"[yellow]Skipping {name}: directory not found at {dataset_path}[/]")
        return []

    try:
        parser = parser_cls(dataset_path, **kwargs)
        records = parser.parse()
        return records
    except FileNotFoundError as e:
        console.print(f"[yellow]Skipping {name}: {e}[/]")
        return []
    except Exception as e:
        console.print(f"[red]Error parsing {name}: {e}[/]")
        logging.exception("Failed to parse %s", name)
        return []


def print_summary(all_records: dict[str, list[GeoRecord]]) -> None:
    """Print a summary table of parsed records."""
    table = Table(title="Dataset Parsing Summary")
    table.add_column("Dataset", style="cyan")
    table.add_column("Records", justify="right", style="green")
    table.add_column("Lat Range", style="white")
    table.add_column("Lon Range", style="white")
    table.add_column("Status", style="bold")

    total = 0
    for name, records in all_records.items():
        n = len(records)
        total += n
        if n == 0:
            table.add_row(name, "0", "-", "-", "[yellow]SKIPPED[/]")
        else:
            lats = [r.lat for r in records]
            lons = [r.lon for r in records]
            lat_range = f"{min(lats):.4f} — {max(lats):.4f}"
            lon_range = f"{min(lons):.4f} — {max(lons):.4f}"
            table.add_row(name, str(n), lat_range, lon_range, "[green]OK[/]")

    table.add_section()
    table.add_row("[bold]Total[/]", f"[bold]{total}[/]", "", "", "")

    console.print()
    console.print(table)
    console.print()


def main():
    parser = argparse.ArgumentParser(description="Parse UAV geo-localization datasets")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root directory containing dataset folders (default: data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for CSV/JSON exports (default: <data-dir>/georecords/)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_REGISTRY.keys()),
        default=None,
        help="Specific datasets to parse (default: all available)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "both"],
        default="both",
        help="Export format (default: both)",
    )
    parser.add_argument(
        "--uavvisloc-drone-positions", action="store_true",
        help="Parse UAV-VisLoc at drone-position level (~6774 records) instead of flight-level (11)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir / "georecords").resolve()

    datasets_to_parse = args.datasets or list(DATASET_REGISTRY.keys())

    console.print(f"[bold]SeasonalGeo Dataset Parser[/]")
    console.print(f"Data directory: {data_dir}")
    console.print(f"Output directory: {output_dir}")
    console.print(f"Datasets: {', '.join(datasets_to_parse)}")
    console.print()

    # Parse each dataset
    all_records: dict[str, list[GeoRecord]] = {}
    unified: list[GeoRecord] = []

    for name in datasets_to_parse:
        console.print(f"[bold cyan]Parsing {name}...[/]")
        if name == "uavvisloc" and args.uavvisloc_drone_positions:
            console.print("  [dim](drone-position mode: one record per drone image)[/]")
            # First check if dataset exists via flight-level parse
            flight_records = parse_dataset(name, data_dir)
            if flight_records:
                dir_name, parser_cls = DATASET_REGISTRY[name]
                parser_inst = parser_cls(data_dir / dir_name)
                records = parser_inst.parse_drone_positions()
            else:
                records = []
        else:
            records = parse_dataset(name, data_dir)
        all_records[name] = records
        unified.extend(records)

    # Print summary
    print_summary(all_records)

    # Export
    if unified:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use any parser instance for export methods
        from seasonalgeo.parsers.base import BaseParser

        class _Exporter(BaseParser):
            def parse(self):
                return []

        exporter = _Exporter.__new__(_Exporter)

        if args.format in ("csv", "both"):
            csv_path = output_dir / "all_georecords.csv"
            exporter.to_csv(unified, csv_path)
            console.print(f"Exported CSV: {csv_path}")

            # Per-dataset CSVs
            for name, records in all_records.items():
                if records:
                    exporter.to_csv(records, output_dir / f"{name}.csv")

        if args.format in ("json", "both"):
            json_path = output_dir / "all_georecords.json"
            exporter.to_json(unified, json_path)
            console.print(f"Exported JSON: {json_path}")

        console.print(f"\n[bold green]Done! {len(unified)} total GeoRecords exported.[/]")
    else:
        console.print("[yellow]No records to export.[/]")


if __name__ == "__main__":
    main()
