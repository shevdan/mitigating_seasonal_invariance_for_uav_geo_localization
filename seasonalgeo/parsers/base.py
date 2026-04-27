"""Abstract base parser for dataset coordinate extraction."""

import csv
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from seasonalgeo.models.schema import GeoRecord

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """Base class for all dataset parsers.

    Subclasses must implement `parse()` to extract GeoRecords from a dataset.
    """

    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root)
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")

    @abstractmethod
    def parse(self) -> list[GeoRecord]:
        """Parse dataset and return list of GeoRecords."""
        ...

    def validate(self, records: list[GeoRecord]) -> list[GeoRecord]:
        """Filter out invalid records and log warnings."""
        valid = []
        seen_ids = set()

        for r in records:
            # Check for duplicate IDs
            if r.location_id in seen_ids:
                logger.warning("Duplicate location_id: %s — skipping", r.location_id)
                continue
            seen_ids.add(r.location_id)

            # Check coordinate ranges
            if not (-90 <= r.lat <= 90):
                logger.warning(
                    "Invalid latitude %.6f for %s — skipping", r.lat, r.location_id
                )
                continue
            if not (-180 <= r.lon <= 180):
                logger.warning(
                    "Invalid longitude %.6f for %s — skipping", r.lon, r.location_id
                )
                continue

            # Check tile dimensions
            if r.original_tile_width <= 0 or r.original_tile_height <= 0:
                logger.warning(
                    "Invalid tile size %dx%d for %s — skipping",
                    r.original_tile_width,
                    r.original_tile_height,
                    r.location_id,
                )
                continue

            valid.append(r)

        n_dropped = len(records) - len(valid)
        if n_dropped > 0:
            logger.info("Validation: dropped %d / %d records", n_dropped, len(records))

        return valid

    def to_csv(self, records: list[GeoRecord], output_path: str | Path) -> None:
        """Export GeoRecords to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "location_id",
            "dataset",
            "lat",
            "lon",
            "bbox_min_lat",
            "bbox_min_lon",
            "bbox_max_lat",
            "bbox_max_lon",
            "original_tile_path",
            "original_tile_width",
            "original_tile_height",
            "original_zoom_level",
            "split",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow({
                    "location_id": r.location_id,
                    "dataset": r.dataset.value,
                    "lat": r.lat,
                    "lon": r.lon,
                    "bbox_min_lat": r.bbox.min_lat,
                    "bbox_min_lon": r.bbox.min_lon,
                    "bbox_max_lat": r.bbox.max_lat,
                    "bbox_max_lon": r.bbox.max_lon,
                    "original_tile_path": r.original_tile_path,
                    "original_tile_width": r.original_tile_width,
                    "original_tile_height": r.original_tile_height,
                    "original_zoom_level": r.original_zoom_level,
                    "split": r.split or "",
                })

        logger.info("Exported %d records to %s", len(records), output_path)

    def to_json(self, records: list[GeoRecord], output_path: str | Path) -> None:
        """Export GeoRecords to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [r.to_dict() for r in records]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Exported %d records to %s", len(records), output_path)
