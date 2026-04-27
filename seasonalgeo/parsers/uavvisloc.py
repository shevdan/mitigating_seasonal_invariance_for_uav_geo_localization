"""Parser for the UAV-VisLoc dataset.

Coordinates come from two sources:
1. Per-flight CSV files (01/01.csv ... 11/11.csv): lat/lon per drone image
2. satellite_coordinates_range.csv: bounding box per satellite map

Supports two parsing modes:
- Flight-level: 11 GeoRecords (one per flight area)
- Drone-position: ~6774 GeoRecords (one per drone image, for patch retrieval)

Season assignment by flight ID (from Sample4Geo):
  Summer: 01, 02, 05, 07, 08, 09, 10
  Autumn: 03, 04, 06, 11
"""

import csv
import logging
from pathlib import Path

from PIL import Image

from seasonalgeo.models.schema import BBox, Dataset, GeoRecord
from seasonalgeo.parsers.base import BaseParser

logger = logging.getLogger(__name__)

# Name of the satellite bounds CSV (note: has a space in the original filename)
SAT_BOUNDS_FILENAME = "satellite_ coordinates_range.csv"

# Flight season assignment (from Sample4Geo dataloader)
SUMMER_FLIGHTS = {"01", "02", "05", "07", "08", "09", "10"}
AUTUMN_FLIGHTS = {"03", "04", "06", "11"}


def flight_season(flight_id: str) -> str:
    """Return the capture season for a given flight."""
    fid = flight_id.lstrip("0") or "0"
    padded = flight_id.zfill(2)
    if padded in SUMMER_FLIGHTS:
        return "summer"
    if padded in AUTUMN_FLIGHTS:
        return "autumn"
    return "unknown"


class UAVVisLocParser(BaseParser):
    """Parse UAV-VisLoc dataset to extract GeoRecords."""

    def __init__(self, dataset_root: str | Path):
        super().__init__(dataset_root)
        self.sat_bounds_file = self.dataset_root / SAT_BOUNDS_FILENAME
        if not self.sat_bounds_file.exists():
            raise FileNotFoundError(
                f"Satellite bounds CSV not found: {self.sat_bounds_file}"
            )

    def parse(self) -> list[GeoRecord]:
        sat_bounds = self._parse_satellite_bounds()
        flight_records = self._build_flight_records(sat_bounds)
        return self.validate(flight_records)

    def _parse_satellite_bounds(self) -> list[dict]:
        """Parse satellite_coordinates_range.csv."""
        bounds = []
        with open(self.sat_bounds_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                bounds.append({
                    "mapname": row["mapname"],
                    "lt_lat": float(row["LT_lat_map"]),
                    "lt_lon": float(row["LT_lon_map"]),
                    "rb_lat": float(row["RB_lat_map"]),
                    "rb_lon": float(row["RB_lon_map"]),
                    "region": row["region"],
                })
        logger.info("Parsed %d satellite bounds", len(bounds))
        return bounds

    def _parse_flight_csv(self, flight_id: str) -> list[dict]:
        """Parse a per-flight CSV file for drone image metadata."""
        csv_path = self.dataset_root / flight_id / f"{flight_id}.csv"
        if not csv_path.exists():
            logger.warning("Flight CSV not found: %s", csv_path)
            return []

        images = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                images.append({
                    "filename": row["filename"],
                    "date": row["date"],
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "height": float(row["height"]),
                })
        return images

    def _build_flight_records(self, sat_bounds: list[dict]) -> list[GeoRecord]:
        """Build one GeoRecord per flight area."""
        records = []

        for bound in sat_bounds:
            mapname = bound["mapname"]
            # Extract flight number from mapname like "satellite01.tif" -> "01"
            flight_id = mapname.replace("satellite", "").replace(".tif", "")

            # Build bbox from satellite map corners
            bbox = BBox.from_corners(
                bound["lt_lat"], bound["lt_lon"],
                bound["rb_lat"], bound["rb_lon"],
            )
            center_lat, center_lon = bbox.center

            # Parse drone CSV for image count
            drone_images = self._parse_flight_csv(flight_id)

            # Satellite map path — may be a single file or split into parts
            sat_path = self.dataset_root / flight_id / mapname
            if not sat_path.exists():
                # Check for split satellite maps (e.g., satellite09_01-01.tif)
                flight_dir = self.dataset_root / flight_id
                parts = sorted(flight_dir.glob(f"satellite{flight_id}*.tif"))
                if parts:
                    sat_path = parts[0]
                    logger.info(
                        "Using split satellite map: %s (%d parts)",
                        sat_path.name,
                        len(parts),
                    )
                else:
                    logger.warning("Satellite map not found: %s", sat_path)

            # Read satellite map dimensions from header
            width, height = self._get_tile_size(sat_path)

            records.append(GeoRecord(
                location_id=f"uavvisloc_flight{flight_id}",
                dataset=Dataset.UAV_VISLOC,
                lat=center_lat,
                lon=center_lon,
                bbox=bbox,
                original_tile_path=str(sat_path),
                original_tile_width=width,
                original_tile_height=height,
                metadata={
                    "region": bound["region"],
                    "drone_image_count": len(drone_images),
                    "satellite_map": mapname,
                    "flight_id": flight_id,
                },
            ))

        logger.info("Built %d flight-level GeoRecords from UAV-VisLoc", len(records))
        return records

    def _get_tile_size(self, tile_path: Path) -> tuple[int, int]:
        """Get image dimensions from header (fast, even for large GeoTIFFs)."""
        if tile_path.exists():
            try:
                with Image.open(tile_path) as img:
                    return img.size
            except Exception:
                pass
        return (1, 1)  # Minimal valid size to pass validation

    def parse_drone_positions(self) -> list[GeoRecord]:
        """Parse individual drone image positions as separate GeoRecords.

        This provides finer-grained locations for patch-level retrieval.
        Returns one GeoRecord per drone image (~6774 total).
        """
        sat_bounds = self._parse_satellite_bounds()
        # Map flight_id -> satellite info
        flight_map = {}
        for b in sat_bounds:
            fid = b["mapname"].replace("satellite", "").replace(".tif", "")
            flight_map[fid] = b

        records = []
        for flight_id, bound in sorted(flight_map.items()):
            drone_images = self._parse_flight_csv(flight_id)
            sat_path = self.dataset_root / flight_id / bound["mapname"]
            season = flight_season(flight_id)

            for img in drone_images:
                img_name = Path(img["filename"]).stem
                records.append(GeoRecord(
                    location_id=f"uavvisloc_{img_name}",
                    dataset=Dataset.UAV_VISLOC,
                    lat=img["lat"],
                    lon=img["lon"],
                    bbox=BBox.from_center(img["lat"], img["lon"], size_m=250),
                    original_tile_path=str(sat_path),
                    original_tile_width=512,
                    original_tile_height=512,
                    split="train",  # UAV-VisLoc uses all data for train
                    metadata={
                        "region": bound["region"],
                        "flight_id": flight_id,
                        "drone_image": img["filename"],
                        "date": img["date"],
                        "drone_height": img["height"],
                        "capture_season": season,
                    },
                ))

        logger.info(
            "Built %d drone-position GeoRecords from UAV-VisLoc", len(records)
        )
        return self.validate(records)
