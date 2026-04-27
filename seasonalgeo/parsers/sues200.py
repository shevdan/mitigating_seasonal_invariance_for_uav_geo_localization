"""Parser for the SUES-200 dataset.

SUES-200 contains 200 locations around Shanghai University of Engineering
Science campus but does NOT include GPS coordinates in the dataset files.

This parser requires an external coordinate mapping file (CSV) with format:
    location_id,lat,lon

If no coordinate file is provided, the parser can still enumerate the dataset
structure but will not produce valid GeoRecords for GEE retrieval.
"""

import csv
import logging
from pathlib import Path

from PIL import Image

from seasonalgeo.models.schema import BBox, Dataset, GeoRecord
from seasonalgeo.parsers.base import BaseParser

logger = logging.getLogger(__name__)

# Default bbox half-size for SUES-200 tiles (512x512 from satellite view).
DEFAULT_BBOX_HALF_SIZE_M = 200


class SUES200Parser(BaseParser):
    """Parse SUES-200 dataset.

    Requires an external coordinate CSV file mapping location IDs to lat/lon.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        coords_csv: str | Path | None = None,
        bbox_half_size_m: float = DEFAULT_BBOX_HALF_SIZE_M,
    ):
        super().__init__(dataset_root)
        self.bbox_half_size_m = bbox_half_size_m

        # The actual image directory may be nested
        self.image_root = self._find_image_root()

        # Load external coordinates if provided
        self.coord_map: dict[str, tuple[float, float]] = {}
        if coords_csv is not None:
            self.coord_map = self._load_coords_csv(Path(coords_csv))
        else:
            # Search for a coords file in the dataset root
            for candidate in [
                self.dataset_root / "coordinates.csv",
                self.dataset_root / "coords.csv",
                self.dataset_root / "sues200_coords.csv",
                self.image_root / "coordinates.csv" if self.image_root else None,
            ]:
                if candidate and candidate.exists():
                    self.coord_map = self._load_coords_csv(candidate)
                    break

    def _find_image_root(self) -> Path:
        """Find the directory containing satellite-view/ and drone_view_512/."""
        # Check for nested structure: SUES-200-512x512-V2/SUES-200-512x512/
        for candidate in [
            self.dataset_root / "SUES-200-512x512",
            self.dataset_root,
        ]:
            if (candidate / "satellite-view").exists():
                return candidate

        # Try one level deeper
        for child in self.dataset_root.iterdir():
            if child.is_dir() and (child / "satellite-view").exists():
                return child

        return self.dataset_root

    def parse(self) -> list[GeoRecord]:
        if not self.coord_map:
            logger.warning(
                "No coordinate mapping file found for SUES-200.\n"
                "This dataset does not include GPS coordinates.\n"
                "To use this parser, create a CSV file with columns: "
                "location_id,lat,lon\n"
                "and pass it via coords_csv parameter.\n"
                "The SUES campus is approximately at 31.17°N, 121.19°E."
            )
            return []

        # Find all location folders
        locations = self._find_locations()
        logger.info("Found %d SUES-200 locations", len(locations))

        records = []
        for loc_id, sat_tile in sorted(locations.items()):
            if loc_id not in self.coord_map:
                logger.debug("No coordinates for location %s", loc_id)
                continue

            lat, lon = self.coord_map[loc_id]
            width, height = self._get_tile_size(sat_tile)

            records.append(GeoRecord(
                location_id=f"sues200_{loc_id}",
                dataset=Dataset.SUES_200,
                lat=lat,
                lon=lon,
                bbox=BBox.from_center(lat, lon, self.bbox_half_size_m),
                original_tile_path=str(sat_tile),
                original_tile_width=width,
                original_tile_height=height,
                metadata={"scene_id": loc_id},
            ))

        logger.info(
            "Built %d GeoRecords from SUES-200 (%d locations had coordinates)",
            len(records),
            len(records),
        )
        return self.validate(records)

    def _find_locations(self) -> dict[str, Path]:
        """Find all location folders and their satellite tiles."""
        sat_dir = self.image_root / "satellite-view"
        if not sat_dir.exists():
            logger.error("satellite-view directory not found in %s", self.image_root)
            return {}

        locations: dict[str, Path] = {}
        for d in sorted(sat_dir.iterdir()):
            if d.is_dir() and d.name.isdigit():
                # Find satellite image (e.g., 0.png)
                tiles = list(d.glob("*.png")) + list(d.glob("*.jpg"))
                if tiles:
                    locations[d.name] = tiles[0]

        return locations

    def _load_coords_csv(self, csv_path: Path) -> dict[str, tuple[float, float]]:
        """Load location_id -> (lat, lon) mapping from CSV."""
        coord_map: dict[str, tuple[float, float]] = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                loc_id = row["location_id"].strip()
                lat = float(row["lat"])
                lon = float(row["lon"])
                coord_map[loc_id] = (lat, lon)

        logger.info(
            "Loaded %d coordinate mappings from %s", len(coord_map), csv_path
        )
        return coord_map

    def _get_tile_size(self, tile_path: Path) -> tuple[int, int]:
        if tile_path.exists():
            try:
                with Image.open(tile_path) as img:
                    return img.size
            except Exception:
                pass
        return (512, 512)

    def enumerate_locations(self) -> list[str]:
        """List all location IDs (works without coordinates)."""
        locations = self._find_locations()
        return sorted(locations.keys())
