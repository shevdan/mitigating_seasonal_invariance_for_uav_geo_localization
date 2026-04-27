"""Parser for the DenseUAV dataset.

GPS coordinates are stored in Dense_GPS_ALL.txt with format:
    train/satellite/000000/H80.tif E120.387... N30.324... 94.606

Each location folder has 6 entries (H80, H90, H100 + _old variants) sharing
the same coordinates. We deduplicate to one GeoRecord per location.
"""

import logging
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

from seasonalgeo.models.schema import BBox, Dataset, GeoRecord
from seasonalgeo.parsers.base import BaseParser

logger = logging.getLogger(__name__)

# Approx ground extent of a 512px DenseUAV satellite tile.
# DenseUAV tiles are Google Earth screenshots at ~zoom 19 (~0.3 m/px),
# so 512px covers ~150m. Half-extent = 75m.
DEFAULT_BBOX_HALF_SIZE_M = 75


class DenseUAVParser(BaseParser):
    """Parse DenseUAV dataset to extract GeoRecords."""

    def __init__(
        self,
        dataset_root: str | Path,
        gps_file: str = "Dense_GPS_ALL.txt",
        bbox_half_size_m: float = DEFAULT_BBOX_HALF_SIZE_M,
    ):
        super().__init__(dataset_root)
        self.gps_file = self.dataset_root / gps_file
        if not self.gps_file.exists():
            raise FileNotFoundError(f"GPS file not found: {self.gps_file}")
        self.bbox_half_size_m = bbox_half_size_m

    def parse(self) -> list[GeoRecord]:
        raw_entries = self._parse_gps_file()
        grouped = self._group_by_location(raw_entries)
        records = self._build_records(grouped)
        return self.validate(records)

    def _parse_gps_file(self) -> list[dict]:
        """Parse Dense_GPS_ALL.txt into raw entries."""
        entries = []
        with open(self.gps_file) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 4:
                    logger.warning("Malformed line %d: %s", line_num, line)
                    continue

                path_str, lon_str, lat_str, height_str = parts

                # Parse E<lon> N<lat>
                lon_match = re.match(r"E([\d.]+)", lon_str)
                lat_match = re.match(r"N([\d.]+)", lat_str)
                if not lon_match or not lat_match:
                    logger.warning("Cannot parse coords on line %d: %s", line_num, line)
                    continue

                entries.append({
                    "path": path_str,
                    "lon": float(lon_match.group(1)),
                    "lat": float(lat_match.group(1)),
                    "height": float(height_str),
                })

        logger.info("Parsed %d entries from %s", len(entries), self.gps_file.name)
        return entries

    def _group_by_location(self, entries: list[dict]) -> dict[str, list[dict]]:
        """Group entries by location folder (e.g., '000000')."""
        grouped: dict[str, list[dict]] = defaultdict(list)
        for e in entries:
            # Extract location ID from path like "train/satellite/000000/H80.tif"
            parts = Path(e["path"]).parts
            # Location folder is the second-to-last part
            if len(parts) >= 2:
                loc_id = parts[-2]
                grouped[loc_id].append(e)
        return grouped

    def _build_records(self, grouped: dict[str, list[dict]]) -> list[GeoRecord]:
        """Build one GeoRecord per unique location."""
        records = []
        for loc_id, entries in sorted(grouped.items()):
            # All entries for a location share coordinates — use the first
            entry = entries[0]
            lat, lon = entry["lat"], entry["lon"]

            # Find the canonical satellite tile (prefer H80.tif without _old)
            canonical = self._find_canonical_tile(entries)
            tile_path = self.dataset_root / canonical["path"]

            # Get tile dimensions
            width, height = self._get_tile_size(tile_path)

            split = Path(canonical["path"]).parts[0]  # "train" or "test"

            records.append(GeoRecord(
                location_id=f"denseuav_{loc_id}",
                dataset=Dataset.DENSE_UAV,
                lat=lat,
                lon=lon,
                bbox=BBox.from_center(lat, lon, self.bbox_half_size_m),
                original_tile_path=str(tile_path),
                original_tile_width=width,
                original_tile_height=height,
                split=split,
                metadata={
                    "height": entry["height"],
                    "num_variants": len(entries),
                },
            ))

        logger.info("Built %d unique GeoRecords from DenseUAV", len(records))
        return records

    def _find_canonical_tile(self, entries: list[dict]) -> dict:
        """Pick the canonical tile: prefer H80.tif (no _old suffix)."""
        for e in entries:
            fname = Path(e["path"]).name
            if fname == "H80.tif":
                return e
        # Fallback: first entry
        return entries[0]

    def _get_tile_size(self, tile_path: Path) -> tuple[int, int]:
        """Get image dimensions. Returns (512, 512) as default if file missing."""
        if tile_path.exists():
            try:
                with Image.open(tile_path) as img:
                    return img.size
            except Exception:
                pass
        return (512, 512)
