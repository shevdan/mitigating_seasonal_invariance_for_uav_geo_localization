"""Parser for the University-1652 dataset.

GPS coordinates come from per-building KML files downloaded from:
    https://drive.google.com/file/d/1PL8fVky9KZg7XESsuS5NCsYRyYAwui3S/view

Each building has its own KML file (e.g., first-key/0000.kml) containing
a Point with lon,lat,alt coordinates. Building IDs (0000-1651) map to
folder names in train/satellite/ and test/gallery_satellite/.
"""

import logging
from pathlib import Path

from PIL import Image

from seasonalgeo.models.schema import BBox, Dataset, GeoRecord
from seasonalgeo.parsers.base import BaseParser
from seasonalgeo.utils.geo import parse_kml_placemarks

logger = logging.getLogger(__name__)

# Default bbox half-size for University-1652 tiles.
# Tiles are 512x512 from Google Maps at approx zoom 18 (~0.6 m/px),
# giving ~300m ground extent per tile.
DEFAULT_BBOX_HALF_SIZE_M = 150
DEFAULT_ZOOM = 18


class University1652Parser(BaseParser):
    """Parse University-1652 dataset using per-building KML files."""

    def __init__(
        self,
        dataset_root: str | Path,
        kml_dir: str | Path | None = None,
        bbox_half_size_m: float = DEFAULT_BBOX_HALF_SIZE_M,
    ):
        super().__init__(dataset_root)
        self.bbox_half_size_m = bbox_half_size_m

        # Find KML directory
        if kml_dir is not None:
            self.kml_dir = Path(kml_dir)
        else:
            self.kml_dir = self._find_kml_dir()

        if self.kml_dir is None or not self.kml_dir.exists():
            raise FileNotFoundError(
                "KML coordinate directory not found.\n"
                "Download from: https://drive.google.com/file/d/"
                "1PL8fVky9KZg7XESsuS5NCsYRyYAwui3S/view\n"
                f"Extract to: {self.dataset_root}/first-key/"
            )

    def _find_kml_dir(self) -> Path | None:
        """Search for a directory containing per-building KML files."""
        # Check common locations
        for candidate in [
            self.dataset_root / "first-key",
            self.dataset_root / "kml",
        ]:
            if candidate.is_dir() and list(candidate.glob("*.kml")):
                logger.info("Found KML directory: %s", candidate)
                return candidate
        return None

    def parse(self) -> list[GeoRecord]:
        # Parse all per-building KML files
        coord_map = self._parse_all_kmls()
        logger.info("Parsed coordinates for %d buildings", len(coord_map))

        # Find all satellite tile folders
        tile_folders = self._find_satellite_folders()
        logger.info("Found %d satellite tile folders", len(tile_folders))

        # Build GeoRecords
        records = []
        matched = 0
        for folder_id, tile_path in sorted(tile_folders.items()):
            if folder_id not in coord_map:
                logger.debug("No coordinates for folder %s", folder_id)
                continue

            lat, lon = coord_map[folder_id]
            width, height = self._get_tile_size(tile_path)

            # Detect split from path: train/satellite/... or test/gallery_satellite/...
            rel = tile_path.relative_to(self.dataset_root)
            split = rel.parts[0] if rel.parts[0] in ("train", "test") else None

            records.append(GeoRecord(
                location_id=f"university1652_{folder_id}",
                dataset=Dataset.UNIVERSITY_1652,
                lat=lat,
                lon=lon,
                bbox=BBox.from_center(lat, lon, self.bbox_half_size_m),
                original_tile_path=str(tile_path),
                original_tile_width=width,
                original_tile_height=height,
                original_zoom_level=DEFAULT_ZOOM,
                split=split,
                metadata={"building_id": folder_id},
            ))
            matched += 1

        logger.info(
            "Matched %d / %d folders to coordinates", matched, len(tile_folders)
        )
        return self.validate(records)

    def _parse_all_kmls(self) -> dict[str, tuple[float, float]]:
        """Parse each per-building KML file to extract coordinates."""
        coord_map: dict[str, tuple[float, float]] = {}

        for kml_file in sorted(self.kml_dir.glob("*.kml")):
            building_id = kml_file.stem  # e.g., "0000"
            try:
                placemarks = parse_kml_placemarks(str(kml_file))
                if placemarks:
                    pm = placemarks[0]
                    coord_map[building_id] = (pm["lat"], pm["lon"])
            except Exception as e:
                logger.warning("Failed to parse %s: %s", kml_file.name, e)

        return coord_map

    def _find_satellite_folders(self) -> dict[str, Path]:
        """Find all satellite tile folders across train and test splits."""
        folders: dict[str, Path] = {}

        # Train satellite folders
        train_sat = self.dataset_root / "train" / "satellite"
        if train_sat.exists():
            for d in train_sat.iterdir():
                if d.is_dir() and d.name.isdigit():
                    tile = self._find_satellite_tile(d)
                    if tile:
                        folders[d.name] = tile

        # Test satellite folders
        test_sat = self.dataset_root / "test" / "gallery_satellite"
        if test_sat.exists():
            for d in test_sat.iterdir():
                if d.is_dir() and d.name.isdigit():
                    if d.name not in folders:
                        tile = self._find_satellite_tile(d)
                        if tile:
                            folders[d.name] = tile

        return folders

    def _find_satellite_tile(self, folder: Path) -> Path | None:
        """Find the satellite image file in a folder."""
        for ext in (".jpg", ".jpeg", ".png", ".tif"):
            candidates = list(folder.glob(f"*{ext}"))
            if candidates:
                return candidates[0]
        return None

    def _get_tile_size(self, tile_path: Path) -> tuple[int, int]:
        if tile_path.exists():
            try:
                with Image.open(tile_path) as img:
                    return img.size
            except Exception:
                pass
        return (512, 512)
