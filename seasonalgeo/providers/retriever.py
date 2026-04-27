"""Seasonal imagery retrieval orchestrator."""

import datetime
import logging
from pathlib import Path

from seasonalgeo.models.schema import (
    GeoRecord,
    Provider,
    Season,
    SeasonalImage,
)
from seasonalgeo.output.writer import build_filename, build_output_dir, write_location_metadata
from seasonalgeo.providers.base import BaseProvider
from seasonalgeo.utils.seasons import get_season_dates

logger = logging.getLogger(__name__)

PROVIDER_ENUM_MAP = {
    "sentinel2": Provider.SENTINEL2,
    "planet": Provider.PLANET,
    "landsat8": Provider.LANDSAT8,
    "arcgis": Provider.ARCGIS,
}


class SeasonalRetriever:
    """Orchestrates seasonal satellite imagery retrieval for GeoRecords."""

    def __init__(
        self,
        provider: BaseProvider,
        config: dict,
        output_dir: str | Path,
    ):
        self.provider = provider
        self.config = config
        self.output_dir = Path(output_dir)

    def retrieve_location(
        self,
        geo_record: GeoRecord,
        seasons: list[Season] | None = None,
        years: list[int] | None = None,
    ) -> list[SeasonalImage]:
        """Retrieve seasonal imagery for a single location.

        Returns list of SeasonalImage for each successfully retrieved image.
        """
        if seasons is None:
            seasons = list(Season)
        if years is None:
            yr = self.config.get("year_range", [2023, 2023])
            years = list(range(yr[0], yr[1] + 1))

        loc_dir = build_output_dir(self.output_dir, geo_record)
        results: list[SeasonalImage] = []

        for year in years:
            for season in seasons:
                result = self._retrieve_single(geo_record, season, year, loc_dir)
                if result is not None:
                    results.append(result)

        # Write metadata
        if results:
            write_location_metadata(loc_dir, geo_record, results)

        return results

    @staticmethod
    def _valid_pixel_ratio(tif_path: Path) -> float:
        """Compute fraction of non-black pixels in a GeoTIFF."""
        import numpy as np
        import rasterio

        with rasterio.open(tif_path) as src:
            # Sum across all bands — pixel is black if all bands are 0
            data = src.read()  # (bands, H, W)
            pixel_sum = data.sum(axis=0)  # (H, W)
            total = pixel_sum.size
            valid = int(np.count_nonzero(pixel_sum))
        return valid / total if total > 0 else 0.0

    def _retrieve_single(
        self,
        geo_record: GeoRecord,
        season: Season,
        year: int,
        loc_dir: Path,
    ) -> SeasonalImage | None:
        """Retrieve a single seasonal image for one location."""
        start_date, end_date = get_season_dates(season, year)
        pname = self.provider.provider_name
        label = f"{geo_record.location_id} {season.value} {year}"

        # Resume support: skip if both tif and jpg already exist
        tif_check = loc_dir / build_filename(season, year, pname, "tif")
        jpg_check = loc_dir / build_filename(season, year, pname, "jpg")
        if tif_check.exists() and jpg_check.exists():
            logger.debug("Skipping %s — already exists", label)
            return None

        try:
            # Query collection
            collection = self.provider.query(
                geo_record.bbox, start_date, end_date
            )
            count = self.provider.get_image_count(collection)

            if count == 0:
                logger.info("No images for %s — skipping", label)
                return None

            logger.info("Found %d images for %s", count, label)

            # Composite (Sentinel-2) or best-scene selection (Planet)
            # Pass bbox for coverage-aware selection (Planet)
            if pname == "planet":
                selected = self.provider.composite(
                    collection, bbox=geo_record.bbox,
                )
            else:
                selected = self.provider.composite(collection)

            # Export GeoTIFF
            tif_name = build_filename(season, year, pname, "tif")
            tif_path = loc_dir / tif_name
            self.provider.export_image(
                selected, geo_record.bbox, tif_path,
                bands=self.config.get("bands_all"),
            )

            # For Planet: validate coverage and retry with next-best scene
            min_valid_ratio = self.config.get("min_valid_pixel_ratio", 0.85)
            if pname == "planet" and count > 1:
                valid_ratio = self._valid_pixel_ratio(tif_path)
                tried = {selected["id"]}
                # Try other scenes if coverage is too low
                remaining = [s for s in collection if s["id"] not in tried]
                # Sort remaining by coverage desc, then cloud_cover asc
                remaining.sort(
                    key=lambda s: (
                        -s.get("_coverage", 0),
                        s["properties"].get("cloud_cover", 1.0),
                    )
                )
                max_retries = min(4, len(remaining))
                retry = 0
                while valid_ratio < min_valid_ratio and retry < max_retries:
                    candidate = remaining[retry]
                    retry += 1
                    logger.info(
                        "Scene %s has %.0f%% valid pixels (< %.0f%%), "
                        "trying scene %s...",
                        selected["id"], valid_ratio * 100,
                        min_valid_ratio * 100, candidate["id"],
                    )
                    selected = candidate
                    self.provider.export_image(
                        selected, geo_record.bbox, tif_path,
                        bands=self.config.get("bands_all"),
                    )
                    valid_ratio = self._valid_pixel_ratio(tif_path)

                if valid_ratio < min_valid_ratio:
                    logger.warning(
                        "%s: best valid pixel ratio is %.0f%% "
                        "(tried %d scenes)",
                        label, valid_ratio * 100, len(tried) + retry,
                    )

            # Export RGB JPEG and compute stats — provider-specific
            jpg_name = build_filename(season, year, pname, "jpg")
            jpg_path = loc_dir / jpg_name
            metadata = {"source_image_count": count}

            if pname == "planet":
                self.provider.export_rgb_jpeg(tif_path, jpg_path)
                stats = self.provider.get_scene_stats(selected, tif_path)
                cloud_cover = stats.get("cloud_cover", 0)
                metadata["scene_id"] = stats.get("scene_id")
                metadata["acquired"] = stats.get("acquired")
            elif pname == "arcgis":
                self.provider.export_rgb_jpeg(tif_path, jpg_path)
                stats = self.provider.get_composite_stats(selected, geo_record.bbox)
                cloud_cover = 0
                metadata["release_id"] = stats.get("release_id")
                metadata["release_date"] = stats.get("release_date")
                metadata["capture_date"] = stats.get("capture_date")
                metadata["resolution_m"] = stats.get("resolution_m")
                metadata["source_name"] = stats.get("source_name")
            else:
                self.provider.export_rgb_jpeg(selected, geo_record.bbox, jpg_path)
                stats = self.provider.get_composite_stats(selected, geo_record.bbox)
                cloud_cover = 0

            provider_enum = PROVIDER_ENUM_MAP.get(pname, Provider.SENTINEL2)

            return SeasonalImage(
                location_id=geo_record.location_id,
                season=season,
                year=year,
                provider=provider_enum,
                file_path=str(jpg_path),
                geotiff_path=str(tif_path),
                cloud_cover_pct=cloud_cover,
                ndvi_mean=stats.get("ndvi_mean"),
                pixel_count=stats.get("valid_pixel_count", 0),
                retrieval_date=datetime.datetime.now(),
                composite_start=start_date,
                composite_end=end_date,
                metadata=metadata,
            )

        except Exception as e:
            logger.error("Failed to retrieve %s: %s", label, e, exc_info=True)
            return None
