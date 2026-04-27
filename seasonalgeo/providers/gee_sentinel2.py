"""Google Earth Engine Sentinel-2 L2A provider."""

import datetime
import io
import logging
import time
from pathlib import Path
from typing import Any

import ee
import numpy as np
import requests
from PIL import Image

from seasonalgeo.models.schema import BBox
from seasonalgeo.providers.base import BaseProvider

logger = logging.getLogger(__name__)

COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"

# SCL band values for masking
SCL_CLOUD = [8, 9, 10]   # cloud_medium, cloud_high, cirrus
SCL_SHADOW = [3]          # cloud_shadow
SCL_MASK_VALUES = SCL_CLOUD + SCL_SHADOW


class Sentinel2Provider(BaseProvider):
    """Sentinel-2 L2A imagery via Google Earth Engine."""

    def __init__(self, config: dict):
        self.config = config
        self._initialized = False

    @property
    def provider_name(self) -> str:
        return "sentinel2"

    def authenticate(self) -> None:
        """Initialize GEE with service account credentials."""
        gee_cfg = self.config.get("gee", {})
        project_id = gee_cfg.get("project_id")
        service_account = gee_cfg.get("service_account")
        key_file = gee_cfg.get("key_file")

        if service_account and key_file:
            key_path = Path(key_file).expanduser()
            credentials = ee.ServiceAccountCredentials(service_account, str(key_path))
            ee.Initialize(credentials, project=project_id)
        elif project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()

        self._initialized = True
        logger.info("GEE authenticated (project: %s)", project_id or "default")

    def query(
        self,
        bbox: BBox,
        date_start: datetime.date,
        date_end: datetime.date,
        max_cloud_pct: float | None = None,
    ) -> ee.ImageCollection:
        """Query Sentinel-2 L2A collection filtered by bbox, date, and cloud cover."""
        if max_cloud_pct is None:
            max_cloud_pct = self.config.get("max_cloud_cover_metadata", 30)

        geometry = ee.Geometry.Rectangle([
            bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat
        ])

        collection = (
            ee.ImageCollection(COLLECTION_ID)
            .filterBounds(geometry)
            .filterDate(str(date_start), str(date_end))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        )

        return collection

    def get_image_count(self, collection: ee.ImageCollection) -> int:
        """Get number of images in collection (makes a server call)."""
        return collection.size().getInfo()

    def composite(
        self, collection: ee.ImageCollection, method: str | None = None
    ) -> ee.Image:
        """Create a cloud-masked temporal composite."""
        if method is None:
            method = self.config.get("composite_method", "median")

        # Apply cloud mask to each image
        masked = collection.map(self._apply_cloud_mask)

        if method == "median":
            return masked.median()
        elif method == "mean":
            return masked.mean()
        elif method == "mosaic":
            return masked.mosaic()
        else:
            raise ValueError(f"Unknown composite method: {method}")

    def compute_ndvi(self, image: ee.Image) -> ee.Image:
        """Compute NDVI: (B8 - B4) / (B8 + B4)."""
        return image.normalizedDifference(["B8", "B4"]).rename("NDVI")

    def export_image(
        self,
        image: ee.Image,
        bbox: BBox,
        output_path: Path,
        scale: float | None = None,
        bands: list[str] | None = None,
    ) -> Path:
        """Download image via getDownloadURL and save as GeoTIFF.

        Fast for small tiles (~50x50 px at 10m/px for 500m bbox).
        """
        if scale is None:
            scale = self.config.get("scale", 10)
        if bands is None:
            bands = self.config.get("bands_rgb", ["B4", "B3", "B2"])

        geometry = ee.Geometry.Rectangle([
            bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat
        ])

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Select bands and clip to region
        export_img = image.select(bands).clip(geometry)

        url = export_img.getDownloadURL({
            "region": geometry,
            "scale": scale,
            "format": "GEO_TIFF",
        })

        response = self._download_with_retry(url)
        output_path.write_bytes(response.content)

        logger.debug("Saved GeoTIFF: %s (%d bytes)", output_path, len(response.content))
        return output_path

    def export_rgb_jpeg(
        self,
        image: ee.Image,
        bbox: BBox,
        output_path: Path,
        scale: float | None = None,
        quality: int | None = None,
    ) -> Path:
        """Download and save as RGB JPEG with visualization parameters."""
        if scale is None:
            scale = self.config.get("scale", 10)
        if quality is None:
            quality = self.config.get("export", {}).get("jpg_quality", 95)

        bands = self.config.get("bands_rgb", ["B4", "B3", "B2"])
        geometry = ee.Geometry.Rectangle([
            bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat
        ])

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Visualize: scale Sentinel-2 reflectance (0-10000) to 0-255
        vis_img = image.select(bands).clip(geometry).visualize(
            min=0, max=3000, bands=bands
        )

        url = vis_img.getDownloadURL({
            "region": geometry,
            "scale": scale,
            "format": "PNG",
        })

        response = self._download_with_retry(url)

        # Convert PNG to JPEG
        png_img = Image.open(io.BytesIO(response.content)).convert("RGB")
        png_img.save(str(output_path), "JPEG", quality=quality)

        logger.debug("Saved JPEG: %s", output_path)
        return output_path

    def get_composite_stats(
        self, composite: ee.Image, bbox: BBox, scale: float | None = None
    ) -> dict:
        """Compute statistics for a composite over a bbox.

        Returns dict with cloud_pct, ndvi_mean, valid_pixel_count.
        """
        if scale is None:
            scale = self.config.get("scale", 10)

        geometry = ee.Geometry.Rectangle([
            bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat
        ])

        # NDVI
        ndvi = self.compute_ndvi(composite)
        ndvi_stats = ndvi.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
            geometry=geometry,
            scale=scale,
            maxPixels=1e6,
        ).getInfo()

        return {
            "ndvi_mean": ndvi_stats.get("NDVI_mean"),
            "valid_pixel_count": ndvi_stats.get("NDVI_count", 0),
        }

    @staticmethod
    def _apply_cloud_mask(image: ee.Image) -> ee.Image:
        """Mask clouds and shadows using the SCL band."""
        scl = image.select("SCL")
        # Create mask: 1 where pixel is NOT cloud/shadow
        mask = scl.neq(3)  # cloud_shadow
        for val in [8, 9, 10]:  # cloud_medium, cloud_high, cirrus
            mask = mask.And(scl.neq(val))
        return image.updateMask(mask)

    def _download_with_retry(self, url: str) -> requests.Response:
        """Download with retry logic."""
        retry_count = self.config.get("retrieval", {}).get("retry_count", 3)
        retry_delay = self.config.get("retrieval", {}).get("retry_delay_s", 30)

        for attempt in range(retry_count + 1):
            try:
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                return response
            except (requests.RequestException, Exception) as e:
                if attempt < retry_count:
                    logger.warning(
                        "Download failed (attempt %d/%d): %s. Retrying in %ds...",
                        attempt + 1, retry_count, e, retry_delay,
                    )
                    time.sleep(retry_delay)
                else:
                    raise
