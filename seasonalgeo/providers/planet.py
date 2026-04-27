"""Planet Labs PlanetScope provider.

Supports two download methods:
- "tiles" (default): Uses Scene Tiles API — small 256×256 tile patches,
  charges against Scene Tiles quota (100k tiles/month). Best for small bboxes.
- "scene": Downloads full scene asset via Data API, charges against Scene
  Downloads quota (km²-based). Wastes quota for small AOIs.
"""

import datetime
import logging
import math
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds

from seasonalgeo.models.schema import BBox
from seasonalgeo.providers.base import BaseProvider

logger = logging.getLogger(__name__)

# PlanetScope defaults
ITEM_TYPE = "PSScene"
ASSET_TYPE = "ortho_analytic_4b_sr"  # 4-band surface reflectance
TILE_SIZE = 256  # Standard XYZ tile size in pixels
TILES_API_BASE = "https://tiles.planet.com/data/v1"


class QuotaTracker:
    """Track cumulative usage against monthly quota."""

    def __init__(
        self,
        monthly_limit_km2: float = 3000.0,
        monthly_limit_tiles: int = 100_000,
        warn_at_pct: float = 80.0,
    ):
        self.monthly_limit_km2 = monthly_limit_km2
        self.monthly_limit_tiles = monthly_limit_tiles
        self.warn_at_pct = warn_at_pct
        self._used_km2: float = 0.0
        self._used_tiles: int = 0

    def check(self, area_km2: float) -> bool:
        """Return True if area fits within remaining scene download quota."""
        return (self._used_km2 + area_km2) <= self.monthly_limit_km2

    def check_tiles(self, n_tiles: int) -> bool:
        """Return True if tiles fit within remaining tile quota."""
        return (self._used_tiles + n_tiles) <= self.monthly_limit_tiles

    def add(self, area_km2: float) -> None:
        self._used_km2 += area_km2
        pct = (self._used_km2 / self.monthly_limit_km2) * 100
        if pct >= self.warn_at_pct:
            logger.warning(
                "Scene download quota at %.1f%% (%.1f / %.0f km²)",
                pct, self._used_km2, self.monthly_limit_km2,
            )

    def add_tiles(self, n_tiles: int) -> None:
        self._used_tiles += n_tiles
        pct = (self._used_tiles / self.monthly_limit_tiles) * 100
        if pct >= self.warn_at_pct:
            logger.warning(
                "Tile quota at %.1f%% (%d / %d tiles)",
                pct, self._used_tiles, self.monthly_limit_tiles,
            )

    @property
    def used_km2(self) -> float:
        return self._used_km2

    @property
    def used_tiles(self) -> int:
        return self._used_tiles

    @property
    def remaining_km2(self) -> float:
        return max(0.0, self.monthly_limit_km2 - self._used_km2)

    @property
    def remaining_tiles(self) -> int:
        return max(0, self.monthly_limit_tiles - self._used_tiles)


def bbox_area_km2(bbox: BBox) -> float:
    """Approximate area of a bbox in km²."""
    return (bbox.width_m * bbox.height_m) / 1_000_000


# ---------------------------------------------------------------------------
# XYZ tile math helpers
# ---------------------------------------------------------------------------

def _lat_lon_to_tile_frac(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    """Convert lat/lon to fractional tile coordinates at given zoom."""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def _tile_to_lat_lon(tx: int, ty: int, zoom: int) -> tuple[float, float]:
    """Get lat/lon of the top-left corner of tile (tx, ty)."""
    n = 2 ** zoom
    lon = tx / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * ty / n))))
    return lat, lon


def _choose_zoom(bbox: BBox, native_res_m: float = 3.0) -> int:
    """Choose zoom level closest to the sensor's native resolution.

    PlanetScope native is 3m/px. We pick the zoom where tile pixel size
    is closest to native_res_m, so we get real detail without fake upsampling.
    """
    cos_lat = abs(math.cos(math.radians(bbox.center[0])))
    best_z = 15
    best_diff = float("inf")
    for z in range(12, 20):
        mpp = 156543.03 * cos_lat / (2 ** z)
        diff = abs(mpp - native_res_m)
        if diff < best_diff:
            best_diff = diff
            best_z = z
    return best_z


# ---------------------------------------------------------------------------
# PlanetProvider
# ---------------------------------------------------------------------------

class PlanetProvider(BaseProvider):
    """PlanetScope imagery via Planet Labs SDK.

    Uses best-scene selection (lowest cloud cover) instead of compositing.
    Default download method is "tiles" (Scene Tiles API) to preserve km² quota.
    """

    def __init__(self, config: dict):
        self.config = config
        self._pl = None  # planet.Planet sync client
        self._api_key: str | None = None
        self.quota = QuotaTracker(
            monthly_limit_km2=config.get("quota_limit_km2", 3000.0),
            monthly_limit_tiles=config.get("quota_limit_tiles", 100_000),
            warn_at_pct=config.get("warn_at_pct", 80.0),
        )
        # "tiles" (default) or "scene"
        self.download_method = config.get("download_method", "tiles")

    @property
    def provider_name(self) -> str:
        return "planet"

    def authenticate(self) -> None:
        """Authenticate with Planet API using API key."""
        import planet

        api_key = self.config.get("api_key") or os.environ.get("PL_API_KEY")
        if not api_key:
            raise ValueError(
                "Planet API key not found. Set PL_API_KEY env var "
                "or add api_key to planet config YAML."
            )

        self._api_key = api_key
        os.environ["PL_API_KEY"] = api_key
        self._pl = planet.Planet()
        logger.info("Planet API authenticated (download_method=%s)", self.download_method)

    def query(
        self,
        bbox: BBox,
        date_start: datetime.date,
        date_end: datetime.date,
        max_cloud_pct: float | None = None,
    ) -> list[dict]:
        """Search PlanetScope scenes within bbox and date range."""
        from planet import data_filter

        if max_cloud_pct is None:
            max_cloud_pct = self.config.get("max_cloud_cover", 30)

        item_type = self.config.get("item_type", ITEM_TYPE)
        search_limit = self.config.get("retrieval", {}).get("search_limit", 100)

        geojson_geom = {
            "type": "Polygon",
            "coordinates": [[
                [bbox.min_lon, bbox.min_lat],
                [bbox.max_lon, bbox.min_lat],
                [bbox.max_lon, bbox.max_lat],
                [bbox.min_lon, bbox.max_lat],
                [bbox.min_lon, bbox.min_lat],
            ]],
        }

        combined_filter = data_filter.and_filter([
            data_filter.permission_filter(),
            data_filter.geometry_filter(geojson_geom),
            data_filter.date_range_filter(
                "acquired",
                gte=datetime.datetime.combine(date_start, datetime.time.min),
                lte=datetime.datetime.combine(date_end, datetime.time.max),
            ),
            data_filter.range_filter("cloud_cover", lte=max_cloud_pct / 100.0),
        ])

        items = list(self._pl.data.search(
            [item_type],
            search_filter=combined_filter,
            limit=search_limit,
        ))

        # Sort by cloud_cover ascending
        items.sort(key=lambda x: x["properties"].get("cloud_cover", 1.0))

        logger.info(
            "Planet search: %d scenes found (max_cloud=%.0f%%)",
            len(items), max_cloud_pct,
        )
        return items

    def get_image_count(self, collection: list[dict]) -> int:
        return len(collection)

    @staticmethod
    def _bbox_coverage(scene: dict, bbox: BBox) -> float:
        """Compute fraction of bbox covered by scene geometry (0.0–1.0)."""
        from shapely.geometry import box, shape

        scene_geom = shape(scene["geometry"])
        bbox_geom = box(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)

        if bbox_geom.area == 0:
            return 0.0

        intersection = scene_geom.intersection(bbox_geom)
        return intersection.area / bbox_geom.area

    def composite(
        self,
        collection: list[dict],
        method: str = "best",
        bbox: BBox | None = None,
        min_coverage: float = 0.95,
    ) -> dict:
        """Select the best single scene (highest coverage, then lowest cloud).

        If bbox is provided, scenes are ranked by coverage of the bbox first,
        then by cloud_cover. Only scenes with coverage >= min_coverage are
        considered; if none qualify, the scene with highest coverage is used.
        """
        if not collection:
            raise ValueError("No scenes available for selection")

        if bbox is not None:
            # Annotate scenes with coverage
            for scene in collection:
                scene["_coverage"] = self._bbox_coverage(scene, bbox)

            # Filter to scenes that fully cover the bbox
            good = [s for s in collection if s["_coverage"] >= min_coverage]
            if good:
                # Among fully-covering scenes, pick lowest cloud cover
                good.sort(key=lambda x: x["properties"].get("cloud_cover", 1.0))
                best = good[0]
            else:
                # No scene fully covers bbox — pick highest coverage
                collection.sort(key=lambda x: -x["_coverage"])
                best = collection[0]
                logger.warning(
                    "No scene fully covers bbox (best coverage: %.1f%%). "
                    "Image may have black patches.",
                    best["_coverage"] * 100,
                )
        else:
            best = collection[0]  # Already sorted by cloud_cover

        cloud = best["properties"].get("cloud_cover", -1)
        coverage = best.get("_coverage")
        cov_str = f", coverage={coverage:.1f}%" if coverage is not None else ""
        logger.info(
            "Selected scene %s (cloud_cover=%.1f%%%s)",
            best["id"], cloud * 100, cov_str,
        )
        return best

    # ------------------------------------------------------------------
    # Export: route to tiles or full-scene download
    # ------------------------------------------------------------------

    def export_image(
        self,
        image: dict,
        bbox: BBox,
        output_path: Path,
        scale: float | None = None,
        bands: list[str] | None = None,
    ) -> Path:
        """Export image to GeoTIFF. Routes to tiles or full-scene method."""
        if self.download_method == "tiles":
            return self._export_via_tiles(image, bbox, output_path)
        else:
            return self._export_full_scene(image, bbox, output_path)

    # ------------------------------------------------------------------
    # Tiles-based export (uses Scene Tiles quota)
    # ------------------------------------------------------------------

    def _export_via_tiles(
        self,
        scene: dict,
        bbox: BBox,
        output_path: Path,
        zoom: int | None = None,
    ) -> Path:
        """Download XYZ tiles covering bbox, stitch, clip, save as GeoTIFF.

        Uses the Scene Tiles API which charges per tile (not per km²).
        Each tile is 256×256 pixels. Tiles are RGB uint8 (visual).
        """
        import requests

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        scene_id = scene["id"]
        item_type = self.config.get("item_type", ITEM_TYPE)

        if zoom is None:
            zoom = _choose_zoom(bbox)

        # Determine tile range covering the bbox
        fx_min, fy_min = _lat_lon_to_tile_frac(bbox.max_lat, bbox.min_lon, zoom)
        fx_max, fy_max = _lat_lon_to_tile_frac(bbox.min_lat, bbox.max_lon, zoom)

        tx_min, tx_max = int(fx_min), int(fx_max)
        ty_min, ty_max = int(fy_min), int(fy_max)

        n_tiles = (tx_max - tx_min + 1) * (ty_max - ty_min + 1)

        # Check tile quota
        if not self.quota.check_tiles(n_tiles):
            raise RuntimeError(
                f"Tile quota would be exceeded: need {n_tiles}, "
                f"remaining {self.quota.remaining_tiles}"
            )

        logger.info(
            "Downloading %d tiles at zoom %d for scene %s",
            n_tiles, zoom, scene_id,
        )

        # Download tiles
        session = requests.Session()
        session.headers["Authorization"] = f"api-key {self._api_key}"

        tiles: dict[tuple[int, int], np.ndarray] = {}
        fetched = 0
        for tx in range(tx_min, tx_max + 1):
            for ty in range(ty_min, ty_max + 1):
                url = f"{TILES_API_BASE}/{item_type}/{scene_id}/{zoom}/{tx}/{ty}.png"
                resp = session.get(url, timeout=30)
                resp.raise_for_status()
                fetched += 1
                if n_tiles > 10 and fetched % 10 == 0:
                    logger.info("  Tiles: %d/%d", fetched, n_tiles)
                tile_img = np.array(Image.open(BytesIO(resp.content)))
                tiles[(tx, ty)] = tile_img

        self.quota.add_tiles(n_tiles)

        # Stitch tiles into one image
        nx = tx_max - tx_min + 1
        ny = ty_max - ty_min + 1
        channels = 3  # RGB

        stitched = np.zeros((ny * TILE_SIZE, nx * TILE_SIZE, channels), dtype=np.uint8)
        for (tx, ty), img in tiles.items():
            xi = tx - tx_min
            yi = ty - ty_min
            h, w = img.shape[:2]
            # Tiles may be RGBA — take only RGB
            rgb = img[:, :, :3] if img.ndim == 3 and img.shape[2] >= 3 else img
            if rgb.ndim == 2:
                rgb = np.stack([rgb] * 3, axis=-1)
            stitched[
                yi * TILE_SIZE : yi * TILE_SIZE + h,
                xi * TILE_SIZE : xi * TILE_SIZE + w,
            ] = rgb[:h, :w, :channels]

        # Compute geographic bounds of stitched image
        stitch_lat_max, stitch_lon_min = _tile_to_lat_lon(tx_min, ty_min, zoom)
        stitch_lat_min, stitch_lon_max = _tile_to_lat_lon(tx_max + 1, ty_max + 1, zoom)

        stitch_h, stitch_w = stitched.shape[:2]

        # Clip to exact bbox (pixel coordinates within stitched image)
        px_left = (bbox.min_lon - stitch_lon_min) / (stitch_lon_max - stitch_lon_min) * stitch_w
        px_right = (bbox.max_lon - stitch_lon_min) / (stitch_lon_max - stitch_lon_min) * stitch_w
        # Y axis is inverted: top of image = max latitude
        py_top = (stitch_lat_max - bbox.max_lat) / (stitch_lat_max - stitch_lat_min) * stitch_h
        py_bottom = (stitch_lat_max - bbox.min_lat) / (stitch_lat_max - stitch_lat_min) * stitch_h

        px_left = max(0, int(px_left))
        px_right = min(stitch_w, int(math.ceil(px_right)))
        py_top = max(0, int(py_top))
        py_bottom = min(stitch_h, int(math.ceil(py_bottom)))

        clipped = stitched[py_top:py_bottom, px_left:px_right]

        if clipped.size == 0:
            raise ValueError(
                f"Empty clip result for scene {scene_id}. "
                f"Tile range: ({tx_min}-{tx_max}, {ty_min}-{ty_max}), zoom={zoom}"
            )

        logger.info(
            "Tiles stitched and clipped: %dx%d px (zoom %d, %d tiles)",
            clipped.shape[1], clipped.shape[0], zoom, n_tiles,
        )

        # Save as 3-band GeoTIFF in EPSG:4326
        transform = from_bounds(
            bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat,
            clipped.shape[1], clipped.shape[0],
        )
        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "width": clipped.shape[1],
            "height": clipped.shape[0],
            "count": 3,
            "crs": "EPSG:4326",
            "transform": transform,
        }
        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(3):
                dst.write(clipped[:, :, i], i + 1)

        return output_path

    # ------------------------------------------------------------------
    # Full-scene export (uses Scene Downloads quota — expensive!)
    # ------------------------------------------------------------------

    def _export_full_scene(
        self,
        image: dict,
        bbox: BBox,
        output_path: Path,
    ) -> Path:
        """Activate, download full scene, and clip to bbox.

        WARNING: Downloads entire scene (~24×8 km), charges full area
        against Scene Downloads quota. Use tiles method instead.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        scene_id = image["id"]
        asset_type = self.config.get("asset_type", ASSET_TYPE)
        item_type = self.config.get("item_type", ITEM_TYPE)

        # Check quota
        area_km2 = bbox_area_km2(bbox)
        if not self.quota.check(area_km2):
            raise RuntimeError(
                f"Quota would be exceeded: need {area_km2:.2f} km², "
                f"remaining {self.quota.remaining_km2:.2f} km²"
            )

        # Get asset, activate, wait for ready
        logger.info("Activating asset %s/%s...", scene_id, asset_type)
        asset = self._pl.data.get_asset(
            item_type_id=item_type,
            item_id=scene_id,
            asset_type_id=asset_type,
        )
        self._pl.data.activate_asset(asset)

        timeout = self.config.get("activation_timeout_s", 600)
        poll = self.config.get("activation_poll_interval_s", 10)
        max_attempts = max(1, timeout // poll)

        def _log_wait(status):
            logger.info("  Waiting for activation... (status: %s)", status)

        asset = self._pl.data.wait_asset(
            asset, delay=poll, max_attempts=max_attempts, callback=_log_wait,
        )
        logger.info("Asset active, downloading %s...", scene_id)

        # Download full scene to output directory
        tmp_path = self._pl.data.download_asset(
            asset,
            directory=str(output_path.parent),
            overwrite=True,
            progress_bar=True,
        )
        tmp_path = Path(tmp_path)

        # Clip to bbox
        logger.info("Clipping to bbox...")
        self._clip_to_bbox(tmp_path, output_path, bbox)

        # Clean up full scene if different from output
        if tmp_path != output_path and tmp_path.exists():
            tmp_path.unlink()

        # Track quota (NOTE: Planet actually charges for full scene area,
        # not just the bbox, but we only track the bbox here)
        self.quota.add(area_km2)

        return output_path

    # ------------------------------------------------------------------
    # RGB JPEG export
    # ------------------------------------------------------------------

    def export_rgb_jpeg(
        self,
        geotiff_path: Path,
        output_path: Path,
        quality: int | None = None,
    ) -> Path:
        """Generate RGB JPEG from a downloaded PlanetScope GeoTIFF."""
        if quality is None:
            quality = self.config.get("export", {}).get("jpg_quality", 95)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(geotiff_path) as src:
            band_count = src.count

            if band_count == 3:
                # Tiles output: already RGB uint8
                red = src.read(1).astype(np.float32)
                green = src.read(2).astype(np.float32)
                blue = src.read(3).astype(np.float32)
            elif band_count >= 4:
                # Full-scene SR output: [Blue, Green, Red, NIR]
                red = src.read(3).astype(np.float32)
                green = src.read(2).astype(np.float32)
                blue = src.read(1).astype(np.float32)
            else:
                raise ValueError(f"Unexpected band count: {band_count}")

        def normalize(band):
            valid = band[band > 0]
            if valid.size == 0:
                return np.zeros_like(band, dtype=np.uint8)
            lo = np.percentile(valid, 2)
            hi = np.percentile(valid, 98)
            if hi <= lo:
                hi = lo + 1
            band = np.clip(band, lo, hi)
            return ((band - lo) / (hi - lo) * 255).astype(np.uint8)

        rgb = np.stack([normalize(red), normalize(green), normalize(blue)], axis=-1)
        img = Image.fromarray(rgb)
        img.save(str(output_path), "JPEG", quality=quality)

        logger.debug("Saved JPEG: %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Stats / NDVI
    # ------------------------------------------------------------------

    def compute_ndvi(self, geotiff_path: Path) -> float | None:
        """Compute mean NDVI from a PlanetScope GeoTIFF.

        Only works for full-scene SR downloads (4 bands with NIR).
        Returns None for tiles-based downloads (RGB only, no NIR).
        """
        with rasterio.open(geotiff_path) as src:
            if src.count < 4:
                return None
            nir = src.read(4).astype(np.float64)
            red = src.read(3).astype(np.float64)

        denominator = nir + red
        valid = denominator > 0
        ndvi = np.where(valid, (nir - red) / denominator, np.nan)
        return float(np.nanmean(ndvi))

    def get_scene_stats(self, scene_metadata: dict, geotiff_path: Path) -> dict:
        """Compute stats for a downloaded scene."""
        ndvi_mean = self.compute_ndvi(geotiff_path)

        with rasterio.open(geotiff_path) as src:
            data = src.read(1)
            valid_count = int(np.count_nonzero(data > 0))

        props = scene_metadata.get("properties", {})
        return {
            "cloud_cover": props.get("cloud_cover", 0) * 100,
            "ndvi_mean": ndvi_mean,
            "valid_pixel_count": valid_count,
            "scene_id": scene_metadata.get("id"),
            "acquired": props.get("acquired"),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clip_to_bbox(src_path: Path, dst_path: Path, bbox: BBox) -> None:
        """Clip a GeoTIFF to the given bbox using rasterio.

        Reprojects bbox from EPSG:4326 to the raster's CRS (typically UTM)
        before computing the clip window.
        """
        from pyproj import Transformer
        from rasterio.windows import from_bounds as win_from_bounds

        with rasterio.open(src_path) as src:
            transformer = Transformer.from_crs(
                "EPSG:4326", src.crs, always_xy=True,
            )
            min_x, min_y = transformer.transform(bbox.min_lon, bbox.min_lat)
            max_x, max_y = transformer.transform(bbox.max_lon, bbox.max_lat)

            window = win_from_bounds(
                min_x, min_y, max_x, max_y,
                transform=src.transform,
            )
            window = window.round_offsets().round_lengths()
            data = src.read(window=window)
            transform = src.window_transform(window)
            profile = src.profile.copy()
            profile.update(
                width=data.shape[2],
                height=data.shape[1],
                transform=transform,
            )

        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data)
