"""ArcGIS World Imagery provider.

Downloads high-resolution satellite tiles from Esri's World Imagery service.
Supports temporal snapshots via the Wayback archive (releases since Feb 2014).

Seasonal selection strategy:
  1. Scan Wayback releases to discover distinct capture dates for the location
  2. Match capture dates to the target season (by month)
  3. Prefer the most recent capture in the target season
  4. Fallback: if no capture in target season, use the closest season available

Tile URL (latest):
    https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}
Wayback tile URL:
    https://wayback.maptiles.arcgis.com/.../MapServer/tile/{release}/{z}/{y}/{x}
Per-release metadata:
    Each release has its own metadata service with resolution-specific layers.
    Layer 4 = 30cm, Layer 5 = 60cm, Layer 6 = 1.2m, etc.
"""

import datetime
import logging
import math
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds

from seasonalgeo.models.schema import BBox, Season
from seasonalgeo.providers.base import BaseProvider

logger = logging.getLogger(__name__)

TILE_SIZE = 256  # Standard XYZ tile size

# Season month mapping
MONTH_TO_SEASON = {
    1: Season.WINTER, 2: Season.WINTER,
    3: Season.SPRING, 4: Season.SPRING, 5: Season.SPRING,
    6: Season.SUMMER, 7: Season.SUMMER, 8: Season.SUMMER,
    9: Season.AUTUMN, 10: Season.AUTUMN, 11: Season.AUTUMN,
    12: Season.WINTER,
}

# Ordered for fallback proximity (e.g., if spring missing, try summer then autumn)
SEASON_FALLBACK_ORDER = {
    Season.SPRING: [Season.SUMMER, Season.AUTUMN, Season.WINTER],
    Season.SUMMER: [Season.SPRING, Season.AUTUMN, Season.WINTER],
    Season.AUTUMN: [Season.SUMMER, Season.WINTER, Season.SPRING],
    Season.WINTER: [Season.AUTUMN, Season.SPRING, Season.SUMMER],
}

# Resolution layers in the metadata service (try most detailed first)
METADATA_LAYERS = [4, 5, 6, 7, 8]  # 30cm, 60cm, 1.2m, 2.4m, 4.8m


# ---------------------------------------------------------------------------
# XYZ tile math (Web Mercator / EPSG:3857)
# ---------------------------------------------------------------------------

def _lat_lon_to_tile_frac(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    """Convert lat/lon to fractional tile coordinates."""
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


# ---------------------------------------------------------------------------
# Wayback release + capture info
# ---------------------------------------------------------------------------

@dataclass
class WaybackRelease:
    """A single Wayback snapshot release."""
    release_num: int
    title: str
    release_id: str
    date: datetime.date | None
    metadata_url: str | None = None


@dataclass
class CaptureInfo:
    """Actual imagery capture metadata for a location in a specific release."""
    capture_date: datetime.date
    season: Season
    resolution_m: float | None
    source_name: str | None
    release_num: int
    release_id: str
    release_date: datetime.date | None


# ---------------------------------------------------------------------------
# ArcGISProvider
# ---------------------------------------------------------------------------

class ArcGISProvider(BaseProvider):
    """ArcGIS World Imagery provider with seasonal Wayback selection.

    Downloads XYZ tiles from Esri's World Imagery service.
    Scans Wayback releases to find imagery captured in the target season.
    Falls back to the closest available season if the target isn't available.
    """

    def __init__(self, config: dict):
        self.config = config
        self._session = None
        self._wayback_releases: list[WaybackRelease] | None = None
        self._wayback_config: dict | None = None  # raw JSON for metadata URLs
        # Cache: (rounded_lat, rounded_lon) → list[CaptureInfo]
        self._capture_cache: dict[tuple[float, float], list[CaptureInfo]] = {}

    @property
    def provider_name(self) -> str:
        return "arcgis"

    def authenticate(self) -> None:
        """No authentication needed for ArcGIS World Imagery tiles."""
        import requests

        self._session = requests.Session()
        self._session.headers["User-Agent"] = (
            "SeasonalGeo/1.0 (academic research; UCU thesis)"
        )
        logger.info("ArcGIS provider ready (no auth required)")

    # ------------------------------------------------------------------
    # Wayback releases
    # ------------------------------------------------------------------

    def _load_wayback_releases(self) -> list[WaybackRelease]:
        """Fetch and parse the Wayback config to get all release snapshots."""
        if self._wayback_releases is not None:
            return self._wayback_releases

        url = self.config.get(
            "wayback_config_url",
            "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json",
        )
        resp = self._session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        self._wayback_config = data

        releases = []
        for release_num_str, item in data.items():
            title = item.get("itemTitle", "")
            layer_id = item.get("layerIdentifier", "")
            meta_url = item.get("metadataLayerUrl")

            # Parse date from title: "World Imagery (Wayback YYYY-MM-DD)"
            date = None
            if "Wayback " in title:
                date_str = title.split("Wayback ")[-1].rstrip(")")
                try:
                    date = datetime.date.fromisoformat(date_str)
                except ValueError:
                    pass

            releases.append(WaybackRelease(
                release_num=int(release_num_str),
                title=title,
                release_id=layer_id,
                date=date,
                metadata_url=meta_url,
            ))

        # Sort by date descending (newest first)
        releases.sort(key=lambda r: r.date or datetime.date.min, reverse=True)
        self._wayback_releases = releases
        logger.info("Loaded %d Wayback releases (newest: %s)",
                     len(releases), releases[0].date if releases else "none")
        return releases

    # ------------------------------------------------------------------
    # Per-release metadata: discover actual capture dates
    # ------------------------------------------------------------------

    def _query_release_metadata(
        self, release: WaybackRelease, bbox: BBox,
    ) -> CaptureInfo | None:
        """Query a specific release's metadata service for capture date."""
        if not release.metadata_url:
            return None

        center = bbox.center
        point_geom = f"{center[1]},{center[0]}"  # lon,lat

        for layer_id in METADATA_LAYERS:
            query_url = f"{release.metadata_url}/{layer_id}/query"
            params = {
                "where": "1=1",
                "geometry": point_geom,
                "geometryType": "esriGeometryPoint",
                "inSR": "4326",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "SRC_DATE2,SRC_RES,NICE_NAME",
                "returnGeometry": "false",
                "f": "json",
                "resultRecordCount": "1",
            }
            try:
                resp = self._session.get(query_url, params=params, timeout=10)
                data = resp.json()
                features = data.get("features", [])
                if features:
                    attrs = features[0]["attributes"]
                    capture_ms = attrs.get("SRC_DATE2")
                    if not capture_ms:
                        continue
                    capture_date = datetime.datetime.fromtimestamp(
                        capture_ms / 1000, tz=datetime.timezone.utc
                    ).date()
                    season = MONTH_TO_SEASON[capture_date.month]
                    return CaptureInfo(
                        capture_date=capture_date,
                        season=season,
                        resolution_m=attrs.get("SRC_RES"),
                        source_name=attrs.get("NICE_NAME"),
                        release_num=release.release_num,
                        release_id=release.release_id,
                        release_date=release.date,
                    )
            except Exception:
                continue
        return None

    def _discover_captures(self, bbox: BBox) -> list[CaptureInfo]:
        """Scan Wayback releases to find all distinct capture dates for a location.

        Uses a sparse scan strategy: check releases at intervals, then fill in
        around transition points. This minimizes API calls (~10-20 instead of 100+).
        """
        # Check cache (round to ~100m grid to share across nearby locations)
        cache_key = (round(bbox.center[0], 3), round(bbox.center[1], 3))
        if cache_key in self._capture_cache:
            return self._capture_cache[cache_key]

        releases = self._load_wayback_releases()
        # Filter to releases with dates
        dated = [r for r in releases if r.date is not None]

        # Sparse scan: check every Nth release to find distinct captures
        step = max(1, len(dated) // 15)  # ~15 samples across all releases
        sample_indices = list(range(0, len(dated), step))
        # Always include first and last
        if len(dated) - 1 not in sample_indices:
            sample_indices.append(len(dated) - 1)

        seen_dates: set[datetime.date] = set()
        captures: list[CaptureInfo] = []

        logger.info(
            "Scanning %d Wayback releases (sampling %d) for capture dates...",
            len(dated), len(sample_indices),
        )

        for idx in sample_indices:
            rel = dated[idx]
            info = self._query_release_metadata(rel, bbox)
            if info and info.capture_date not in seen_dates:
                seen_dates.add(info.capture_date)
                captures.append(info)
                logger.debug(
                    "  Found capture: %s (%s) in release %s",
                    info.capture_date, info.season.value, rel.release_id,
                )

        # If we found transitions, refine around them to find exact boundaries
        # (and potentially more captures we missed between samples)
        if len(captures) > 1 and step > 2:
            # Check midpoints between samples where captures changed
            extra_indices = []
            for i in range(len(sample_indices) - 1):
                idx_a, idx_b = sample_indices[i], sample_indices[i + 1]
                info_a = self._query_release_metadata(dated[idx_a], bbox)
                info_b = self._query_release_metadata(dated[idx_b], bbox)
                if (info_a and info_b
                        and info_a.capture_date != info_b.capture_date):
                    mid = (idx_a + idx_b) // 2
                    if mid not in sample_indices:
                        extra_indices.append(mid)

            for idx in extra_indices:
                rel = dated[idx]
                info = self._query_release_metadata(rel, bbox)
                if info and info.capture_date not in seen_dates:
                    seen_dates.add(info.capture_date)
                    captures.append(info)

        # Sort by capture date descending (most recent first)
        captures.sort(key=lambda c: c.capture_date, reverse=True)

        logger.info(
            "Found %d distinct captures: %s",
            len(captures),
            ", ".join(f"{c.capture_date} ({c.season.value})" for c in captures),
        )

        self._capture_cache[cache_key] = captures
        return captures

    # ------------------------------------------------------------------
    # Query / composite interface (BaseProvider)
    # ------------------------------------------------------------------

    def query(
        self,
        bbox: BBox,
        date_start: datetime.date,
        date_end: datetime.date,
        max_cloud_pct: float | None = None,
    ) -> list[dict]:
        """Find Wayback releases with imagery matching the target season.

        Scans releases for actual capture dates, then returns candidates
        ordered by: (1) target season match, (2) most recent capture.
        """
        captures = self._discover_captures(bbox)

        # Determine target season from date range
        mid_date = date_start + (date_end - date_start) // 2
        target_season = MONTH_TO_SEASON[mid_date.month]

        # Separate into: exact match, then fallback seasons
        exact = [c for c in captures if c.season == target_season]
        fallback_order = SEASON_FALLBACK_ORDER[target_season]

        # Build ordered candidate list: exact matches first, then fallbacks
        ordered: list[CaptureInfo] = list(exact)
        for fb_season in fallback_order:
            ordered.extend(c for c in captures if c.season == fb_season)

        if not ordered:
            logger.warning("No captures found for bbox at (%.4f, %.4f)",
                           bbox.center[0], bbox.center[1])
            return []

        # Log what we found
        if exact:
            logger.info(
                "ArcGIS: %d captures in %s, using %s (captured %s)",
                len(exact), target_season.value,
                exact[0].capture_date, exact[0].season.value,
            )
        else:
            best = ordered[0]
            logger.warning(
                "No %s capture found. Falling back to %s (captured %s)",
                target_season.value, best.season.value, best.capture_date,
            )

        # Convert to dicts for the retriever interface
        results = []
        for cap in ordered:
            results.append({
                "release_num": cap.release_num,
                "release_id": cap.release_id,
                "release_date": cap.release_date.isoformat() if cap.release_date else None,
                "capture_date": cap.capture_date.isoformat(),
                "capture_season": cap.season.value,
                "target_season": target_season.value,
                "is_target_season": cap.season == target_season,
                "resolution_m": cap.resolution_m,
                "source_name": cap.source_name,
            })

        return results

    def get_image_count(self, collection: list[dict]) -> int:
        return len(collection)

    def composite(self, collection: list[dict], method: str = "best") -> dict:
        """Select the best Wayback release.

        Prefers releases where imagery was captured in the target season.
        Among those, picks the most recent capture date.
        """
        if not collection:
            raise ValueError("No Wayback releases available")

        # collection is already ordered: target season first, then fallbacks
        best = collection[0]

        if not best.get("is_target_season"):
            logger.warning(
                "Using %s imagery (captured %s) instead of %s — "
                "saving under requested season label",
                best["capture_season"], best["capture_date"],
                best["target_season"],
            )

        logger.info(
            "Selected release %s: captured %s (%s), resolution %.2fm",
            best["release_id"],
            best["capture_date"],
            best["capture_season"],
            best.get("resolution_m") or 0,
        )
        return best

    # ------------------------------------------------------------------
    # Tile download, stitch, clip
    # ------------------------------------------------------------------

    def export_image(
        self,
        image: dict,
        bbox: BBox,
        output_path: Path,
        scale: float | None = None,
        bands: list[str] | None = None,
    ) -> Path:
        """Download ArcGIS tiles, stitch, clip to bbox, save as GeoTIFF."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        zoom = self.config.get("zoom", 19)
        release_num = image.get("release_num")
        delay = self.config.get("request_delay_s", 0.05)

        # Use Wayback URL if we have a release, otherwise latest imagery
        if release_num:
            base_url = self.config.get(
                "wayback_tile_url",
                "https://wayback.maptiles.arcgis.com/arcgis/rest/services/"
                "World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/{release}/{z}/{y}/{x}",
            )
        else:
            base_url = self.config.get(
                "tile_url",
                "https://services.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}",
            )

        # Determine tile range
        fx_min, fy_min = _lat_lon_to_tile_frac(bbox.max_lat, bbox.min_lon, zoom)
        fx_max, fy_max = _lat_lon_to_tile_frac(bbox.min_lat, bbox.max_lon, zoom)

        tx_min, tx_max = int(fx_min), int(fx_max)
        ty_min, ty_max = int(fy_min), int(fy_max)
        n_tiles = (tx_max - tx_min + 1) * (ty_max - ty_min + 1)

        logger.info(
            "Downloading %d ArcGIS tiles at zoom %d (release: %s)",
            n_tiles, zoom, image.get("release_id", "latest"),
        )

        tiles, blank_count = self._download_tiles(
            base_url, release_num, zoom,
            tx_min, tx_max, ty_min, ty_max, delay,
        )

        if blank_count > 0:
            logger.warning(
                "%d/%d tiles were blank at zoom %d", blank_count, n_tiles, zoom,
            )

        # If all tiles blank at primary zoom, try fallback zoom
        fallback_zoom = self.config.get("fallback_zoom")
        if blank_count == n_tiles and fallback_zoom and fallback_zoom != zoom:
            logger.info("All tiles blank at zoom %d, retrying at zoom %d",
                        zoom, fallback_zoom)
            zoom = fallback_zoom
            fx_min, fy_min = _lat_lon_to_tile_frac(bbox.max_lat, bbox.min_lon, zoom)
            fx_max, fy_max = _lat_lon_to_tile_frac(bbox.min_lat, bbox.max_lon, zoom)
            tx_min, tx_max = int(fx_min), int(fx_max)
            ty_min, ty_max = int(fy_min), int(fy_max)

            tiles, blank_count = self._download_tiles(
                base_url, release_num, zoom,
                tx_min, tx_max, ty_min, ty_max, delay,
            )

        # Stitch, clip, save
        return self._stitch_and_save(
            tiles, bbox, zoom, tx_min, tx_max, ty_min, ty_max,
            n_tiles, output_path,
        )

    def _download_tiles(
        self,
        base_url: str,
        release_num: int | None,
        zoom: int,
        tx_min: int, tx_max: int,
        ty_min: int, ty_max: int,
        delay: float,
    ) -> tuple[dict[tuple[int, int], np.ndarray], int]:
        """Download XYZ tiles. Returns (tiles_dict, blank_count)."""
        tiles: dict[tuple[int, int], np.ndarray] = {}
        blank_count = 0
        n_tiles = (tx_max - tx_min + 1) * (ty_max - ty_min + 1)
        fetched = 0

        for ty in range(ty_min, ty_max + 1):
            for tx in range(tx_min, tx_max + 1):
                if release_num:
                    url = base_url.format(release=release_num, z=zoom, y=ty, x=tx)
                else:
                    url = base_url.format(z=zoom, y=ty, x=tx)

                resp = self._session.get(url, timeout=30)
                fetched += 1

                if resp.status_code in (404, 204):
                    blank_count += 1
                    tiles[(tx, ty)] = np.zeros(
                        (TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8,
                    )
                else:
                    resp.raise_for_status()
                    tile_img = np.array(
                        Image.open(BytesIO(resp.content)).convert("RGB"),
                    )
                    tiles[(tx, ty)] = tile_img

                if delay > 0:
                    time.sleep(delay)
                if n_tiles > 10 and fetched % 10 == 0:
                    logger.info("  Tiles: %d/%d", fetched, n_tiles)

        return tiles, blank_count

    def _stitch_and_save(
        self,
        tiles: dict[tuple[int, int], np.ndarray],
        bbox: BBox,
        zoom: int,
        tx_min: int, tx_max: int,
        ty_min: int, ty_max: int,
        n_tiles: int,
        output_path: Path,
    ) -> Path:
        """Stitch tiles, clip to bbox, save as GeoTIFF."""
        nx = tx_max - tx_min + 1
        ny = ty_max - ty_min + 1
        stitched = np.zeros((ny * TILE_SIZE, nx * TILE_SIZE, 3), dtype=np.uint8)

        for (tx, ty), img in tiles.items():
            xi = tx - tx_min
            yi = ty - ty_min
            h, w = img.shape[:2]
            stitched[
                yi * TILE_SIZE: yi * TILE_SIZE + h,
                xi * TILE_SIZE: xi * TILE_SIZE + w,
            ] = img[:h, :w, :3]

        # Compute geographic bounds of stitched image
        stitch_lat_max, stitch_lon_min = _tile_to_lat_lon(tx_min, ty_min, zoom)
        stitch_lat_min, stitch_lon_max = _tile_to_lat_lon(tx_max + 1, ty_max + 1, zoom)
        stitch_h, stitch_w = stitched.shape[:2]

        # Clip to exact bbox
        px_left = (bbox.min_lon - stitch_lon_min) / (stitch_lon_max - stitch_lon_min) * stitch_w
        px_right = (bbox.max_lon - stitch_lon_min) / (stitch_lon_max - stitch_lon_min) * stitch_w
        py_top = (stitch_lat_max - bbox.max_lat) / (stitch_lat_max - stitch_lat_min) * stitch_h
        py_bottom = (stitch_lat_max - bbox.min_lat) / (stitch_lat_max - stitch_lat_min) * stitch_h

        px_left = max(0, int(px_left))
        px_right = min(stitch_w, int(math.ceil(px_right)))
        py_top = max(0, int(py_top))
        py_bottom = min(stitch_h, int(math.ceil(py_bottom)))

        clipped = stitched[py_top:py_bottom, px_left:px_right]

        if clipped.size == 0:
            raise ValueError(
                f"Empty clip result. Tile range: ({tx_min}-{tx_max}, "
                f"{ty_min}-{ty_max}), zoom={zoom}"
            )

        logger.info(
            "Stitched and clipped: %dx%d px (zoom %d, %d tiles)",
            clipped.shape[1], clipped.shape[0], zoom, n_tiles,
        )

        # Save as 3-band GeoTIFF (EPSG:4326)
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
    # RGB JPEG export
    # ------------------------------------------------------------------

    def export_rgb_jpeg(
        self,
        geotiff_path: Path,
        output_path: Path,
        quality: int | None = None,
    ) -> Path:
        """Generate RGB JPEG from ArcGIS GeoTIFF (already RGB uint8)."""
        if quality is None:
            quality = self.config.get("export", {}).get("jpg_quality", 95)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(geotiff_path) as src:
            red = src.read(1)
            green = src.read(2)
            blue = src.read(3)

        rgb = np.stack([red, green, blue], axis=-1)
        img = Image.fromarray(rgb)
        img.save(str(output_path), "JPEG", quality=quality)

        logger.debug("Saved JPEG: %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_composite_stats(self, image: dict, bbox: BBox) -> dict:
        """Return stats for the downloaded image."""
        return {
            "cloud_cover": 0,
            "ndvi_mean": None,
            "valid_pixel_count": 0,
            "release_id": image.get("release_id"),
            "release_date": image.get("release_date"),
            "capture_date": image.get("capture_date"),
            "capture_season": image.get("capture_season"),
            "target_season": image.get("target_season"),
            "is_target_season": image.get("is_target_season"),
            "resolution_m": image.get("resolution_m"),
            "source_name": image.get("source_name"),
        }
