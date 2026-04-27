"""Core data models for SeasonalGeo pipeline."""

from dataclasses import dataclass, field, asdict
from enum import Enum
from math import cos, radians
from typing import Optional
import datetime
import json


class Dataset(Enum):
    UNIVERSITY_1652 = "university1652"
    SUES_200 = "sues200"
    DENSE_UAV = "denseuav"
    UAV_VISLOC = "uavvisloc"


class Season(Enum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"


class Provider(Enum):
    SENTINEL2 = "sentinel2"
    LANDSAT8 = "landsat8"
    PLANET = "planet"
    ARCGIS = "arcgis"


@dataclass
class BBox:
    """Geographic bounding box in WGS84."""

    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.min_lat + self.max_lat) / 2,
            (self.min_lon + self.max_lon) / 2,
        )

    @property
    def width_m(self) -> float:
        """Approximate east-west extent in meters."""
        center_lat = (self.min_lat + self.max_lat) / 2
        return (self.max_lon - self.min_lon) * 111_320 * abs(cos(radians(center_lat)))

    @property
    def height_m(self) -> float:
        """Approximate north-south extent in meters."""
        return (self.max_lat - self.min_lat) * 111_320

    @classmethod
    def from_center(cls, lat: float, lon: float, size_m: float = 500) -> "BBox":
        """Create bbox from center point and half-size in meters."""
        delta_lat = size_m / 111_320
        delta_lon = size_m / (111_320 * abs(cos(radians(lat))))
        return cls(
            min_lat=lat - delta_lat,
            min_lon=lon - delta_lon,
            max_lat=lat + delta_lat,
            max_lon=lon + delta_lon,
        )

    @classmethod
    def from_corners(
        cls, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> "BBox":
        """Create bbox from two corner points (order-agnostic)."""
        return cls(
            min_lat=min(lat1, lat2),
            min_lon=min(lon1, lon2),
            max_lat=max(lat1, lat2),
            max_lon=max(lon1, lon2),
        )

    @property
    def as_ee_geometry(self):
        """Convert to Earth Engine geometry (deferred import)."""
        import ee
        return ee.Geometry.Rectangle(
            [self.min_lon, self.min_lat, self.max_lon, self.max_lat]
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GeoRecord:
    """Unified representation of a single location across all datasets."""

    location_id: str
    dataset: Dataset
    lat: float
    lon: float
    bbox: BBox
    original_tile_path: str
    original_tile_width: int
    original_tile_height: int
    original_zoom_level: Optional[int] = None
    split: Optional[str] = None  # "train" or "test"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dataset"] = self.dataset.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class SeasonalImage:
    """A single retrieved seasonal satellite image."""

    location_id: str
    season: Season
    year: int
    provider: Provider
    file_path: str
    geotiff_path: Optional[str] = None
    cloud_cover_pct: float = 0.0
    pixel_count: int = 0
    ndvi_mean: Optional[float] = None
    alignment_score: Optional[float] = None
    retrieval_date: Optional[datetime.datetime] = None
    composite_start: Optional[datetime.date] = None
    composite_end: Optional[datetime.date] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class LocationOutput:
    """Complete multi-seasonal output for one location."""

    geo_record: GeoRecord
    seasonal_images: list[SeasonalImage] = field(default_factory=list)
    completeness: float = 0.0
    quality_score: float = 0.0
