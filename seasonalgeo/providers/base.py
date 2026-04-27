"""Abstract base class for satellite imagery providers."""

import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from seasonalgeo.models.schema import BBox


class BaseProvider(ABC):
    """Base class for satellite imagery providers (GEE, Planet, etc.)."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short identifier for this provider (e.g. 'sentinel2', 'planet')."""
        ...

    @abstractmethod
    def authenticate(self) -> None:
        """Authenticate with the provider API."""
        ...

    @abstractmethod
    def query(
        self,
        bbox: BBox,
        date_start: datetime.date,
        date_end: datetime.date,
        max_cloud_pct: float = 30.0,
    ) -> Any:
        """Query for available imagery within bbox and date range."""
        ...

    @abstractmethod
    def get_image_count(self, collection: Any) -> int:
        """Get count of images/scenes in the queried collection."""
        ...

    @abstractmethod
    def composite(self, collection: Any, method: str = "median") -> Any:
        """Create a temporal composite or select best scene from a collection."""
        ...

    @abstractmethod
    def export_image(
        self,
        image: Any,
        bbox: BBox,
        output_path: Path,
        scale: float = 10.0,
        bands: list[str] | None = None,
    ) -> Path:
        """Export an image to a local file. Returns the saved path."""
        ...
