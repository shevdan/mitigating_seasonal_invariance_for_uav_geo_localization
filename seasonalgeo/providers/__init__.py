"""Satellite imagery providers for SeasonalGeo pipeline."""

from seasonalgeo.providers.base import BaseProvider

__all__ = ["BaseProvider", "Sentinel2Provider", "PlanetProvider"]


def __getattr__(name):
    """Lazy imports to avoid requiring all optional dependencies."""
    if name == "Sentinel2Provider":
        from seasonalgeo.providers.gee_sentinel2 import Sentinel2Provider
        return Sentinel2Provider
    if name == "PlanetProvider":
        from seasonalgeo.providers.planet import PlanetProvider
        return PlanetProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
