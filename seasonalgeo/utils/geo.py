"""Geographic utility functions."""

from math import radians, cos, sin, asin, sqrt
from xml.etree import ElementTree as ET


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two WGS84 points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6_371_000 * 2 * asin(sqrt(a))


def meters_per_pixel(zoom: int, lat: float) -> float:
    """Ground resolution in meters/pixel for a Web Mercator tile at given zoom and latitude."""
    return 156543.03392 * cos(radians(lat)) / (2**zoom)


def estimate_tile_extent_m(tile_size_px: int, zoom: int, lat: float) -> float:
    """Estimate the ground extent (in meters) of a square tile."""
    return tile_size_px * meters_per_pixel(zoom, lat)


def parse_kml_placemarks(kml_path: str) -> list[dict]:
    """Parse a KML file and extract Placemark name + coordinates.

    Returns list of dicts with keys: name, lat, lon, alt (altitude).
    """
    tree = ET.parse(kml_path)
    root = tree.getroot()

    # Handle KML namespace
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    placemarks = []
    for pm in root.iter(f"{ns}Placemark"):
        name_el = pm.find(f"{ns}name")
        name = name_el.text.strip() if name_el is not None and name_el.text else ""

        # Try Point coordinates first
        point = pm.find(f".//{ns}Point/{ns}coordinates")
        if point is not None and point.text:
            coords_text = point.text.strip()
            parts = coords_text.split(",")
            lon = float(parts[0])
            lat = float(parts[1])
            alt = float(parts[2]) if len(parts) > 2 else 0.0
            placemarks.append({"name": name, "lat": lat, "lon": lon, "alt": alt})
            continue

        # Try LookAt as fallback
        lookat = pm.find(f".//{ns}LookAt")
        if lookat is not None:
            lat_el = lookat.find(f"{ns}latitude")
            lon_el = lookat.find(f"{ns}longitude")
            if lat_el is not None and lon_el is not None:
                placemarks.append({
                    "name": name,
                    "lat": float(lat_el.text),
                    "lon": float(lon_el.text),
                    "alt": 0.0,
                })

    return placemarks
