"""Reverse-geocode GPS coordinates to place names via OpenStreetMap Nominatim."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

_USER_AGENT = "WikiPicture/0.1 (travel photo Wikipedia tool)"
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
_MIN_REQUEST_INTERVAL = 1.0  # seconds – Nominatim rate-limit

# Timestamp of the last Nominatim request (module-level).
_last_request_time: float = 0.0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class LocationInfo:
    """Reverse-geocoding result for a single coordinate pair."""

    latitude: float
    longitude: float
    display_name: str | None = None
    place_name: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    country_code: str | None = None


# ---------------------------------------------------------------------------
# Haversine helper
# ---------------------------------------------------------------------------

_EARTH_RADIUS_M = 6_371_000.0


def _haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Return the great-circle distance in **meters** between two points."""
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return _EARTH_RADIUS_M * c


# ---------------------------------------------------------------------------
# Nominatim reverse geocoding
# ---------------------------------------------------------------------------

def _extract_place_name(data: dict) -> str | None:
    """Pick the most specific name from Nominatim's response."""
    # The top-level "name" field is usually the most specific label
    # (e.g. "Eiffel Tower").  Fall back through address keys.
    if data.get("name"):
        return data["name"]

    address = data.get("address", {})
    for key in (
        "tourism",
        "amenity",
        "building",
        "historic",
        "leisure",
        "shop",
        "village",
        "hamlet",
        "suburb",
        "neighbourhood",
        "town",
        "city",
    ):
        if address.get(key):
            return address[key]
    return None


def _extract_city(address: dict) -> str | None:
    """Return the city (or closest equivalent) from the address dict."""
    for key in ("city", "town", "village", "hamlet", "municipality"):
        if address.get(key):
            return address[key]
    return None


def reverse_geocode(lat: float, lon: float) -> LocationInfo:
    """Query Nominatim and return a :class:`LocationInfo` for *lat*/*lon*.

    Enforces at least 1 s between successive HTTP requests.  On any
    error the returned ``LocationInfo`` will have ``None`` for every
    optional field.
    """
    global _last_request_time  # noqa: PLW0603

    # Rate-limit
    elapsed = time.monotonic() - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)

    params = {
        "format": "jsonv2",
        "lat": str(lat),
        "lon": str(lon),
        "zoom": "18",
        "addressdetails": "1",
    }
    headers = {"User-Agent": _USER_AGENT}

    try:
        _last_request_time = time.monotonic()
        resp = requests.get(
            _NOMINATIM_URL, params=params, headers=headers, timeout=10
        )
        resp.raise_for_status()
        data: dict = resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Nominatim request failed for (%s, %s): %s", lat, lon, exc)
        return LocationInfo(latitude=lat, longitude=lon)

    if "error" in data:
        logger.warning("Nominatim error for (%s, %s): %s", lat, lon, data["error"])
        return LocationInfo(latitude=lat, longitude=lon)

    address = data.get("address", {})

    return LocationInfo(
        latitude=lat,
        longitude=lon,
        display_name=data.get("display_name"),
        place_name=_extract_place_name(data),
        city=_extract_city(address),
        state=address.get("state"),
        country=address.get("country"),
        country_code=address.get("country_code"),
    )


# ---------------------------------------------------------------------------
# Batch geocoding with deduplication
# ---------------------------------------------------------------------------

_CLUSTER_RADIUS_M = 500.0


def batch_geocode(
    coordinates: list[tuple[float, float]],
) -> list[LocationInfo]:
    """Reverse-geocode a list of *(lat, lon)* tuples.

    Nearby coordinates (within ~500 m) are clustered so that only one
    Nominatim request is made per cluster.  The returned list is
    parallel to *coordinates* — every input gets a result.
    """
    if not coordinates:
        return []

    # Assign each coordinate to a cluster index.
    # cluster_centers: list of (lat, lon) – one per cluster.
    cluster_centers: list[tuple[float, float]] = []
    coord_to_cluster: list[int] = []

    for lat, lon in coordinates:
        matched = False
        for idx, (clat, clon) in enumerate(cluster_centers):
            if _haversine_distance(lat, lon, clat, clon) <= _CLUSTER_RADIUS_M:
                coord_to_cluster.append(idx)
                matched = True
                break
        if not matched:
            coord_to_cluster.append(len(cluster_centers))
            cluster_centers.append((lat, lon))

    logger.debug(
        "Clustered %d coordinates into %d unique locations",
        len(coordinates),
        len(cluster_centers),
    )

    # Geocode each cluster center once.
    cluster_results: list[LocationInfo] = []
    for clat, clon in cluster_centers:
        cluster_results.append(reverse_geocode(clat, clon))

    # Map results back to the original ordering.
    return [cluster_results[ci] for ci in coord_to_cluster]
