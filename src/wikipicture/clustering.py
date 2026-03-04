"""Group photos by location and time to reduce redundant geocoding API calls."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from wikipicture.exif_extractor import PhotoMetadata
from wikipicture.geocoder import _haversine_distance

logger = logging.getLogger(__name__)


@dataclass
class PhotoCluster:
    """A group of photos taken at a similar location."""

    center_lat: float
    center_lon: float
    photos: list[PhotoMetadata] = field(default_factory=list)
    location_name: str | None = None


def _centroid(photos: list[PhotoMetadata]) -> tuple[float, float]:
    """Return the average (lat, lon) of *photos* that have GPS data."""
    lats = [p.latitude for p in photos if p.latitude is not None]
    lons = [p.longitude for p in photos if p.longitude is not None]
    return sum(lats) / len(lats), sum(lons) / len(lons)


def cluster_photos(
    photos: list[PhotoMetadata],
    distance_threshold_m: float = 500.0,
) -> list[PhotoCluster]:
    """Cluster *photos* by spatial proximity using greedy assignment.

    Photos without GPS coordinates are skipped (with a logged warning).
    Returns clusters sorted by number of photos, largest first.
    """
    gps_photos: list[PhotoMetadata] = []
    for photo in photos:
        if photo.latitude is None or photo.longitude is None:
            logger.warning("Skipping %s: no GPS data", photo.filepath)
        else:
            gps_photos.append(photo)

    if not gps_photos:
        return []

    clusters: list[PhotoCluster] = []

    for photo in gps_photos:
        assigned = False
        for cluster in clusters:
            dist = _haversine_distance(
                photo.latitude, photo.longitude,
                cluster.center_lat, cluster.center_lon,
            )
            if dist <= distance_threshold_m:
                cluster.photos.append(photo)
                # Recompute centroid.
                cluster.center_lat, cluster.center_lon = _centroid(cluster.photos)
                assigned = True
                break

        if not assigned:
            clusters.append(
                PhotoCluster(
                    center_lat=photo.latitude,
                    center_lon=photo.longitude,
                    photos=[photo],
                )
            )

    clusters.sort(key=lambda c: len(c.photos), reverse=True)

    logger.debug(
        "Clustered %d photos into %d spatial clusters",
        len(gps_photos),
        len(clusters),
    )
    return clusters


def group_by_time(
    cluster: PhotoCluster,
    time_threshold_hours: float = 1.0,
) -> list[PhotoCluster]:
    """Sub-divide a spatial *cluster* into temporal sub-groups.

    Photos within *time_threshold_hours* of each other are kept together.
    Photos without timestamps are placed in their own group.
    """
    threshold = timedelta(hours=time_threshold_hours)

    timed: list[PhotoMetadata] = []
    untimed: list[PhotoMetadata] = []
    for photo in cluster.photos:
        if photo.timestamp is not None:
            timed.append(photo)
        else:
            untimed.append(photo)

    # Sort by timestamp so greedy grouping works chronologically.
    timed.sort(key=lambda p: p.timestamp)  # type: ignore[arg-type]

    groups: list[list[PhotoMetadata]] = []
    for photo in timed:
        if groups and (photo.timestamp - groups[-1][-1].timestamp) <= threshold:  # type: ignore[operator]
            groups[-1].append(photo)
        else:
            groups.append([photo])

    if untimed:
        groups.append(untimed)

    return [
        PhotoCluster(
            center_lat=_centroid(g)[0],
            center_lon=_centroid(g)[1],
            photos=g,
            location_name=cluster.location_name,
        )
        for g in groups
    ]
