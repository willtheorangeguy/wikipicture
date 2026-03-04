"""Check Wikimedia Commons for existing photos near a GPS location."""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "WikiPicture/0.1 (travel photo Wikipedia tool)"
_RATE_LIMIT_S = 0.2  # 200 ms between API calls

# Categories that are too generic to be useful as upload suggestions.
_GENERIC_CATEGORIES = frozenset(
    {
        "All media files",
        "All free media",
        "Files from Wikimedia Commons",
        "Media needing categories",
        "Pages using the JsonConfig extension",
        "Uploaded with UploadWizard",
        "Self-published work",
        "CC-BY-SA-4.0",
        "CC-BY-SA-3.0",
        "CC-BY-4.0",
        "CC-BY-3.0",
        "CC-BY-2.0",
        "GFDL",
    }
)


class SaturationLevel(enum.Enum):
    """How well-covered a location already is on Commons."""

    NONE = "No coverage"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    SATURATED = "Saturated"


@dataclass
class CommonsResult:
    """Result of a Commons saturation check."""

    latitude: float
    longitude: float
    nearby_image_count: int
    saturation: SaturationLevel
    sample_titles: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    oldest_image_year: int | None = None
    newest_image_year: int | None = None


def _session() -> requests.Session:
    """Return a requests session with the correct User-Agent."""
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _rate_limit() -> None:
    time.sleep(_RATE_LIMIT_S)


def _count_to_saturation(count: int) -> SaturationLevel:
    if count == 0:
        return SaturationLevel.NONE
    if count <= 5:
        return SaturationLevel.LOW
    if count <= 20:
        return SaturationLevel.MEDIUM
    if count <= 50:
        return SaturationLevel.HIGH
    return SaturationLevel.SATURATED


def check_commons_saturation(
    lat: float,
    lon: float,
    radius_m: int = 1000,
) -> CommonsResult:
    """Query Wikimedia Commons for geotagged images near *lat*/*lon*.

    Parameters
    ----------
    lat, lon:
        GPS coordinates (WGS-84).
    radius_m:
        Search radius in metres (max 10 000).

    Returns
    -------
    CommonsResult with saturation level, sample titles, categories and year
    range of existing images.
    """
    sess = _session()

    # --- 1. Geosearch --------------------------------------------------
    geo_params: dict[str, str | int] = {
        "action": "query",
        "list": "geosearch",
        "gscoord": f"{lat}|{lon}",
        "gsradius": radius_m,
        "gsnamespace": 6,
        "gslimit": 50,
        "format": "json",
    }

    try:
        resp = sess.get(COMMONS_API_URL, params=geo_params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.error("Geosearch API request failed: %s", exc)
        return CommonsResult(
            latitude=lat,
            longitude=lon,
            nearby_image_count=0,
            saturation=SaturationLevel.NONE,
        )

    results = data.get("query", {}).get("geosearch", [])
    count = len(results)
    saturation = _count_to_saturation(count)
    sample_titles = [r["title"] for r in results[:5]]

    if count == 0:
        return CommonsResult(
            latitude=lat,
            longitude=lon,
            nearby_image_count=0,
            saturation=SaturationLevel.NONE,
            sample_titles=[],
            categories=[],
        )

    # --- 2. Fetch categories for found images --------------------------
    _rate_limit()

    titles_str = "|".join(r["title"] for r in results[:50])
    cat_params: dict[str, str | int] = {
        "action": "query",
        "prop": "categories",
        "titles": titles_str,
        "cllimit": "max",
        "format": "json",
    }

    categories: list[str] = []
    try:
        resp = sess.get(COMMONS_API_URL, params=cat_params, timeout=30)
        resp.raise_for_status()
        cat_data = resp.json()
        pages = cat_data.get("query", {}).get("pages", {})
        for page in pages.values():
            for cat in page.get("categories", []):
                cat_name = cat["title"].removeprefix("Category:")
                if cat_name not in _GENERIC_CATEGORIES:
                    categories.append(cat_name)
    except (requests.RequestException, ValueError) as exc:
        logger.error("Category fetch failed: %s", exc)

    # Deduplicate while keeping order
    seen: set[str] = set()
    unique_categories: list[str] = []
    for c in categories:
        if c not in seen:
            seen.add(c)
            unique_categories.append(c)

    # --- 3. Fetch upload dates (imageinfo) -----------------------------
    _rate_limit()

    oldest_year: int | None = None
    newest_year: int | None = None

    ii_params: dict[str, str | int] = {
        "action": "query",
        "prop": "imageinfo",
        "titles": titles_str,
        "iiprop": "timestamp",
        "format": "json",
    }

    try:
        resp = sess.get(COMMONS_API_URL, params=ii_params, timeout=30)
        resp.raise_for_status()
        ii_data = resp.json()
        pages = ii_data.get("query", {}).get("pages", {})
        years: list[int] = []
        for page in pages.values():
            for info in page.get("imageinfo", []):
                ts = info.get("timestamp", "")
                if ts:
                    try:
                        years.append(int(ts[:4]))
                    except (ValueError, IndexError):
                        pass
        if years:
            oldest_year = min(years)
            newest_year = max(years)
    except (requests.RequestException, ValueError) as exc:
        logger.error("Image info fetch failed: %s", exc)

    return CommonsResult(
        latitude=lat,
        longitude=lon,
        nearby_image_count=count,
        saturation=saturation,
        sample_titles=sample_titles,
        categories=unique_categories,
        oldest_image_year=oldest_year,
        newest_image_year=newest_year,
    )


def get_upload_categories(
    lat: float,
    lon: float,
    location_name: str | None = None,
) -> list[str]:
    """Suggest Commons categories for uploading a photo taken at *lat*/*lon*.

    Queries nearby images, collects their categories, filters out generic
    ones and optionally adds a location-based category.
    """
    result = check_commons_saturation(lat, lon)
    cats = [c for c in result.categories if c not in _GENERIC_CATEGORIES]

    if location_name:
        cats.append(location_name)

    # Deduplicate and sort
    seen: set[str] = set()
    unique: list[str] = []
    for c in cats:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return sorted(unique)


def check_freshness(result: CommonsResult) -> str:
    """Return a human-readable freshness assessment of existing coverage."""
    if result.nearby_image_count == 0:
        return "No existing photos"

    if result.oldest_image_year is not None and result.newest_image_year is not None:
        current_year = time.gmtime().tm_year
        if result.newest_image_year < current_year - 5:
            return f"Photos are outdated (oldest from {result.oldest_image_year})"
        return f"Recent coverage exists (newest from {result.newest_image_year})"

    return "No existing photos"
