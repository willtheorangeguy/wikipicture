"""Extract GPS location data and metadata from JPEG and HEIC/HEIF photos."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

logger = logging.getLogger(__name__)

_PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".heic", ".heif"}


@dataclass
class PhotoMetadata:
    """Metadata extracted from a geotagged photo."""

    filepath: Path
    latitude: float | None = None
    longitude: float | None = None
    timestamp: datetime | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    width: int | None = None
    height: int | None = None


def _dms_to_decimal(dms_tuple: tuple, ref: str) -> float:
    """Convert EXIF GPS DMS (degrees, minutes, seconds) to decimal degrees.

    Args:
        dms_tuple: Tuple of (degrees, minutes, seconds), where each element
            is a float or an IFDRational.
        ref: One of 'N', 'S', 'E', 'W'.

    Returns:
        Decimal degrees (negative for S/W).
    """
    degrees = float(dms_tuple[0])
    minutes = float(dms_tuple[1])
    seconds = float(dms_tuple[2])
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def _parse_gps_info(exif_data: dict) -> tuple[float | None, float | None]:
    """Extract latitude and longitude from decoded EXIF data."""
    gps_info_raw = exif_data.get("GPSInfo")
    if not gps_info_raw:
        return None, None

    # GPSInfo may already be decoded (tag names) or still numeric tag IDs.
    if isinstance(gps_info_raw, dict):
        gps_info: dict = {}
        for key, value in gps_info_raw.items():
            decoded_key = GPSTAGS.get(key, key) if isinstance(key, int) else key
            gps_info[decoded_key] = value
    else:
        return None, None

    lat = lon = None
    try:
        if "GPSLatitude" in gps_info and "GPSLatitudeRef" in gps_info:
            lat = _dms_to_decimal(gps_info["GPSLatitude"], gps_info["GPSLatitudeRef"])
    except (TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
        logger.warning("Failed to parse GPS latitude: %s", exc)

    try:
        if "GPSLongitude" in gps_info and "GPSLongitudeRef" in gps_info:
            lon = _dms_to_decimal(
                gps_info["GPSLongitude"], gps_info["GPSLongitudeRef"]
            )
    except (TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
        logger.warning("Failed to parse GPS longitude: %s", exc)

    return lat, lon


def _parse_datetime(value: str | None) -> datetime | None:
    """Parse EXIF DateTimeOriginal string to a datetime object."""
    if not value:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    logger.warning("Unrecognised datetime format: %s", value)
    return None


def _decode_exif(raw_exif: dict) -> dict[str, object]:
    """Decode numeric EXIF tag IDs to human-readable names."""
    decoded: dict[str, object] = {}
    for tag_id, value in raw_exif.items():
        tag_name = TAGS.get(tag_id, tag_id)
        decoded[tag_name] = value
    return decoded


def _extract_from_jpeg(filepath: Path) -> dict[str, object]:
    """Open a JPEG and return decoded EXIF data plus image dimensions."""
    with Image.open(filepath) as img:
        width, height = img.size
        raw_exif = getattr(img, "_getexif", lambda: None)()
        decoded = _decode_exif(raw_exif) if raw_exif else {}
        decoded["_width"] = width
        decoded["_height"] = height
        return decoded


def _extract_from_heif(filepath: Path) -> dict[str, object]:
    """Open a HEIC/HEIF file and return decoded EXIF data plus dimensions."""
    import pillow_heif  # lazy import to avoid hard crash if not installed

    pillow_heif.register_heif_opener()
    with Image.open(filepath) as img:
        width, height = img.size
        exif_obj = img.getexif()
        raw_exif = dict(exif_obj) if exif_obj else {}
        # Include IFD GPS sub-dict when present.
        from PIL.ExifTags import IFD

        gps_ifd = exif_obj.get_ifd(IFD.GPSInfo)
        if gps_ifd:
            raw_exif[0x8825] = gps_ifd  # GPSInfo tag number
        decoded = _decode_exif(raw_exif) if raw_exif else {}
        decoded["_width"] = width
        decoded["_height"] = height
        return decoded


def extract_metadata(filepath: Path) -> PhotoMetadata:
    """Extract metadata from a single photo file.

    Supports JPEG (.jpg/.jpeg) and HEIC/HEIF (.heic/.heif).
    Returns a PhotoMetadata instance; missing fields default to None.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    try:
        if suffix in (".jpg", ".jpeg"):
            exif_data = _extract_from_jpeg(filepath)
        elif suffix in (".heic", ".heif"):
            exif_data = _extract_from_heif(filepath)
        else:
            logger.warning("Unsupported file extension: %s", filepath)
            return PhotoMetadata(filepath=filepath)
    except Exception as exc:
        logger.warning("Failed to read EXIF from %s: %s", filepath, exc)
        return PhotoMetadata(filepath=filepath)

    latitude, longitude = _parse_gps_info(exif_data)
    timestamp = _parse_datetime(exif_data.get("DateTimeOriginal"))
    camera_make = exif_data.get("Make")
    camera_model = exif_data.get("Model")
    if isinstance(camera_make, str):
        camera_make = camera_make.strip()
    else:
        camera_make = None
    if isinstance(camera_model, str):
        camera_model = camera_model.strip()
    else:
        camera_model = None

    return PhotoMetadata(
        filepath=filepath,
        latitude=latitude,
        longitude=longitude,
        timestamp=timestamp,
        camera_make=camera_make,
        camera_model=camera_model,
        width=exif_data.get("_width"),
        height=exif_data.get("_height"),
    )


def scan_directory(directory: Path) -> list[PhotoMetadata]:
    """Recursively scan *directory* for photo files and extract metadata.

    Finds .jpg, .jpeg, .heic, .heif files (case-insensitive).  Files that
    fail to parse are logged as warnings and skipped.
    """
    directory = Path(directory).resolve()
    results: list[PhotoMetadata] = []

    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in _PHOTO_EXTENSIONS:
            try:
                metadata = extract_metadata(path)
                results.append(metadata)
            except Exception as exc:
                logger.warning("Skipping %s: %s", path, exc)

    return results
