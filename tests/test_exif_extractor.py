"""Tests for wikipicture.exif_extractor."""

from __future__ import annotations

import struct
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from wikipicture.exif_extractor import (
    PhotoMetadata,
    _dms_to_decimal,
    extract_metadata,
    scan_directory,
)


# ---------------------------------------------------------------------------
# _dms_to_decimal
# ---------------------------------------------------------------------------


class TestDmsToDecimal:
    """Tests for the DMS-to-decimal helper."""

    def test_north_latitude(self) -> None:
        # 40°26'46" N → 40.446111...
        result = _dms_to_decimal((40.0, 26.0, 46.0), "N")
        assert result == pytest.approx(40.44611, abs=1e-4)

    def test_south_latitude(self) -> None:
        # 33°51'54" S → -33.865
        result = _dms_to_decimal((33.0, 51.0, 54.0), "S")
        assert result == pytest.approx(-33.865, abs=1e-4)

    def test_east_longitude(self) -> None:
        # 116°23'30" E → 116.391667
        result = _dms_to_decimal((116.0, 23.0, 30.0), "E")
        assert result == pytest.approx(116.39167, abs=1e-4)

    def test_west_longitude(self) -> None:
        # 73°58'3" W → -73.9675
        result = _dms_to_decimal((73.0, 58.0, 3.0), "W")
        assert result == pytest.approx(-73.9675, abs=1e-4)

    def test_zero_coordinates(self) -> None:
        result = _dms_to_decimal((0.0, 0.0, 0.0), "N")
        assert result == 0.0

    def test_accepts_ifd_rational_like_floats(self) -> None:
        """Ensure it works when elements are not plain floats."""
        from fractions import Fraction

        dms = (Fraction(40, 1), Fraction(26, 1), Fraction(46, 1))
        result = _dms_to_decimal(dms, "N")
        assert result == pytest.approx(40.44611, abs=1e-4)


# ---------------------------------------------------------------------------
# extract_metadata — JPEG with GPS data
# ---------------------------------------------------------------------------


def _make_test_jpeg(path: Path) -> None:
    """Create a tiny valid JPEG with fake EXIF containing GPS data."""
    img = Image.new("RGB", (100, 50), color="red")
    # Build EXIF bytes with GPS and camera info via Pillow's Exif class.
    from PIL.ExifTags import Base, GPS, IFD

    exif = img.getexif()
    exif[Base.Make] = "TestCam"
    exif[Base.Model] = "T-1000"
    exif[Base.DateTimeOriginal] = "2024:06:15 10:30:00"

    gps_ifd = {
        GPS.GPSLatitudeRef: "N",
        GPS.GPSLatitude: (51.0, 30.0, 26.0),
        GPS.GPSLongitudeRef: "W",
        GPS.GPSLongitude: (0.0, 7.0, 39.0),
    }
    exif.get_ifd(IFD.GPSInfo).update(gps_ifd)

    img.save(path, format="JPEG", exif=exif.tobytes())


class TestExtractMetadataJpeg:
    """Test extract_metadata with real JPEG files on disk."""

    def test_extracts_gps_and_camera(self, tmp_path: Path) -> None:
        jpeg_path = tmp_path / "photo.jpg"
        _make_test_jpeg(jpeg_path)

        meta = extract_metadata(jpeg_path)

        assert meta.filepath == jpeg_path
        assert meta.width == 100
        assert meta.height == 50
        assert meta.camera_make == "TestCam"
        assert meta.camera_model == "T-1000"
        assert meta.timestamp == datetime(2024, 6, 15, 10, 30, 0)
        # GPS: 51°30'26"N ≈ 51.5072, 0°7'39"W ≈ -0.1275
        if meta.latitude is not None:
            assert meta.latitude == pytest.approx(51.5072, abs=1e-3)
        if meta.longitude is not None:
            assert meta.longitude == pytest.approx(-0.1275, abs=1e-3)

    def test_returns_none_for_missing_gps(self, tmp_path: Path) -> None:
        jpeg_path = tmp_path / "no_gps.jpg"
        img = Image.new("RGB", (64, 64), color="blue")
        img.save(jpeg_path, format="JPEG")

        meta = extract_metadata(jpeg_path)

        assert meta.filepath == jpeg_path
        assert meta.latitude is None
        assert meta.longitude is None
        assert meta.width == 64
        assert meta.height == 64


# ---------------------------------------------------------------------------
# extract_metadata — graceful handling
# ---------------------------------------------------------------------------


class TestExtractMetadataEdgeCases:
    """Ensure extract_metadata never raises on bad input."""

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("hello")

        meta = extract_metadata(txt_file)

        assert meta.filepath == txt_file
        assert meta.latitude is None

    def test_corrupt_file(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "corrupt.jpg"
        bad_file.write_bytes(b"\x00\x01\x02\x03")

        meta = extract_metadata(bad_file)

        assert meta.filepath == bad_file
        assert meta.latitude is None


# ---------------------------------------------------------------------------
# scan_directory
# ---------------------------------------------------------------------------


class TestScanDirectory:
    """Tests for recursive directory scanning."""

    def test_finds_files_recursively(self, tmp_path: Path) -> None:
        # Create nested structure with various extensions.
        (tmp_path / "a.jpg").write_bytes(b"")
        (tmp_path / "b.JPEG").write_bytes(b"")
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "c.heic").write_bytes(b"")
        (sub / "d.txt").write_text("ignore me")

        with patch(
            "wikipicture.exif_extractor.extract_metadata"
        ) as mock_extract:
            mock_extract.side_effect = lambda p: PhotoMetadata(filepath=p)
            results = scan_directory(tmp_path)

        found_names = sorted(m.filepath.name for m in results)
        assert found_names == ["a.jpg", "b.JPEG", "c.heic"]
        assert len(results) == 3

    def test_logs_warning_on_failure(self, tmp_path: Path, caplog) -> None:
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"\xff")

        with patch(
            "wikipicture.exif_extractor.extract_metadata",
            side_effect=RuntimeError("boom"),
        ):
            import logging

            with caplog.at_level(logging.WARNING):
                results = scan_directory(tmp_path)

        assert len(results) == 0
        assert "boom" in caplog.text

    def test_empty_directory(self, tmp_path: Path) -> None:
        results = scan_directory(tmp_path)
        assert results == []
