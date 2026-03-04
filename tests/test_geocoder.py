"""Tests for wikipicture.geocoder."""

from __future__ import annotations

import math
from unittest.mock import patch, MagicMock

import pytest
import requests

from wikipicture.geocoder import (
    LocationInfo,
    _haversine_distance,
    batch_geocode,
    reverse_geocode,
)


# ---------------------------------------------------------------------------
# _haversine_distance
# ---------------------------------------------------------------------------


class TestHaversineDistance:
    """Verify haversine against known reference distances."""

    def test_same_point_is_zero(self):
        assert _haversine_distance(48.8584, 2.2945, 48.8584, 2.2945) == 0.0

    def test_paris_to_london(self):
        # ~343 km accepted within 1 %
        dist = _haversine_distance(48.8566, 2.3522, 51.5074, -0.1278)
        assert 339_000 < dist < 347_000

    def test_new_york_to_los_angeles(self):
        # ~3 944 km accepted within 1 %
        dist = _haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
        assert 3_900_000 < dist < 3_990_000

    def test_antipodal_points(self):
        # North pole to south pole ≈ 20 015 km
        dist = _haversine_distance(90, 0, -90, 0)
        assert 20_000_000 < dist < 20_030_000

    def test_short_distance(self):
        # Two points ~111 m apart (≈ 0.001° latitude at equator)
        dist = _haversine_distance(0.0, 0.0, 0.001, 0.0)
        assert 100 < dist < 120


# ---------------------------------------------------------------------------
# Clustering logic (batch_geocode)
# ---------------------------------------------------------------------------


class TestBatchGeocodeClustering:
    """Test deduplication / clustering without hitting the network."""

    @patch("wikipicture.geocoder.reverse_geocode")
    def test_nearby_coords_grouped(self, mock_rg: MagicMock):
        """Two points < 500 m apart should produce only one API call."""
        mock_rg.return_value = LocationInfo(
            latitude=48.8584, longitude=2.2945, display_name="Eiffel Tower"
        )

        coords = [
            (48.8584, 2.2945),
            (48.8588, 2.2950),  # ~55 m away
        ]
        results = batch_geocode(coords)

        assert mock_rg.call_count == 1
        assert len(results) == 2
        # Both entries should reference the same LocationInfo
        assert results[0] is results[1]

    @patch("wikipicture.geocoder.reverse_geocode")
    def test_far_coords_separate(self, mock_rg: MagicMock):
        """Two points far apart should produce two API calls."""
        mock_rg.side_effect = [
            LocationInfo(latitude=48.8584, longitude=2.2945, display_name="Paris"),
            LocationInfo(latitude=40.7128, longitude=-74.0060, display_name="NYC"),
        ]

        coords = [
            (48.8584, 2.2945),   # Paris
            (40.7128, -74.0060),  # New York
        ]
        results = batch_geocode(coords)

        assert mock_rg.call_count == 2
        assert len(results) == 2
        assert results[0].display_name == "Paris"
        assert results[1].display_name == "NYC"

    @patch("wikipicture.geocoder.reverse_geocode")
    def test_empty_input(self, mock_rg: MagicMock):
        assert batch_geocode([]) == []
        mock_rg.assert_not_called()

    @patch("wikipicture.geocoder.reverse_geocode")
    def test_mixed_clusters(self, mock_rg: MagicMock):
        """Three coords: two nearby + one far produces two API calls."""
        loc_a = LocationInfo(latitude=0.0, longitude=0.0, display_name="A")
        loc_b = LocationInfo(latitude=10.0, longitude=10.0, display_name="B")
        mock_rg.side_effect = [loc_a, loc_b]

        coords = [
            (0.0, 0.0),
            (0.001, 0.001),  # ~157 m from first – same cluster
            (10.0, 10.0),    # far away – new cluster
        ]
        results = batch_geocode(coords)

        assert mock_rg.call_count == 2
        assert results[0] is results[1]
        assert results[2].display_name == "B"


# ---------------------------------------------------------------------------
# reverse_geocode – mock the HTTP layer
# ---------------------------------------------------------------------------


_SAMPLE_RESPONSE = {
    "place_id": 123,
    "name": "Eiffel Tower",
    "display_name": "Eiffel Tower, Avenue Anatole France, Paris, France",
    "address": {
        "tourism": "Eiffel Tower",
        "road": "Avenue Anatole France",
        "city": "Paris",
        "state": "Île-de-France",
        "country": "France",
        "country_code": "fr",
    },
}


class TestReverseGeocode:
    """Test JSON parsing of reverse_geocode with mocked HTTP."""

    @patch("wikipicture.geocoder.time")
    @patch("wikipicture.geocoder.requests.get")
    def test_successful_response(self, mock_get: MagicMock, mock_time: MagicMock):
        mock_time.monotonic.return_value = 1000.0
        mock_time.sleep = MagicMock()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _SAMPLE_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        info = reverse_geocode(48.8584, 2.2945)

        assert info.latitude == 48.8584
        assert info.longitude == 2.2945
        assert info.display_name == "Eiffel Tower, Avenue Anatole France, Paris, France"
        assert info.place_name == "Eiffel Tower"
        assert info.city == "Paris"
        assert info.state == "Île-de-France"
        assert info.country == "France"
        assert info.country_code == "fr"

    @patch("wikipicture.geocoder.time")
    @patch("wikipicture.geocoder.requests.get")
    def test_api_error_returns_none_fields(
        self, mock_get: MagicMock, mock_time: MagicMock
    ):
        mock_time.monotonic.return_value = 1000.0
        mock_time.sleep = MagicMock()

        mock_get.side_effect = requests.ConnectionError("no network")

        info = reverse_geocode(0.0, 0.0)

        assert info.latitude == 0.0
        assert info.longitude == 0.0
        assert info.display_name is None
        assert info.place_name is None
        assert info.city is None

    @patch("wikipicture.geocoder.time")
    @patch("wikipicture.geocoder.requests.get")
    def test_nominatim_error_field(self, mock_get: MagicMock, mock_time: MagicMock):
        """Nominatim returns an error object for coordinates in the ocean."""
        mock_time.monotonic.return_value = 1000.0
        mock_time.sleep = MagicMock()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "Unable to geocode"}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        info = reverse_geocode(0.0, 0.0)

        assert info.display_name is None
        assert info.place_name is None

    @patch("wikipicture.geocoder.time")
    @patch("wikipicture.geocoder.requests.get")
    def test_fallback_place_name(self, mock_get: MagicMock, mock_time: MagicMock):
        """When top-level 'name' is absent, fall back to address keys."""
        mock_time.monotonic.return_value = 1000.0
        mock_time.sleep = MagicMock()

        response = {
            "display_name": "Some Road, Springfield, IL, USA",
            "address": {
                "road": "Some Road",
                "city": "Springfield",
                "state": "Illinois",
                "country": "United States",
                "country_code": "us",
            },
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = response
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        info = reverse_geocode(39.7817, -89.6501)

        # Falls through to "city" in _extract_place_name
        assert info.place_name == "Springfield"
        assert info.city == "Springfield"
