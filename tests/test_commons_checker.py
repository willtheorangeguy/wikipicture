"""Tests for wikipicture.commons_checker."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import responses

from wikipicture.commons_checker import (
    COMMONS_API_URL,
    CommonsResult,
    SaturationLevel,
    _count_to_saturation,
    check_commons_saturation,
    check_freshness,
    get_upload_categories,
)


# ---------- SaturationLevel thresholds ----------


@pytest.mark.parametrize(
    "count, expected",
    [
        (0, SaturationLevel.NONE),
        (1, SaturationLevel.LOW),
        (5, SaturationLevel.LOW),
        (6, SaturationLevel.MEDIUM),
        (20, SaturationLevel.MEDIUM),
        (21, SaturationLevel.HIGH),
        (50, SaturationLevel.HIGH),
        (51, SaturationLevel.SATURATED),
        (100, SaturationLevel.SATURATED),
    ],
)
def test_count_to_saturation(count: int, expected: SaturationLevel) -> None:
    assert _count_to_saturation(count) == expected


def test_saturation_values() -> None:
    assert SaturationLevel.NONE.value == "No coverage"
    assert SaturationLevel.LOW.value == "Low"
    assert SaturationLevel.MEDIUM.value == "Medium"
    assert SaturationLevel.HIGH.value == "High"
    assert SaturationLevel.SATURATED.value == "Saturated"


# ---------- Helpers for mocked API responses ----------

def _geosearch_response(n: int) -> dict:
    """Build a fake geosearch JSON response with *n* results."""
    items = [
        {
            "pageid": 1000 + i,
            "ns": 6,
            "title": f"File:Photo_{i}.jpg",
            "lat": 48.8584 + i * 0.0001,
            "lon": 2.2945 + i * 0.0001,
            "dist": 10.0 * i,
            "primary": "",
        }
        for i in range(n)
    ]
    return {"batchcomplete": "", "query": {"geosearch": items}}


def _categories_response(titles: list[str]) -> dict:
    """Build a fake categories JSON response."""
    pages: dict[str, dict] = {}
    for idx, title in enumerate(titles):
        pages[str(idx)] = {
            "pageid": 1000 + idx,
            "ns": 6,
            "title": title,
            "categories": [
                {"ns": 14, "title": "Category:Paris"},
                {"ns": 14, "title": "Category:Eiffel Tower"},
                {"ns": 14, "title": "Category:All media files"},
            ],
        }
    return {"batchcomplete": "", "query": {"pages": pages}}


def _imageinfo_response(titles: list[str], years: list[int] | None = None) -> dict:
    """Build a fake imageinfo JSON response."""
    if years is None:
        years = [2018 + i for i in range(len(titles))]
    pages: dict[str, dict] = {}
    for idx, title in enumerate(titles):
        pages[str(idx)] = {
            "pageid": 1000 + idx,
            "ns": 6,
            "title": title,
            "imageinfo": [
                {"timestamp": f"{years[idx]}-06-15T12:00:00Z"}
            ],
        }
    return {"batchcomplete": "", "query": {"pages": pages}}


# ---------- Mocked geosearch ----------


@responses.activate
@patch("wikipicture.commons_checker._rate_limit")
def test_check_commons_saturation_no_results(mock_rl) -> None:
    responses.add(
        responses.GET,
        COMMONS_API_URL,
        json=_geosearch_response(0),
        status=200,
    )

    result = check_commons_saturation(48.8584, 2.2945)

    assert result.nearby_image_count == 0
    assert result.saturation == SaturationLevel.NONE
    assert result.sample_titles == []
    assert result.categories == []


@responses.activate
@patch("wikipicture.commons_checker._rate_limit")
def test_check_commons_saturation_low(mock_rl) -> None:
    n = 3
    titles = [f"File:Photo_{i}.jpg" for i in range(n)]

    responses.add(responses.GET, COMMONS_API_URL, json=_geosearch_response(n), status=200)
    responses.add(responses.GET, COMMONS_API_URL, json=_categories_response(titles), status=200)
    responses.add(responses.GET, COMMONS_API_URL, json=_imageinfo_response(titles), status=200)

    result = check_commons_saturation(48.8584, 2.2945)

    assert result.nearby_image_count == n
    assert result.saturation == SaturationLevel.LOW
    assert len(result.sample_titles) == n
    assert "Paris" in result.categories
    assert "Eiffel Tower" in result.categories
    # "All media files" should be filtered out
    assert "All media files" not in result.categories


@responses.activate
@patch("wikipicture.commons_checker._rate_limit")
def test_check_commons_saturation_medium(mock_rl) -> None:
    n = 15
    titles = [f"File:Photo_{i}.jpg" for i in range(n)]

    responses.add(responses.GET, COMMONS_API_URL, json=_geosearch_response(n), status=200)
    responses.add(responses.GET, COMMONS_API_URL, json=_categories_response(titles[:5]), status=200)
    responses.add(
        responses.GET,
        COMMONS_API_URL,
        json=_imageinfo_response(titles[:5]),
        status=200,
    )

    result = check_commons_saturation(48.8584, 2.2945)

    assert result.nearby_image_count == n
    assert result.saturation == SaturationLevel.MEDIUM
    assert len(result.sample_titles) == 5  # capped at 5


@responses.activate
@patch("wikipicture.commons_checker._rate_limit")
def test_check_commons_api_error_returns_none(mock_rl) -> None:
    responses.add(responses.GET, COMMONS_API_URL, status=500)

    result = check_commons_saturation(0.0, 0.0)

    assert result.saturation == SaturationLevel.NONE
    assert result.nearby_image_count == 0


# ---------- Category extraction ----------


@responses.activate
@patch("wikipicture.commons_checker._rate_limit")
def test_categories_deduplication(mock_rl) -> None:
    """Categories should be deduplicated across images."""
    n = 2
    titles = [f"File:Photo_{i}.jpg" for i in range(n)]

    responses.add(responses.GET, COMMONS_API_URL, json=_geosearch_response(n), status=200)
    responses.add(responses.GET, COMMONS_API_URL, json=_categories_response(titles), status=200)
    responses.add(responses.GET, COMMONS_API_URL, json=_imageinfo_response(titles), status=200)

    result = check_commons_saturation(48.8584, 2.2945)

    # Paris and Eiffel Tower appear for each image but should appear once
    assert result.categories.count("Paris") == 1
    assert result.categories.count("Eiffel Tower") == 1


@responses.activate
@patch("wikipicture.commons_checker._rate_limit")
@patch("wikipicture.commons_checker.check_commons_saturation")
def test_get_upload_categories_filters_and_sorts(mock_check, mock_rl) -> None:
    mock_check.return_value = CommonsResult(
        latitude=48.8584,
        longitude=2.2945,
        nearby_image_count=5,
        saturation=SaturationLevel.LOW,
        sample_titles=["File:A.jpg"],
        categories=["Eiffel Tower", "Paris", "All media files"],
    )

    cats = get_upload_categories(48.8584, 2.2945, location_name="7th arrondissement")

    assert "All media files" not in cats
    assert "Eiffel Tower" in cats
    assert "7th arrondissement" in cats
    # Should be sorted
    assert cats == sorted(cats)


# ---------- Freshness ----------


def test_freshness_no_photos() -> None:
    result = CommonsResult(
        latitude=0, longitude=0, nearby_image_count=0,
        saturation=SaturationLevel.NONE,
    )
    assert check_freshness(result) == "No existing photos"


def test_freshness_outdated() -> None:
    result = CommonsResult(
        latitude=0, longitude=0, nearby_image_count=10,
        saturation=SaturationLevel.MEDIUM,
        oldest_image_year=2005,
        newest_image_year=2010,
    )
    assert "outdated" in check_freshness(result).lower()
    assert "2005" in check_freshness(result)


def test_freshness_recent() -> None:
    import time
    current_year = time.gmtime().tm_year

    result = CommonsResult(
        latitude=0, longitude=0, nearby_image_count=10,
        saturation=SaturationLevel.MEDIUM,
        oldest_image_year=2015,
        newest_image_year=current_year,
    )
    msg = check_freshness(result)
    assert "recent" in msg.lower()
    assert str(current_year) in msg


def test_freshness_no_years() -> None:
    result = CommonsResult(
        latitude=0, longitude=0, nearby_image_count=5,
        saturation=SaturationLevel.LOW,
        oldest_image_year=None,
        newest_image_year=None,
    )
    assert check_freshness(result) == "No existing photos"
