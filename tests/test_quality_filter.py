"""Tests for the quality_filter module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from wikipicture.quality_filter import QualityAssessment, assess_quality, batch_assess


@pytest.fixture()
def sharp_image(tmp_path: Path) -> Path:
    """Create a large, sharp (high-frequency) test image."""
    width, height = 2000, 1500  # 3 MP
    # Checkerboard pattern produces high Laplacian variance
    arr = np.zeros((height, width), dtype=np.uint8)
    arr[::2, ::2] = 255
    arr[1::2, 1::2] = 255
    img = Image.fromarray(arr, mode="L")
    path = tmp_path / "sharp.png"
    img.save(path)
    return path


@pytest.fixture()
def blurry_image(tmp_path: Path) -> Path:
    """Create a large, uniform (blurry) test image."""
    width, height = 2000, 1500  # 3 MP
    arr = np.full((height, width), 128, dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    path = tmp_path / "blurry.png"
    img.save(path)
    return path


@pytest.fixture()
def small_image(tmp_path: Path) -> Path:
    """Create a tiny image that fails the resolution check."""
    width, height = 100, 100  # 0.01 MP
    arr = np.zeros((height, width), dtype=np.uint8)
    arr[::2, ::2] = 255
    arr[1::2, 1::2] = 255
    img = Image.fromarray(arr, mode="L")
    path = tmp_path / "small.png"
    img.save(path)
    return path


class TestAssessQuality:
    def test_sharp_high_res_passes(self, sharp_image: Path) -> None:
        result = assess_quality(sharp_image)
        assert result.resolution_ok is True
        assert result.is_sharp is True
        assert result.overall_suitable is True
        assert result.reason is None
        assert result.width == 2000
        assert result.height == 1500
        assert result.megapixels == 3.0

    def test_small_image_fails_resolution(self, small_image: Path) -> None:
        result = assess_quality(small_image)
        assert result.resolution_ok is False
        assert result.overall_suitable is False
        assert result.reason is not None
        assert "Resolution too low" in result.reason

    def test_blurry_image_fails_sharpness(self, blurry_image: Path) -> None:
        result = assess_quality(blurry_image)
        assert result.blur_score < 100.0
        assert result.is_sharp is False
        assert result.overall_suitable is False
        assert result.reason is not None
        assert "blurry" in result.reason

    def test_blur_score_higher_for_sharp(
        self, sharp_image: Path, blurry_image: Path
    ) -> None:
        sharp_result = assess_quality(sharp_image)
        blurry_result = assess_quality(blurry_image)
        assert sharp_result.blur_score > blurry_result.blur_score

    def test_reason_combines_both_issues(self, tmp_path: Path) -> None:
        """A tiny uniform image should report both resolution and blur."""
        arr = np.full((50, 50), 128, dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        path = tmp_path / "tiny_uniform.png"
        img.save(path)

        result = assess_quality(path)
        assert result.overall_suitable is False
        assert "Resolution too low" in result.reason
        assert "blurry" in result.reason

    def test_custom_thresholds(self, sharp_image: Path) -> None:
        result = assess_quality(
            sharp_image, min_megapixels=10.0, min_blur_score=1e12
        )
        assert result.resolution_ok is False
        assert result.is_sharp is False
        assert result.overall_suitable is False

    def test_missing_file_returns_error(self, tmp_path: Path) -> None:
        result = assess_quality(tmp_path / "nonexistent.jpg")
        assert result.overall_suitable is False
        assert result.reason is not None
        assert result.reason.startswith("Error:")


class TestBatchAssess:
    def test_batch_returns_parallel_list(
        self, sharp_image: Path, small_image: Path
    ) -> None:
        results = batch_assess([sharp_image, small_image])
        assert len(results) == 2
        assert results[0].filepath == sharp_image
        assert results[1].filepath == small_image
        assert results[0].overall_suitable is True
        assert results[1].overall_suitable is False

    def test_batch_empty_list(self) -> None:
        results = batch_assess([])
        assert results == []
