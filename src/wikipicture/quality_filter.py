"""Assess photo quality for Wikimedia Commons upload suitability."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

logger = logging.getLogger(__name__)


@dataclass
class QualityAssessment:
    """Result of a single image quality check."""

    filepath: Path
    resolution_ok: bool
    width: int
    height: int
    megapixels: float
    blur_score: float
    is_sharp: bool
    overall_suitable: bool
    reason: str | None


def assess_quality(
    filepath: Path,
    min_megapixels: float = 2.0,
    min_blur_score: float = 100.0,
) -> QualityAssessment:
    """Assess whether an image meets quality requirements.

    Parameters
    ----------
    filepath:
        Path to the image file (JPEG, PNG, HEIC/HEIF, etc.).
    min_megapixels:
        Minimum resolution in megapixels.
    min_blur_score:
        Minimum Laplacian variance to consider the image sharp.
    """
    try:
        img = Image.open(filepath)
        width, height = img.size
        megapixels = round(width * height / 1_000_000, 2)

        resolution_ok = width * height >= min_megapixels * 1_000_000

        gray = np.array(img.convert("L"))
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        is_sharp = blur_score >= min_blur_score
        overall_suitable = resolution_ok and is_sharp

        reasons: list[str] = []
        if not resolution_ok:
            reasons.append(
                f"Resolution too low ({megapixels}MP, need {min_megapixels}MP)"
            )
        if not is_sharp:
            reasons.append(f"Image appears blurry (score: {blur_score:.1f})")
        reason = "; ".join(reasons) if reasons else None

        return QualityAssessment(
            filepath=filepath,
            resolution_ok=resolution_ok,
            width=width,
            height=height,
            megapixels=megapixels,
            blur_score=blur_score,
            is_sharp=is_sharp,
            overall_suitable=overall_suitable,
            reason=reason,
        )
    except Exception as exc:
        logger.warning("Failed to assess %s: %s", filepath, exc)
        return QualityAssessment(
            filepath=filepath,
            resolution_ok=False,
            width=0,
            height=0,
            megapixels=0.0,
            blur_score=0.0,
            is_sharp=False,
            overall_suitable=False,
            reason=f"Error: {exc}",
        )


def batch_assess(
    filepaths: list[Path],
    min_megapixels: float = 2.0,
    min_blur_score: float = 100.0,
) -> list[QualityAssessment]:
    """Assess quality for a batch of images.

    Returns a parallel list of :class:`QualityAssessment` results (one per
    input path, in the same order).
    """
    results: list[QualityAssessment] = []
    for fp in filepaths:
        result = assess_quality(fp, min_megapixels, min_blur_score)
        if not result.overall_suitable:
            logger.warning("%s: not suitable — %s", fp, result.reason)
        results.append(result)
    return results
