"""Combine signals from Wikipedia, Commons, and photo quality into a priority score."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from wikipicture.commons_checker import CommonsResult, SaturationLevel, check_freshness
from wikipicture.quality_filter import QualityAssessment
from wikipicture.wiki_analyzer import WikiArticle, check_article_image_need


@dataclass
class ScoredArticle:
    """A Wikipedia article with its individual image-need score."""

    article: WikiArticle
    score: float
    reason: str


@dataclass
class PhotoOpportunity:
    """A scored photo opportunity combining all analysis signals."""

    filepath: Path
    latitude: float
    longitude: float
    location_name: str
    score: float
    score_breakdown: dict[str, float]
    best_article: WikiArticle | None
    commons_result: CommonsResult | None
    quality: QualityAssessment | None
    recommendation: str
    reasons: list[str] = field(default_factory=list)
    top_articles: list[ScoredArticle] = field(default_factory=list)


def _pick_best_article(articles: list[WikiArticle]) -> WikiArticle | None:
    """Select the article most in need of a photo."""
    if not articles:
        return None
    # Prefer needs_photo=True, then lowest image_count
    return min(articles, key=lambda a: (not a.needs_photo, a.image_count))


def _rank_articles(
    articles: list[WikiArticle],
    max_candidates: int = 5,
) -> list[ScoredArticle]:
    """Score every article individually and return the top *max_candidates* sorted by score desc."""
    scored = []
    for article in articles:
        s, r = _score_article_need(article)
        scored.append(ScoredArticle(article=article, score=s, reason=r))
    scored.sort(key=lambda sa: sa.score, reverse=True)
    return scored[:max_candidates]


def _score_article_need(article: WikiArticle | None) -> tuple[float, str]:
    """Score 0-40 based on Wikipedia article image need."""
    if article is None:
        return 0.0, "No relevant Wikipedia article found"
    if article.needs_photo:
        return 40.0, f"Article '{article.title}' is tagged as needing a photo"
    if article.image_count == 0:
        return 35.0, f"Article '{article.title}' has no images"
    if article.image_count <= 2:
        return 20.0, f"Article '{article.title}' has only {article.image_count} image(s)"
    if article.image_count <= 5:
        return 10.0, f"Article '{article.title}' has {article.image_count} images"
    return 5.0, f"Article '{article.title}' already has {article.image_count} images"


def _score_commons_saturation(commons: CommonsResult | None) -> tuple[float, str]:
    """Score 0-30 based on Commons coverage."""
    if commons is None:
        return 15.0, "Commons coverage not checked"
    mapping = {
        SaturationLevel.NONE: (30.0, "No existing Commons coverage"),
        SaturationLevel.LOW: (25.0, "Low Commons coverage"),
        SaturationLevel.MEDIUM: (15.0, "Medium Commons coverage"),
        SaturationLevel.HIGH: (5.0, "High Commons coverage"),
        SaturationLevel.SATURATED: (0.0, "Location is saturated on Commons"),
    }
    return mapping[commons.saturation]


def _score_quality(quality: QualityAssessment | None) -> tuple[float, str]:
    """Score 0-15 based on photo quality."""
    if quality is None:
        return 10.0, "Photo quality not assessed (benefit of the doubt)"
    if quality.overall_suitable:
        return 15.0, "Photo meets quality standards"
    if quality.resolution_ok and not quality.is_sharp:
        return 8.0, "Good resolution but image may be blurry"
    if quality.is_sharp and not quality.resolution_ok:
        return 5.0, "Sharp image but resolution is low"
    return 0.0, f"Photo does not meet quality standards ({quality.reason})"


def _score_freshness(commons: CommonsResult | None) -> tuple[float, str]:
    """Score 0-15 based on how recent existing Commons photos are."""
    if commons is None or commons.nearby_image_count == 0:
        return 15.0, "No existing photos at this location"
    if commons.newest_image_year is None:
        return 15.0, "No existing photos at this location"
    current_year = time.gmtime().tm_year
    age = current_year - commons.newest_image_year
    if age > 5:
        return 10.0, f"Existing photos are outdated (newest from {commons.newest_image_year})"
    if age >= 2:
        return 5.0, f"Existing photos are a few years old (newest from {commons.newest_image_year})"
    return 0.0, f"Recent photos already exist (newest from {commons.newest_image_year})"


def _recommendation(score: float) -> str:
    """Map a numeric score to a human-readable recommendation."""
    if score >= 70:
        return "Highly recommended"
    if score >= 45:
        return "Recommended"
    if score >= 25:
        return "Maybe"
    return "Not recommended"


def score_opportunity(
    articles: list[WikiArticle],
    commons: CommonsResult | None,
    quality: QualityAssessment | None,
    filepath: Path,
    lat: float,
    lon: float,
    location_name: str,
    max_article_candidates: int = 5,
) -> PhotoOpportunity:
    """Score a photo opportunity by combining all analysis signals.

    Parameters
    ----------
    articles:
        Wikipedia articles found near the photo location.
    commons:
        Commons saturation check result, or ``None`` if not checked.
    quality:
        Photo quality assessment, or ``None`` if not assessed.
    filepath:
        Path to the photo file.
    lat, lon:
        GPS coordinates of the photo.
    location_name:
        Human-readable location name.
    max_article_candidates:
        Maximum number of ranked article candidates to include (default 5).

    Returns
    -------
    A fully scored :class:`PhotoOpportunity`.
    """
    best_article = _pick_best_article(articles)
    top_articles = _rank_articles(articles, max_candidates=max_article_candidates)

    article_score, article_reason = _score_article_need(best_article)
    commons_score, commons_reason = _score_commons_saturation(commons)
    quality_score, quality_reason = _score_quality(quality)
    freshness_score, freshness_reason = _score_freshness(commons)

    total = article_score + commons_score + quality_score + freshness_score
    breakdown = {
        "article_need": article_score,
        "commons_saturation": commons_score,
        "photo_quality": quality_score,
        "freshness_bonus": freshness_score,
    }
    reasons = [article_reason, commons_reason, quality_reason, freshness_reason]

    return PhotoOpportunity(
        filepath=filepath,
        latitude=lat,
        longitude=lon,
        location_name=location_name,
        score=total,
        score_breakdown=breakdown,
        best_article=best_article,
        commons_result=commons,
        quality=quality,
        recommendation=_recommendation(total),
        reasons=reasons,
        top_articles=top_articles,
    )


def rank_opportunities(
    opportunities: list[PhotoOpportunity],
) -> list[PhotoOpportunity]:
    """Sort photo opportunities by score descending."""
    return sorted(opportunities, key=lambda o: o.score, reverse=True)
