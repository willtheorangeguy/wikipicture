"""Tests for wikipicture.scorer."""

from __future__ import annotations

from pathlib import Path

from wikipicture.commons_checker import CommonsResult, SaturationLevel
from wikipicture.quality_filter import QualityAssessment
from wikipicture.scorer import PhotoOpportunity, rank_opportunities, score_opportunity
from wikipicture.wiki_analyzer import WikiArticle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _article(
    title: str = "Test Article",
    pageid: int = 1,
    image_count: int = 0,
    needs_photo: bool = False,
) -> WikiArticle:
    return WikiArticle(
        title=title,
        pageid=pageid,
        url=f"https://en.wikipedia.org/?curid={pageid}",
        image_count=image_count,
        needs_photo=needs_photo,
        extract=None,
    )


def _commons(
    saturation: SaturationLevel = SaturationLevel.NONE,
    count: int = 0,
    oldest: int | None = None,
    newest: int | None = None,
) -> CommonsResult:
    return CommonsResult(
        latitude=48.8,
        longitude=2.3,
        nearby_image_count=count,
        saturation=saturation,
        oldest_image_year=oldest,
        newest_image_year=newest,
    )


def _quality(
    overall_suitable: bool = True,
    resolution_ok: bool = True,
    is_sharp: bool = True,
) -> QualityAssessment:
    return QualityAssessment(
        filepath=Path("photo.jpg"),
        resolution_ok=resolution_ok,
        width=4000,
        height=3000,
        megapixels=12.0,
        blur_score=200.0,
        is_sharp=is_sharp,
        overall_suitable=overall_suitable,
        reason=None if overall_suitable else "not suitable",
    )


_FP = Path("photo.jpg")


# ---------------------------------------------------------------------------
# High-score scenario
# ---------------------------------------------------------------------------

def test_high_score_needs_photo_no_commons_good_quality():
    """needs_photo article + no commons + good quality → high score."""
    articles = [_article(needs_photo=True)]
    commons = _commons(SaturationLevel.NONE, count=0)
    quality = _quality(overall_suitable=True)

    opp = score_opportunity(articles, commons, quality, _FP, 48.8, 2.3, "Paris")

    assert opp.score >= 70
    assert opp.recommendation == "Highly recommended"
    assert opp.best_article is not None
    assert opp.best_article.needs_photo is True


# ---------------------------------------------------------------------------
# Low-score scenario
# ---------------------------------------------------------------------------

def test_low_score_well_illustrated_saturated():
    """Well-illustrated article + saturated commons → low score."""
    import time

    current_year = time.gmtime().tm_year
    articles = [_article(image_count=10)]
    commons = _commons(SaturationLevel.SATURATED, count=60, oldest=current_year - 1, newest=current_year)
    quality = _quality(overall_suitable=False, resolution_ok=False, is_sharp=False)

    opp = score_opportunity(articles, commons, quality, _FP, 48.8, 2.3, "Paris")

    assert opp.score < 25
    assert opp.recommendation == "Not recommended"


# ---------------------------------------------------------------------------
# Score breakdown adds up
# ---------------------------------------------------------------------------

def test_score_breakdown_sums_to_total():
    articles = [_article(image_count=1)]
    commons = _commons(SaturationLevel.LOW, count=3, oldest=2020, newest=2021)
    quality = _quality(overall_suitable=True)

    opp = score_opportunity(articles, commons, quality, _FP, 48.8, 2.3, "Paris")

    assert sum(opp.score_breakdown.values()) == opp.score
    assert set(opp.score_breakdown.keys()) == {
        "article_need",
        "commons_saturation",
        "photo_quality",
        "freshness_bonus",
    }


# ---------------------------------------------------------------------------
# Recommendation thresholds
# ---------------------------------------------------------------------------

def test_recommendation_highly_recommended():
    articles = [_article(needs_photo=True)]
    commons = _commons(SaturationLevel.NONE, count=0)
    quality = _quality(overall_suitable=True)

    opp = score_opportunity(articles, commons, quality, _FP, 48.8, 2.3, "X")
    assert opp.recommendation == "Highly recommended"
    assert opp.score >= 70


def test_recommendation_recommended():
    """Moderate signals should yield 'Recommended'."""
    import time

    current_year = time.gmtime().tm_year
    articles = [_article(image_count=2)]
    commons = _commons(SaturationLevel.LOW, count=3, oldest=current_year - 3, newest=current_year - 3)
    quality = _quality(overall_suitable=True)

    opp = score_opportunity(articles, commons, quality, _FP, 48.8, 2.3, "X")
    # 20 + 25 + 15 + 5 = 65
    assert opp.recommendation == "Recommended"
    assert 45 <= opp.score < 70


def test_recommendation_maybe():
    """Weak signals should yield 'Maybe'."""
    import time

    current_year = time.gmtime().tm_year
    articles = [_article(image_count=4)]
    commons = _commons(SaturationLevel.MEDIUM, count=15, oldest=current_year - 1, newest=current_year)
    quality = _quality(overall_suitable=False, resolution_ok=True, is_sharp=False)

    opp = score_opportunity(articles, commons, quality, _FP, 48.8, 2.3, "X")
    # 10 + 15 + 8 + 0 = 33
    assert opp.recommendation == "Maybe"
    assert 25 <= opp.score < 45


def test_recommendation_not_recommended():
    """Poor signals should yield 'Not recommended'."""
    import time

    current_year = time.gmtime().tm_year
    articles = [_article(image_count=10)]
    commons = _commons(SaturationLevel.SATURATED, count=60, oldest=current_year, newest=current_year)
    quality = _quality(overall_suitable=False, resolution_ok=False, is_sharp=False)

    opp = score_opportunity(articles, commons, quality, _FP, 48.8, 2.3, "X")
    # 5 + 0 + 0 + 0 = 5
    assert opp.score < 25
    assert opp.recommendation == "Not recommended"


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def test_ranking_sorts_by_score_descending():
    opps = [
        PhotoOpportunity(
            filepath=_FP, latitude=0, longitude=0, location_name="A",
            score=30, score_breakdown={}, best_article=None,
            commons_result=None, quality=None, recommendation="Maybe",
            reasons=[],
        ),
        PhotoOpportunity(
            filepath=_FP, latitude=0, longitude=0, location_name="B",
            score=80, score_breakdown={}, best_article=None,
            commons_result=None, quality=None, recommendation="Highly recommended",
            reasons=[],
        ),
        PhotoOpportunity(
            filepath=_FP, latitude=0, longitude=0, location_name="C",
            score=55, score_breakdown={}, best_article=None,
            commons_result=None, quality=None, recommendation="Recommended",
            reasons=[],
        ),
    ]
    ranked = rank_opportunities(opps)
    assert [o.location_name for o in ranked] == ["B", "C", "A"]
    assert ranked[0].score > ranked[1].score > ranked[2].score


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_no_articles():
    opp = score_opportunity([], None, None, _FP, 0, 0, "Nowhere")
    assert opp.best_article is None
    assert opp.score_breakdown["article_need"] == 0


def test_no_commons():
    articles = [_article(image_count=0)]
    opp = score_opportunity(articles, None, None, _FP, 0, 0, "Nowhere")
    # commons not checked → 15 pts midpoint; freshness → 15 pts
    assert opp.score_breakdown["commons_saturation"] == 15
    assert opp.score_breakdown["freshness_bonus"] == 15


def test_no_quality():
    articles = [_article(image_count=0)]
    commons = _commons(SaturationLevel.NONE, count=0)
    opp = score_opportunity(articles, commons, None, _FP, 0, 0, "Nowhere")
    assert opp.score_breakdown["photo_quality"] == 10


def test_all_none_edge_case():
    opp = score_opportunity([], None, None, _FP, 0, 0, "")
    assert 0 <= opp.score <= 100
    assert isinstance(opp.reasons, list)
    assert len(opp.reasons) == 4


def test_best_article_picks_needs_photo_first():
    a1 = _article(title="A", image_count=5, needs_photo=False)
    a2 = _article(title="B", image_count=0, needs_photo=True)
    a3 = _article(title="C", image_count=0, needs_photo=False)

    opp = score_opportunity([a1, a2, a3], None, None, _FP, 0, 0, "X")
    assert opp.best_article is not None
    assert opp.best_article.title == "B"


def test_quality_resolution_ok_not_sharp():
    quality = _quality(overall_suitable=False, resolution_ok=True, is_sharp=False)
    opp = score_opportunity([], None, quality, _FP, 0, 0, "X")
    assert opp.score_breakdown["photo_quality"] == 8


def test_quality_sharp_not_resolution_ok():
    quality = _quality(overall_suitable=False, resolution_ok=False, is_sharp=True)
    opp = score_opportunity([], None, quality, _FP, 0, 0, "X")
    assert opp.score_breakdown["photo_quality"] == 5
