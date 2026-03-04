"""Tests for wikipicture.wiki_analyzer."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from wikipicture.wiki_analyzer import (
    WikiArticle,
    _get_article_details,
    _is_real_photo,
    check_article_image_need,
    search_articles,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_article(**overrides) -> WikiArticle:
    defaults = {
        "title": "Test Article",
        "pageid": 1,
        "url": "https://en.wikipedia.org/?curid=1",
        "image_count": 0,
        "needs_photo": False,
        "extract": None,
        "categories": [],
    }
    defaults.update(overrides)
    return WikiArticle(**defaults)


def _mock_api_response(json_data: dict) -> MagicMock:
    """Create a fake requests.Response."""
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# _is_real_photo
# ---------------------------------------------------------------------------

class TestIsRealPhoto:
    def test_normal_image_is_photo(self):
        assert _is_real_photo("File:Eiffel Tower from Champ de Mars.jpg") is True

    def test_commons_logo_is_not_photo(self):
        assert _is_real_photo("File:Commons-logo.svg") is False

    def test_flag_is_not_photo(self):
        assert _is_real_photo("File:Flag of France.svg") is False

    def test_icon_is_not_photo(self):
        assert _is_real_photo("File:Icon something.png") is False


# ---------------------------------------------------------------------------
# _get_article_details
# ---------------------------------------------------------------------------

class TestGetArticleDetails:
    """Test _get_article_details with mocked API responses."""

    DETAIL_RESPONSE = {
        "query": {
            "pages": {
                "42": {
                    "pageid": 42,
                    "title": "Test Page",
                    "images": [
                        {"title": "File:Photo1.jpg"},
                        {"title": "File:Photo2.jpg"},
                        {"title": "File:Commons-logo.svg"},
                    ],
                    "categories": [
                        {"title": "Category:Buildings"},
                        {"title": "Category:Articles needing photos"},
                    ],
                    "extract": "This is a short extract.",
                }
            }
        }
    }

    @patch("wikipicture.wiki_analyzer._rate_limit")
    def test_counts_real_images(self, _mock_rl):
        session = MagicMock()
        session.get.return_value = _mock_api_response(self.DETAIL_RESPONSE)

        result = _get_article_details(42, session=session)

        # 2 real photos (Commons-logo filtered out)
        assert result["image_count"] == 2

    @patch("wikipicture.wiki_analyzer._rate_limit")
    def test_detects_needs_photo_category(self, _mock_rl):
        session = MagicMock()
        session.get.return_value = _mock_api_response(self.DETAIL_RESPONSE)

        result = _get_article_details(42, session=session)

        assert result["needs_photo"] is True

    @patch("wikipicture.wiki_analyzer._rate_limit")
    def test_no_needs_photo_when_absent(self, _mock_rl):
        response = {
            "query": {
                "pages": {
                    "99": {
                        "pageid": 99,
                        "images": [{"title": "File:Nice.jpg"}],
                        "categories": [{"title": "Category:Parks"}],
                        "extract": "A park.",
                    }
                }
            }
        }
        session = MagicMock()
        session.get.return_value = _mock_api_response(response)

        result = _get_article_details(99, session=session)

        assert result["needs_photo"] is False
        assert result["image_count"] == 1

    @patch("wikipicture.wiki_analyzer._rate_limit")
    def test_extract_and_categories(self, _mock_rl):
        session = MagicMock()
        session.get.return_value = _mock_api_response(self.DETAIL_RESPONSE)

        result = _get_article_details(42, session=session)

        assert result["extract"] == "This is a short extract."
        assert "Buildings" in result["categories"]
        assert "Articles needing photos" in result["categories"]

    @patch("wikipicture.wiki_analyzer._rate_limit")
    def test_url_format(self, _mock_rl):
        session = MagicMock()
        session.get.return_value = _mock_api_response(self.DETAIL_RESPONSE)

        result = _get_article_details(42, session=session)

        assert result["url"] == "https://en.wikipedia.org/?curid=42"

    @patch("wikipicture.wiki_analyzer._rate_limit")
    def test_handles_empty_response(self, _mock_rl):
        session = MagicMock()
        session.get.return_value = _mock_api_response({"query": {"pages": {"7": {}}}})

        result = _get_article_details(7, session=session)

        assert result["image_count"] == 0
        assert result["needs_photo"] is False
        assert result["extract"] is None
        assert result["categories"] == []


# ---------------------------------------------------------------------------
# check_article_image_need
# ---------------------------------------------------------------------------

class TestCheckArticleImageNeed:
    def test_tagged_needs_photo(self):
        article = _make_article(needs_photo=True, image_count=5)
        assert check_article_image_need(article) == "Needs photo (tagged)"

    def test_no_images(self):
        article = _make_article(image_count=0)
        assert check_article_image_need(article) == "No images"

    def test_few_images_one(self):
        article = _make_article(image_count=1)
        assert check_article_image_need(article) == "Few images (1)"

    def test_few_images_two(self):
        article = _make_article(image_count=2)
        assert check_article_image_need(article) == "Few images (2)"

    def test_well_illustrated(self):
        article = _make_article(image_count=5)
        assert check_article_image_need(article) == "Well-illustrated"

    def test_tagged_takes_precedence_over_no_images(self):
        article = _make_article(needs_photo=True, image_count=0)
        assert check_article_image_need(article) == "Needs photo (tagged)"


# ---------------------------------------------------------------------------
# search_articles — deduplication & merging
# ---------------------------------------------------------------------------

class TestSearchArticles:
    """Test search_articles with mocked HTTP calls."""

    TEXT_SEARCH_RESPONSE = {
        "query": {
            "search": [
                {"title": "Eiffel Tower", "pageid": 10},
                {"title": "Champ de Mars", "pageid": 20},
            ]
        }
    }

    GEO_SEARCH_RESPONSE = {
        "query": {
            "geosearch": [
                {"title": "Eiffel Tower", "pageid": 10},  # duplicate
                {"title": "Trocadéro", "pageid": 30},
            ]
        }
    }

    DETAIL_TEMPLATE = {
        "query": {
            "pages": {
                "{pid}": {
                    "images": [],
                    "categories": [],
                    "extract": "Summary.",
                }
            }
        }
    }

    def _details_side_effect(self, url, params=None, **kwargs):
        """Return a mock response based on the request params."""
        params = params or {}
        resp = MagicMock()
        resp.raise_for_status.return_value = None

        if "list" in params:
            if params["list"] == "search":
                resp.json.return_value = self.TEXT_SEARCH_RESPONSE
            elif params["list"] == "geosearch":
                resp.json.return_value = self.GEO_SEARCH_RESPONSE
        elif "pageids" in params:
            pid = str(params["pageids"])
            resp.json.return_value = {
                "query": {
                    "pages": {
                        pid: {
                            "pageid": int(pid),
                            "images": [{"title": "File:Photo.jpg"}],
                            "categories": [],
                            "extract": "A summary.",
                        }
                    }
                }
            }
        else:
            resp.json.return_value = {}

        return resp

    @patch("wikipicture.wiki_analyzer._rate_limit")
    def test_deduplication(self, _mock_rl):
        session = MagicMock()
        session.get.side_effect = self._details_side_effect

        with patch("wikipicture.wiki_analyzer._make_session", return_value=session):
            articles = search_articles("Eiffel Tower", lat=48.858, lon=2.2945)

        pageids = [a.pageid for a in articles]
        # "Eiffel Tower" (pageid=10) should appear only once
        assert pageids.count(10) == 1
        # All three unique articles present
        assert set(pageids) == {10, 20, 30}

    @patch("wikipicture.wiki_analyzer._rate_limit")
    def test_text_search_only(self, _mock_rl):
        session = MagicMock()
        session.get.side_effect = self._details_side_effect

        with patch("wikipicture.wiki_analyzer._make_session", return_value=session):
            articles = search_articles("Eiffel Tower")

        # No geo search without coords
        pageids = [a.pageid for a in articles]
        assert set(pageids) == {10, 20}

    @patch("wikipicture.wiki_analyzer._rate_limit")
    def test_returns_at_most_10(self, _mock_rl):
        many_results = {
            "query": {
                "search": [
                    {"title": f"Article {i}", "pageid": i} for i in range(15)
                ]
            }
        }

        def side_effect(url, params=None, **kwargs):
            params = params or {}
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            if "list" in params:
                resp.json.return_value = many_results
            elif "pageids" in params:
                pid = str(params["pageids"])
                resp.json.return_value = {
                    "query": {"pages": {pid: {"images": [], "categories": []}}}
                }
            else:
                resp.json.return_value = {}
            return resp

        session = MagicMock()
        session.get.side_effect = side_effect

        with patch("wikipicture.wiki_analyzer._make_session", return_value=session):
            articles = search_articles("Some Place")

        assert len(articles) <= 10

    @patch("wikipicture.wiki_analyzer._rate_limit")
    def test_handles_api_error(self, _mock_rl):
        session = MagicMock()
        session.get.return_value = _mock_api_response({})

        with patch("wikipicture.wiki_analyzer._make_session", return_value=session):
            articles = search_articles("Nowhere")

        assert articles == []
