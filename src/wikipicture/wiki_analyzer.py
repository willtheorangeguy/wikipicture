"""Search Wikipedia for articles about a location and check if they need images."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "WikiPicture/0.1 (travel photo Wikipedia tool)"
RATE_LIMIT_SECONDS = 0.2

# Common non-photo image prefixes to filter out when counting real photos.
_NON_PHOTO_PREFIXES = (
    "Flag of",
    "Coat of arms",
    "Icon",
    "Logo",
    "Symbol",
    "Pictogram",
    "Kit ",
    "Commons-logo",
    "Wiki",
    "Ambox",
    "Question book",
    "Edit-clear",
    "Text document",
    "Folder Hexagonal",
    "Crystal Clear",
    "Nuvola",
    "Disambig",
    "Stub",
    "Padlock",
    "Lock-",
    "Semi-protection",
    "Increase",
    "Decrease",
    "Steady",
    "Green arrow",
    "Red arrow",
)

# Category substrings indicating an article needs photos.
_NEEDS_PHOTO_INDICATORS = (
    "needing photos",
    "lacking photos",
    "wikipedia requested photographs",
    "articles with missing images",
)


@dataclass
class WikiArticle:
    """A Wikipedia article with metadata about its images."""

    title: str
    pageid: int
    url: str
    image_count: int
    needs_photo: bool
    extract: str | None
    categories: list[str] = field(default_factory=list)


def _make_session() -> requests.Session:
    """Create a requests session with the correct User-Agent."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


_last_request_time: float = 0.0


def _rate_limit() -> None:
    """Enforce a minimum delay between API calls."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < RATE_LIMIT_SECONDS:
        time.sleep(RATE_LIMIT_SECONDS - elapsed)
    _last_request_time = time.monotonic()


def _api_get(session: requests.Session, params: dict) -> dict:
    """Make a rate-limited GET request to the Wikipedia API."""
    params.setdefault("format", "json")
    _rate_limit()
    try:
        resp = session.get(API_URL, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        logger.exception("Wikipedia API request failed")
        return {}


def _is_real_photo(image_title: str) -> bool:
    """Return True if the image title likely represents a real photograph."""
    name = image_title.removeprefix("File:").removesuffix(".svg")
    return not any(name.startswith(prefix) for prefix in _NON_PHOTO_PREFIXES)


def _get_article_details(pageid: int, session: requests.Session | None = None) -> dict:
    """Fetch image count, categories, extract, and URL for a single article.

    Parameters
    ----------
    pageid:
        The Wikipedia page ID.
    session:
        An optional pre-configured requests session.

    Returns
    -------
    dict with keys: image_count, needs_photo, extract, categories, url
    """
    if session is None:
        session = _make_session()

    params = {
        "action": "query",
        "pageids": pageid,
        "prop": "images|categories|extracts",
        "imlimit": "max",
        "cllimit": "max",
        "exsentences": 2,
        "exintro": True,
        "explaintext": True,
    }

    data = _api_get(session, params)
    pages = data.get("query", {}).get("pages", {})
    page = pages.get(str(pageid), {})

    # Count real photos.
    images = page.get("images", [])
    photo_count = sum(1 for img in images if _is_real_photo(img.get("title", "")))

    # Gather categories and check for needs-photo indicators.
    raw_categories = page.get("categories", [])
    categories = [cat.get("title", "").removeprefix("Category:") for cat in raw_categories]
    needs_photo = any(
        indicator in cat.lower()
        for cat in categories
        for indicator in _NEEDS_PHOTO_INDICATORS
    )

    extract: str | None = page.get("extract") or None
    url = f"https://en.wikipedia.org/?curid={pageid}"

    return {
        "image_count": photo_count,
        "needs_photo": needs_photo,
        "extract": extract,
        "categories": categories,
        "url": url,
    }


def search_articles(
    location_name: str,
    lat: float | None = None,
    lon: float | None = None,
) -> list[WikiArticle]:
    """Search Wikipedia for articles about *location_name*.

    Parameters
    ----------
    location_name:
        Free-text location name (e.g. "Eiffel Tower").
    lat, lon:
        Optional coordinates.  When provided an additional geo-search
        within a 10 km radius is performed and the results are merged.

    Returns
    -------
    Up to 10 :class:`WikiArticle` objects, most relevant first.
    """
    session = _make_session()
    seen_pageids: set[int] = set()
    articles: list[WikiArticle] = []

    # --- text search ---
    text_params = {
        "action": "query",
        "list": "search",
        "srsearch": location_name,
        "srlimit": 10,
    }
    text_data = _api_get(session, text_params)
    for item in text_data.get("query", {}).get("search", []):
        pid = item["pageid"]
        if pid not in seen_pageids:
            seen_pageids.add(pid)
            articles.append(
                _stub_article(title=item["title"], pageid=pid)
            )

    # --- geo search ---
    if lat is not None and lon is not None:
        geo_params = {
            "action": "query",
            "list": "geosearch",
            "gscoord": f"{lat}|{lon}",
            "gsradius": 10000,
            "gslimit": 10,
        }
        geo_data = _api_get(session, geo_params)
        for item in geo_data.get("query", {}).get("geosearch", []):
            pid = item["pageid"]
            if pid not in seen_pageids:
                seen_pageids.add(pid)
                articles.append(
                    _stub_article(title=item["title"], pageid=pid)
                )

    # Trim to 10 before fetching details (most relevant first).
    articles = articles[:10]

    # Fill in details for each article.
    for article in articles:
        details = _get_article_details(article.pageid, session=session)
        article.image_count = details["image_count"]
        article.needs_photo = details["needs_photo"]
        article.extract = details["extract"]
        article.categories = details["categories"]
        article.url = details["url"]

    return articles


def _stub_article(title: str, pageid: int) -> WikiArticle:
    """Create a placeholder WikiArticle before details are fetched."""
    return WikiArticle(
        title=title,
        pageid=pageid,
        url="",
        image_count=0,
        needs_photo=False,
        extract=None,
    )


def check_article_image_need(article: WikiArticle) -> str:
    """Return a human-readable assessment of an article's image situation.

    Possible return values:
    - ``"Needs photo (tagged)"`` – article has a needs-photo category
    - ``"No images"`` – zero photos detected
    - ``"Few images (N)"`` – 1-2 photos
    - ``"Well-illustrated"`` – 3+ photos
    """
    if article.needs_photo:
        return "Needs photo (tagged)"
    if article.image_count == 0:
        return "No images"
    if article.image_count <= 2:
        return f"Few images ({article.image_count})"
    return "Well-illustrated"
