"""Shared HTTP session setup for the web APIs used by WikiPicture."""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from wikipicture import __version__

# Wikimedia's User-Agent policy
# (https://meta.wikimedia.org/wiki/User-Agent_policy) requires a descriptive
# agent string with contact information; requests without it are aggressively
# rate-limited (429 Too Many Requests). Nominatim has the same requirement.
USER_AGENT = (
    f"WikiPicture/{__version__} "
    "(https://github.com/willtheorangeguy/wikipicture) "
    f"python-requests/{requests.__version__}"
)

# Retry transient failures, honouring the server's Retry-After header on 429.
_RETRY = Retry(
    total=4,
    backoff_factor=1.5,
    status_forcelist=(429, 500, 502, 503),
    allowed_methods=("GET",),
    respect_retry_after_header=True,
)


def make_session() -> requests.Session:
    """Return a session with a policy-compliant User-Agent and retry handling."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.mount("https://", HTTPAdapter(max_retries=_RETRY))
    return session
