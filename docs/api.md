# WikiPicture — API

No API of its own. It consumes three public, unauthenticated ones.

## OpenStreetMap Nominatim

Reverse geocoding: cluster centroid → place name.

| | |
|---|---|
| Auth | None |
| Rate limit | **1 request/second, absolute** |
| Enforced by | `geocoder.py`, module-level timestamp and `time.sleep` |
| User-Agent | Required, and set in `http_client.py` |

Nominatim's usage policy is not advisory — ignoring it gets your IP blocked, and the service is
volunteer-funded. The one-second interval is why clustering exists.

## Wikipedia API

Two searches per location: by place name, and geosearch within 10 km. For each article, the image
count and whether it carries a "needs photo" tag.

Unauthenticated, and generous, but the same User-Agent policy applies.

## Wikimedia Commons API

Geosearch around each location to count nearby images and read their categories and dates. Feeds
the `SaturationLevel` and the freshness score.

## The shared session

`http_client.make_session()` returns a `requests.Session` used by all three modules:

```python
USER_AGENT = (
    f"WikiPicture/{__version__} "
    "(https://github.com/willtheorangeguy/wikipicture) "
    f"python-requests/{requests.__version__}"
)
```

Name, version, and a contact URL — what Wikimedia's
[User-Agent policy](https://meta.wikimedia.org/wiki/User-Agent_policy) requires. A generic or
absent agent is aggressively rate-limited.

Retries: 4 attempts, 1.5× backoff, on 429/500/502/503, `Retry-After` respected, `GET` only —
so a retry can never repeat a non-idempotent request.

## Caching

Every external result is cached in `~/.wikipicture/cache.db`:

| Table | Keyed by |
|---|---|
| `geocode_cache` | Coordinates |
| `wiki_cache` | Coordinates |
| `commons_cache` | Coordinates |
| `photo_cache` | Path, invalidated by mtime |

A second run over the same library makes almost no requests. `--no-cache` bypasses this — worth
avoiding except when debugging, since it means repeating rate-limited work.

## If you fork this

Keep the rate limiting and the User-Agent. They are the difference between a well-behaved client
and one that gets a shared IP range blocked from services other people depend on.

## Reference

- [Nominatim usage policy](https://operations.osmfoundation.org/policies/nominatim/)
- [Wikimedia User-Agent policy](https://meta.wikimedia.org/wiki/User-Agent_policy)
- [MediaWiki API](https://www.mediawiki.org/wiki/API:Main_page)
