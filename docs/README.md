# WikiPicture — Documentation

A CLI that reads GPS from your photos, works out which Wikipedia articles near those places need
images, and ranks your photos by how useful they would be.

```
src/wikipicture/
├── cli.py               the pipeline, Click commands
├── exif_extractor.py    GPS and camera metadata from JPEG/HEIC
├── clustering.py        group nearby photos (Haversine)
├── geocoder.py          Nominatim reverse geocoding, 1 req/sec
├── wiki_analyzer.py     Wikipedia search and image-need analysis
├── commons_checker.py   Commons geosearch, saturation, freshness
├── quality_filter.py    resolution and blur
├── scorer.py            0–100 with a breakdown
├── report.py            Jinja2 → HTML
├── cache.py             SQLite at ~/.wikipicture/cache.db
├── http_client.py       shared session, compliant User-Agent, retries
└── network_paths.py     Windows UNC handling
```

## Pages

- [Quickstart](./quickstart.md) — a first report
- [Installation](./installation.md) — Python, HEIC, OpenCV
- [Configuration](./configuration.md) — every flag and what it changes
- [Architecture](./architecture.md) — the eight-stage pipeline
- [API](./api.md) — the three external APIs, and how they are used politely
- [Development](./development.md) — tests, mocked HTTP, conventions
- [FAQ](./faq.md) — privacy, cost, accuracy, what to do with the report
- [Troubleshooting](./troubleshooting.md) — no GPS, no articles, rate limits
- [Roadmap](./roadmap.md) — direction and non-goals
- [Known issues](./internal/known-issues.md) — recorded defects

## What leaves your machine

| Data | Goes to |
|---|---|
| Cluster centre coordinates | Nominatim, Wikipedia, Commons |
| Nothing else | — |

**Your photos are never uploaded.** They are read locally for EXIF and for the blur check; only
the coordinates of clustered locations are sent, and clustering means one lookup per place rather
than one per photo.

All three APIs are public and unauthenticated, so there is no key and no account. Results are
cached in SQLite, so a second run over the same library is nearly free.

## Being a good citizen

Nominatim's usage policy requires a descriptive User-Agent with contact information and at most
one request per second. Both are implemented — `http_client.py` sets the agent, `geocoder.py`
enforces the interval, and retries honour `Retry-After`.

This matters: these are volunteer-funded services, and a tool that hammers them gets the whole
user base blocked. If you fork this, keep the rate limiting.
