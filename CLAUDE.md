# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WikiPicture is a Python CLI tool that scans geotagged travel photos and identifies Wikipedia articles that need images at those locations, helping photographers contribute to Wikipedia and Wikimedia Commons.

## Commands

```bash
# Install for development
pip install -e ".[dev]"

# Run all tests
python -m pytest tests/ -v --tb=short

# Run a single test file
python -m pytest tests/test_scorer.py -v

# Run a single test by name
python -m pytest tests/test_scorer.py::TestScoredArticle::test_score_breakdown -v

# Run with coverage
python -m pytest tests/ --cov=wikipicture --cov-report=html

# Run the CLI during development (adds src/ to path)
python main.py scan /path/to/photos --output report.html
python main.py clear-cache --older-than 30
```

## Architecture

The tool runs a linear pipeline, defined in `src/wikipicture/cli.py` (`_run_pipeline()`, `_process_cluster()`):

1. **`exif_extractor.py`** — Scans a directory for JPEG/HEIC photos, extracts GPS coordinates and camera metadata via EXIF. Returns `PhotoMetadata` dataclasses.
2. **`clustering.py`** — Groups nearby photos by Haversine distance (default 500 m). Returns `PhotoCluster` dataclasses, each with a centroid lat/lon.
3. **`geocoder.py`** — Reverse-geocodes each cluster centroid via OSM Nominatim (1 req/sec rate limit). Returns `LocationInfo`.
4. **`wiki_analyzer.py`** — Searches Wikipedia by text and geosearch (10 km radius) for articles missing images. Returns `WikiArticle` list.
5. **`commons_checker.py`** — Checks Wikimedia Commons geosearch to assess how saturated an area is with existing photos. Returns `CommonsResult` with a `SaturationLevel` enum.
6. **`quality_filter.py`** — Assesses each photo's resolution (megapixels) and sharpness (OpenCV Laplacian variance). Returns `QualityAssessment`.
7. **`scorer.py`** — Combines the above signals into a 0–100 score per `PhotoOpportunity`. Score breakdown: article need (0–40 pts), Commons saturation (0–30), photo quality (0–15), freshness bonus (0–15).
8. **`report.py`** — Renders `src/wikipicture/templates/report.html.j2` (Jinja2) into an interactive HTML report with a sortable table.

### Caching

`cache.py` wraps all external API calls in a SQLite cache at `~/.wikipicture/cache.db`. The four tables (`geocode_cache`, `wiki_cache`, `commons_cache`, `photo_cache`) are keyed by coordinates/path and invalidated by file mtime. The cache is a context manager used in `_run_pipeline()` and bypassed with `--no-cache`.

### HTTP Client

`http_client.py` creates a `requests.Session` with a Wikimedia-policy-compliant User-Agent and a retry strategy (4 retries, 1.5× backoff, respects `Retry-After`). All three external API modules (geocoder, wiki_analyzer, commons_checker) share this session.

### External APIs

All three APIs are public and unauthenticated:
- **OSM Nominatim** — reverse geocoding, 1 req/sec enforced in `geocoder.py`
- **Wikipedia API** — article search and metadata
- **Wikimedia Commons API** — geosearch and category data

The `responses` library is used in tests to mock all HTTP calls — no real network access occurs during `pytest`.

## Key Conventions

- Source lives under `src/wikipicture/`; tests mirror module names under `tests/`.
- Each pipeline stage has a dedicated module with its own dataclass(es) as the primary output type.
- `main.py` at the repo root is a dev-only shim that prepends `src/` to `sys.path` before invoking `cli.main()`.
- Windows UNC path handling (`\\server\share`) is isolated in `network_paths.py` and only activated on `sys.platform == 'win32'`.
- CI runs on Python 3.10, 3.11, and 3.12 via `.github/workflows/ci.yml`. Publishing to PyPI is handled by `.github/workflows/publish.yml` using OIDC trusted publishing.
