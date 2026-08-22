# WikiPicture — Architecture

A linear pipeline, one module per stage, each producing a dataclass the next one consumes.

```text
photos → EXIF GPS → clusters → geocode → Wikipedia → Commons → quality → score → report
```

`cli.py`'s `_run_pipeline()` and `_process_cluster()` drive it.

## 1. `exif_extractor.py`

Walks the directory for JPEG and HEIC/HEIF, reads EXIF GPS and camera metadata, and returns
`PhotoMetadata`. Photos without coordinates are skipped — they are the input this tool cannot use.

## 2. `clustering.py`

Groups photos within `--cluster-distance` (500 m default) by Haversine distance, producing
`PhotoCluster` with a centroid.

This is the stage that makes the tool practical. Geocoding runs at one request per second, so a
thousand photos of one holiday would be seventeen minutes of lookups; clustered into thirty
places, it is thirty seconds.

## 3. `geocoder.py`

Reverse-geocodes each centroid via Nominatim, returning `LocationInfo`. A module-level timestamp
enforces the one-request-per-second minimum interval — Nominatim's usage policy, not a
guess.

## 4. `wiki_analyzer.py`

Two searches per location — by place name, and geosearch within 10 km — then `WikiArticle` per
result with an image count and whether the article carries a "needs photo" tag.

## 5. `commons_checker.py`

Geosearch on Commons to count nearby images, plus their categories and date range, producing
`CommonsResult` with a `SaturationLevel` enum (`NONE` … `SATURATED`) and oldest/newest image
years.

## 6. `quality_filter.py`

Resolution in megapixels, and sharpness via OpenCV's Laplacian variance — the standard cheap blur
metric: a blurred image has less high-frequency detail, so the variance of its Laplacian is low.

Skipped entirely with `--skip-quality-check`, in which case scoring gives the benefit of the
doubt rather than zero.

## 7. `scorer.py`

Four independent signals into one 0–100 score:

| Component | Max | Question |
|---|---|---|
| Article need | 40 | Does an article here lack images? |
| Commons saturation | 30 | Is the place already photographed? |
| Photo quality | 15 | Is your photo good enough? |
| Freshness | 15 | Is the existing coverage old? |

Every component returns a `(score, reason)` pair, so the report can explain each number rather
than presenting a total. That is the right shape for a recommendation tool — a bare score would
be unarguable.

`_pick_best_article` selects by `(not needs_photo, image_count)` for the score;
`_rank_articles` scores every candidate and keeps the top `--max-article-candidates` for display.

Two problems in the freshness path — a "no existing photos" message and full marks awarded where
photos exist but their dates could not be parsed — are in
[`internal/known-issues.md`](./internal/known-issues.md).

## 8. `report.py`

Renders `templates/report.html.j2` into a self-contained sortable HTML table with thumbnails,
links, and colour-coded priority, then optionally opens it.

## Cross-cutting

**`cache.py`** — SQLite at `~/.wikipicture/cache.db`, four tables keyed by coordinate or path,
photo entries invalidated by mtime. Used as a context manager in `_run_pipeline`, bypassed by
`--no-cache`. Without it a re-run would repeat every rate-limited lookup, so the cache is what
makes iterating on a library bearable.

**`http_client.py`** — one `requests.Session` shared by all three API modules, carrying a
Wikimedia-policy-compliant User-Agent (name, version, project URL) and a retry strategy: 4
retries, 1.5× backoff, `Retry-After` respected on 429.

**`network_paths.py`** — Windows UNC paths (`\server\share`), isolated here and active only on
`sys.platform == 'win32'`, so the platform-specific case does not leak into the scanner.

## Design notes

**One dataclass per stage.** Each module's output is the next's input, so a stage can be tested
by constructing its input directly — which is what the test suite does.

**Every network call is mocked in tests** with `responses`. No test touches the network.

**Nothing is uploaded.** The pipeline ends at an HTML file. Coordinates go out; photos do not.
