# WikiPicture — Quickstart

## 1. Install

```bash
pip install -e .
```

## 2. Scan

```bash
wikipicture /path/to/photos --output report.html
```

The report opens in your browser when it finishes.

**Start small.** Geocoding is limited to one request per second, so the first run over a large
library takes a while. Try `--limit 50` first.

## 3. What happens

1. Photos are scanned for GPS in EXIF — anything without coordinates is skipped.
2. Nearby photos are clustered (500 m by default), so one place costs one set of lookups.
3. Each cluster centre is reverse-geocoded, then searched on Wikipedia and Commons.
4. Your photos are checked for resolution and blur.
5. Everything is scored out of 100 and written to the report.

Every external result is cached in `~/.wikipicture/cache.db`, so a second run is quick.

## 4. Reading the report

Each photo gets a score and a recommendation:

| Score | Recommendation |
|---|---|
| 70+ | Highly recommended |
| 45–69 | Recommended |
| 25–44 | Maybe |
| under 25 | Not recommended |

The breakdown shows where the points came from:

| Component | Max | Rewards |
|---|---|---|
| Article need | 40 | An article tagged "needs photo", or with few or no images |
| Commons saturation | 30 | A place with little existing coverage |
| Photo quality | 15 | Sharp and high enough resolution |
| Freshness | 15 | Existing coverage being old, or absent |

Sort by score and start at the top.

## 5. Then upload — yourself

WikiPicture recommends; it does not upload. Read
[Commons' licensing and scope](https://commons.wikimedia.org/wiki/Commons:Licensing) before
contributing, and add the photo to the article once it is there.

## Useful flags

```bash
wikipicture ./photos --limit 50                 # try a subset first
wikipicture ./photos --skip-quality-check       # skip blur detection (faster)
wikipicture ./photos --cluster-distance 1000    # coarser grouping, fewer lookups
wikipicture ./photos --no-open-report           # don't launch a browser
wikipicture clear-cache --older-than 30         # drop cache entries over 30 days
```

Note the path must come **first** for the `scan` shortcut to work — see
[Configuration](./configuration.md).
