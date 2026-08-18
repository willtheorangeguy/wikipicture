# WikiPicture — Configuration

Command-line flags only. No config file, no environment variables.

## `scan`

| Flag | Default | Effect |
|---|---|---|
| `PHOTO_DIR` | required | Directory to scan |
| `-o`, `--output` | `wikipicture_report.html` | Report path |
| `--limit` | none | Maximum photos to process |
| `--skip-quality-check` | off | Skip blur and resolution assessment |
| `--no-cache` | off | Ignore and do not write the cache |
| `--open-report` / `--no-open-report` | open | Launch a browser when finished |
| `--cluster-distance` | `500` | Metres; how close photos must be to group |
| `--max-article-candidates` | `5` | Ranked articles kept per photo (1–10) |
| `-v`, `--verbose` | off | DEBUG logging |

## `clear-cache`

| Flag | Effect |
|---|---|
| `--older-than N` | Drop entries older than N days; omit to clear everything |

## The `scan` shortcut, and its edge

`wikipicture ./photos` works because the CLI injects `scan` when the first argument is neither a
known command nor an option.

That last condition matters: **options must come after the path**, or the shortcut does not fire.

```bash
wikipicture ./photos --output report.html      # works
wikipicture scan --output report.html ./photos # works
wikipicture --output report.html ./photos      # fails: "no such option"
```

Recorded in [`internal/known-issues.md`](./internal/known-issues.md).

## `--cluster-distance`

The most consequential setting. It decides what counts as one place, which decides how many
lookups happen and therefore how long the run takes.

| Value | Effect |
|---|---|
| 100–200 m | Distinguishes buildings; many more lookups |
| 500 m (default) | A neighbourhood or a large site |
| 1000 m+ | A town centre; far fewer lookups, coarser matches |

At one geocode per second, halving the cluster count halves the wall time.

## `--skip-quality-check`

Skips the OpenCV work. Two effects: the run is faster, and every photo scores 10 of the 15
quality points as "not assessed (benefit of the doubt)" rather than 15 or 0.

So skipping it does not make photos score worse — it makes the score less discriminating.

## `--max-article-candidates`

How many ranked articles the report lists per photo. The **score** always uses the single best
article; this only changes how many alternatives you see.

## Scoring

Fixed in `scorer.py`, totalling 100:

| Component | Max | Rule |
|---|---|---|
| Article need | 40 | 40 "needs photo" · 35 no images · 20 ≤2 · 10 ≤5 · 5 more |
| Commons saturation | 30 | 30 none · 25 low · 15 medium · 5 high · 0 saturated |
| Photo quality | 15 | 15 suitable · 8 sharp-but-low-res issue · 5 · 0 · 10 unassessed |
| Freshness | 15 | 15 none · 10 over 5 years · 5 two to five · 0 recent |

Recommendation thresholds: 70 highly recommended, 45 recommended, 25 maybe.

## Cache location

`~/.wikipicture/cache.db`, not configurable. `--no-cache` bypasses it entirely.
