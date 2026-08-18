# WikiPicture — Troubleshooting

## No photos found

The scan looks for JPEG and HEIC/HEIF **with GPS in EXIF**. Anything else is skipped silently.

| Check | How |
|---|---|
| Do they have GPS? | Any EXIF viewer, or `exiftool -GPS* photo.jpg` |
| Right format? | Other formats are not scanned |
| Right directory? | The scan is over the directory you name |
| HEIC support loaded? | `python -c "import pillow_heif"` |

Location services being off when the photo was taken, or an editor stripping EXIF on export, are
the two usual causes — and both are invisible until you look.

## `no such option: --output`

The implied `scan` command only fires when the first argument is a path. Put the directory first:

```bash
wikipicture ./photos --output report.html      # works
wikipicture scan --output report.html ./photos # works
```

Recorded in [`internal/known-issues.md`](./internal/known-issues.md).

## It is extremely slow

Geocoding is capped at one request per second, by Nominatim's policy. Time scales with the number
of **clusters**:

```bash
wikipicture ./photos --limit 50                 # try a subset
wikipicture ./photos --cluster-distance 1000    # fewer, larger clusters
```

Re-runs use the cache and are much faster. `--skip-quality-check` removes the OpenCV work, which
helps on large libraries.

## HTTP 429 / rate limited

The client retries with backoff and honours `Retry-After`, so short bursts resolve themselves. If
it persists, you may have been temporarily blocked — wait, and do not run with `--no-cache`,
which repeats every lookup.

## No Wikipedia articles for a location

Genuinely possible. The search covers a 10 km radius by name and geosearch; remote places may
have nothing notable, and an article that does not exist cannot need a photo.

Check the reverse-geocoded place name in the report — an unhelpful name (a road, a postcode)
produces poor text search results.

## Every photo scores low

Usually a well-photographed place: Commons saturation is 30 of the 100 points and freshness
another 15, so somewhere already covered scores low by design. That is the tool working.

If quality is the culprit, the breakdown says so.

## "No existing photos at this location" but Commons shows plenty

A bug. When Commons image dates cannot be parsed, freshness reports no photos and awards full
marks. Cross-check the Commons count in the same row. Recorded in
[`internal/known-issues.md`](./internal/known-issues.md).

## `ImportError` on startup

The CLI catches import failures and reports them. Usually an incomplete install:

```bash
pip install -e ".[dev]"
```

If OpenCV is the problem, check you do not have both `opencv-python` and
`opencv-python-headless` installed — keep the headless one.

## Report is empty or malformed

The template is `src/wikipicture/templates/report.html.j2`, packaged with the module. If it is
missing, the install is incomplete — reinstall rather than copying the file.

## Cache seems stale

```bash
wikipicture clear-cache --older-than 30
wikipicture clear-cache
```

Photo entries invalidate on mtime, but API results do not expire on their own — an article that
gained images since your last run stays cached until cleared.

## Windows: UNC path not found

`\server\share` handling is in `network_paths.py` and only active on Windows. Map the share to a
drive letter as a workaround, and report the path that failed.

## Still stuck

[Open an issue](https://github.com/willtheorangeguy/wikipicture/issues/new/choose) with the
command, the error, your OS and Python version, and roughly how many photos are involved.
