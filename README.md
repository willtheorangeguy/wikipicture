# WikiPicture

Scan your geotagged travel photos and discover which Wikipedia articles need images at those locations.

## Features

- **EXIF GPS extraction** from JPEG and HEIC/HEIF photos
- **Reverse geocoding** via OpenStreetMap Nominatim
- **Wikipedia analysis** — finds articles missing images or tagged "needs photo"
- **Wikimedia Commons saturation check** — see how many photos already exist nearby
- **Photo quality screening** — resolution and blur detection
- **Opportunity scoring** — prioritizes your best upload candidates
- **HTML report** — sortable table with thumbnails, links, and color-coded priority

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Scan a folder of photos and generate a report
wikipicture /path/to/photos --output report.html

# Limit to a specific number of photos
wikipicture /path/to/photos --output report.html --limit 50

# Skip quality filtering
wikipicture /path/to/photos --output report.html --skip-quality-check
```

## How It Works

1. Scans your photo folder for JPEG and HEIC/HEIF files with GPS data
2. Clusters nearby photos to reduce API calls
3. Reverse-geocodes each unique location via Nominatim
4. Searches Wikipedia for articles about each location
5. Checks if those articles need images (few/no photos, "needs photo" tags)
6. Searches Wikimedia Commons to see how saturated each location is
7. Scores each photo by upload opportunity and generates an HTML report

## API Usage

This tool uses **public, unauthenticated** Wikimedia and OpenStreetMap APIs.
Rate limits are respected automatically (1 request/second for Nominatim).
Results are cached in a local SQLite database so re-runs are fast.

## License

MIT
