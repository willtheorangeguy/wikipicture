<!-- Logo -->
<h1 align="center">WikiPicture</h1>

<!-- Copy -->
<h4 align="center">Finds the Wikipedia articles your travel photos could illustrate — by reading where the photos were taken.</h4>

<!-- Badges -->
<div align="center">
  <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/willtheorangeguy/wikipicture/ci.yml?label=ci">
  <img alt="GitHub Issues" src="https://img.shields.io/github/issues/willtheorangeguy/wikipicture">
  <img alt="GitHub Pull Requests" src="https://img.shields.io/github/issues-pr/willtheorangeguy/wikipicture">
  <img alt="License" src="https://img.shields.io/github/license/willtheorangeguy/wikipicture">
</div>

<!-- Navigation -->
<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#support">Support</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## Key Features

- Reads GPS from JPEG and HEIC/HEIF photos, and clusters nearby ones so a hundred shots of one place cost one lookup.
- Finds Wikipedia articles near each location that have no images, few images, or a "needs photo" tag.
- Checks Wikimedia Commons to see how well covered the place already is — and how recently.
- Screens your photos for resolution and blur, so it does not recommend uploading a shaky one.
- Scores each photo out of 100 and explains every point in the breakdown.
- Produces a sortable HTML report with thumbnails and links.
- Public, unauthenticated APIs only: no keys, no account, and a local SQLite cache so re-runs are fast.

## Installation

```bash
pip install -e .
```

Python 3.10+. See [`docs/installation.md`](docs/installation.md).

## Usage

```bash
wikipicture /path/to/photos --output report.html
wikipicture /path/to/photos --limit 50 --skip-quality-check
wikipicture clear-cache --older-than 30
```

The `scan` subcommand is implied when the first argument is a path — but only when it comes first. See [`docs/configuration.md`](docs/configuration.md).

## Documentation

Full documentation lives in [`docs/`](docs/README.md):
[Quickstart](docs/quickstart.md) · [Installation](docs/installation.md) · [Configuration](docs/configuration.md) · [Architecture](docs/architecture.md) · [API](docs/api.md) · [Development](docs/development.md) · [FAQ](docs/faq.md) · [Troubleshooting](docs/troubleshooting.md) · [Roadmap](docs/roadmap.md)

## Support

Open a [GitHub Discussion](https://github.com/willtheorangeguy/wikipicture/discussions/new) or file an [issue](https://github.com/willtheorangeguy/wikipicture/issues/new/choose).

## Contributing

Contributions welcome. See the org-wide [Contributing Guide](https://github.com/willtheorangeguy/.github/blob/main/CONTRIBUTING.md) and [Code of Conduct](https://github.com/willtheorangeguy/.github/blob/main/CODE_OF_CONDUCT.md).

## Credits

Uses [OpenStreetMap Nominatim](https://nominatim.org/), the [Wikipedia API](https://www.mediawiki.org/wiki/API:Main_page), and [Wikimedia Commons](https://commons.wikimedia.org/). Built with [Pillow](https://python-pillow.org/), [OpenCV](https://opencv.org/), [Click](https://click.palletsprojects.com/), and [Jinja2](https://jinja.palletsprojects.com/).

## License

MIT — see [`LICENSE.md`](LICENSE.md).

> It tells you where your photos would be useful. Uploading them is still your decision, and Commons has its own standards worth reading first.
