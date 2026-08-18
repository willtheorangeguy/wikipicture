# WikiPicture — Installation

## Requirements

| | |
|---|---|
| Python | 3.10, 3.11, or 3.12 (what CI covers) |
| Network | For the three public APIs; no keys needed |
| Disk | A SQLite cache in `~/.wikipicture/` |

## Install

```bash
git clone https://github.com/willtheorangeguy/wikipicture.git
cd wikipicture
pip install -e ".[dev]"
```

## What comes with it

| Dependency | For |
|---|---|
| `Pillow` | JPEG reading and EXIF |
| `pillow-heif` | HEIC/HEIF — iPhone photos |
| `opencv-python-headless` | Laplacian blur detection |
| `requests` | The three APIs |
| `click` | The CLI |
| `Jinja2` | The HTML report |
| `tqdm` | Progress bars |

`opencv-python-headless` is the largest of these. The **headless** build is deliberate: it omits
the GUI libraries, which matters on a server and avoids a common conflict with a separately
installed `opencv-python`. If you have both, pip will let you, and imports become unpredictable —
keep one.

`tqdm` is imported defensively (`try/except ImportError`), so a missing progress bar degrades
rather than failing.

## HEIC support

`pillow-heif` covers iPhone photos, which is most of the modern travel-photo case. It is a
required dependency, so nothing extra is needed — but if HEIC files are skipped, confirm it
imported cleanly rather than assuming the photos lack GPS.

## Development shim

```bash
python main.py scan /path/to/photos --output report.html
```

`main.py` at the repository root prepends `src/` to `sys.path` and calls `cli.main()`. It exists
so the tool can be run without installing; the installed `wikipicture` command is the real entry
point.

## Verify

```bash
pytest tests/ -v
wikipicture --version
wikipicture ./a-few-photos --limit 5 --no-open-report
```

The test suite mocks every HTTP call with `responses`, so it needs no network.

## The cache

Created on first run at `~/.wikipicture/cache.db`. Four tables — geocoding, Wikipedia, Commons,
and photo quality — keyed by coordinate or path, and invalidated by file mtime.

```bash
wikipicture clear-cache                    # everything
wikipicture clear-cache --older-than 30    # entries over 30 days old
```

Deleting the file is equivalent.

## Uninstall

```bash
pip uninstall wikipicture
rm -rf ~/.wikipicture
```
