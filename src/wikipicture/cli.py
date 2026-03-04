"""Command-line interface for WikiPicture."""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict
from pathlib import Path

import click

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

try:
    from wikipicture.exif_extractor import scan_directory
    from wikipicture.clustering import cluster_photos
    from wikipicture.geocoder import reverse_geocode
    from wikipicture.wiki_analyzer import search_articles
    from wikipicture.commons_checker import check_commons_saturation
    from wikipicture.quality_filter import assess_quality
    from wikipicture.scorer import score_opportunity, rank_opportunities
    from wikipicture.report import generate_report, open_report
    from wikipicture.cache import Cache
except ImportError as exc:  # pragma: no cover
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

logger = logging.getLogger("wikipicture")


def _setup_logging(verbose: bool) -> None:
    """Configure root logger for the wikipicture package."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.setLevel(level)
    logger.handlers = [handler]


def _progress(iterable, **kwargs):
    """Wrap *iterable* in a tqdm progress bar when available."""
    if tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


class _DefaultGroup(click.Group):
    """A Click group that delegates to 'scan' when no subcommand is given."""

    def parse_args(self, ctx, args):
        # If the first argument looks like a path (not a known command), inject 'scan'.
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = ["scan"] + args
        return super().parse_args(ctx, args)


@click.group(cls=_DefaultGroup)
@click.version_option(package_name="wikipicture")
def main() -> None:
    """WikiPicture — find Wikipedia articles that need your travel photos."""


@main.command()
@click.argument("photo_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-o", "--output", default="wikipicture_report.html", type=click.Path(path_type=Path),
              help="Output HTML report path.")
@click.option("--limit", default=None, type=int, help="Maximum number of photos to process.")
@click.option("--skip-quality-check", is_flag=True, help="Skip blur/resolution quality filtering.")
@click.option("--no-cache", is_flag=True, help="Disable caching.")
@click.option("--open-report/--no-open-report", "open_report_flag", default=True, help="Auto-open report in browser.")
@click.option("--cluster-distance", default=500, type=float,
              help="Distance threshold in metres for grouping photos.")
@click.option("--max-article-candidates", default=5, type=click.IntRange(1, 10),
              help="Number of ranked Wikipedia article candidates per photo (1–10, default 5).")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose (DEBUG) logging.")
def scan(
    photo_dir: Path,
    output: Path,
    limit: int | None,
    skip_quality_check: bool,
    no_cache: bool,
    open_report_flag: bool,
    cluster_distance: float,
    max_article_candidates: int,
    verbose: bool,
) -> None:
    """Scan geotagged photos and match them to Wikipedia articles."""
    if _IMPORT_ERROR is not None:  # pragma: no cover
        raise click.ClickException(
            f"Required dependency is missing: {_IMPORT_ERROR}"
        )

    _setup_logging(verbose)

    # -- 1. Cache ----------------------------------------------------------
    cache: Cache | None = None
    if not no_cache:
        cache = Cache()
        logger.debug("Cache enabled")

    try:
        _run_pipeline(photo_dir, output, limit, skip_quality_check,
                      cache, open_report_flag, cluster_distance, max_article_candidates)
    except KeyboardInterrupt:
        click.echo("\nInterrupted — partial results were not saved.")
        sys.exit(130)
    finally:
        if cache is not None:
            cache.close()


def _run_pipeline(
    photo_dir: Path,
    output: Path,
    limit: int | None,
    skip_quality_check: bool,
    cache: Cache | None,
    should_open_report: bool,
    cluster_distance: float,
    max_article_candidates: int = 5,
) -> None:
    # -- 2. Scan directory -------------------------------------------------
    click.echo(f"Scanning {photo_dir} for geotagged photos …")
    all_photos = scan_directory(photo_dir)

    geotagged = [p for p in all_photos if p.latitude is not None and p.longitude is not None]
    if limit is not None:
        geotagged = geotagged[:limit]

    if not geotagged:
        click.echo("No geotagged photos found.")
        return

    click.echo(f"Found {len(geotagged)} geotagged photo(s) out of {len(all_photos)} total.")

    # -- 3. Cluster --------------------------------------------------------
    clusters = cluster_photos(geotagged, distance_threshold_m=cluster_distance)
    click.echo(f"Grouped into {len(clusters)} location cluster(s).")

    # -- 4. Process each cluster -------------------------------------------
    opportunities: list = []

    for cluster in _progress(clusters, desc="Analysing locations", unit="loc"):
        try:
            _process_cluster(cluster, cache, skip_quality_check, opportunities, max_article_candidates)
        except KeyboardInterrupt:
            raise
        except Exception:
            logger.warning(
                "Failed to process cluster at (%.4f, %.4f) — skipping.",
                cluster.center_lat, cluster.center_lon,
                exc_info=True,
            )

    # -- 5. Rank -----------------------------------------------------------
    ranked = rank_opportunities(opportunities)

    # -- 6. Report ---------------------------------------------------------
    stats = {
        "total_photos": len(all_photos),
        "photos_with_gps": len(geotagged),
        "unique_locations": len(clusters),
    }
    report_path = generate_report(ranked, stats, output)
    click.echo(f"\nReport written to {report_path}")

    # -- 7. Summary --------------------------------------------------------
    _print_summary(ranked)

    # -- 8. Open -----------------------------------------------------------
    if should_open_report:
        open_report(report_path)


def _process_cluster(
    cluster,
    cache: Cache | None,
    skip_quality_check: bool,
    opportunities: list,
    max_article_candidates: int = 5,
) -> None:
    lat, lon = cluster.center_lat, cluster.center_lon

    # -- Reverse geocode ---------------------------------------------------
    location_info = None
    cached_geo = cache.get_geocode(lat, lon) if cache else None
    if cached_geo is not None:
        location_name = cached_geo.get("display_name", "Unknown")
    else:
        location_info = reverse_geocode(lat, lon)
        location_name = location_info.display_name
        if cache is not None:
            cache.set_geocode(lat, lon, asdict(location_info))

    cluster.location_name = location_name

    # -- Wikipedia search --------------------------------------------------
    wiki_key = f"{round(lat, 4)},{round(lon, 4)}"
    cached_wiki = cache.get_wiki(wiki_key) if cache else None
    if cached_wiki is not None:
        articles = cached_wiki
    else:
        articles_objs = search_articles(location_name, lat=lat, lon=lon)
        articles = [asdict(a) for a in articles_objs]
        if cache is not None:
            cache.set_wiki(wiki_key, articles)

    # Reconstruct dataclass objects for the scorer
    from wikipicture.wiki_analyzer import WikiArticle
    article_objs = [WikiArticle(**a) for a in articles]

    # -- Commons saturation ------------------------------------------------
    cached_commons = cache.get_commons(lat, lon) if cache else None
    if cached_commons is not None:
        commons_dict = cached_commons
    else:
        commons_result = check_commons_saturation(lat, lon)
        commons_dict = asdict(commons_result)
        # Convert enum to its string value for JSON serialization
        commons_dict["saturation"] = commons_dict["saturation"].value
        if cache is not None:
            cache.set_commons(lat, lon, commons_dict)

    from wikipicture.commons_checker import CommonsResult, SaturationLevel
    # Restore enum for the saturation field
    raw = dict(commons_dict)
    raw["saturation"] = SaturationLevel(raw["saturation"])
    commons_obj = CommonsResult(**raw)

    # -- Quality + scoring for each photo ----------------------------------
    for photo in cluster.photos:
        quality_obj = None
        if not skip_quality_check:
            quality_obj = assess_quality(photo.filepath)

        opp = score_opportunity(
            articles=article_objs,
            commons=commons_obj,
            quality=quality_obj,
            filepath=photo.filepath,
            lat=photo.latitude,
            lon=photo.longitude,
            location_name=location_name,
            max_article_candidates=max_article_candidates,
        )
        opportunities.append(opp)


def _print_summary(ranked: list) -> None:
    """Print the top 5 opportunities to the terminal."""
    if not ranked:
        click.echo("No opportunities found.")
        return

    click.echo("\n── Top opportunities ──")
    for i, opp in enumerate(ranked[:5], 1):
        article_title = opp.best_article.title if opp.best_article else "—"
        click.echo(
            f"  {i}. [{opp.score:.0f}] {opp.location_name}  →  {article_title}"
            f"  ({opp.recommendation})"
        )
    remaining = len(ranked) - 5
    if remaining > 0:
        click.echo(f"  … and {remaining} more in the report.")


@main.command("clear-cache")
@click.option("--older-than", default=None, type=int,
              help="Only clear entries older than this many days.")
def clear_cache(older_than: int | None) -> None:
    """Clear the WikiPicture cache."""
    if _IMPORT_ERROR is not None:  # pragma: no cover
        raise click.ClickException(
            f"Required dependency is missing: {_IMPORT_ERROR}"
        )

    with Cache() as c:
        deleted = c.clear(older_than_days=older_than)
    click.echo(f"Cleared {deleted} cache entry/entries.")
