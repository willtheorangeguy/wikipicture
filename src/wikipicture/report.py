"""HTML report generator for WikiPicture."""

from __future__ import annotations

import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

TEMPLATES_DIR = Path(__file__).parent / "templates"


def generate_report(
    opportunities: list,
    stats: dict,
    output_path: Path,
) -> Path:
    """Render an HTML report and write it to *output_path*.

    Parameters
    ----------
    opportunities:
        Scored opportunity objects to display in the report.
    stats:
        Dict with keys ``total_photos``, ``photos_with_gps``,
        ``unique_locations``.
    output_path:
        Destination file path for the generated HTML report.

    Returns
    -------
    Path
        The *output_path* that was written to.
    """
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template("report.html.j2")

    html = template.render(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        total_photos=stats.get("total_photos", 0),
        photos_with_gps=stats.get("photos_with_gps", 0),
        unique_locations=stats.get("unique_locations", 0),
        opportunities=opportunities,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def open_report(path: Path) -> None:
    """Open the HTML report in the default web browser."""
    webbrowser.open(Path(path).resolve().as_uri())
