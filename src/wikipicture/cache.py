"""SQLite-based caching layer for WikiPicture.

Caches geocoding results, Wikipedia article lookups, Wikimedia Commons
queries, and photo metadata so that re-runs don't re-query APIs for
already-processed items.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path.home() / ".wikipicture"


class Cache:
    """Persistent SQLite cache for API responses and photo metadata."""

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            db_path = _DEFAULT_DIR / "cache.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Opening cache at %s", self._db_path)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    # -- context manager --------------------------------------------------

    def __enter__(self) -> Cache:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()

    # -- schema -----------------------------------------------------------

    def _init_db(self) -> None:
        """Create cache tables if they don't already exist."""
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS geocode_cache ("
                "  key TEXT PRIMARY KEY,"
                "  location_json TEXT NOT NULL,"
                "  created_at TEXT NOT NULL"
                ")"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS wiki_cache ("
                "  key TEXT PRIMARY KEY,"
                "  articles_json TEXT NOT NULL,"
                "  created_at TEXT NOT NULL"
                ")"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS commons_cache ("
                "  key TEXT PRIMARY KEY,"
                "  result_json TEXT NOT NULL,"
                "  created_at TEXT NOT NULL"
                ")"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS photo_cache ("
                "  filepath TEXT PRIMARY KEY,"
                "  metadata_json TEXT NOT NULL,"
                "  quality_json TEXT NOT NULL,"
                "  file_mtime REAL NOT NULL,"
                "  created_at TEXT NOT NULL"
                ")"
            )

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _geo_key(lat: float, lon: float, decimals: int) -> str:
        return f"{round(lat, decimals)},{round(lon, decimals)}"

    # -- geocode ----------------------------------------------------------

    def get_geocode(self, lat: float, lon: float) -> dict | None:
        """Return cached geocode result or ``None``."""
        key = self._geo_key(lat, lon, decimals=4)
        row = self._conn.execute(
            "SELECT location_json FROM geocode_cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        logger.debug("Cache hit (geocode) for %s", key)
        return json.loads(row["location_json"])

    def set_geocode(self, lat: float, lon: float, data: dict) -> None:
        """Store a geocode result."""
        key = self._geo_key(lat, lon, decimals=4)
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO geocode_cache (key, location_json, created_at) "
                "VALUES (?, ?, ?)",
                (key, json.dumps(data), self._now()),
            )

    # -- wiki -------------------------------------------------------------

    def get_wiki(self, location_key: str) -> list[dict] | None:
        """Return cached Wikipedia articles or ``None``."""
        row = self._conn.execute(
            "SELECT articles_json FROM wiki_cache WHERE key = ?", (location_key,)
        ).fetchone()
        if row is None:
            return None
        logger.debug("Cache hit (wiki) for %s", location_key)
        return json.loads(row["articles_json"])

    def set_wiki(self, location_key: str, data: list[dict]) -> None:
        """Store Wikipedia article results."""
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO wiki_cache (key, articles_json, created_at) "
                "VALUES (?, ?, ?)",
                (location_key, json.dumps(data), self._now()),
            )

    # -- commons ----------------------------------------------------------

    def get_commons(self, lat: float, lon: float) -> dict | None:
        """Return cached Commons result or ``None``."""
        key = self._geo_key(lat, lon, decimals=3)
        row = self._conn.execute(
            "SELECT result_json FROM commons_cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        logger.debug("Cache hit (commons) for %s", key)
        return json.loads(row["result_json"])

    def set_commons(self, lat: float, lon: float, data: dict) -> None:
        """Store a Commons query result."""
        key = self._geo_key(lat, lon, decimals=3)
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO commons_cache (key, result_json, created_at) "
                "VALUES (?, ?, ?)",
                (key, json.dumps(data), self._now()),
            )

    # -- photo ------------------------------------------------------------

    def get_photo(self, filepath: Path) -> dict | None:
        """Return cached photo data or ``None`` if missing / stale."""
        key = str(filepath)
        row = self._conn.execute(
            "SELECT metadata_json, quality_json, file_mtime FROM photo_cache "
            "WHERE filepath = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        try:
            current_mtime = filepath.stat().st_mtime
        except OSError:
            return None
        if current_mtime != row["file_mtime"]:
            logger.debug("Cache stale (photo mtime changed) for %s", key)
            return None
        logger.debug("Cache hit (photo) for %s", key)
        return {
            "metadata": json.loads(row["metadata_json"]),
            "quality": json.loads(row["quality_json"]),
        }

    def set_photo(self, filepath: Path, metadata: dict, quality: dict) -> None:
        """Store photo metadata and quality assessment."""
        key = str(filepath)
        try:
            mtime = filepath.stat().st_mtime
        except OSError:
            logger.warning("Cannot stat %s; skipping cache write", key)
            return
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO photo_cache "
                "(filepath, metadata_json, quality_json, file_mtime, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (key, json.dumps(metadata), json.dumps(quality), mtime, self._now()),
            )

    # -- maintenance ------------------------------------------------------

    def clear(self, older_than_days: int | None = None) -> int:
        """Delete cache entries. Returns total number of rows removed.

        If *older_than_days* is ``None`` every entry is deleted; otherwise
        only entries whose ``created_at`` is older than the cutoff are removed.
        """
        tables = ["geocode_cache", "wiki_cache", "commons_cache", "photo_cache"]
        deleted = 0
        with self._conn:
            if older_than_days is None:
                for table in tables:
                    cur = self._conn.execute(f"DELETE FROM {table}")  # noqa: S608
                    deleted += cur.rowcount
            else:
                cutoff = (
                    datetime.now(timezone.utc) - timedelta(days=older_than_days)
                ).isoformat()
                for table in tables:
                    cur = self._conn.execute(
                        f"DELETE FROM {table} WHERE created_at < ?",  # noqa: S608
                        (cutoff,),
                    )
                    deleted += cur.rowcount
        logger.info("Cleared %d cache entries", deleted)
        return deleted

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
        logger.debug("Cache closed")
