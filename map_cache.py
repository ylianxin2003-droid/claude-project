"""
Map data cache — stores pre-computed grid maps on disk.

Cache directory: ``data/cache/`` (auto-created).
Primary format: Parquet.  Fallback: CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import CACHE_DIR

logger = logging.getLogger(__name__)

_CACHE_ROOT = Path(CACHE_DIR)


def _ensure_cache_dir() -> None:
    """Create the cache directory tree if it does not exist."""
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def _sanitise(name: str) -> str:
    """Replace characters unsafe for filenames."""
    for ch in (":", " ", "/", "\\", "?", "*", "|", "<", ">", '"', "'"):
        name = name.replace(ch, "-")
    # Collapse repeated dashes.
    while "--" in name:
        name = name.replace("--", "-")
    return name.strip("-")


def _cache_key(
    model: str,
    variable: str,
    timestamp: str,
    resolution: float,
    region: str,
) -> str:
    """Build a safe filename stem from cache key components."""
    return _sanitise(
        f"{model}_{variable}_{timestamp}_{resolution}_{region}"
    )


def get_cache_path(
    model: str,
    variable: str,
    timestamp: str,
    resolution: float,
    region: str,
) -> Path:
    """Return the expected Parquet path for a cache key."""
    _ensure_cache_dir()
    stem = _cache_key(model, variable, timestamp, resolution, region)
    return _CACHE_ROOT / f"{stem}.parquet"


def cache_exists(
    model: str,
    variable: str,
    timestamp: str,
    resolution: float,
    region: str,
) -> bool:
    """Return ``True`` when a cached file (Parquet or CSV) exists."""
    path = get_cache_path(model, variable, timestamp, resolution, region)
    if path.exists():
        return True
    return path.with_suffix(".csv").exists()


def load_cached_map(
    model: str,
    variable: str,
    timestamp: str,
    resolution: float,
    region: str,
) -> pd.DataFrame:
    """Load a cached map DataFrame.

    Tries Parquet first, then CSV.  Returns an empty DataFrame when
    no cache file is found.
    """
    path = get_cache_path(model, variable, timestamp, resolution, region)
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.warning("Failed to read Parquet cache %s: %s", path, exc)

    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception as exc:
            logger.warning("Failed to read CSV cache %s: %s", csv_path, exc)

    return pd.DataFrame()


def save_cached_map(
    df: pd.DataFrame,
    model: str,
    variable: str,
    timestamp: str,
    resolution: float,
    region: str,
) -> Path | None:
    """Save *df* to the cache.

    Prefers Parquet; falls back to CSV when ``pyarrow`` is unavailable.
    Returns the path written, or ``None`` on failure.
    """
    if df.empty:
        logger.info("Skipping cache save — empty DataFrame.")
        return None

    path = get_cache_path(model, variable, timestamp, resolution, region)
    try:
        df.to_parquet(path, index=False)
        logger.info("Cached map → %s", path)
        return path
    except ImportError:
        logger.debug("pyarrow not available — saving CSV fallback.")
    except Exception as exc:
        logger.warning("Parquet save failed (%s), trying CSV fallback.", exc)

    # CSV fallback
    csv_path = path.with_suffix(".csv")
    try:
        df.to_csv(csv_path, index=False)
        logger.info("Cached map (CSV) → %s", csv_path)
        return csv_path
    except Exception as exc:
        logger.warning("CSV cache save failed: %s", exc)
        return None
