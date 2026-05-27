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


def list_cached_maps() -> list[dict[str, object]]:
    """List all cached maps with metadata.

    Returns a list of dicts with keys: model, variable, timestamp, resolution,
    region, file_size, file_path.
    """
    _ensure_cache_dir()
    results: list[dict[str, object]] = []
    if not _CACHE_ROOT.exists():
        return results

    for path in sorted(_CACHE_ROOT.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in (".parquet", ".csv"):
            continue
        stem = path.stem
        parts = stem.split("_")
        # Expected format: model_variable_timestamp_resolution_region
        if len(parts) < 5:
            continue
        try:
            region = parts[-1]
            resolution = float(parts[-2])
            timestamp = parts[-3].replace("-", ":")
            # variable and model may contain underscores — rejoin
            variable = parts[-4]
            model = "_".join(parts[:-4])
            file_size = path.stat().st_size
            results.append({
                "model": model,
                "variable": variable,
                "timestamp": timestamp,
                "resolution": resolution,
                "region": region,
                "file_size": file_size,
                "file_path": str(path),
            })
        except (ValueError, IndexError):
            continue

    return results


def count_cached_maps() -> dict[str, object]:
    """Return cache statistics: total count and total size in bytes.

    Returns a dict with keys: count, total_size, readable_size.
    """
    maps = list_cached_maps()
    total_size = sum(m.get("file_size", 0) for m in maps)  # type: ignore[arg-type]
    # Human-readable size
    if total_size < 1024:
        readable = f"{total_size} B"
    elif total_size < 1024 * 1024:
        readable = f"{total_size / 1024:.1f} KB"
    elif total_size < 1024 * 1024 * 1024:
        readable = f"{total_size / (1024 * 1024):.1f} MB"
    else:
        readable = f"{total_size / (1024 * 1024 * 1024):.2f} GB"
    return {
        "count": len(maps),
        "total_size": total_size,
        "readable_size": readable,
    }


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
