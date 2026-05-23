"""
Fixed-grid map builder.

Combines grid generation, SERENE API point-sampling, and disk cache
into a single ``build_fixed_map()`` entry point.
"""

from __future__ import annotations

import logging

import pandas as pd

from config import DEFAULT_REGION, MAP_RESOLUTION_DEFAULT
from grid_generator import generate_region_grid
from map_cache import cache_exists, load_cached_map, save_cached_map
from serene_client import SereneClient

logger = logging.getLogger(__name__)

# Standardised output columns (order matches existing data_loader conventions).
_STANDARD_COLS = ["time", "lat", "lon", "alt", "variable", "value", "model", "region"]


def build_fixed_map(
    model: str,
    timestamp: str,
    variable: str,
    region: str = DEFAULT_REGION,
    resolution: float = MAP_RESOLUTION_DEFAULT,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Build a fixed-resolution grid map for a single *variable*.

    Parameters
    ----------
    model : str
        Model name (``"AIDA"`` or ``"TOMIRIS"``).
    timestamp : str
        ISO 8601 timestamp used for the cache key (not sent to the API
        until SERENE documents a time parameter).
    variable : str
        Single variable name, e.g. ``"TEC"``.
    region : str
        Named region key (see :mod:`grid_generator`).
    resolution : float
        Grid spacing in degrees.
    use_cache : bool
        If ``True`` (default), return a cached result when available.
    force_refresh : bool
        If ``True``, bypass the cache and re-fetch from the API.

    Returns
    -------
    tuple[pd.DataFrame, str]
        ``(map_df, status_message)``.  *map_df* is empty when no data
        could be obtained.
    """
    # ── 1. Cache hit ──────────────────────────────────────────────────────
    if use_cache and not force_refresh and cache_exists(model, variable, timestamp, resolution, region):
        df = load_cached_map(model, variable, timestamp, resolution, region)
        if not df.empty:
            return df, f"Loaded from cache ({len(df)} rows)."

    # ── 2. Generate grid points ───────────────────────────────────────────
    grid_df = generate_region_grid(region, resolution=resolution)
    total = len(grid_df)
    logger.info("Grid: %d points for region=%s @ %.1f°", total, region, resolution)

    # ── 3. Fetch from SERENE API (point-by-point) ─────────────────────────
    client = SereneClient()
    frames: list[pd.DataFrame] = []
    success = 0

    # Future: when SERENE supports a batch endpoint, replace this loop
    # with a single batch call (see SERENE_BATCH_SIZE in config.py).
    for _, row in grid_df.iterrows():
        lat, lon = float(row["lat"]), float(row["lon"])
        ok, _msg, data = client._request(
            "POST",
            "/api/calc/",
            data={"latitude": lat, "longitude": lon},
        )
        if not ok or data is None:
            continue

        parsed = client.parse_response_to_dataframe(data, model=model)
        if parsed.empty:
            continue

        if "lat" not in parsed.columns:
            parsed["lat"] = lat
        if "lon" not in parsed.columns:
            parsed["lon"] = lon

        frames.append(parsed)
        success += 1

    if not frames:
        return pd.DataFrame(), (
            f"No data: all {total} grid point(s) failed or returned empty "
            f"from SERENE API."
        )

    result = pd.concat(frames, ignore_index=True)

    # ── 4. Standardise columns ────────────────────────────────────────────
    result["region"] = region
    if "model" not in result.columns:
        result["model"] = model
    if "alt" not in result.columns:
        result["alt"] = None
    if "time" not in result.columns:
        result["time"] = timestamp

    for col in _STANDARD_COLS:
        if col not in result.columns:
            result[col] = None

    result = result[_STANDARD_COLS]

    # ── 5. Save to cache ──────────────────────────────────────────────────
    if use_cache:
        save_cached_map(result, model, variable, timestamp, resolution, region)

    return result, f"Map built: {success}/{total} grid point(s) OK ({len(result)} rows)."
