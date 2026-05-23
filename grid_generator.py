"""
Fixed grid point generator for ionospheric map sampling.

Produces DataFrames of (lat, lon) points at a given resolution
for named regions used by the dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Region bounding boxes ─────────────────────────────────────────────────────
# Each region: lat_min, lat_max, lon_min, lon_max (degrees)

REGIONS: dict[str, dict[str, float]] = {
    "global":        {"lat_min": -90.0, "lat_max": 90.0,  "lon_min": -180.0, "lon_max": 180.0},
    "europe":        {"lat_min":  35.0, "lat_max": 70.0,  "lon_min":  -10.0, "lon_max":  40.0},
    "north_atlantic": {"lat_min": 40.0, "lat_max": 65.0,  "lon_min":  -60.0, "lon_max":   0.0},
    "polar_north":   {"lat_min":  60.0, "lat_max": 90.0,  "lon_min": -180.0, "lon_max": 180.0},
    "polar_south":   {"lat_min": -90.0, "lat_max": -60.0, "lon_min": -180.0, "lon_max": 180.0},
    "uk":            {"lat_min":  49.0, "lat_max": 60.0,  "lon_min":  -10.0, "lon_max":   2.0},
}


def _arange_fixed(start: float, stop: float, step: float) -> np.ndarray:
    """``np.arange`` that always includes *stop* when step divides evenly."""
    # Use linspace so the endpoint is guaranteed; round to 6 decimal places
    # to avoid floating-point wobble.
    n = int(round((stop - start) / step)) + 1
    return np.round(np.linspace(start, stop, n), decimals=6)


def generate_global_grid(resolution: float = 2.5) -> pd.DataFrame:
    """Generate a global lat/lon grid at *resolution* degrees.

    Returns a DataFrame with columns ``lat``, ``lon``.
    """
    return generate_region_grid("global", resolution=resolution)


def generate_region_grid(region: str = "global", resolution: float = 2.5) -> pd.DataFrame:
    """Generate a lat/lon grid for a named *region*.

    Parameters
    ----------
    region : str
        One of ``global``, ``europe``, ``north_atlantic``, ``polar_north``,
        ``polar_south``, ``uk``.  Unknown values fall back to ``global``.
    resolution : float
        Grid spacing in degrees (default 2.5).

    Returns
    -------
    pd.DataFrame
        Columns: ``lat``, ``lon`` (both float64).  No duplicate rows.
    """
    bounds = REGIONS.get(region.lower())
    if bounds is None:
        bounds = REGIONS["global"]

    lats = _arange_fixed(bounds["lat_min"], bounds["lat_max"], float(resolution))
    lons = _arange_fixed(bounds["lon_min"], bounds["lon_max"], float(resolution))

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    points = pd.DataFrame({
        "lat": lat_grid.ravel().astype(float),
        "lon": lon_grid.ravel().astype(float),
    })
    return points.drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)


def list_regions() -> list[str]:
    """Return the list of known region names."""
    return list(REGIONS.keys())


def region_bounds(region: str) -> dict[str, float] | None:
    """Return the bounding box for *region*, or ``None`` if unknown."""
    return REGIONS.get(region.lower())
