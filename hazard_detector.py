"""
Hazard detection engine for fixed-grid ionospheric maps.

Analyses a spatial map (and optionally a previous map) to detect hazards
based on configurable prototype thresholds for value, spatial gradient,
and temporal change rate.

**These are academic prototype thresholds — not official ICAO values.**
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config import HAZARD_THRESHOLDS, MAP_RESOLUTION_DEFAULT

logger = logging.getLogger(__name__)

RISK_PRIORITY = {"Severe": 0, "Warning": 1, "Watch": 2, "Normal": 3}

# Variable → hazard type mapping
_VARIABLE_HAZARD_MAP: dict[str, str] = {
    "TEC": "GNSS positioning risk",
    "TECU": "GNSS positioning risk",
    "MUF3000": "HF communication risk",
    "MUF3000_depression": "HF communication risk",
    "foF2": "HF communication risk",
    "foF2_depression": "HF communication risk",
    "hmF2": "General ionospheric disturbance",
    "NmF2": "General ionospheric disturbance",
    "ionospheric_disturbance": "General ionospheric disturbance",
}


def _hazard_type_for(variable: str) -> str:
    """Map a variable name to its hazard category."""
    name = variable.lower()
    for key, htype in _VARIABLE_HAZARD_MAP.items():
        if key.lower() == name:
            return htype
    if "tec" in name and "dep" not in name:
        return "GNSS positioning risk"
    if "muf" in name or "fof2" in name:
        return "HF communication risk"
    if "hmf2" in name or "nmf2" in name:
        return "General ionospheric disturbance"
    if "disturb" in name:
        return "General ionospheric disturbance"
    return "General ionospheric disturbance"


def _infer_resolution(df: pd.DataFrame) -> float:
    """Guess grid resolution from sorted unique lat/lon values."""
    if df.empty:
        return MAP_RESOLUTION_DEFAULT
    lats = sorted(df["lat"].dropna().unique())
    lons = sorted(df["lon"].dropna().unique())
    dlat = (lats[1] - lats[0]) if len(lats) > 1 else MAP_RESOLUTION_DEFAULT
    dlon = (lons[1] - lons[0]) if len(lons) > 1 else MAP_RESOLUTION_DEFAULT
    return float(max(dlat, dlon))


def _compute_spatial_gradients(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a ``spatial_gradient`` column (value per degree) to each row."""
    result = df.copy()
    result["spatial_gradient"] = np.nan

    resolution = _infer_resolution(df)

    for var in df["variable"].dropna().unique():
        var_mask = result["variable"] == var
        var_df = result[var_mask]
        if var_df.empty:
            continue

        lats = sorted(var_df["lat"].dropna().unique())
        lons = sorted(var_df["lon"].dropna().unique())
        if len(lats) < 2 or len(lons) < 2:
            # Can't compute 2-D gradient for a single row / column strip.
            result.loc[var_mask, "spatial_gradient"] = 0.0
            continue

        pivot = var_df.pivot_table(
            index="lat", columns="lon", values="value", aggfunc="first",
        )
        pivot = pivot.reindex(index=lats, columns=lons)
        values_2d = pivot.values

        dlat = (lats[1] - lats[0]) if len(lats) > 1 else resolution
        dlon = (lons[1] - lons[0]) if len(lons) > 1 else resolution

        grad_lat, grad_lon = np.gradient(values_2d.astype(float), dlat, dlon)
        grad_mag = np.sqrt(grad_lat ** 2 + grad_lon ** 2)

        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                gval = grad_mag[i, j]
                if not np.isnan(gval):
                    point_mask = (
                        (result["lat"] == lat)
                        & (result["lon"] == lon)
                        & (result["variable"] == var)
                    )
                    result.loc[point_mask, "spatial_gradient"] = float(gval)

    # Fill any remaining NaN with 0.
    result["spatial_gradient"] = result["spatial_gradient"].fillna(0.0)
    return result


def _compute_temporal_change(
    current: pd.DataFrame,
    previous: pd.DataFrame,
) -> pd.DataFrame:
    """Compute point-wise temporal change rate (abs value diff per hour)."""
    if previous is None or previous.empty:
        current["temporal_change"] = 0.0
        return current

    prev = previous[["lat", "lon", "variable", "value", "time"]].copy()
    prev = prev.rename(columns={"value": "prev_value", "time": "prev_time"})

    merged = current.merge(prev, on=["lat", "lon", "variable"], how="left")
    merged["prev_value"] = pd.to_numeric(merged["prev_value"], errors="coerce")

    # Time delta in hours
    merged["time"] = pd.to_datetime(merged["time"], errors="coerce")
    merged["prev_time"] = pd.to_datetime(merged["prev_time"], errors="coerce")
    delta_hours = (merged["time"] - merged["prev_time"]).dt.total_seconds().abs() / 3600.0

    merged["temporal_change"] = np.where(
        merged["prev_value"].notna() & (delta_hours > 0),
        (merged["value"] - merged["prev_value"]).abs() / delta_hours,
        0.0,
    )
    return merged


def _classify_from_thresholds(
    value: float,
    gradient: float,
    change_rate: float,
    variable: str,
    thresholds: dict[str, dict[str, float]] | None = None,
) -> str:
    """Return the highest risk level across value, gradient, and change rate.

    Uses *thresholds* when provided, otherwise falls back to
    :data:`config.HAZARD_THRESHOLDS`.
    """
    t = thresholds or HAZARD_THRESHOLDS
    cfg = _find_hazard_config(variable, t)
    if cfg is None:
        return "Normal"

    def _level(metric: str, val: float) -> str:
        w = cfg.get(f"{metric}_watch")
        wn = cfg.get(f"{metric}_warning")
        s = cfg.get(f"{metric}_severe")
        if w is None:
            return "Normal"
        if val >= (s or float("inf")):
            return "Severe"
        if val >= (wn or float("inf")):
            return "Warning"
        if val >= w:
            return "Watch"
        return "Normal"

    levels = [
        _level("value", abs(float(value))),
        _level("gradient", float(gradient)),
        _level("change_rate", float(change_rate)),
    ]
    return min(levels, key=lambda l: RISK_PRIORITY.get(l, 3))


def _find_hazard_config(
    variable: str,
    thresholds: dict[str, dict[str, float]],
) -> dict[str, float] | None:
    """Match a variable name to its hazard threshold config."""
    if variable in thresholds:
        return thresholds[variable]
    name = variable.lower()
    for key, cfg in thresholds.items():
        if key.lower() == name:
            return cfg
    if "tec" in name and "dep" not in name:
        return thresholds.get("TEC")
    if "muf" in name and "dep" in name:
        return thresholds.get("MUF3000_depression")
    if "fof2" in name and "dep" in name:
        return thresholds.get("foF2_depression")
    return None


def detect_hazards_from_map(
    current_map: pd.DataFrame,
    previous_map: pd.DataFrame | None = None,
    variable: str | None = None,
    thresholds: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Detect hazards from a fixed-grid ionospheric map.

    Parameters
    ----------
    current_map : pd.DataFrame
        Columns: ``time, lat, lon, variable, value, model, region``.
    previous_map : pd.DataFrame or None
        Same schema as *current_map* for temporal comparison.
    variable : str or None
        Restrict detection to one variable.  ``None`` runs on all variables.
    thresholds : dict or None
        Custom threshold table (default: :data:`config.HAZARD_THRESHOLDS`).

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp, region, variable, hazard_type, risk_level,
        reason, max_value, mean_value, max_gradient, max_change_rate, model``.
        Rows with ``risk_level == "Normal"`` are excluded.
    """
    if current_map.empty:
        return pd.DataFrame()

    work = current_map.copy()
    if variable:
        work = work[work["variable"] == variable]
    if work.empty:
        return pd.DataFrame()

    # Ensure numeric value column.
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work.dropna(subset=["value"])

    # ── Spatial gradients ─────────────────────────────────────────────────
    work = _compute_spatial_gradients(work)

    # ── Temporal change rate ──────────────────────────────────────────────
    work = _compute_temporal_change(work, previous_map)

    # ── Per-region, per-variable aggregation ──────────────────────────────
    hazards: list[dict[str, Any]] = []
    group_cols = ["region", "variable"]
    if "model" in work.columns:
        group_cols.append("model")

    for keys, grp in work.groupby(group_cols, sort=False):
        if isinstance(keys, tuple):
            region_val = keys[0]
            var_val = keys[1]
            model_val = keys[2] if len(keys) > 2 else grp["model"].iloc[0] if "model" in grp.columns else "unknown"
        else:
            region_val = keys
            var_val = grp["variable"].iloc[0]
            model_val = grp["model"].iloc[0] if "model" in grp.columns else "unknown"

        max_val = float(grp["value"].max())
        mean_val = float(grp["value"].mean())
        max_grad = float(grp["spatial_gradient"].max())
        max_change = float(grp.get("temporal_change", pd.Series([0.0])).max())
        time_val = grp["time"].iloc[0] if "time" in grp.columns else "unknown"

        hazard_type = _hazard_type_for(var_val)
        risk = _classify_from_thresholds(max_val, max_grad, max_change, var_val, thresholds)

        if risk == "Normal":
            continue

        reason_parts = [f"{var_val} max = {max_val:.2f}"]
        if max_grad > 0:
            reason_parts.append(f"max spatial gradient = {max_grad:.2f}/deg")
        if max_change > 0:
            reason_parts.append(f"max temporal change = {max_change:.2f}/hr")
        reason_parts.append(f"risk level: {risk}")

        hazards.append({
            "timestamp": time_val,
            "region": region_val,
            "variable": var_val,
            "hazard_type": hazard_type,
            "risk_level": risk,
            "reason": "; ".join(reason_parts),
            "max_value": max_val,
            "mean_value": mean_val,
            "max_gradient": max_grad,
            "max_change_rate": max_change,
            "model": model_val,
        })

    if not hazards:
        return pd.DataFrame()

    result = pd.DataFrame(hazards)
    result = result.sort_values("risk_level", key=lambda s: s.map(RISK_PRIORITY))
    return result.reset_index(drop=True)
