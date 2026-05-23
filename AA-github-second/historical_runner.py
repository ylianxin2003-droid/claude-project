"""
Historical analysis runner.

Walks a time range at fixed steps, builds a fixed-grid map at each step,
detects hazards, and generates prototype advisories.  Designed for offline /
cached operation — every timestamp failure is logged as a warning without
crashing the whole run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from alert_engine import generate_alerts_from_hazards
from config import DEFAULT_REGION, DEFAULT_TIME_STEP_HOURS, MAP_RESOLUTION_DEFAULT
from hazard_detector import detect_hazards_from_map
from map_builder import build_fixed_map

logger = logging.getLogger(__name__)


@dataclass
class RunSummary:
    """Metadata collected during a historical analysis run."""

    start_time: str = ""
    end_time: str = ""
    time_step_hours: int = DEFAULT_TIME_STEP_HOURS
    map_count: int = 0
    alert_count: int = 0
    cache_hits: int = 0
    api_calls_estimated: int = 0
    failures: int = 0
    messages: list[str] = field(default_factory=list)


def _parse_dt(ts: str) -> datetime:
    """Parse an ISO timestamp into a timezone-aware datetime."""
    try:
        return pd.to_datetime(ts).tz_localize("UTC") if pd.to_datetime(ts).tz is None else pd.to_datetime(ts)
    except Exception:
        return datetime.now(timezone.utc)


def run_historical_analysis(
    model: str,
    variable: str,
    start_time: str,
    end_time: str,
    time_step_hours: int = DEFAULT_TIME_STEP_HOURS,
    region: str = DEFAULT_REGION,
    resolution: float = MAP_RESOLUTION_DEFAULT,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, RunSummary]:
    """Run hazard detection across a historical time window.

    Parameters
    ----------
    model : str
        ``"AIDA"`` or ``"TOMIRIS"``.
    variable : str
        Single variable name, e.g. ``"TEC"``.
    start_time : str
        ISO 8601 start of the window.
    end_time : str
        ISO 8601 end of the window.
    time_step_hours : int
        Interval between consecutive maps in hours.
    region : str
        Named region (see :mod:`grid_generator`).
    resolution : float
        Grid spacing in degrees.
    use_cache : bool
        Prefer cached maps when available.
    force_refresh : bool
        Ignore cache and re-fetch from the API.

    Returns
    -------
    tuple
        ``(maps_metadata, hazards_df, alerts_df, run_summary)``.
        DataFrames are empty when no data was collected.
    """
    summary = RunSummary(
        start_time=start_time,
        end_time=end_time,
        time_step_hours=time_step_hours,
    )

    start_dt = _parse_dt(start_time)
    end_dt = _parse_dt(end_time)

    if start_dt >= end_dt:
        summary.messages.append("start_time must be before end_time.")
        return [], pd.DataFrame(), pd.DataFrame(), summary

    timestamps: list[datetime] = []
    current = start_dt
    while current <= end_dt:
        timestamps.append(current)
        current += timedelta(hours=time_step_hours)

    maps_meta: list[dict[str, Any]] = []
    all_hazards: list[pd.DataFrame] = []
    all_alerts: list[pd.DataFrame] = []
    previous_map: pd.DataFrame | None = None

    for ts in timestamps:
        ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            map_df, msg = build_fixed_map(
                model=model,
                timestamp=ts_str,
                variable=variable,
                region=region,
                resolution=resolution,
                use_cache=use_cache,
                force_refresh=force_refresh,
            )
        except Exception as exc:
            summary.failures += 1
            summary.messages.append(f"[{ts_str}] build_fixed_map error: {exc}")
            logger.warning("build_fixed_map failed for %s: %s", ts_str, exc)
            continue

        cache_hit = "cache" in msg.lower()
        if cache_hit:
            summary.cache_hits += 1
        if "API" in msg or "grid point" in msg:
            summary.api_calls_estimated += 1  # rough tally

        if map_df.empty:
            summary.failures += 1
            summary.messages.append(f"[{ts_str}] map empty: {msg}")
            logger.warning("Empty map at %s: %s", ts_str, msg)
            continue

        summary.map_count += 1

        try:
            hazards = detect_hazards_from_map(
                current_map=map_df,
                previous_map=previous_map,
                variable=variable,
            )
        except Exception as exc:
            summary.messages.append(f"[{ts_str}] hazard detection error: {exc}")
            logger.warning("Hazard detection failed at %s: %s", ts_str, exc)
            previous_map = map_df
            continue

        if not hazards.empty:
            all_hazards.append(hazards)

        try:
            alerts = generate_alerts_from_hazards(hazards)
        except Exception as exc:
            summary.messages.append(f"[{ts_str}] alert generation error: {exc}")
            logger.warning("Alert generation failed at %s: %s", ts_str, exc)
            previous_map = map_df
            continue

        if not alerts.empty:
            all_alerts.append(alerts)
            summary.alert_count += len(alerts)

        maps_meta.append({
            "timestamp": ts_str,
            "rows": len(map_df),
            "cache_hit": cache_hit,
            "status": msg,
        })

        previous_map = map_df

    hazards_df = pd.concat(all_hazards, ignore_index=True) if all_hazards else pd.DataFrame()
    alerts_df = pd.concat(all_alerts, ignore_index=True) if all_alerts else pd.DataFrame()

    return maps_meta, hazards_df, alerts_df, summary
