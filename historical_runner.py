"""
Historical analysis runner.

Walks a time range at fixed steps, builds a fixed-grid map at each step,
detects hazards, and generates prototype advisories.  Designed for offline /
cached operation — every timestamp failure is logged as a warning without
crashing the whole run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from alert_engine import generate_alerts_from_hazards
from config import DEFAULT_REGION, DEFAULT_TIME_STEP_HOURS, HISTORY_DIR, MAP_RESOLUTION_DEFAULT
from hazard_detector import detect_hazards_from_map
from map_builder import build_fixed_map
from map_cache import cache_exists, load_cached_map

logger = logging.getLogger(__name__)

_HISTORY_ROOT = Path(HISTORY_DIR)


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
    allow_api: bool = False,
    progress_callback: Any | None = None,
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
    allow_api : bool
        When ``False`` (default), historical replay is cache-only;
        timestamps without a cached map are skipped.  When ``True``, live
        SERENE API calls are permitted for missing timestamps.
    progress_callback : callable or None
        ``(done: int, total: int, status: str)`` — called after each
        timestamp step.

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

    if start_dt > end_dt:
        summary.messages.append("start_time must be before or equal to end_time.")
        return [], pd.DataFrame(), pd.DataFrame(), summary

    if not allow_api and (not use_cache or force_refresh):
        summary.messages.append(
            "Cache-only historical replay requires use_cache=True and force_refresh=False."
        )
        return [], pd.DataFrame(), pd.DataFrame(), summary

    timestamps: list[datetime] = []
    current = start_dt
    while current <= end_dt:
        timestamps.append(current)
        current += timedelta(hours=time_step_hours)

    total_steps = len(timestamps)
    maps_meta: list[dict[str, Any]] = []
    all_hazards: list[pd.DataFrame] = []
    all_alerts: list[pd.DataFrame] = []
    previous_map: pd.DataFrame | None = None
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3

    for idx, ts in enumerate(timestamps):
        ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S")

        if progress_callback:
            progress_callback(idx, total_steps, f"[{idx + 1}/{total_steps}] {ts_str} — {variable}")

        if not allow_api:
            if not cache_exists(model, variable, ts_str, resolution, region):
                summary.failures += 1
                summary.messages.append(f"[{ts_str}] skipped: no cache in cache-only mode")
                if progress_callback:
                    progress_callback(idx + 1, total_steps, f"[{idx + 1}/{total_steps}] skipped — {variable}")
                continue

            map_df = load_cached_map(model, variable, ts_str, resolution, region)
            msg = f"Loaded from cache ({len(map_df)} rows)."
        else:
            try:
                map_df, msg = build_fixed_map(
                    model=model,
                    timestamp=ts_str,
                    variable=variable,
                    region=region,
                    resolution=resolution,
                    use_cache=use_cache,
                    force_refresh=force_refresh,
                    allow_api=allow_api,
                )
            except Exception as exc:
                summary.failures += 1
                consecutive_failures += 1
                summary.messages.append(f"[{ts_str}] build_fixed_map error: {exc}")
                logger.warning("build_fixed_map failed for %s: %s", ts_str, exc)
                if progress_callback:
                    progress_callback(idx + 1, total_steps, f"[{idx + 1}/{total_steps}] failed — {variable}")
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    summary.messages.append(
                        f"Stopped early: {consecutive_failures} consecutive failures "
                        f"(last: [{ts_str}] {exc})."
                    )
                    break
                continue

        cache_hit = "cache" in msg.lower() or "loaded from cache" in msg.lower()
        if cache_hit:
            summary.cache_hits += 1
            consecutive_failures = 0
        if "API" in msg or "grid point" in msg:
            summary.api_calls_estimated += 1

        if map_df.empty:
            summary.failures += 1
            consecutive_failures += 1
            summary.messages.append(f"[{ts_str}] map empty: {msg}")
            logger.warning("Empty map at %s: %s", ts_str, msg)
            if progress_callback:
                progress_callback(idx + 1, total_steps, f"[{idx + 1}/{total_steps}] empty — {variable}")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                summary.messages.append(
                    f"Stopped early: {consecutive_failures} consecutive empty maps "
                    f"(last: [{ts_str}])."
                )
                break
            continue

        # Success — reset failure counter.
        consecutive_failures = 0
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

        if progress_callback:
            progress_callback(idx + 1, total_steps, f"[{idx + 1}/{total_steps}] complete — {variable}")

    if progress_callback:
        progress_callback(total_steps, total_steps, "Complete")

    hazards_df = pd.concat(all_hazards, ignore_index=True) if all_hazards else pd.DataFrame()
    alerts_df = pd.concat(all_alerts, ignore_index=True) if all_alerts else pd.DataFrame()

    return maps_meta, hazards_df, alerts_df, summary


# ── Persistence ──────────────────────────────────────────────────────────────


def _ensure_history_dir() -> None:
    _HISTORY_ROOT.mkdir(parents=True, exist_ok=True)


def _sanitise_run_id(raw: str) -> str:
    for ch in (":", " ", "/", "\\", "?", "*", "|", "<", ">", '"', "'"):
        raw = raw.replace(ch, "-")
    while "--" in raw:
        raw = raw.replace("--", "-")
    return raw.strip("-")


def save_historical_run(
    hazards_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    summary: RunSummary,
    model: str,
    variable: str,
    region: str,
) -> str | None:
    """Persist a completed historical run to ``data/history/``.

    Returns the *run_id* on success, or ``None`` if there is nothing to save.
    """
    if hazards_df.empty and alerts_df.empty:
        logger.info("Nothing to save — both hazards and alerts are empty.")
        return None

    _ensure_history_dir()
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = _sanitise_run_id(f"{now}_{model}_{variable}_{region}")

    run_dir = _HISTORY_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not hazards_df.empty:
            hazards_df.to_parquet(run_dir / "hazards.parquet", index=False)
        if not alerts_df.empty:
            alerts_df.to_parquet(run_dir / "alerts.parquet", index=False)
    except Exception:
        # Parquet fallback → CSV
        if not hazards_df.empty:
            hazards_df.to_csv(run_dir / "hazards.csv", index=False)
        if not alerts_df.empty:
            alerts_df.to_csv(run_dir / "alerts.csv", index=False)

    metadata = {
        "run_id": run_id,
        "created_at": now,
        "model": model,
        "variable": variable,
        "region": region,
        "start_time": summary.start_time,
        "end_time": summary.end_time,
        "time_step_hours": summary.time_step_hours,
        "map_count": summary.map_count,
        "alert_count": summary.alert_count,
        "cache_hits": summary.cache_hits,
        "api_calls_estimated": summary.api_calls_estimated,
        "failures": summary.failures,
        "hazards_rows": len(hazards_df),
        "alerts_rows": len(alerts_df),
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    logger.info("Saved historical run → %s", run_dir)
    return run_id


def load_historical_run(
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, RunSummary | None]:
    """Load a previously saved historical run.

    Returns ``(hazards_df, alerts_df, summary)``.  DataFrames are empty when
    the run directory or its files are missing.
    """
    run_dir = _HISTORY_ROOT / run_id
    if not run_dir.is_dir():
        logger.warning("Historical run not found: %s", run_dir)
        return pd.DataFrame(), pd.DataFrame(), None

    hazards = pd.DataFrame()
    alerts = pd.DataFrame()

    h_parquet = run_dir / "hazards.parquet"
    h_csv = run_dir / "hazards.csv"
    if h_parquet.exists():
        hazards = pd.read_parquet(h_parquet)
    elif h_csv.exists():
        hazards = pd.read_csv(h_csv)

    a_parquet = run_dir / "alerts.parquet"
    a_csv = run_dir / "alerts.csv"
    if a_parquet.exists():
        alerts = pd.read_parquet(a_parquet)
    elif a_csv.exists():
        alerts = pd.read_csv(a_csv)

    summary = None
    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            summary = RunSummary(
                start_time=meta.get("start_time", ""),
                end_time=meta.get("end_time", ""),
                time_step_hours=meta.get("time_step_hours", DEFAULT_TIME_STEP_HOURS),
                map_count=meta.get("map_count", 0),
                alert_count=meta.get("alert_count", 0),
                cache_hits=meta.get("cache_hits", 0),
                api_calls_estimated=meta.get("api_calls_estimated", 0),
                failures=meta.get("failures", 0),
            )
        except Exception as exc:
            logger.warning("Failed to read metadata for %s: %s", run_id, exc)

    return hazards, alerts, summary


def list_historical_runs() -> list[dict[str, Any]]:
    """List all saved historical runs with their metadata.

    Returns a list of metadata dicts sorted newest-first.
    """
    _ensure_history_dir()
    runs: list[dict[str, Any]] = []
    for run_dir in sorted(_HISTORY_ROOT.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    runs.append(json.load(fh))
            except Exception:
                continue
    return runs


def delete_historical_run(run_id: str) -> bool:
    """Delete a saved historical run.  Returns ``True`` on success."""
    import shutil
    run_dir = _HISTORY_ROOT / run_id
    if run_dir.is_dir():
        shutil.rmtree(run_dir)
        logger.info("Deleted historical run → %s", run_dir)
        return True
    return False
