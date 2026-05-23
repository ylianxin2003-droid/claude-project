"""
Unified data loader.

Single entry-point :func:`load_data` for the Streamlit dashboard.
SERENE API is primary; local JSON sample file is the automatic fallback.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from serene_client import SereneClient

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LOCAL_FILE = _PROJECT_ROOT / "data" / "latest_aida_grid.json"


def resolve_local_file(local_file: str | Path | None = None) -> Path:
    """Find sample JSON — supports ``data/`` or repo-root layouts (GitHub clone)."""
    if local_file:
        path = Path(local_file)
        if path.exists():
            return path

    candidates = [
        _PROJECT_ROOT / "data" / "latest_aida_grid.json",
        _PROJECT_ROOT / "latest_aida_grid.json",
        _PROJECT_ROOT / "data" / "test_aida_grid.json",
        _PROJECT_ROOT / "test_aida_grid.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


@dataclass
class LoadStatus:
    """Metadata about a data loading operation."""

    source: str = "unknown"
    ok: bool = False
    message: str = ""
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_data(
    source: str = "api",
    model: str = "AIDA",
    start_time: str | None = None,
    end_time: str | None = None,
    variables: list[str] | None = None,
    region: dict[str, float] | None = None,
    local_file: str | Path | None = None,
    grid_step: float = 10.0,
    progress_callback: Any | None = None,
) -> tuple[pd.DataFrame, LoadStatus]:
    """Load ionospheric data from SERENE API or a local sample file.

    Parameters
    ----------
    source : str
        ``"api"`` or ``"local"``.
    model : str
        ``"AIDA"`` or ``"TOMIRIS"``.
    start_time, end_time : str, optional
        ISO 8601 timestamps passed to the API client.
    variables : list[str], optional
        Variable names to request (when supported by the API).
    region : dict, optional
        ``{"lat_min", "lat_max", "lon_min", "lon_max"}``.
    local_file : str or Path, optional
        Fallback JSON path.
    grid_step : float
        Grid spacing (degrees) for point-sampling fallback.

    Returns
    -------
    tuple[pd.DataFrame, LoadStatus]
        Standardised data and a status object describing the outcome.
    """
    local_path = resolve_local_file(local_file)
    status = LoadStatus()
    api_warnings: list[str] = []

    if source == "api":
        client = SereneClient()
        r = region or {
            "lat_min": -90.0,
            "lat_max": 90.0,
            "lon_min": -180.0,
            "lon_max": 180.0,
        }

        ok, msg, raw = client.fetch_model_output(
            model=model,
            start_time=start_time or "",
            end_time=end_time or "",
            variables=variables,
            region=r,
            grid_step=grid_step,
            progress_callback=progress_callback,
        )

        if not ok:
            api_warnings.append(msg)
        elif raw is not None:
            df = client.parse_response_to_dataframe(raw, model=model)
            if variables and not df.empty and "variable" in df.columns:
                df = df[df["variable"].isin(variables)]

            if not df.empty:
                status.source = "api"
                status.ok = True
                status.message = f"API connected — {len(df)} rows loaded from SERENE."
                status.metadata = {"model": model, "api_message": msg}
                return df, status

            api_warnings.append(msg or "API returned empty or unparseable data.")
        else:
            api_warnings.append(msg or "API returned no data.")

        status.warnings.extend(api_warnings)

    # ── Local path (explicit or fallback) ───────────────────────────────────
    if not local_path.exists():
        status.source = "none"
        status.ok = False
        if source == "api":
            status.message = (
                "API failed, using local fallback — local file not found. "
                "No data available."
            )
        else:
            status.message = "Local file not found. No data available."
        return pd.DataFrame(), status

    try:
        with local_path.open("r", encoding="utf-8") as fh:
            product = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        status.source = "none"
        status.ok = False
        status.message = f"Cannot read local file: {exc}"
        status.warnings.append(str(exc))
        return pd.DataFrame(), status

    df = _parse_aida_grid_json(product, model=model)
    if variables and not df.empty and "variable" in df.columns:
        df = df[df["variable"].isin(variables)]

    meta = product.get("metadata", {})
    status.metadata = meta

    if df.empty:
        status.source = "none"
        status.ok = False
        status.message = "Local file loaded but contained no usable data."
        return pd.DataFrame(), status

    if source == "api":
        status.source = "local_fallback"
        status.ok = True
        warn = api_warnings[0][:120] if api_warnings else "API unavailable"
        status.message = (
            f"API failed ({warn}), using local fallback — "
            f"{len(df)} rows from {meta.get('model_time', 'unknown time')}."
        )
    else:
        status.source = "local"
        status.ok = True
        status.message = (
            f"Local file loaded — {len(df)} rows from "
            f"{meta.get('model_time', 'unknown time')}."
        )

    return df, status


def _parse_aida_grid_json(product: dict[str, Any], model: str = "AIDA") -> pd.DataFrame:
    """Convert bundled AIDA grid JSON to the standard long-form schema."""
    coords = product.get("coordinates", {})
    lats: list[float] = coords.get("lat", [])
    lons: list[float] = coords.get("lon", [])
    vars_dict: dict[str, Any] = product.get("variables", {})
    meta = product.get("metadata", {})
    model_time = meta.get("model_time", "unknown")

    rows: list[dict[str, Any]] = []
    for var_name, var_info in vars_dict.items():
        if not isinstance(var_info, dict):
            continue
        values = var_info.get("values", [])
        for ilat, lat in enumerate(lats):
            if ilat >= len(values):
                continue
            row_vals = values[ilat]
            for ilon, lon in enumerate(lons):
                if ilon >= len(row_vals):
                    continue
                rows.append({
                    "time": model_time,
                    "lat": lat,
                    "lon": lon,
                    "alt": None,
                    "variable": var_name,
                    "value": row_vals[ilon],
                    "model": model,
                })

    return pd.DataFrame(rows)
