"""
Unified data loader.

Single entry-point :func:`load_data` for the Streamlit dashboard.
SERENE API is primary; local JSON sample file is the automatic fallback.
"""

from __future__ import annotations

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


def discover_variables(local_file: str | Path | None = None) -> list[str]:
    """Auto-discover available variables from local JSON.

    Delegates to :func:`variable_registry.get_available_variables`.
    Returns only the variable name list (no metadata).
    """
    from variable_registry import get_available_variables

    vars_list, _metadata = get_available_variables(
        source="auto",
        local_file=resolve_local_file(local_file),
    )
    return vars_list


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
    """Load ionospheric data from SERENE API.

    Local file mode is disabled — this project uses SERENE API only.

    Parameters
    ----------
    source : str
        ``"api"`` (default). ``"local"`` returns an empty DataFrame with a warning.
    model : str
        ``"AIDA"`` or ``"TOMIRIS"``.
    start_time, end_time : str, optional
        ISO 8601 timestamps passed to the API client.
    variables : list[str], optional
        Variable names to request (when supported by the API).
    region : dict, optional
        ``{"lat_min", "lat_max", "lon_min", "lon_max"}``.
    local_file : str or Path, optional
        Ignored — local file mode is disabled.
    grid_step : float
        Grid spacing (degrees) for point-sampling.

    Returns
    -------
    tuple[pd.DataFrame, LoadStatus]
        Standardised data and a status object describing the outcome.
    """
    status = LoadStatus()

    # ── Local mode is disabled ─────────────────────────────────────────────
    if source == "local":
        status.source = "none"
        status.ok = False
        status.message = "Local file mode is disabled. This project uses SERENE API only."
        return pd.DataFrame(), status

    # ── API mode ───────────────────────────────────────────────────────────
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
        status.source = "api"
        status.ok = False
        status.message = msg or "SERENE API request failed."
        status.warnings.append(msg)
        return pd.DataFrame(), status

    if raw is None:
        status.source = "api"
        status.ok = False
        status.message = msg or "SERENE API returned no data."
        return pd.DataFrame(), status

    df = client.parse_response_to_dataframe(raw, model=model)
    if variables and not df.empty and "variable" in df.columns:
        df = df[df["variable"].isin(variables)]

    if df.empty:
        status.source = "api"
        status.ok = False
        status.message = msg or "SERENE API returned empty or unparseable data."
        return pd.DataFrame(), status

    status.source = "api"
    status.ok = True
    status.message = f"API connected — {len(df)} rows loaded from SERENE."
    status.metadata = {"model": model, "api_message": msg}
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
        var_unit = var_info.get("units", "")
        var_desc = var_info.get("description", "")
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
                    "unit": var_unit,
                    "description": var_desc,
                })

    return pd.DataFrame(rows)
