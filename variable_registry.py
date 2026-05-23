"""
Variable registry — single source of truth for ionospheric variable discovery.

Every variable list in the dashboard should flow through this module.
No other file should hardcode variable names.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── Canonical default list ───────────────────────────────────────────────────

_DEFAULT_VARIABLES = [
    "TEC",
    "MUF3000",
    "foF2",
    "hmF2",
    "NmF2",
    "MUF3000_depression",
    "foF2_depression",
]


def get_default_variables() -> list[str]:
    """Return the canonical default variable list (preserved casing)."""
    return list(_DEFAULT_VARIABLES)


# ── DataFrame discovery ──────────────────────────────────────────────────────


def discover_variables_from_dataframe(
    df: pd.DataFrame,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Extract variables and per-variable metadata from a loaded DataFrame.

    If *df* is empty or has no ``variable`` column, falls back to
    :func:`get_default_variables`.

    Returns
    -------
    tuple[list[str], dict]
        ``(variables_list, metadata)`` where *metadata* is keyed by variable
        name and each value has ``unit``, ``description``, ``min``, ``max``.
    """
    if df is None or df.empty or "variable" not in df.columns:
        return get_default_variables(), {}

    var_names = sorted(df["variable"].dropna().unique().tolist())
    if not var_names:
        return get_default_variables(), {}

    metadata: dict[str, dict[str, Any]] = {}
    for var in var_names:
        var_df = df[df["variable"] == var]
        vals = pd.to_numeric(var_df["value"], errors="coerce").dropna()
        entry: dict[str, Any] = {
            "unit": "",
            "description": "",
            "min": float(vals.min()) if not vals.empty else None,
            "max": float(vals.max()) if not vals.empty else None,
        }
        for col in ("unit", "units"):
            if col in var_df.columns:
                first = var_df[col].dropna()
                if not first.empty:
                    entry["unit"] = str(first.iloc[0])
                    break
        if "description" in var_df.columns:
            first = var_df["description"].dropna()
            if not first.empty:
                entry["description"] = str(first.iloc[0])
        metadata[var] = entry

    return var_names, metadata


# ── Local JSON discovery ─────────────────────────────────────────────────────


def discover_variables_from_local_json(
    path: str | Path,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Read variable keys and metadata from a local AIDA grid JSON file.

    Each variable in the JSON is expected to have ``units``, ``description``,
    ``min``, ``max``, and ``values``.  Missing fields become empty strings or
    ``None``.

    Returns
    -------
    tuple[list[str], dict]
        ``(variables_list, variable_metadata)``.
        Falls back to :func:`get_default_variables` if the file is missing
        or unreadable.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Local JSON not found: %s", path)
        return get_default_variables(), {}

    try:
        with path.open("r", encoding="utf-8") as fh:
            product = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Cannot read local JSON %s: %s", path, exc)
        return get_default_variables(), {}

    vars_dict: dict[str, Any] = product.get("variables", {})
    if not vars_dict:
        return get_default_variables(), {}

    var_names = sorted(vars_dict.keys())
    metadata: dict[str, dict[str, Any]] = {}
    for var_name, var_info in vars_dict.items():
        if isinstance(var_info, dict):
            metadata[var_name] = {
                "unit": var_info.get("units", ""),
                "description": var_info.get("description", ""),
                "min": var_info.get("min"),
                "max": var_info.get("max"),
            }
        else:
            metadata[var_name] = {
                "unit": "",
                "description": "",
                "min": None,
                "max": None,
            }

    return var_names, metadata


# ── API discovery ────────────────────────────────────────────────────────────


def discover_variables_from_api(
    client: Any = None,
    model: str = "AIDA",
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Discover variables from the SERENE API.

    Falls back to :func:`get_default_variables` on any failure — never raises.

    Returns
    -------
    tuple[list[str], dict]
        ``(variables_list, metadata)``.  Metadata is empty for API-discovered
        variables (the current API endpoint does not return units or ranges).
    """
    if client is None:
        from serene_client import SereneClient  # local import avoids circular ref
        client = SereneClient()

    try:
        ok, _msg, variables = client.fetch_available_variables(model=model)
        if ok and variables:
            return list(variables), {}
    except Exception as exc:
        logger.warning("API variable discovery failed: %s", exc)

    return get_default_variables(), {}


# ── Unified entry point ──────────────────────────────────────────────────────


def get_available_variables(
    source: str = "auto",
    df: pd.DataFrame | None = None,
    local_file: str | Path | None = None,
    client: Any = None,
    model: str = "AIDA",
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Return available variables from the best available source.

    Priority (first non-empty result wins):

    1. *df* — if a non-empty DataFrame is provided
    2. *local_file* — if the path exists on disk
    3. *client* / SERENE API — if reachable
    4. :func:`get_default_variables` — hardcoded fallback

    Never raises.  Variables are deduplicated with original casing preserved.

    Parameters
    ----------
    source : str
        ``"auto"``, ``"local"``, ``"api"``, or ``"default"``.
    df : pd.DataFrame or None
        Already-loaded data.
    local_file : str or Path or None
        Path to a local AIDA grid JSON file.
    client : SereneClient or None
        Pre-configured API client.
    model : str
        Model name passed to the API (default ``"AIDA"``).

    Returns
    -------
    tuple[list[str], dict]
        ``(variables_list, variable_metadata)``.
    """
    metadata: dict[str, dict[str, Any]] = {}

    # 1. DataFrame — enrich with local JSON metadata when available
    if df is not None and not df.empty:
        vars_list, metadata = discover_variables_from_dataframe(df)
        if vars_list:
            # Merge metadata from local JSON (units, descriptions) on top of
            # DataFrame-computed min/max so the caller gets both.
            if local_file:
                _merge_json_metadata(metadata, local_file)
            return _deduplicate(vars_list), metadata

    # 2. Local JSON
    if local_file:
        path = Path(local_file)
        if path.exists():
            vars_list, metadata = discover_variables_from_local_json(path)
            if vars_list and vars_list != get_default_variables():
                return _deduplicate(vars_list), metadata

    # 3. API
    if source in ("api", "auto"):
        try:
            vars_list, api_meta = discover_variables_from_api(client=client, model=model)
            if vars_list and vars_list != get_default_variables():
                return _deduplicate(vars_list), api_meta
        except Exception:
            pass

    # 4. Defaults
    return get_default_variables(), {}


# ── Internal helpers ─────────────────────────────────────────────────────────


def _deduplicate(variables: list[str]) -> list[str]:
    """Deduplicate variable names preserving original casing and order."""
    seen: set[str] = set()
    result: list[str] = []
    for v in variables:
        if v.lower() not in seen:
            seen.add(v.lower())
            result.append(v)
    return result


def _merge_json_metadata(metadata: dict[str, dict[str, Any]], local_file: str | Path) -> None:
    """Enrich *metadata* in-place with unit/description from a local JSON file.

    Only fills keys that are currently empty — DataFrame-computed min/max
    are left untouched.
    """
    path = Path(local_file)
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as fh:
            product = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return

    vars_dict: dict[str, Any] = product.get("variables", {})
    for var_name, entry in metadata.items():
        var_info = vars_dict.get(var_name)
        if not isinstance(var_info, dict):
            continue
        if not entry.get("unit"):
            entry["unit"] = var_info.get("units", "")
        if not entry.get("description"):
            entry["description"] = var_info.get("description", "")
