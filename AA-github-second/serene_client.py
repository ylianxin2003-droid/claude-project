"""
SERENE API client.

Official request format (Birmingham SERENE)::

    curl -X POST \\
      -H "Authorization: Token <SERENE_API_TOKEN>" \\
      -d latitude=52.4862 -d longitude=1.8904 \\
      https://spaceweather.bham.ac.uk/api/calc/

- Method: POST
- Auth: ``Authorization: Token <token>``  (not Bearer)
- Body: form-urlencoded ``latitude`` and ``longitude`` only

Endpoint paths are listed in ``ENDPOINTS`` for future API expansion.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    SERENE_API_BASE_URL,
    SERENE_API_TIMEOUT,
    SERENE_API_TOKEN,
    SERENE_AUTH_SCHEME,
)

# Max /api/calc/ calls per load (each point is one HTTP request).
MAX_GRID_POINTS = int(os.getenv("SERENE_MAX_GRID_POINTS", "30"))

logger = logging.getLogger(__name__)

# ── API Endpoints ───────────────────────────────────────────────────────────
# Official (2026): POST /api/calc/  — see module docstring for curl example.
# ★ Add future endpoints here when SERENE publishes expanded API docs.
ENDPOINTS: dict[str, str] = {
    "calc": "/api/calc/",
    # Placeholders (not yet documented for Birmingham deployment):
    "health": "/api/health/",
    "models": "/api/models/",
    "variables": "/api/variables/",
    "model_output": "/api/model-output/",
}

# Coordinates used for connection smoke tests (official curl example).
_OFFICIAL_TEST_LAT = 52.4862
_OFFICIAL_TEST_LON = 1.8904
# ─────────────────────────────────────────────────────────────────────────────


class SereneAPIError(Exception):
    """Non-fatal SERENE API error with a user-readable message."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class SereneClient:
    """HTTP client for the SERENE API.

    Parameters
    ----------
    base_url : str, optional
        Override ``SERENE_API_BASE_URL``.
    token : str, optional
        Override ``SERENE_API_TOKEN``.
    timeout : int, optional
        Request timeout in seconds.
    auth_scheme : str, optional
        Official SERENE value is ``Token`` (override ``SERENE_AUTH_SCHEME``).
    """

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: int | None = None,
        auth_scheme: str | None = None,
    ) -> None:
        self.base_url = (base_url or SERENE_API_BASE_URL).rstrip("/")
        self.token = token or SERENE_API_TOKEN
        self.timeout = timeout if timeout is not None else SERENE_API_TIMEOUT
        self.auth_scheme = (auth_scheme or SERENE_AUTH_SCHEME).strip()

        self._session = requests.Session()
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    # ── Low-level request ───────────────────────────────────────────────────

    def _auth_headers(self) -> dict[str, str]:
        if not self.token:
            return {}
        # Official: Authorization: Token <token>
        return {"Authorization": f"{self.auth_scheme} {self.token}"}

    @staticmethod
    def _calc_form(lat: float, lon: float) -> dict[str, float]:
        """Form body for POST /api/calc/ (official fields only)."""
        return {"latitude": float(lat), "longitude": float(lon)}

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> tuple[bool, str, Any]:
        """Execute an HTTP request.

        Returns
        -------
        tuple[bool, str, Any]
            ``(success, message, parsed_body_or_none)``
        """
        if not self.base_url:
            return False, "SERENE_API_BASE_URL is not configured.", None

        url = f"{self.base_url}{endpoint}"
        headers = self._auth_headers()

        try:
            response = self._session.request(
                method=method.upper(),
                url=url,
                headers=headers,
                params=params,
                json=json,
                data=data,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout:
            msg = f"SERENE API request timed out after {self.timeout}s: {url}"
            logger.warning(msg)
            return False, msg, None
        except requests.exceptions.ConnectionError as exc:
            msg = (
                f"Cannot connect to SERENE API at {self.base_url}. "
                f"Check SERENE_API_BASE_URL and network. ({exc})"
            )
            logger.warning(msg)
            return False, msg, None
        except requests.exceptions.RequestException as exc:
            msg = f"SERENE API request failed: {exc}"
            logger.warning(msg)
            return False, msg, None

        if response.status_code == 401:
            msg = (
                "SERENE API returned 401 Unauthorized. "
                "Check SERENE_API_TOKEN. Official auth is: Authorization: Token <token>."
            )
            logger.warning(msg)
            return False, msg, None

        if response.status_code == 403:
            msg = "SERENE API returned 403 Forbidden. Token may lack permission."
            logger.warning(msg)
            return False, msg, None

        if response.status_code == 404:
            msg = f"SERENE API endpoint not found (404): {url}"
            logger.warning(msg)
            return False, msg, None

        if response.status_code >= 500:
            msg = f"SERENE API server error ({response.status_code}): {url}"
            logger.warning(msg)
            return False, msg, None

        if not response.ok:
            msg = f"SERENE API unexpected status {response.status_code}: {url}"
            logger.warning(msg)
            return False, msg, None

        if not response.content:
            return True, "OK (empty response body)", None

        try:
            body = response.json()
        except ValueError:
            text = response.text.strip()
            if not text:
                return True, "OK (empty response)", None
            return True, "OK (non-JSON response)", text

        if body is None or body == "" or body == [] or body == {}:
            return True, "OK (no data in response)", body

        return True, "OK", body

    # ── High-level API methods ──────────────────────────────────────────────

    def test_connection(self) -> tuple[bool, str]:
        """Verify connectivity and authentication."""
        if not self.base_url:
            return False, "SERENE_API_BASE_URL is not configured."
        if not self.token:
            return False, "SERENE_API_TOKEN is not configured."

        # Official smoke test: POST /api/calc/ (same as documented curl example).
        ok, msg, _ = self._request(
            "POST",
            ENDPOINTS["calc"],
            data=self._calc_form(_OFFICIAL_TEST_LAT, _OFFICIAL_TEST_LON),
        )
        if ok:
            return (
                True,
                f"Connected to SERENE API at {self.base_url} "
                f"(POST {ENDPOINTS['calc']} OK).",
            )

        if "401" in msg or "403" in msg:
            return False, msg

        return False, msg

    def fetch_available_models(self) -> tuple[bool, str, list[str]]:
        """List models exposed by the API.

        Falls back to a static list when the models endpoint is not yet available.
        """
        ok, msg, data = self._request("GET", ENDPOINTS["models"])
        if ok and data is not None:
            models = _extract_string_list(data, keys=("models", "results", "data"))
            if models:
                return True, msg, models

        # ★ Placeholder until SERENE documents GET /api/models/
        return True, "Using default model list (models endpoint not available).", ["AIDA", "TOMIRIS"]

    def fetch_available_variables(self, model: str | None = None) -> tuple[bool, str, list[str]]:
        """List variables for a model."""
        params: dict[str, Any] = {}
        if model:
            params["model"] = model

        ok, msg, data = self._request("GET", ENDPOINTS["variables"], params=params or None)
        if ok and data is not None:
            variables = _extract_string_list(data, keys=("variables", "results", "data"))
            if variables:
                return True, msg, variables

        # ★ Placeholder defaults for dashboard prototyping
        defaults = ["TEC", "MUF3000", "foF2", "MUF3000_depression", "foF2_depression"]
        note = "Using default variable list (variables endpoint not available)."
        if not ok and "404" in msg:
            return True, note, defaults
        if ok:
            return True, note, defaults
        return False, msg, defaults

    def fetch_model_output(
        self,
        model: str,
        start_time: str,
        end_time: str,
        variables: list[str] | None = None,
        region: dict[str, float] | None = None,
        grid_step: float = 10.0,
        progress_callback: Any | None = None,
    ) -> tuple[bool, str, Any]:
        """Fetch data by sampling ``POST /api/calc/`` on a lat/lon grid.

        The official SERENE API is point-based (``latitude`` / ``longitude`` form
        fields only).  ``model``, ``start_time``, and ``end_time`` are kept in the
        signature for dashboard compatibility but are not sent to ``/api/calc/``
        until SERENE documents those parameters.
        """
        del variables, start_time, end_time  # reserved for future API versions

        r = region or {
            "lat_min": -90.0,
            "lat_max": 90.0,
            "lon_min": -180.0,
            "lon_max": 180.0,
        }
        ok_grid, msg_grid, batch = self._fetch_calc_grid(
            lat_min=r["lat_min"],
            lat_max=r["lat_max"],
            lon_min=r["lon_min"],
            lon_max=r["lon_max"],
            lat_step=grid_step,
            lon_step=grid_step,
            model=model,
            progress_callback=progress_callback,
        )
        if ok_grid and batch:
            return True, (
                f"Grid sampled via official POST {ENDPOINTS['calc']} "
                f"({len(batch)} point(s))."
            ), batch

        return False, msg_grid or "No data returned from SERENE API.", None

    @staticmethod
    def estimate_grid_points(
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        lat_step: float,
        lon_step: float,
        max_points: int = MAX_GRID_POINTS,
    ) -> tuple[int, float, float]:
        """Return (point_count, effective_lat_step, effective_lon_step)."""
        step_lat, step_lon = float(lat_step), float(lon_step)
        for _ in range(20):
            lats = np.arange(lat_min, lat_max + step_lat / 2, step_lat)
            lons = np.arange(lon_min, lon_max + step_lon / 2, step_lon)
            count = len(lats) * len(lons)
            if count <= max_points:
                return count, step_lat, step_lon
            step_lat = min(step_lat * 1.5, 45.0)
            step_lon = min(step_lon * 1.5, 45.0)
        return max_points, step_lat, step_lon

    def _fetch_calc_grid(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        lat_step: float,
        lon_step: float,
        model: str,
        progress_callback: Any | None = None,
    ) -> tuple[bool, str, list[dict[str, Any]]]:
        """Sample official ``POST /api/calc/`` on a lat/lon grid."""
        n_pts, eff_lat_step, eff_lon_step = self.estimate_grid_points(
            lat_min, lat_max, lon_min, lon_max, lat_step, lon_step,
        )
        lats = np.arange(lat_min, lat_max + eff_lat_step / 2, eff_lat_step)
        lons = np.arange(lon_min, lon_max + eff_lon_step / 2, eff_lon_step)
        points = [(float(lat), float(lon)) for lat in lats for lon in lons]

        results: list[dict[str, Any]] = []
        success_count = 0
        total = len(points)

        for i, (lat, lon) in enumerate(points):
            if progress_callback:
                progress_callback(i + 1, total)
            ok, _msg, data = self._request(
                "POST",
                ENDPOINTS["calc"],
                data=self._calc_form(lat, lon),
            )
            if ok and data is not None:
                success_count += 1
                results.append({
                    "lat": lat,
                    "lon": lon,
                    "model": model,
                    "response": data,
                })

        if success_count == 0:
            return False, "All /api/calc/ point requests failed or returned empty data.", []

        note = ""
        if eff_lat_step > lat_step or eff_lon_step > lon_step:
            note = (
                f" Grid auto-widened to {eff_lat_step:.1f}°×{eff_lon_step:.1f}° "
                f"(max {MAX_GRID_POINTS} API calls)."
            )
        return (
            True,
            f"{success_count}/{total} grid point(s) returned data.{note}",
            results,
        )

    # ── Response parsing ────────────────────────────────────────────────────

    def parse_response_to_dataframe(
        self,
        response_data: Any,
        model: str | None = None,
    ) -> pd.DataFrame:
        """Convert API JSON into the standard long-form schema."""
        if response_data is None:
            logger.warning("parse_response_to_dataframe received None.")
            return pd.DataFrame()

        # Batch wrapper from grid fallback
        if isinstance(response_data, list) and response_data and isinstance(response_data[0], dict):
            if "response" in response_data[0]:
                frames = []
                for item in response_data:
                    sub = self.parse_response_to_dataframe(
                        item.get("response"),
                        model=item.get("model") or model,
                    )
                    if sub.empty:
                        continue
                    if "lat" not in sub.columns and item.get("lat") is not None:
                        sub["lat"] = item["lat"]
                    if "lon" not in sub.columns and item.get("lon") is not None:
                        sub["lon"] = item["lon"]
                    frames.append(sub)
                if frames:
                    return pd.concat(frames, ignore_index=True)
                return pd.DataFrame()

        records = _extract_records(response_data)
        if not records:
            logger.warning("Could not extract records from SERENE response.")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = _normalise_columns(df)

        if model and "model" not in df.columns:
            df["model"] = model

        for col in ("time", "timestamp", "date", "datetime"):
            if col in df.columns:
                try:
                    df["time"] = pd.to_datetime(df[col])
                except (ValueError, TypeError):
                    pass
                break

        return df

    # ── Backward-compatible helpers (used by scripts / tests) ───────────────

    def fetch_space_weather(self, lat: float, lon: float) -> tuple[bool, str, Any]:
        """Single-point query — official ``POST /api/calc/`` format."""
        return self._request(
            "POST",
            ENDPOINTS["calc"],
            data=self._calc_form(lat, lon),
        )


# ── Internal helpers ────────────────────────────────────────────────────────


def _extract_string_list(data: Any, keys: tuple[str, ...]) -> list[str]:
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        for key in keys:
            if key in data and isinstance(data[key], list):
                return [str(x) for x in data[key]]
    return []


def _parse_calc_text_response(text: str) -> list[dict[str, Any]]:
    """Parse official /api/calc/ plain-text lines, e.g. ``TEC: 7.85``."""
    records: list[dict[str, Any]] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        name, _, raw_val = line.partition(":")
        name = name.strip()
        raw_val = raw_val.strip()
        try:
            value = float(raw_val)
        except ValueError:
            value = raw_val
        records.append({"variable": name, "value": value})
    if not records and text.strip():
        records.append({"value": text.strip()})
    return records


def _extract_records(response_data: Any) -> list[dict[str, Any]]:
    if isinstance(response_data, str):
        parsed = _parse_calc_text_response(response_data)
        return parsed if parsed else [{"value": response_data}]

    if isinstance(response_data, list):
        if response_data and isinstance(response_data[0], dict) and "response" in response_data[0]:
            return []
        return [r for r in response_data if isinstance(r, dict)]

    if isinstance(response_data, dict):
        for key in ("data", "results", "records", "output", "variables", "grid", "parameters"):
            if key not in response_data:
                continue
            candidate = response_data[key]
            if isinstance(candidate, list):
                return [r for r in candidate if isinstance(r, dict)]
            if isinstance(candidate, dict):
                return _columnar_to_records(candidate)

        flat = _flatten_dict(response_data)
        return flat if flat else [response_data]

    return []


def _columnar_to_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    if not data:
        return []
    arrays = {k: v if isinstance(v, (list, tuple)) else [v] for k, v in data.items()}
    length = max(len(v) for v in arrays.values())
    return [
        {key: values[i] if i < len(values) else None for key, values in arrays.items()}
        for i in range(length)
    ]


def _flatten_dict(data: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key, val in data.items():
        if isinstance(val, dict) and "value" in val:
            record: dict[str, Any] = {"variable": key}
            record.update(val)
            records.append(record)
    return records


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "timestamp": "time",
        "date": "time",
        "datetime": "time",
        "t": "time",
        "latitude": "lat",
        "longitude": "lon",
        "long": "lon",
        "altitude": "alt",
        "height": "alt",
        "h": "alt",
        "var": "variable",
        "param": "variable",
        "parameter": "variable",
        "field": "variable",
        "val": "value",
        "data_value": "value",
    }
    rename = {k: v for k, v in mapping.items() if k in df.columns and v not in df.columns}
    return df.rename(columns=rename)
