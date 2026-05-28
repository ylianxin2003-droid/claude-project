"""
SERENE geomagnetic indices loader and risk classifier.

This module uses SERENE's downloadable Kp/ap index CSV. It does not use
``/api/calc/`` and does not claim retrospective AIDA map retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Any

import pandas as pd
import requests

SERENE_KP_AP_URL = "https://serene.bham.ac.uk/resources/download/Indices__Kp_ap.csv/"

RISK_PRIORITY = {"Severe": 0, "Warning": 1, "Watch": 2, "Normal": 3}
G_SCALE_PRIORITY = {
    "G5 Extreme": 0,
    "G4 Severe": 1,
    "G3 Strong": 2,
    "G2 Moderate": 3,
    "G1 Minor": 4,
    "G0 Below storm": 5,
}

_IMPACT_BY_G_SCALE: dict[str, tuple[str, str, str]] = {
    "G0 Below storm": (
        "Normal",
        "No storm-level geomagnetic risk indicated by Kp.",
        "Routine monitoring. SERENE AIDA/API map analysis can still be used for ionospheric details.",
    ),
    "G1 Minor": (
        "Watch",
        "Minor geomagnetic storm conditions.",
        "Monitor HF and GNSS-sensitive operations, especially at high latitudes.",
    ),
    "G2 Moderate": (
        "Watch",
        "Moderate geomagnetic storm conditions.",
        "HF propagation at high latitudes may fade; GNSS accuracy should be monitored.",
    ),
    "G3 Strong": (
        "Warning",
        "Strong geomagnetic storm conditions.",
        "Intermittent HF and satellite navigation problems are possible; review operational backups.",
    ),
    "G4 Severe": (
        "Severe",
        "Severe geomagnetic storm conditions.",
        "HF propagation can become sporadic and satellite navigation may degrade for hours.",
    ),
    "G5 Extreme": (
        "Severe",
        "Extreme geomagnetic storm conditions.",
        "Widespread HF disruption and prolonged satellite navigation degradation are possible.",
    ),
}


@dataclass
class IndicesLoadStatus:
    """Status metadata for a SERENE indices load."""

    ok: bool = False
    source_url: str = SERENE_KP_AP_URL
    message: str = ""
    rows: int = 0


def classify_kp_risk(kp: float | int | str | None) -> tuple[str, str]:
    """Return ``(g_scale, prototype_risk_level)`` for a Kp value."""
    try:
        value = float(kp)
    except (TypeError, ValueError):
        return "G0 Below storm", "Normal"

    if value >= 9.0:
        return "G5 Extreme", "Severe"
    if value >= 8.0:
        return "G4 Severe", "Severe"
    if value >= 7.0:
        return "G3 Strong", "Warning"
    if value >= 6.0:
        return "G2 Moderate", "Watch"
    if value >= 5.0:
        return "G1 Minor", "Watch"
    return "G0 Below storm", "Normal"


def impact_for_g_scale(g_scale: str) -> tuple[str, str, str]:
    """Return ``(risk_level, impact, interpretation)`` for a G-scale label."""
    return _IMPACT_BY_G_SCALE.get(g_scale, _IMPACT_BY_G_SCALE["G0 Below storm"])


def add_kp_risk_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise SERENE Kp/ap rows and attach risk metadata."""
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    if "time" not in work.columns or "Kp" not in work.columns:
        raise ValueError("SERENE Kp/ap data must contain 'time' and 'Kp' columns.")

    work["time"] = pd.to_datetime(work["time"], errors="coerce", utc=True)
    work["Kp"] = pd.to_numeric(work["Kp"], errors="coerce")
    if "ap" in work.columns:
        work["ap"] = pd.to_numeric(work["ap"], errors="coerce")
    if "rAp" in work.columns:
        work["rAp"] = pd.to_numeric(work["rAp"], errors="coerce")

    work = work.dropna(subset=["time", "Kp"]).sort_values("time").reset_index(drop=True)
    if work.empty:
        return work

    classified = work["Kp"].apply(classify_kp_risk)
    work["g_scale"] = classified.apply(lambda item: item[0])
    work["risk_level"] = classified.apply(lambda item: item[1])
    work["risk_rank"] = work["risk_level"].map(RISK_PRIORITY).fillna(3).astype(int)
    work["g_rank"] = work["g_scale"].map(G_SCALE_PRIORITY).fillna(5).astype(int)
    work["variable"] = "Kp"
    work["alert_type"] = "Geomagnetic storm risk"
    work["region"] = "Global geomagnetic"
    work["possible_aviation_impact"] = work["g_scale"].apply(
        lambda g: impact_for_g_scale(g)[1]
    )
    work["interpretation"] = work["g_scale"].apply(lambda g: impact_for_g_scale(g)[2])
    return work


def load_kp_ap_indices(
    url: str = SERENE_KP_AP_URL,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, IndicesLoadStatus]:
    """Download SERENE Kp/ap index data and return a normalised DataFrame."""
    http = session or requests.Session()
    try:
        response = http.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        return pd.DataFrame(), IndicesLoadStatus(
            ok=False,
            source_url=url,
            message=f"Could not load SERENE Kp/ap indices: {exc}",
        )

    try:
        raw = pd.read_csv(StringIO(response.text))
        df = add_kp_risk_columns(raw)
    except Exception as exc:
        return pd.DataFrame(), IndicesLoadStatus(
            ok=False,
            source_url=url,
            message=f"Could not parse SERENE Kp/ap indices: {exc}",
        )

    return df, IndicesLoadStatus(
        ok=True,
        source_url=url,
        message=f"Loaded {len(df):,} SERENE Kp/ap interval(s).",
        rows=len(df),
    )


def filter_indices_by_time(
    df: pd.DataFrame,
    start_time: Any,
    end_time: Any,
) -> pd.DataFrame:
    """Return rows whose timestamp lies inside the inclusive UTC window."""
    if df.empty:
        return pd.DataFrame()

    start = pd.to_datetime(start_time, errors="coerce", utc=True)
    end = pd.to_datetime(end_time, errors="coerce", utc=True)
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Invalid indices time range.")
    if start > end:
        raise ValueError("Start time must be before or equal to end time.")

    work = df.copy()
    work["time"] = pd.to_datetime(work["time"], errors="coerce", utc=True)
    return work[(work["time"] >= start) & (work["time"] <= end)].reset_index(drop=True)


def risk_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count intervals by G scale and prototype risk level."""
    if df.empty:
        return pd.DataFrame(columns=["g_scale", "risk_level", "intervals"])
    counts = (
        df.groupby(["g_scale", "risk_level"], dropna=False)
        .size()
        .reset_index(name="intervals")
    )
    counts["g_rank"] = counts["g_scale"].map(G_SCALE_PRIORITY).fillna(5).astype(int)
    return counts.sort_values("g_rank").drop(columns=["g_rank"]).reset_index(drop=True)


def daily_peak_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise daily peak Kp/ap and storm interval counts."""
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    work["date"] = pd.to_datetime(work["time"], utc=True).dt.date
    rows: list[dict[str, Any]] = []
    for day, grp in work.groupby("date", sort=True):
        max_kp = float(grp["Kp"].max())
        g_scale, risk_level = classify_kp_risk(max_kp)
        max_ap = float(grp["ap"].max()) if "ap" in grp.columns else None
        rows.append({
            "date": day,
            "max_Kp": max_kp,
            "max_ap": max_ap,
            "peak_g_scale": g_scale,
            "peak_risk_level": risk_level,
            "storm_intervals_Kp_ge_5": int((grp["Kp"] >= 5.0).sum()),
            "strong_intervals_Kp_ge_7": int((grp["Kp"] >= 7.0).sum()),
            "severe_intervals_Kp_ge_8": int((grp["Kp"] >= 8.0).sum()),
        })
    return pd.DataFrame(rows)


def build_indices_alerts(df: pd.DataFrame, minimum_kp: float = 5.0) -> pd.DataFrame:
    """Build advisory-like rows from Kp intervals at or above *minimum_kp*."""
    if df.empty:
        return pd.DataFrame()

    events = df[pd.to_numeric(df["Kp"], errors="coerce") >= float(minimum_kp)].copy()
    if events.empty:
        return pd.DataFrame()

    events["timestamp"] = events["time"]
    events["value"] = events["Kp"]
    events["threshold_info"] = events.apply(
        lambda row: f"Kp={row['Kp']:.1f}, ap={row.get('ap', float('nan')):.0f}, {row['g_scale']}",
        axis=1,
    )
    events["reason"] = events["threshold_info"]
    keep_cols = [
        "timestamp",
        "region",
        "variable",
        "alert_type",
        "risk_level",
        "g_scale",
        "value",
        "ap",
        "threshold_info",
        "reason",
        "possible_aviation_impact",
        "interpretation",
    ]
    return events[[col for col in keep_cols if col in events.columns]].reset_index(drop=True)
