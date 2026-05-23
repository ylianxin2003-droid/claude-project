"""
Plotly-based visualisation functions for the aviation space weather dashboard.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Colour maps ─────────────────────────────────────────────────────────────

RISK_COLORS: dict[str, str] = {
    "Normal": "#2ecc71",
    "Watch": "#f1c40f",
    "Warning": "#e67e22",
    "Severe": "#e74c3c",
}

ALERT_TYPE_COLORS: dict[str, str] = {
    "GNSS positioning risk": "#3498db",
    "HF communication risk": "#9b59b6",
    "General ionospheric disturbance": "#e74c3c",
}


# ── Time series ─────────────────────────────────────────────────────────────


def create_time_series_plot(
    df: pd.DataFrame,
    variable: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Create a time-series line plot for one or all variables.

    Parameters
    ----------
    df : DataFrame
        Must contain at least ``time``, ``value``, and ``variable`` columns.
    variable : str, optional
        Filter to a single variable.  If ``None``, plot all variables.
    title : str, optional
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for time-series plot.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    work = df.copy()
    if "time" not in work.columns:
        # Try to use the index or create a synthetic time axis.
        work["time"] = pd.to_datetime("now")
    work["time"] = pd.to_datetime(work["time"])

    if variable:
        work = work[work["variable"] == variable]

    if work.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data for variable '{variable}'.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    # Aggregate: mean value per time step per variable.
    grouped = work.groupby(["time", "variable"], as_index=False)["value"].mean()

    if grouped["variable"].nunique() <= 1:
        fig = px.line(
            grouped, x="time", y="value", color="variable",
            title=title or "Ionospheric parameter over time",
            labels={"value": "Value", "time": "Time", "variable": "Variable"},
        )
    else:
        fig = make_subplots(
            rows=grouped["variable"].nunique(),
            cols=1,
            shared_xaxes=True,
            subplot_titles=list(grouped["variable"].unique()),
        )
        for i, var in enumerate(grouped["variable"].unique()):
            sub = grouped[grouped["variable"] == var]
            fig.add_trace(
                go.Scatter(x=sub["time"], y=sub["value"], mode="lines+markers", name=var),
                row=i + 1, col=1,
            )
        fig.update_layout(
            title_text=title or "Ionospheric parameters over time",
            height=250 * grouped["variable"].nunique(),
        )

    fig.update_layout(template="plotly_white", hovermode="x unified")
    return fig


# ── Map plot ────────────────────────────────────────────────────────────────


def create_map_plot(
    df: pd.DataFrame,
    variable: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Create a scatter-geo map of the data.

    Expects ``lat``, ``lon``, ``value`` columns.

    Parameters
    ----------
    df : DataFrame
    variable : str, optional
        Filter to one variable.
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for map plot.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    work = df.copy()
    if variable:
        work = work[work["variable"] == variable]

    # Keep maps responsive for large local grid files.
    if len(work) > 3000:
        work = work.sample(n=3000, random_state=42)

    if "lat" not in work.columns or "lon" not in work.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Data does not contain lat/lon columns for map display.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    # If multiple time steps, use the latest.
    if "time" in work.columns:
        work["time"] = pd.to_datetime(work["time"])
        work = work[work["time"] == work["time"].max()]

    fig = px.scatter_geo(
        work,
        lat="lat",
        lon="lon",
        color="value",
        size="value",
        hover_name="variable" if "variable" in work.columns else None,
        hover_data=["value", "variable"] if "variable" in work.columns else ["value"],
        title=title or f"Global {variable or 'ionospheric'} map",
        color_continuous_scale="Plasma",
        projection="natural earth",
    )
    fig.update_geos(
        showcoastlines=True,
        coastlinecolor="gray",
        showland=True,
        landcolor="lightgray",
        showocean=True,
        oceancolor="aliceblue",
    )
    fig.update_layout(template="plotly_white", height=500)
    return fig


# ── Alert timeline ──────────────────────────────────────────────────────────


def create_alert_timeline(alerts: pd.DataFrame) -> go.Figure:
    """Create a Gantt-like timeline of alerts colour-coded by risk level.

    Parameters
    ----------
    alerts : DataFrame
        Output from :func:`alert_engine.generate_alerts`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if alerts.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No alerts to display — all parameters within normal range.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    needed_cols = {"timestamp", "alert_type", "risk_level"}
    if not needed_cols.issubset(alerts.columns):
        fig = go.Figure()
        fig.add_annotation(
            text="Alert data missing required columns for timeline.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    work = alerts.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.sort_values("timestamp")

    # Group by alert_type and assign y-position.
    alert_types = work["alert_type"].unique()
    y_map = {t: i for i, t in enumerate(alert_types)}

    fig = go.Figure()
    for _, row in work.iterrows():
        risk = row.get("risk_level", "Normal")
        fig.add_trace(go.Scatter(
            x=[row["timestamp"]],
            y=[y_map.get(row.get("alert_type", "Unknown"), 0)],
            mode="markers",
            marker=dict(
                size=14,
                color=RISK_COLORS.get(risk, "#95a5a6"),
                symbol="diamond",
                line=dict(width=1, color="black"),
            ),
            name=f"{row.get('alert_type', '?')} — {risk}",
            text=f"{row.get('region', '?')}<br>{row.get('reason', '')}",
            hoverinfo="text+name",
        ))

    fig.update_yaxes(
        tickvals=list(y_map.values()),
        ticktext=list(y_map.keys()),
    )
    fig.update_layout(
        title="ICAO-style prototype alert timeline",
        xaxis_title="Time",
        yaxis_title="Alert type",
        template="plotly_white",
        height=300 + 60 * len(alert_types),
        showlegend=False,
    )
    return fig


# ── Alert summary ───────────────────────────────────────────────────────────


def create_alert_summary(alerts: pd.DataFrame) -> go.Figure:
    """Create a bar chart summarising alert counts by type and risk level.

    Parameters
    ----------
    alerts : DataFrame
        Output from :func:`alert_engine.generate_alerts`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if alerts.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No alerts — all parameters within normal range.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    if "alert_type" not in alerts.columns or "risk_level" not in alerts.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Alert data missing required columns.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    counts = alerts.groupby(["alert_type", "risk_level"]).size().reset_index(name="count")

    # Ensure consistent risk level ordering.
    risk_order = ["Normal", "Watch", "Warning", "Severe"]
    counts["risk_level"] = pd.Categorical(
        counts["risk_level"], categories=risk_order, ordered=True
    )
    counts = counts.sort_values(["alert_type", "risk_level"])

    fig = px.bar(
        counts,
        x="alert_type",
        y="count",
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        title="Alert summary by type and risk level",
        labels={"count": "Number of advisories", "alert_type": "Alert type", "risk_level": "Risk level"},
        barmode="stack",
        category_orders={"risk_level": risk_order},
    )
    fig.update_layout(template="plotly_white", height=400)
    return fig


# ── Fixed map plot ───────────────────────────────────────────────────────────


def create_fixed_map_plot(
    df: pd.DataFrame,
    variable: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Scatter-geo map of a fixed-resolution grid.

    Parameters
    ----------
    df : DataFrame
        Must contain ``lat``, ``lon``, ``value`` columns.
    variable : str, optional
        Filter to one variable.
    title : str, optional
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df.empty:
        return empty_figure("No fixed map data to display.")

    work = df.copy()
    if variable and "variable" in work.columns:
        work = work[work["variable"] == variable]

    if work.empty:
        return empty_figure(f"No data for variable '{variable}'.")

    if "lat" not in work.columns or "lon" not in work.columns:
        return empty_figure("Fixed map data missing lat/lon columns.")

    # Use latest time step if multiple.
    if "time" in work.columns:
        work["time"] = pd.to_datetime(work["time"], errors="coerce")
        work = work[work["time"] == work["time"].max()]

    # Down-sample large grids for responsive rendering.
    if len(work) > 3000:
        work = work.sample(n=3000, random_state=42)

    fig = px.scatter_geo(
        work,
        lat="lat",
        lon="lon",
        color="value",
        size="value",
        hover_name="variable" if "variable" in work.columns else None,
        hover_data=["value", "variable"] if "variable" in work.columns else ["value"],
        title=title or f"Fixed grid map — {variable or 'ionospheric parameter'}",
        color_continuous_scale="Plasma",
        projection="natural earth",
    )
    fig.update_geos(
        showcoastlines=True,
        coastlinecolor="gray",
        showland=True,
        landcolor="lightgray",
        showocean=True,
        oceancolor="aliceblue",
    )
    fig.update_layout(template="plotly_white", height=500)
    return fig


# ── Hazard map plot ──────────────────────────────────────────────────────────


def create_hazard_map_plot(df: pd.DataFrame) -> go.Figure:
    """Visualise detected hazards on a map.

    Each hazard region is shown as a filled marker colour-coded by risk level.

    Parameters
    ----------
    df : DataFrame
        Output from :func:`hazard_detector.detect_hazards_from_map`.
        Expected columns: ``region``, ``hazard_type``, ``risk_level``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df.empty:
        return empty_figure("No hazard data to display.")

    from grid_generator import region_bounds

    fig = go.Figure()
    hazard_colors = {"Normal": "#2ecc71", "Watch": "#f1c40f", "Warning": "#e67e22", "Severe": "#e74c3c"}

    for _, row in df.iterrows():
        region_name = str(row.get("region", "global"))
        bounds = region_bounds(region_name)
        if bounds is None:
            bounds = region_bounds("global")
        center_lat = (bounds["lat_min"] + bounds["lat_max"]) / 2
        center_lon = (bounds["lon_min"] + bounds["lon_max"]) / 2
        risk = row.get("risk_level", "Normal")

        fig.add_trace(go.Scattergeo(
            lon=[center_lon],
            lat=[center_lat],
            mode="markers",
            marker=dict(
                size=18,
                color=hazard_colors.get(risk, "#95a5a6"),
                symbol="diamond",
                line=dict(width=1, color="black"),
            ),
            name=f"{row.get('hazard_type', '?')} — {risk}",
            text=(
                f"Region: {region_name}<br>"
                f"Variable: {row.get('variable', '?')}<br>"
                f"Risk: {risk}<br>"
                f"{row.get('reason', '')}"
            ),
            hoverinfo="text+name",
        ))

    fig.update_geos(
        showcoastlines=True,
        coastlinecolor="gray",
        showland=True,
        landcolor="lightgray",
        showocean=True,
        oceancolor="aliceblue",
    )
    fig.update_layout(
        title="Hazard detection map",
        template="plotly_white",
        height=450,
        showlegend=False,
    )
    return fig


# ── Historical summary plot ──────────────────────────────────────────────────


def create_historical_summary_plot(alerts: pd.DataFrame) -> go.Figure:
    """Stacked bar chart of alert counts by risk level over time.

    Parameters
    ----------
    alerts : pd.DataFrame
        Must contain ``timestamp`` and ``risk_level`` columns.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if alerts.empty:
        return empty_figure("No alert data for historical summary.")

    if "timestamp" not in alerts.columns or "risk_level" not in alerts.columns:
        return empty_figure("Alert data missing timestamp or risk_level columns.")

    work = alerts.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"])

    if work.empty:
        return empty_figure("No valid timestamps in alert data.")

    risk_colors = {"Normal": "#2ecc71", "Watch": "#f1c40f", "Warning": "#e67e22", "Severe": "#e74c3c"}
    risk_order = ["Normal", "Watch", "Warning", "Severe"]

    # Bucket by hour.
    work["hour_bucket"] = work["timestamp"].dt.floor("h")

    counts = (
        work.groupby(["hour_bucket", "risk_level"])
        .size()
        .reset_index(name="count")
    )
    counts["risk_level"] = pd.Categorical(counts["risk_level"], categories=risk_order, ordered=True)
    counts = counts.sort_values(["hour_bucket", "risk_level"])

    fig = px.bar(
        counts,
        x="hour_bucket",
        y="count",
        color="risk_level",
        color_discrete_map=risk_colors,
        title="Historical alert summary by risk level",
        labels={
            "count": "Number of advisories",
            "hour_bucket": "Time",
            "risk_level": "Risk level",
        },
        barmode="stack",
        category_orders={"risk_level": risk_order},
    )
    fig.update_layout(template="plotly_white", height=400)
    return fig


# ── Utility ─────────────────────────────────────────────────────────────────


def empty_figure(message: str = "No data to display.") -> go.Figure:
    """Return an empty figure with a centred annotation."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="#7f8c8d"),
    )
    fig.update_layout(template="plotly_white", height=300)
    return fig
