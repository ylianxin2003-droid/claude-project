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


# ── Variable summary table ────────────────────────────────────────────────────


def create_variable_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-variable summary DataFrame for display.

    Returns a DataFrame with columns:
        variable, unit, description, min, max, mean, valid_points, risk_relevance

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least ``variable`` and ``value`` columns.
        ``unit`` and ``description`` columns are used when present.

    Returns
    -------
    pd.DataFrame
        One row per variable, sorted by variable name.
    """
    if df is None or df.empty or "variable" not in df.columns:
        return pd.DataFrame(columns=[
            "variable", "unit", "description", "min", "max", "mean", "valid_points", "risk_relevance",
        ])

    rows: list[dict[str, object]] = []
    var_names = sorted(df["variable"].dropna().unique())

    for var in var_names:
        var_df = df[df["variable"] == var]
        vals = pd.to_numeric(var_df["value"], errors="coerce").dropna()

        unit = ""
        if "unit" in var_df.columns:
            first = var_df["unit"].dropna()
            if not first.empty:
                unit = str(first.iloc[0])
        description = ""
        if "description" in var_df.columns:
            first = var_df["description"].dropna()
            if not first.empty:
                description = str(first.iloc[0])

        risk_rel = _risk_relevance(var)

        rows.append({
            "variable": var,
            "unit": unit,
            "description": description,
            "min": round(float(vals.min()), 3) if not vals.empty else None,
            "max": round(float(vals.max()), 3) if not vals.empty else None,
            "mean": round(float(vals.mean()), 3) if not vals.empty else None,
            "valid_points": len(vals),
            "risk_relevance": risk_rel,
        })

    return pd.DataFrame(rows)


def _risk_relevance(variable: str) -> str:
    """Map a variable name to its risk relevance category."""
    name = variable.lower()
    if "tec" in name and "dep" not in name:
        return "GNSS positioning risk"
    if "muf" in name or "fof2" in name:
        return "HF communication risk"
    if "hmf2" in name or "nmf2" in name:
        return "General ionospheric monitoring"
    return "General ionospheric monitoring"


# ── Multi-variable time series ────────────────────────────────────────────────


def create_multi_variable_time_series(
    df: pd.DataFrame,
    normalize: bool = False,
    title: str | None = None,
) -> go.Figure:
    """Time-series plot with one subplot per variable.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``time``, ``variable``, and ``value`` columns.
    normalize : bool
        If True, apply min-max normalisation per variable so that variables
        with different units can be compared on the same scale.
    title : str, optional
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df is None or df.empty:
        return empty_figure("No data available for time-series plot.")

    if "variable" not in df.columns or "value" not in df.columns:
        return empty_figure("Data missing required columns (variable, value).")

    work = df.copy()
    if "time" in work.columns:
        work["time"] = pd.to_datetime(work["time"], errors="coerce")
    else:
        work["time"] = pd.to_datetime("now")

    grouped = work.groupby(["time", "variable"], as_index=False)["value"].mean()
    var_names = sorted(grouped["variable"].dropna().unique())

    if not var_names:
        return empty_figure("No variables with valid data.")

    if normalize:
        for var in var_names:
            mask = grouped["variable"] == var
            vmin = grouped.loc[mask, "value"].min()
            vmax = grouped.loc[mask, "value"].max()
            if vmax > vmin:
                grouped.loc[mask, "value"] = (grouped.loc[mask, "value"] - vmin) / (vmax - vmin)
            else:
                grouped.loc[mask, "value"] = 0.5
        yaxis_title = "Normalised value (0–1)"
    else:
        yaxis_title = "Value"

    n_vars = len(var_names)
    fig = make_subplots(
        rows=n_vars, cols=1,
        shared_xaxes=True,
        subplot_titles=list(var_names),
        vertical_spacing=0.03,
    )

    for i, var in enumerate(var_names):
        sub = grouped[grouped["variable"] == var]
        fig.add_trace(
            go.Scatter(
                x=sub["time"], y=sub["value"],
                mode="lines+markers", name=var,
                showlegend=False,
            ),
            row=i + 1, col=1,
        )

    fig.update_layout(
        title_text=title or "All variables — time series",
        height=max(300, 180 * n_vars),
        template="plotly_white",
        hovermode="x unified",
        yaxis_title=yaxis_title,
    )
    return fig


# ── Variable map grid ─────────────────────────────────────────────────────────


def create_variable_map_grid(df: pd.DataFrame) -> go.Figure:
    """Subplot grid of scatter-geo maps, one per variable.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``lat``, ``lon``, ``variable``, ``value`` columns.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df is None or df.empty:
        return empty_figure("No data available for map grid.")

    if "variable" not in df.columns:
        return empty_figure("Data missing 'variable' column for map grid.")

    var_names = sorted(df["variable"].dropna().unique())
    if not var_names:
        return empty_figure("No variables in data.")

    n_vars = len(var_names)
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=list(var_names),
        specs=[[{"type": "scattergeo"} for _ in range(n_cols)] for _ in range(n_rows)],
        vertical_spacing=0.05,
        horizontal_spacing=0.02,
    )

    for i, var in enumerate(var_names):
        row = i // n_cols + 1
        col = i % n_cols + 1
        var_df = df[df["variable"] == var]
        if len(var_df) > 1500:
            var_df = var_df.sample(n=1500, random_state=42)

        # Latest time slice if multiple.
        if "time" in var_df.columns:
            var_df["time"] = pd.to_datetime(var_df["time"], errors="coerce")
            var_df = var_df[var_df["time"] == var_df["time"].max()]

        fig.add_trace(
            go.Scattergeo(
                lat=var_df["lat"], lon=var_df["lon"],
                mode="markers",
                marker=dict(
                    size=4,
                    color=var_df["value"],
                    colorscale="Plasma",
                    showscale=(i == 0),
                    colorbar=dict(title="Value", x=0.02, len=0.3) if i == 0 else None,
                ),
                name=var,
                text=var_df["value"].round(3),
                hoverinfo="text+name",
                showlegend=False,
            ),
            row=row, col=col,
        )

    fig.update_geos(
        showcoastlines=True, coastlinecolor="gray",
        showland=True, landcolor="lightgray",
        showocean=True, oceancolor="aliceblue",
    )
    fig.update_layout(
        title="All variables — maps",
        template="plotly_white",
        height=280 * n_rows,
    )
    return fig


# ── Variable card data ────────────────────────────────────────────────────────


def create_variable_card_data(df: pd.DataFrame) -> list[dict[str, object]]:
    """Build per-variable summary cards for use with ``st.metric``.

    Returns a list of dicts, one per variable, with keys:
        variable, unit, description, min, max, mean, count, risk_relevance

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``variable`` and ``value`` columns.

    Returns
    -------
    list[dict]
    """
    if df is None or df.empty or "variable" not in df.columns:
        return []

    cards: list[dict[str, object]] = []
    for var in sorted(df["variable"].dropna().unique()):
        var_df = df[df["variable"] == var]
        vals = pd.to_numeric(var_df["value"], errors="coerce").dropna()
        if vals.empty:
            continue

        unit = ""
        if "unit" in var_df.columns:
            first = var_df["unit"].dropna()
            if not first.empty:
                unit = str(first.iloc[0])
        description = ""
        if "description" in var_df.columns:
            first = var_df["description"].dropna()
            if not first.empty:
                description = str(first.iloc[0])

        cards.append({
            "variable": var,
            "unit": unit,
            "description": description,
            "min": round(float(vals.min()), 3),
            "max": round(float(vals.max()), 3),
            "mean": round(float(vals.mean()), 3),
            "count": len(vals),
            "risk_relevance": _risk_relevance(var),
        })

    return cards


# ── Risk timeline ────────────────────────────────────────────────────────────


def create_risk_timeline(
    hazards_df: pd.DataFrame,
    title: str | None = None,
) -> go.Figure:
    """Line chart of risk level over time per variable.

    Each risk level is mapped to a numeric Y value (Normal=0, Watch=1,
    Warning=2, Severe=3) and plotted as a stepped line.

    Parameters
    ----------
    hazards_df : pd.DataFrame
        Must contain ``timestamp``, ``variable``, ``risk_level`` columns.
    title : str, optional
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if hazards_df.empty:
        return empty_figure("No hazard data for risk timeline.")

    needed = {"timestamp", "variable", "risk_level"}
    if not needed.issubset(hazards_df.columns):
        return empty_figure("Hazard data missing required columns (timestamp, variable, risk_level).")

    risk_value = {"Normal": 0, "Watch": 1, "Warning": 2, "Severe": 3}

    work = hazards_df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"])
    work["risk_num"] = work["risk_level"].map(risk_value).fillna(0)
    work = work.sort_values("timestamp")

    fig = go.Figure()
    for var in sorted(work["variable"].dropna().unique()):
        var_df = work[work["variable"] == var]
        fig.add_trace(go.Scatter(
            x=var_df["timestamp"],
            y=var_df["risk_num"],
            mode="lines+markers",
            name=var,
            line=dict(shape="hv"),
            marker=dict(size=10),
        ))

    fig.update_yaxes(
        tickvals=[0, 1, 2, 3],
        ticktext=["Normal", "Watch", "Warning", "Severe"],
    )
    fig.update_layout(
        title=title or "Risk level timeline by variable",
        xaxis_title="Time",
        yaxis_title="Risk level",
        template="plotly_white",
        height=450,
        hovermode="x unified",
    )
    return fig


def create_hazard_summary_map(
    hazards_df: pd.DataFrame,
    title: str | None = None,
) -> go.Figure:
    """Scatter-geo map of detected hazards colour-coded by risk level.

    Parameters
    ----------
    hazards_df : pd.DataFrame
        Must contain ``region``, ``risk_level``, ``variable`` columns.
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if hazards_df.empty:
        return empty_figure("No hazards to display on map.")

    from grid_generator import region_bounds

    colors = {"Normal": "#2ecc71", "Watch": "#f1c40f", "Warning": "#e67e22", "Severe": "#e74c3c"}

    fig = go.Figure()
    seen: set[tuple[str, str, str]] = set()
    for _, row in hazards_df.iterrows():
        region_name = str(row.get("region", "global"))
        var = str(row.get("variable", "?"))
        risk = str(row.get("risk_level", "Normal"))
        key = (region_name, var, risk)
        if key in seen:
            continue
        seen.add(key)

        bounds = region_bounds(region_name)
        if bounds is None:
            bounds = region_bounds("global")
        center_lat = (bounds["lat_min"] + bounds["lat_max"]) / 2
        center_lon = (bounds["lon_min"] + bounds["lon_max"]) / 2

        fig.add_trace(go.Scattergeo(
            lon=[center_lon],
            lat=[center_lat],
            mode="markers+text",
            marker=dict(
                size=16,
                color=colors.get(risk, "#95a5a6"),
                symbol="diamond",
                line=dict(width=1, color="black"),
            ),
            text=var,
            textposition="top center",
            name=f"{var} — {risk}",
            hovertext=(
                f"Region: {region_name}<br>"
                f"Variable: {var}<br>"
                f"Risk: {risk}<br>"
                f"Value: {row.get('max_value', '?')}<br>"
                f"Reason: {row.get('reason', '')}"
            ),
            hoverinfo="text",
            showlegend=False,
        ))

    fig.update_geos(
        showcoastlines=True, coastlinecolor="gray",
        showland=True, landcolor="lightgray",
        showocean=True, oceancolor="aliceblue",
    )
    fig.update_layout(
        title=title or "Hazard summary map",
        template="plotly_white",
        height=500,
    )

    # Add legend manually
    for level, color in colors.items():
        fig.add_trace(go.Scattergeo(
            lon=[None], lat=[None],
            mode="markers",
            marker=dict(size=12, color=color, symbol="diamond"),
            name=level,
            showlegend=True,
        ))

    return fig


def create_spatial_gradient_map(
    df: pd.DataFrame,
    variable: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Heatmap of spatial gradient magnitude on a map.

    Computes spatial gradients for the given variable, then renders a
    scatter-geo map where marker color encodes gradient magnitude.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``lat``, ``lon``, ``variable``, ``value`` columns.
    variable : str, optional
        Filter to one variable.
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    from hazard_detector import _compute_spatial_gradients

    if df.empty:
        return empty_figure("No data for spatial gradient map.")

    work = df.copy()
    if variable and "variable" in work.columns:
        work = work[work["variable"] == variable]

    if work.empty:
        return empty_figure(f"No data for variable '{variable}'.")

    if "spatial_gradient" not in work.columns:
        try:
            work = _compute_spatial_gradients(work)
        except Exception:
            return empty_figure("Could not compute spatial gradients.")

    work = work.dropna(subset=["spatial_gradient"])
    if work.empty:
        return empty_figure("No spatial gradient data after computation.")

    if len(work) > 3000:
        work = work.sample(n=3000, random_state=42)

    var_label = variable or "all variables"
    fig = px.scatter_geo(
        work,
        lat="lat",
        lon="lon",
        color="spatial_gradient",
        size="spatial_gradient",
        color_continuous_scale="RdYlBu_r",
        title=title or f"Spatial gradient — {var_label}",
        projection="natural earth",
        labels={"spatial_gradient": "Gradient (value/deg)"},
    )
    fig.update_geos(
        showcoastlines=True, coastlinecolor="gray",
        showland=True, landcolor="lightgray",
        showocean=True, oceancolor="aliceblue",
    )
    fig.update_layout(template="plotly_white", height=500)
    return fig


def create_temporal_change_map(
    current_df: pd.DataFrame,
    previous_df: pd.DataFrame | None = None,
    variable: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Heatmap of temporal change rate on a map.

    If *previous_df* is provided, computes the absolute value difference per
    hour between the current and previous map. Otherwise shows a placeholder.

    Parameters
    ----------
    current_df : pd.DataFrame
        Must contain ``lat``, ``lon``, ``variable``, ``value``, ``time``.
    previous_df : pd.DataFrame or None
        Previous time-step data for comparison.
    variable : str, optional
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    from hazard_detector import _compute_temporal_change

    if current_df.empty:
        return empty_figure("No data for temporal change map.")

    work = current_df.copy()
    if variable and "variable" in work.columns:
        work = work[work["variable"] == variable]

    if work.empty:
        return empty_figure(f"No data for variable '{variable}'.")

    work = _compute_temporal_change(work, previous_df)

    if "temporal_change" not in work.columns or work["temporal_change"].max() == 0:
        return empty_figure(
            "No temporal change detected — either no previous data for comparison "
            "or all changes are zero."
        )

    work = work.dropna(subset=["temporal_change"])
    if len(work) > 3000:
        work = work.sample(n=3000, random_state=42)

    var_label = variable or "all variables"
    fig = px.scatter_geo(
        work,
        lat="lat",
        lon="lon",
        color="temporal_change",
        size="temporal_change",
        color_continuous_scale="OrRd",
        title=title or f"Temporal change rate — {var_label}",
        projection="natural earth",
        labels={"temporal_change": "Change rate (value/hr)"},
    )
    fig.update_geos(
        showcoastlines=True, coastlinecolor="gray",
        showland=True, landcolor="lightgray",
        showocean=True, oceancolor="aliceblue",
    )
    fig.update_layout(template="plotly_white", height=500)
    return fig


def create_advisory_card_html(advisory: dict[str, object]) -> str:
    """Render a single ICAO-style prototype advisory as an HTML card.

    Parameters
    ----------
    advisory : dict
        Keys: timestamp, region, variable, alert_type, risk_level, reason,
        possible_aviation_impact, interpretation, disclaimer.

    Returns
    -------
    str
        HTML string suitable for ``st.markdown(..., unsafe_allow_html=True)``.
    """
    risk = str(advisory.get("risk_level", "Normal"))
    colors = {
        "Normal": ("#2ecc71", "#1a7a3a"),
        "Watch": ("#f1c40f", "#8a6d00"),
        "Warning": ("#e67e22", "#8a3d00"),
        "Severe": ("#e74c3c", "#8a1c1c"),
    }
    bg, border = colors.get(risk, ("#95a5a6", "#555"))

    timestamp = advisory.get("timestamp", "—")
    region = advisory.get("region", "—")
    variable = advisory.get("variable", "—")
    alert_type = advisory.get("alert_type", "—")
    reason = advisory.get("reason", "—")
    impact = advisory.get("possible_aviation_impact", "—")
    interpretation = advisory.get("interpretation", "—")
    disclaimer = advisory.get("disclaimer", "")

    return f"""
<div style="
    border-left: 5px solid {border};
    background: {bg}15;
    padding: 16px;
    margin: 10px 0;
    border-radius: 6px;
">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
        <span style="font-weight: bold; font-size: 1.1em; color: {border};">{alert_type}</span>
        <span style="
            background: {bg}; color: white; padding: 3px 12px;
            border-radius: 12px; font-weight: bold; font-size: 0.85em;
        ">{risk.upper()}</span>
    </div>
    <table style="width: 100%; font-size: 0.9em; border-collapse: collapse;">
        <tr><td style="color: #7f8c8d; width: 100px;">Timestamp</td><td>{timestamp}</td></tr>
        <tr><td style="color: #7f8c8d;">Region</td><td>{region}</td></tr>
        <tr><td style="color: #7f8c8d;">Variable</td><td>{variable}</td></tr>
        <tr><td style="color: #7f8c8d;">Reason</td><td>{reason}</td></tr>
    </table>
    <div style="margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 4px;">
        <strong>Aviation impact:</strong> {impact}<br>
        <strong>Interpretation:</strong> {interpretation}
    </div>
    <div style="margin-top: 8px; font-size: 0.78em; color: #7f8c8d; font-style: italic;">
        {disclaimer}
    </div>
</div>"""


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
