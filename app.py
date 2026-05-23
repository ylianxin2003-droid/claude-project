"""
Aviation Space Weather Monitoring & ICAO-style Risk Alert Dashboard.

Run::

    streamlit run app.py
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from alert_engine import DISCLAIMER, generate_alerts, generate_alerts_from_hazards, generate_overall_risk
from config import (
    DEFAULT_REGION,
    DEFAULT_TIME_STEP_HOURS,
    MAP_RESOLUTION_DEFAULT,
    SERENE_API_TOKEN,
    reload_config,
    validate_config,
)
from data_loader import LoadStatus, discover_variables, load_data, resolve_local_file
from grid_generator import list_regions
from hazard_detector import detect_hazards_from_map
from historical_runner import list_historical_runs, load_historical_run, run_historical_analysis, save_historical_run
from map_builder import build_fixed_map
from serene_client import MAX_GRID_POINTS, SereneClient
from visualisation import (
    create_alert_summary,
    create_alert_timeline,
    create_fixed_map_plot,
    create_hazard_map_plot,
    create_historical_summary_plot,
    create_map_plot,
    create_time_series_plot,
)

st.set_page_config(
    page_title="Aviation Space Weather Dashboard",
    page_icon="🛩️",
    layout="wide",
    initial_sidebar_state="expanded",
)

reload_config()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Session state (existing) ─────────────────────────────────────────────────
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame()
if "status" not in st.session_state:
    st.session_state.status = LoadStatus()
if "alerts" not in st.session_state:
    st.session_state.alerts = pd.DataFrame()
if "api_connected" not in st.session_state:
    st.session_state.api_connected = None
if "api_message" not in st.session_state:
    st.session_state.api_message = "Not tested yet."
if "config_warnings" not in st.session_state:
    st.session_state.config_warnings = validate_config()

# ── Session state (live / historical modes) ──────────────────────────────────
if "live_map_df" not in st.session_state:
    st.session_state.live_map_df = pd.DataFrame()
if "live_hazards" not in st.session_state:
    st.session_state.live_hazards = pd.DataFrame()
if "live_alerts" not in st.session_state:
    st.session_state.live_alerts = pd.DataFrame()
if "live_map_status" not in st.session_state:
    st.session_state.live_map_status = ""
if "historical_maps_meta" not in st.session_state:
    st.session_state.historical_maps_meta = []
if "historical_hazards" not in st.session_state:
    st.session_state.historical_hazards = pd.DataFrame()
if "historical_alerts" not in st.session_state:
    st.session_state.historical_alerts = pd.DataFrame()
if "historical_summary" not in st.session_state:
    st.session_state.historical_summary = None
if "historical_run_id" not in st.session_state:
    st.session_state.historical_run_id = None
if "available_variables" not in st.session_state:
    st.session_state.available_variables = discover_variables()

# Auto-load bundled sample data on first visit.
if "bootstrap_done" not in st.session_state:
    _boot_df, _boot_status = load_data(source="local")
    if not _boot_df.empty:
        st.session_state.data = _boot_df
        st.session_state.status = _boot_status
        st.session_state.alerts = generate_alerts(_boot_df)
        # Refresh variable list from actual data.
        if "variable" in _boot_df.columns:
            st.session_state.available_variables = sorted(_boot_df["variable"].dropna().unique().tolist())
    st.session_state.bootstrap_done = True


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar helpers (shared)
# ═══════════════════════════════════════════════════════════════════════════════


def _render_cloud_api_hint() -> None:
    if SERENE_API_TOKEN:
        return
    st.info(
        "**SERENE API 未配置。** 当前展示的是仓库内的样例数据（可正常演示）。\n\n"
        "若要在云端使用实时 API：Streamlit Cloud → 你的应用 → **Settings → Secrets**，粘贴：\n\n"
        "```toml\n"
        "SERENE_API_BASE_URL = \"https://spaceweather.bham.ac.uk\"\n"
        "SERENE_API_TOKEN = \"你的token\"\n"
        "SERENE_API_TIMEOUT = \"30\"\n"
        "SERENE_AUTH_SCHEME = \"Token\"\n"
        "```\n\n"
        "保存后点击 **Reboot app**。"
    )


def _run_api_connection_test() -> None:
    with st.spinner("Testing connection..."):
        ok, msg = SereneClient().test_connection()
        st.session_state.api_connected = ok
        st.session_state.api_message = msg
    if ok:
        st.sidebar.success(msg)
    else:
        st.sidebar.warning(msg)


def _source_label(status: LoadStatus) -> str:
    mapping = {
        "api": "SERENE API",
        "local": "Local sample file",
        "local_fallback": "Local sample file (API fallback)",
        "none": "No data",
    }
    return mapping.get(status.source, status.source)


def _render_variable_summary_table(df: pd.DataFrame, var_options: list[str]) -> None:
    """Render a summary table with one row per variable (min, max, mean, std, risk)."""
    from hazard_detector import _classify_from_thresholds, _hazard_type_for

    rows: list[dict[str, object]] = []
    for var in var_options:
        var_df = df[df["variable"] == var]
        if var_df.empty:
            continue
        vals = pd.to_numeric(var_df["value"], errors="coerce").dropna()
        if vals.empty:
            continue
        max_val = float(vals.max())
        risk = _classify_from_thresholds(max_val, 0.0, 0.0, var)
        rows.append({
            "Variable": var,
            "Min": round(float(vals.min()), 3),
            "Max": round(float(vals.max()), 3),
            "Mean": round(float(vals.mean()), 3),
            "Std": round(float(vals.std()), 3),
            "Count": len(vals),
            "Risk": risk,
            "Hazard Type": _hazard_type_for(var),
        })

    if not rows:
        st.info("No numeric values to summarize.")
        return

    summary_df = pd.DataFrame(rows)

    def _risk_color(val: object) -> str:
        colors = {"Severe": "#ff4b4b", "Warning": "#ffa726", "Watch": "#ffd54f", "Normal": "#66bb6a"}
        return f"background-color: {colors.get(str(val), 'transparent')}; color: white; font-weight: bold"

    styled = summary_df.style.map(_risk_color, subset=["Risk"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar — Existing dashboard mode
# ═══════════════════════════════════════════════════════════════════════════════


def _render_existing_sidebar() -> dict:
    st.sidebar.markdown("# 🛩️ SERENE AIDA")
    st.sidebar.markdown("*Aviation Space Weather Monitor*")
    st.sidebar.markdown("---")

    params: dict = {"mode": "existing"}

    if st.session_state.config_warnings:
        with st.sidebar.expander("Configuration issues", expanded=True):
            for msg in st.session_state.config_warnings:
                st.warning(msg)

    params["source"] = st.sidebar.selectbox(
        "Data source",
        ["local", "api"],
        format_func=lambda s: "SERENE API" if s == "api" else "Local sample file",
        help="Local file loads instantly. SERENE API calls /api/calc/ per grid point.",
    )

    params["model"] = st.sidebar.selectbox("Model", ["AIDA", "TOMIRIS"])

    now = datetime.now(timezone.utc)
    st.sidebar.markdown("#### Time range")
    params["start_time"] = st.sidebar.text_input(
        "Start datetime (ISO 8601)",
        value=(now - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%S"),
    )
    params["end_time"] = st.sidebar.text_input(
        "End datetime (ISO 8601)",
        value=now.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    avail_vars = st.session_state.available_variables
    selected_vars = st.sidebar.multiselect(
        "Variable selection",
        options=avail_vars,
        default=avail_vars,
    )
    params["variables"] = selected_vars or None

    st.sidebar.markdown("#### Region selection (API mode)")
    with st.sidebar.expander("Bounding box & grid step", expanded=params["source"] == "api"):
        lat_min = st.number_input("Lat min", value=45.0, min_value=-90.0, max_value=90.0)
        lat_max = st.number_input("Lat max", value=60.0, min_value=-90.0, max_value=90.0)
        lon_min = st.number_input("Lon min", value=-15.0, min_value=-180.0, max_value=180.0)
        lon_max = st.number_input("Lon max", value=15.0, min_value=-180.0, max_value=180.0)
        params["grid_step"] = st.slider("Grid step (degrees)", 2.0, 30.0, 5.0, 1.0)
        est_n, _, _ = SereneClient.estimate_grid_points(
            lat_min, lat_max, lon_min, lon_max, params["grid_step"], params["grid_step"],
        )
        st.caption(
            f"≈ {est_n} API call(s) (max {MAX_GRID_POINTS}). "
            "Global region can take many minutes."
        )

    params["region"] = {
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
    }

    params["local_file"] = st.sidebar.text_input(
        "Local fallback file path",
        value=str(resolve_local_file()),
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("Test SERENE API connection", use_container_width=True):
        _run_api_connection_test()

    st.sidebar.markdown("---")
    if st.sidebar.button("Load / Refresh data", type="primary", use_container_width=True):
        _do_load(params)

    st.sidebar.caption(
        "Prototype research system — not for operational aviation decision-making."
    )
    return params


def _do_load(params: dict) -> None:
    progress_bar = st.progress(0.0, text="Preparing…")
    progress_state = {"done": 0, "total": 1}

    def _on_api_progress(done: int, total: int) -> None:
        progress_state["done"] = done
        progress_state["total"] = max(total, 1)
        progress_bar.progress(
            done / progress_state["total"],
            text=f"SERENE API: point {done}/{total}…",
        )

    try:
        df, status = load_data(
            source=params["source"],
            model=params["model"],
            start_time=params.get("start_time"),
            end_time=params.get("end_time"),
            variables=params.get("variables"),
            region=params.get("region"),
            local_file=params.get("local_file"),
            grid_step=params.get("grid_step", 5.0),
            progress_callback=_on_api_progress if params["source"] == "api" else None,
        )
        progress_bar.progress(1.0, text="Generating advisories…")
        st.session_state.data = df
        st.session_state.status = status
        st.session_state.alerts = generate_alerts(df) if not df.empty else pd.DataFrame()
        # Refresh variable list from actual loaded data.
        if not df.empty and "variable" in df.columns:
            st.session_state.available_variables = sorted(df["variable"].dropna().unique().tolist())
    finally:
        progress_bar.empty()


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar — Live fixed map mode
# ═══════════════════════════════════════════════════════════════════════════════


def _render_live_sidebar() -> dict:
    st.sidebar.markdown("# 🛩️ Live Fixed Map")
    st.sidebar.markdown("*Fixed-grid ionospheric monitoring*")
    st.sidebar.markdown("---")

    params: dict = {"mode": "live"}

    params["model"] = st.sidebar.selectbox("Model", ["AIDA", "TOMIRIS"], key="live_model")
    params["variable"] = st.sidebar.selectbox(
        "Variable",
        st.session_state.available_variables,
        key="live_var",
    )
    params["region"] = st.sidebar.selectbox(
        "Region",
        list_regions(),
        index=list_regions().index(DEFAULT_REGION),
        key="live_region",
    )
    params["resolution"] = st.sidebar.slider(
        "Resolution (degrees)",
        0.5, 10.0, MAP_RESOLUTION_DEFAULT, 0.5,
        key="live_res",
    )

    now = datetime.now(timezone.utc)
    params["timestamp"] = st.sidebar.text_input(
        "Timestamp (ISO 8601)",
        value=now.strftime("%Y-%m-%dT%H:%M:%S"),
        key="live_ts",
    )

    params["use_cache"] = st.sidebar.checkbox("Use cache", value=True, key="live_use_cache")
    params["force_refresh"] = st.sidebar.checkbox("Force refresh", value=False, key="live_force")

    st.sidebar.markdown("---")
    if st.sidebar.button("Test SERENE API connection", use_container_width=True, key="live_test_api"):
        _run_api_connection_test()

    if st.sidebar.button("Build fixed map", type="primary", use_container_width=True):
        _build_live_map(params)

    st.sidebar.markdown("---")
    _render_shared_status()
    st.sidebar.caption(
        "Prototype research system — not for operational aviation decision-making."
    )
    return params


def _build_live_map(params: dict) -> None:
    with st.spinner("Building fixed map…"):
        try:
            map_df, msg = build_fixed_map(
                model=params["model"],
                timestamp=params["timestamp"],
                variable=params["variable"],
                region=params["region"],
                resolution=params["resolution"],
                use_cache=params["use_cache"],
                force_refresh=params["force_refresh"],
            )
        except Exception as exc:
            st.error(f"build_fixed_map failed: {exc}")
            return

        st.session_state.live_map_df = map_df
        st.session_state.live_map_status = msg

        if map_df.empty:
            st.session_state.live_hazards = pd.DataFrame()
            st.session_state.live_alerts = pd.DataFrame()
            st.warning(f"Map is empty: {msg}")
            return

        try:
            hazards = detect_hazards_from_map(
                current_map=map_df,
                previous_map=None,
                variable=params["variable"],
            )
        except Exception as exc:
            st.warning(f"Hazard detection failed: {exc}")
            hazards = pd.DataFrame()

        st.session_state.live_hazards = hazards

        if not hazards.empty:
            st.session_state.live_alerts = generate_alerts_from_hazards(hazards)
        else:
            st.session_state.live_alerts = pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar — Historical analysis mode
# ═══════════════════════════════════════════════════════════════════════════════


def _render_historical_sidebar() -> dict:
    st.sidebar.markdown("# 📊 Historical Analysis")
    st.sidebar.markdown("*Time-window hazard survey*")
    st.sidebar.markdown("---")

    params: dict = {"mode": "historical"}

    params["model"] = st.sidebar.selectbox("Model", ["AIDA", "TOMIRIS"], key="hist_model")
    params["variable"] = st.sidebar.selectbox(
        "Variable",
        st.session_state.available_variables,
        key="hist_var",
    )
    params["region"] = st.sidebar.selectbox(
        "Region",
        list_regions(),
        index=list_regions().index(DEFAULT_REGION),
        key="hist_region",
    )
    params["resolution"] = st.sidebar.slider(
        "Resolution (degrees)",
        0.5, 10.0, MAP_RESOLUTION_DEFAULT, 0.5,
        key="hist_res",
    )

    st.sidebar.markdown("#### Time window")
    params["start_time"] = st.sidebar.text_input(
        "Start datetime (ISO 8601)",
        value="2024-05-10T00:00:00",
        key="hist_start",
    )
    params["end_time"] = st.sidebar.text_input(
        "End datetime (ISO 8601)",
        value="2024-05-12T23:00:00",
        key="hist_end",
    )
    params["time_step"] = st.sidebar.number_input(
        "Time step (hours)",
        min_value=1, max_value=24, value=DEFAULT_TIME_STEP_HOURS,
        key="hist_step",
    )

    params["use_cache"] = st.sidebar.checkbox("Use cache", value=True, key="hist_use_cache")
    params["force_refresh"] = st.sidebar.checkbox("Force refresh", value=False, key="hist_force")

    # ── Load previous run ──────────────────────────────────────────────────
    saved_runs = list_historical_runs()
    if saved_runs:
        run_options = ["(none)"] + [
            "{} -- {} to {}".format(
                r.get("run_id", "?"),
                r.get("start_time", "")[:16],
                r.get("end_time", "")[:16],
            )
            for r in saved_runs
        ]
        selected = st.sidebar.selectbox(
            "Load saved run",
            run_options,
            key="hist_load_select",
            help="Select a previously saved run to view results without re-running.",
        )
        if st.sidebar.button("Load selected run", use_container_width=True):
            if selected != "(none)":
                run_id = saved_runs[run_options.index(selected) - 1]["run_id"]
                h_df, a_df, r_sum = load_historical_run(run_id)
                st.session_state.historical_hazards = h_df
                st.session_state.historical_alerts = a_df
                st.session_state.historical_summary = r_sum
                st.session_state.historical_maps_meta = []
                st.session_state.historical_run_id = run_id
                st.sidebar.success("Loaded: " + run_id)

    st.sidebar.markdown("---")

    # Case study quick button
    if st.sidebar.button("⚡ Load May 2024 storm case study", use_container_width=True):
        st.session_state.hist_case_study = True

    if st.sidebar.button("Run historical analysis", type="primary", use_container_width=True):
        _run_historical(params)
    elif st.session_state.get("hist_case_study"):
        _run_historical(params)
        st.session_state.hist_case_study = False

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Historical analysis uses cached data when available. "
        "SERENE API currently returns real-time data only."
    )
    return params


def _run_historical(params: dict) -> None:
    with st.spinner("Running historical analysis…"):
        try:
            maps_meta, hazards_df, alerts_df, summary = run_historical_analysis(
                model=params["model"],
                variable=params["variable"],
                start_time=params["start_time"],
                end_time=params["end_time"],
                time_step_hours=params["time_step"],
                region=params["region"],
                resolution=params["resolution"],
                use_cache=params["use_cache"],
                force_refresh=params["force_refresh"],
            )
        except Exception as exc:
            st.error(f"Historical analysis failed: {exc}")
            return

    st.session_state.historical_maps_meta = maps_meta
    st.session_state.historical_hazards = hazards_df
    st.session_state.historical_alerts = alerts_df
    st.session_state.historical_summary = summary

    # Persist to disk so results survive page refresh / restart.
    if summary is not None and summary.map_count > 0:
        try:
            run_id = save_historical_run(
                hazards_df=hazards_df,
                alerts_df=alerts_df,
                summary=summary,
                model=params["model"],
                variable=params["variable"],
                region=params["region"],
            )
            if run_id:
                st.session_state.historical_run_id = run_id
                st.sidebar.success(f"Saved: {run_id}")
        except Exception as exc:
            st.sidebar.warning(f"Could not save results: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# Shared sidebar utilities
# ═══════════════════════════════════════════════════════════════════════════════


def _render_shared_status() -> None:
    """Render compact API / connection status block."""
    if st.session_state.api_connected is True:
        st.sidebar.success(f"API: {st.session_state.api_message}")
    elif st.session_state.api_connected is False:
        st.sidebar.warning(f"API: {st.session_state.api_message}")
    else:
        st.sidebar.info("API: not tested.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main page — Existing dashboard
# ═══════════════════════════════════════════════════════════════════════════════


def _render_connection_panel() -> None:
    st.subheader("SERENE API & data status")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.session_state.api_connected is True:
            st.success(f"API: {st.session_state.api_message}")
        elif st.session_state.api_connected is False:
            st.warning(f"API: {st.session_state.api_message}")
        else:
            st.info("API: not tested — use sidebar **Test SERENE API connection**.")

    status: LoadStatus = st.session_state.status
    with c2:
        st.metric("Current data source", _source_label(status))

    with c3:
        st.metric("Rows loaded", f"{len(st.session_state.data):,}")

    if status.message:
        if status.source == "local_fallback":
            st.warning(status.message)
        elif status.ok:
            st.info(status.message)
        else:
            st.error(status.message)

    for warn in status.warnings:
        st.warning(warn)


def _render_existing_main(params: dict) -> None:
    st.title("Aviation Space Weather Dashboard")
    st.caption(
        "ICAO-style prototype risk monitor — SERENE real-time data & AIDA/TOMIRIS models"
    )

    _render_cloud_api_hint()
    _render_connection_panel()
    st.markdown("---")

    if st.session_state.data.empty:
        st.info(
            "Select a data source in the sidebar and click **Load / Refresh data** "
            "to begin. If SERENE API fails, the dashboard automatically uses the "
            "local fallback file without crashing."
        )
        with st.expander("Quick start"):
            st.markdown(
                """
                1. Copy `.env.example` to `.env` and set `SERENE_API_BASE_URL` and
                   `SERENE_API_TOKEN` (auth uses official `Token` scheme by default).
                2. Click **Test SERENE API connection**, then **Load / Refresh data**.
                4. Advisories shown here are **prototype advisories**, not official ICAO warnings.
                """
            )
        return

    df = st.session_state.data
    alerts = st.session_state.alerts

    # ── ICAO-style risk alert panel ─────────────────────────────────────────
    st.subheader("ICAO-style prototype risk advisories")
    overall, summary = generate_overall_risk(alerts)
    emoji = {"Normal": "🟢", "Watch": "🟡", "Warning": "🟠", "Severe": "🔴"}
    st.markdown(f"**Overall risk:** {emoji.get(overall, '⚪')} {overall}")
    st.caption(summary)
    st.caption(DISCLAIMER)

    if alerts.empty:
        st.success("No active prototype advisories — parameters within normal range.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                create_alert_summary(alerts),
                use_container_width=True,
                key="alert_summary_chart",
            )
        with col_b:
            st.plotly_chart(
                create_alert_timeline(alerts),
                use_container_width=True,
                key="alert_timeline_chart",
            )
        show_cols = [
            c for c in (
                "timestamp", "region", "alert_type", "risk_level",
                "reason", "possible_aviation_impact", "interpretation",
            )
            if c in alerts.columns
        ]
        st.dataframe(alerts[show_cols], use_container_width=True, height=220)

    st.markdown("---")

    # ── All Variables Overview ───────────────────────────────────────────────
    var_options = sorted(df["variable"].dropna().unique()) if "variable" in df.columns else []

    st.subheader("All Variables Overview")
    st.caption(f"{len(var_options)} variable(s) discovered from data source.")

    if var_options:
        # Summary table: one row per variable
        _render_variable_summary_table(df, var_options)

        # Multi-variable time series (all variables in subplots)
        st.subheader("Time series — all variables")
        st.plotly_chart(
            create_time_series_plot(df),
            use_container_width=True,
            key="overview_all_ts",
        )

        # Mini-map grid: 3 columns, one map per variable
        st.subheader("Maps — all variables")
        cols_per_row = 3
        for row_start in range(0, len(var_options), cols_per_row):
            row_vars = var_options[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for i, var in enumerate(row_vars):
                with cols[i]:
                    st.caption(f"**{var}**")
                    st.plotly_chart(
                        create_map_plot(df, variable=var, title=var),
                        use_container_width=True,
                        key=f"overview_map_{var}",
                    )

    st.markdown("---")

    # ── Per-variable exploration ────────────────────────────────────────────
    st.subheader("Data preview")
    st.dataframe(df.head(100), use_container_width=True)

    selected_var = st.selectbox("Variable for plots", var_options or [None])

    col_ts, col_map = st.columns(2)
    with col_ts:
        st.subheader("Time series")
        st.plotly_chart(
            create_time_series_plot(df, variable=selected_var),
            use_container_width=True,
            key="overview_time_series",
        )
    with col_map:
        st.subheader("Map / scatter (lat/lon)")
        st.plotly_chart(
            create_map_plot(df, variable=selected_var),
            use_container_width=True,
            key="overview_map",
        )

    with st.expander("Detailed tabs (GNSS / HF / General / Raw data)"):
        tab_gnss, tab_hf, tab_gen, tab_raw = st.tabs(
            ["GNSS", "HF Communication", "General", "Raw data"]
        )

        with tab_gnss:
            gnss_vars = [v for v in var_options if "tec" in v.lower()]
            for i, var in enumerate(gnss_vars or ["TEC"]):
                if var not in df["variable"].values:
                    continue
                st.plotly_chart(
                    create_map_plot(df, variable=var, title=f"GNSS — {var}"),
                    use_container_width=True,
                    key=f"gnss_map_{var}_{i}",
                )

        with tab_hf:
            hf_vars = [v for v in var_options if "muf" in v.lower() or "fof2" in v.lower()]
            for i, var in enumerate(hf_vars or ["MUF3000"]):
                if var not in df["variable"].values:
                    continue
                st.plotly_chart(
                    create_map_plot(df, variable=var, title=f"HF — {var}"),
                    use_container_width=True,
                    key=f"hf_map_{var}_{i}",
                )

        with tab_gen:
            gen_vars = [v for v in var_options
                        if "hmf2" in v.lower() or "nmf2" in v.lower()]
            if not gen_vars:
                st.info("No general ionospheric monitoring variables (hmF2, NmF2) in current data.")
            for i, var in enumerate(gen_vars):
                st.plotly_chart(
                    create_map_plot(df, variable=var, title=f"General — {var}"),
                    use_container_width=True,
                    key=f"gen_map_{var}_{i}",
                )

        with tab_raw:
            st.json({
                "source": st.session_state.status.source,
                "message": st.session_state.status.message,
                "warnings": st.session_state.status.warnings,
                "metadata": st.session_state.status.metadata,
            })
            st.download_button(
                "Download CSV",
                data=df.to_csv(index=False),
                file_name=f"space_weather_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Main page — Live fixed map mode
# ═══════════════════════════════════════════════════════════════════════════════


def _render_live_main(params: dict) -> None:
    st.title("Live Fixed Map Monitor")
    st.caption(
        "Fixed-grid ionospheric monitoring with hazard detection "
        "and ICAO-style prototype advisories."
    )

    _render_cloud_api_hint()

    # Status row
    c1, c2, c3 = st.columns(3)
    map_df = st.session_state.live_map_df
    with c1:
        if map_df.empty:
            st.info("No fixed map built yet. Configure sidebar → **Build fixed map**.")
        else:
            st.success(f"Map loaded: {len(map_df)} rows")
    with c2:
        st.metric("Region", params.get("region", "—"))
    with c3:
        st.metric("Status", st.session_state.live_map_status or "—")

    st.markdown("---")

    if map_df.empty:
        st.info("Use the sidebar to configure and build a fixed map.")
        return

    # Fixed map plot
    st.subheader(f"Fixed grid map — {params.get('variable', '')}")
    st.plotly_chart(
        create_fixed_map_plot(map_df, variable=params.get("variable")),
        use_container_width=True,
        key="live_fixed_map",
    )

    st.markdown("---")

    # Hazard detection results
    hazards = st.session_state.live_hazards
    st.subheader("Hazard detection")
    if hazards.empty:
        st.success("No hazards detected — all parameters within normal range.")
    else:
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.plotly_chart(
                create_hazard_map_plot(hazards),
                use_container_width=True,
                key="live_hazard_map",
            )
        with col_h2:
            st.dataframe(
                hazards[[
                    c for c in ("variable", "hazard_type", "risk_level", "max_value",
                                "max_gradient", "max_change_rate", "reason")
                    if c in hazards.columns
                ]],
                use_container_width=True,
                height=300,
            )

    st.markdown("---")

    # ICAO-style prototype advisories
    alerts = st.session_state.live_alerts
    st.subheader("ICAO-style prototype advisories")
    st.caption(DISCLAIMER)
    if alerts.empty:
        st.success("No active prototype advisories.")
    else:
        overall, summary = generate_overall_risk(alerts)
        emoji = {"Normal": "🟢", "Watch": "🟡", "Warning": "🟠", "Severe": "🔴"}
        st.markdown(f"**Overall risk:** {emoji.get(overall, '⚪')} {overall}")
        st.caption(summary)

        show_cols = [
            c for c in (
                "timestamp", "region", "alert_type", "risk_level",
                "reason", "possible_aviation_impact", "interpretation",
            )
            if c in alerts.columns
        ]
        st.dataframe(alerts[show_cols], use_container_width=True, height=220)


# ═══════════════════════════════════════════════════════════════════════════════
# Main page — Historical analysis mode
# ═══════════════════════════════════════════════════════════════════════════════


def _render_historical_main(params: dict) -> None:
    st.title("Historical Analysis")
    st.caption(
        "Time-window hazard survey — ICAO-style prototype advisories "
        "for academic demonstration only."
    )

    _render_cloud_api_hint()

    # Show loaded run info
    if st.session_state.get("historical_run_id"):
        st.info("Loaded run: " + st.session_state.historical_run_id)

    # Apply case study preset if triggered.
    if st.session_state.get("hist_case_study"):
        st.info(
            "**May 2024 storm case study loaded.** "
            "2024-05-10 00:00 → 2024-05-12 23:00 UTC, TEC, 1h steps. "
            "Click **Run historical analysis** to start."
        )

    summary = st.session_state.historical_summary

    if summary is None:
        st.info(
            "Configure the time window in the sidebar and click "
            "**Run historical analysis** to begin. "
            "Use the ⚡ **Load May 2024 storm case study** button for a demo preset."
        )
        return

    # Run summary
    st.subheader("Run summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Maps built", summary.map_count)
    with c2:
        st.metric("Alerts generated", summary.alert_count)
    with c3:
        st.metric("Cache hits", summary.cache_hits)
    with c4:
        st.metric("Failures", summary.failures)
    with c5:
        st.metric("Time steps", f"{summary.time_step_hours}h")

    if summary.messages:
        with st.expander(f"Details ({len(summary.messages)} message(s))"):
            for m in summary.messages:
                st.caption(m)

    st.markdown("---")

    alerts_df = st.session_state.historical_alerts
    hazards_df = st.session_state.historical_hazards

    # Alert timeline
    st.subheader("Alert timeline")
    st.plotly_chart(
        create_alert_timeline(alerts_df) if not alerts_df.empty else _empty_plot("No alerts in this run."),
        use_container_width=True,
        key="hist_timeline",
    )

    # Historical summary
    st.subheader("Historical summary")
    if alerts_df.empty:
        st.success("No prototype advisories generated — all timestamps within normal range.")
    else:
        st.plotly_chart(
            create_historical_summary_plot(alerts_df),
            use_container_width=True,
            key="hist_summary_plot",
        )

        overall, overall_msg = generate_overall_risk(alerts_df)
        emoji = {"Normal": "🟢", "Watch": "🟡", "Warning": "🟠", "Severe": "🔴"}
        st.markdown(f"**Peak risk across window:** {emoji.get(overall, '⚪')} {overall}")
        st.caption(overall_msg)

    st.markdown("---")

    # Hazards table
    st.subheader("Hazard records")
    if hazards_df.empty:
        st.info("No hazard records.")
    else:
        show_cols = [
            c for c in (
                "timestamp", "region", "variable", "hazard_type", "risk_level",
                "max_value", "max_gradient", "max_change_rate", "reason",
            )
            if c in hazards_df.columns
        ]
        st.dataframe(hazards_df[show_cols], use_container_width=True, height=250)

    # Alerts table
    st.subheader("ICAO-style prototype advisories")
    st.caption(DISCLAIMER)
    if alerts_df.empty:
        st.success("No advisories generated.")
    else:
        show_cols = [
            c for c in (
                "timestamp", "region", "alert_type", "risk_level",
                "reason", "possible_aviation_impact", "interpretation",
            )
            if c in alerts_df.columns
        ]
        st.dataframe(alerts_df[show_cols], use_container_width=True, height=250)


def _empty_plot(message: str) -> object:
    """Return an empty Plotly figure with a message (lazy import)."""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level sidebar & main dispatcher
# ═══════════════════════════════════════════════════════════════════════════════


def _render_sidebar() -> dict:
    mode = st.sidebar.selectbox(
        "Mode",
        ["Existing dashboard", "Live fixed map", "Historical analysis"],
        help="Switch between operational modes.",
    )

    if mode == "Existing dashboard":
        return _render_existing_sidebar()
    elif mode == "Live fixed map":
        return _render_live_sidebar()
    else:
        return _render_historical_sidebar()


def _render_main(params: dict) -> None:
    mode = params.get("mode", "existing")
    if mode == "existing":
        _render_existing_main(params)
    elif mode == "live":
        _render_live_main(params)
    else:
        _render_historical_main(params)


def main() -> None:
    params = _render_sidebar()
    _render_main(params)


if __name__ == "__main__":
    main()
