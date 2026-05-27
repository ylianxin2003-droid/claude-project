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
    DEFAULT_TIME_STEP_HOURS,
    SERENE_API_TOKEN,
    reload_config,
    validate_config,
)
from data_loader import LoadStatus, discover_variables, load_data
from variable_registry import get_available_variables
from grid_generator import generate_region_grid, list_regions
from hazard_detector import detect_hazards_from_map
from historical_runner import RunSummary, list_historical_runs, load_historical_run, run_historical_analysis, save_historical_run
from map_builder import build_fixed_map
from map_cache import cache_exists, count_cached_maps, list_cached_maps
from serene_client import MAX_GRID_POINTS, SereneClient
from visualisation import (
    create_advisory_card_html,
    create_alert_summary,
    create_alert_timeline,
    create_fixed_map_plot,
    create_hazard_map_plot,
    create_hazard_summary_map,
    create_historical_summary_plot,
    create_map_plot,
    create_multi_variable_time_series,
    create_risk_timeline,
    create_spatial_gradient_map,
    create_temporal_change_map,
    create_time_series_plot,
    create_variable_card_data,
    create_variable_map_grid,
    create_variable_summary_table,
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

MAX_HISTORICAL_STEPS = 30
MAX_HISTORICAL_API_REQUESTS = 300
MAX_HISTORICAL_CACHE_CHECKS = 300
NO_HISTORICAL_CACHE_MESSAGE = (
    "No cached maps found for this historical window.\n\n"
    "Historical replay is cache-only by default.\n\n"
    "Run Live Fixed Map first to create matching cache, or enable "
    "\"Allow live API requests for missing historical cache\".\n\n"
    "Note: current SERENE /api/calc/ does not provide confirmed retrospective retrieval."
)

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
if "historical_running" not in st.session_state:
    st.session_state.historical_running = False
if "historical_debug_plan" not in st.session_state:
    st.session_state.historical_debug_plan = {}
if "hist_case_study" not in st.session_state:
    st.session_state.hist_case_study = False
if "hist_case_study_loaded" not in st.session_state:
    st.session_state.hist_case_study_loaded = False
if "available_variables" not in st.session_state:
    st.session_state.available_variables = discover_variables()
if "available_variable_metadata" not in st.session_state:
    st.session_state.available_variable_metadata = {}
if "show_all_variables" not in st.session_state:
    st.session_state.show_all_variables = True

# No local bootstrap — this project uses SERENE API only.
if "bootstrap_done" not in st.session_state:
    st.session_state.bootstrap_done = True


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar helpers (shared)
# ═══════════════════════════════════════════════════════════════════════════════


def _render_cloud_api_hint() -> None:
    if SERENE_API_TOKEN:
        return
    st.warning(
        "**SERENE API token is not configured.** "
        "This dashboard requires SERENE API access or existing cache. "
        "Local sample fallback is disabled.\n\n"
        "To configure: Streamlit Cloud → your app → **Settings → Secrets**, paste:\n\n"
        "```toml\n"
        "SERENE_API_BASE_URL = \"https://spaceweather.bham.ac.uk\"\n"
        "SERENE_API_TOKEN = \"your-token\"\n"
        "SERENE_API_TIMEOUT = \"30\"\n"
        "SERENE_AUTH_SCHEME = \"Token\"\n"
        "```\n\n"
        "Save and click **Reboot app**."
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



def _render_variable_summary_table(df: pd.DataFrame, var_options: list[str]) -> None:
    """Render a summary table with one row per variable (min, max, mean, std, unit, risk)."""
    from hazard_detector import _classify_from_thresholds, _hazard_type_for

    meta = st.session_state.get("available_variable_metadata", {})

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
        var_meta = meta.get(var, {})
        rows.append({
            "Variable": var,
            "Unit": var_meta.get("unit", ""),
            "Description": var_meta.get("description", ""),
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

    params: dict = {"mode": "existing", "source": "api"}

    if st.session_state.config_warnings:
        with st.sidebar.expander("Configuration issues", expanded=True):
            for msg in st.session_state.config_warnings:
                st.warning(msg)

    st.sidebar.caption("Data source: SERENE API only")

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

    st.sidebar.markdown("#### Variable selection")
    show_all = st.sidebar.checkbox(
        "Show all variables",
        value=st.session_state.show_all_variables,
        help="Load and display all available variables from the data source.",
    )
    st.session_state.show_all_variables = show_all

    avail_vars = st.session_state.available_variables
    if show_all:
        params["variables"] = None  # signal: load all
        st.sidebar.caption(f"Loading all {len(avail_vars)} variable(s).")
    else:
        selected_vars = st.sidebar.multiselect(
            "Select variables",
            options=avail_vars,
            default=avail_vars[:1],
        )
        params["variables"] = selected_vars or None

    st.sidebar.markdown("#### Region selection")
    with st.sidebar.expander("Bounding box & grid step", expanded=True):
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
            source="api",
            model=params["model"],
            start_time=params.get("start_time"),
            end_time=params.get("end_time"),
            variables=params.get("variables"),
            region=params.get("region"),
            grid_step=params.get("grid_step", 5.0),
            progress_callback=_on_api_progress,
        )
        progress_bar.progress(1.0, text="Generating advisories…")
        st.session_state.data = df
        st.session_state.status = status
        st.session_state.alerts = generate_alerts(df) if not df.empty else pd.DataFrame()
        # Refresh variable list and metadata from actual loaded data.
        if not df.empty and "variable" in df.columns:
            _vars, _meta = get_available_variables(source="auto", df=df)
            st.session_state.available_variables = _vars
            st.session_state.available_variable_metadata = _meta
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

    # Variable selection — support show all
    show_all_live = st.sidebar.checkbox(
        "Show all variables",
        value=False,
        key="live_show_all",
        help="Build fixed maps for all available variables.",
    )
    st.session_state.show_all_variables = show_all_live
    if show_all_live:
        params["variables"] = None  # build all
        st.sidebar.caption(f"Will build {len(st.session_state.available_variables)} variable(s).")
    else:
        params["variables"] = [
            st.sidebar.selectbox(
                "Variable",
                st.session_state.available_variables,
                key="live_var",
            )
        ]

    _live_regions = list_regions()
    params["region"] = st.sidebar.selectbox(
        "Region",
        _live_regions,
        index=_live_regions.index("uk") if "uk" in _live_regions else 0,
        key="live_region",
    )
    params["resolution"] = st.sidebar.slider(
        "Resolution (degrees)",
        0.5, 10.0, 10.0, 0.5,
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
    variables = params.get("variables")
    if variables is None:
        variables = st.session_state.available_variables
    if isinstance(variables, str):
        variables = [variables]

    # ── Safety checks ────────────────────────────────────────────────────
    from grid_generator import generate_region_grid
    grid_df = generate_region_grid(params["region"], resolution=params["resolution"])
    est_points = len(grid_df)
    use_cache = params.get("use_cache", True)
    force_refresh = params.get("force_refresh", False)
    is_all_vars = params.get("variables") is None

    # Option A: block all-variables API fetch unless cached.
    if is_all_vars and (force_refresh or not use_cache):
        st.error(
            "**All variables live API fetch is disabled** to prevent excessive "
            "API calls. Please select one variable or use cached maps."
        )
        st.session_state.live_map_df = pd.DataFrame()
        st.session_state.live_map_status = "Blocked: all-variables API fetch disabled."
        st.session_state.live_hazards = pd.DataFrame()
        st.session_state.live_alerts = pd.DataFrame()
        return

    if est_points > 500 and (force_refresh or not use_cache):
        st.error(
            "**Too many API calls for live fixed map.** "
            "Please select a smaller region, use coarser resolution, or use cached data."
        )
        st.session_state.live_map_df = pd.DataFrame()
        st.session_state.live_map_status = f"Blocked: {est_points} points exceeds limit."
        st.session_state.live_hazards = pd.DataFrame()
        st.session_state.live_alerts = pd.DataFrame()
        return

    all_frames: list[pd.DataFrame] = []
    messages: list[str] = []

    with st.spinner(f"Building fixed map(s) for {len(variables)} variable(s)…"):
        for var in variables:
            try:
                map_df, msg = build_fixed_map(
                    model=params["model"],
                    timestamp=params["timestamp"],
                    variable=var,
                    region=params["region"],
                    resolution=params["resolution"],
                    use_cache=params["use_cache"],
                    force_refresh=params["force_refresh"],
                )
            except Exception as exc:
                st.warning(f"build_fixed_map failed for {var}: {exc}")
                continue
            if not map_df.empty:
                all_frames.append(map_df)
            messages.append(f"{var}: {msg}")

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
    else:
        combined = pd.DataFrame()

    st.session_state.live_map_df = combined
    st.session_state.live_map_status = "; ".join(messages) if messages else "No maps built."

    if combined.empty:
        st.session_state.live_hazards = pd.DataFrame()
        st.session_state.live_alerts = pd.DataFrame()
        st.warning("All maps are empty. Check API connectivity or cache.")
        return

    try:
        hazards = detect_hazards_from_map(
            current_map=combined,
            previous_map=None,
            variable=None,  # all variables
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


def _clear_historical_results(summary: RunSummary | None = None) -> None:
    st.session_state.historical_maps_meta = []
    st.session_state.historical_hazards = pd.DataFrame()
    st.session_state.historical_alerts = pd.DataFrame()
    st.session_state.historical_summary = summary
    st.session_state.historical_run_id = None


def _normalise_historical_variables(params: dict) -> list[str]:
    variables = params.get("variables")
    if variables is None:
        variables = st.session_state.available_variables
    if isinstance(variables, str):
        variables = [variables]
    return [str(v) for v in variables if str(v)]


def _apply_historical_case_study_defaults() -> None:
    variables = st.session_state.available_variables
    default_variable = "TEC" if "TEC" in variables else (variables[0] if variables else "")
    st.session_state.hist_show_all = False
    if default_variable:
        st.session_state.hist_var = default_variable
    st.session_state.hist_region = "uk"
    st.session_state.hist_res = 10.0
    st.session_state.hist_start = "2024-05-10T00:00:00"
    st.session_state.hist_end = "2024-05-12T23:00:00"
    st.session_state.hist_step = 12
    st.session_state.hist_use_cache = True
    st.session_state.hist_force = False
    st.session_state.hist_allow_api = False


def _format_history_timestamp(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")


def _build_historical_execution_plan(params: dict) -> dict:
    variables = _normalise_historical_variables(params)
    use_cache = bool(params.get("use_cache", True))
    force_refresh = bool(params.get("force_refresh", False))
    allow_live_api = bool(params.get("allow_live_api", False))
    time_step_hours = max(1, int(params.get("time_step", DEFAULT_TIME_STEP_HOURS)))

    plan: dict = {
        "variables": variables,
        "start_time": params.get("start_time", ""),
        "end_time": params.get("end_time", ""),
        "time_step_hours": time_step_hours,
        "number_of_timestamps": 0,
        "region": params.get("region", "uk"),
        "resolution": params.get("resolution", 10.0),
        "use_cache": use_cache,
        "force_refresh": force_refresh,
        "allow_live_api": allow_live_api,
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_check_skipped": False,
        "total_steps": 0,
        "estimated_grid_points": 0,
        "estimated_api_requests": 0,
        "execution_mode": "live API allowed" if allow_live_api else "cache-only replay",
        "timestamps": [],
        "cached_keys": [],
        "missing_keys": [],
        "error": "",
    }

    if not variables:
        plan["error"] = "No variables selected for historical analysis."
        return plan

    try:
        start_ts = pd.to_datetime(params["start_time"])
        end_ts = pd.to_datetime(params["end_time"])
    except Exception as exc:
        plan["error"] = f"Invalid historical time window: {exc}"
        return plan

    if pd.isna(start_ts) or pd.isna(end_ts):
        plan["error"] = "Invalid historical time window."
        return plan
    if start_ts > end_ts:
        plan["error"] = "Start datetime must be before or equal to end datetime."
        return plan

    try:
        span_hours = (end_ts - start_ts).total_seconds() / 3600
    except Exception as exc:
        plan["error"] = f"Invalid historical time window: {exc}"
        return plan

    timestamp_count = int(span_hours // time_step_hours) + 1
    total_steps = timestamp_count * len(variables)
    plan["number_of_timestamps"] = timestamp_count
    plan["total_steps"] = total_steps

    try:
        grid_df = generate_region_grid(params["region"], resolution=params["resolution"])
        plan["estimated_grid_points"] = len(grid_df)
    except Exception as exc:
        plan["error"] = f"Could not estimate grid points: {exc}"
        return plan

    plan["estimated_api_requests"] = plan["estimated_grid_points"] * total_steps

    if total_steps > MAX_HISTORICAL_CACHE_CHECKS:
        plan["cache_check_skipped"] = True
        return plan

    timestamps = [
        _format_history_timestamp(ts)
        for ts in pd.date_range(start_ts, end_ts, freq=f"{time_step_hours}h")
    ]
    plan["timestamps"] = timestamps

    for variable in variables:
        for ts_str in timestamps:
            key = {
                "model": params["model"],
                "variable": variable,
                "timestamp": ts_str,
                "resolution": params["resolution"],
                "region": params["region"],
            }
            if cache_exists(params["model"], variable, ts_str, params["resolution"], params["region"]):
                plan["cached_keys"].append(key)
            else:
                plan["missing_keys"].append(key)

    plan["cache_hits"] = len(plan["cached_keys"])
    plan["cache_misses"] = len(plan["missing_keys"])
    return plan


def _render_historical_debug_plan(params: dict) -> None:
    plan = _build_historical_execution_plan(params)
    st.session_state.historical_debug_plan = plan

    with st.expander("Debug / execution plan"):
        rows = {
            "variables": ", ".join(plan["variables"]),
            "start_time": plan["start_time"],
            "end_time": plan["end_time"],
            "time_step_hours": plan["time_step_hours"],
            "number_of_timestamps": plan["number_of_timestamps"],
            "region": plan["region"],
            "resolution": plan["resolution"],
            "use_cache": plan["use_cache"],
            "force_refresh": plan["force_refresh"],
            "allow_live_api": plan["allow_live_api"],
            "cache_hits_found": plan["cache_hits"],
            "cache_misses": plan["cache_misses"],
            "total_steps": plan["total_steps"],
            "estimated_grid_points": plan["estimated_grid_points"],
            "estimated_api_requests": plan["estimated_api_requests"],
            "execution_mode": plan["execution_mode"],
        }
        if plan["cache_check_skipped"]:
            rows["cache_check"] = f"skipped above {MAX_HISTORICAL_CACHE_CHECKS} step(s)"
        if plan["error"]:
            rows["error"] = plan["error"]

        debug_df = pd.DataFrame(
            [{"setting": key, "value": str(value)} for key, value in rows.items()]
        )
        st.dataframe(debug_df, hide_index=True, use_container_width=True)


def _render_historical_sidebar() -> dict:
    if st.session_state.get("hist_case_study"):
        _apply_historical_case_study_defaults()
        st.session_state.hist_case_study = False
        st.session_state.hist_case_study_loaded = True

    st.sidebar.markdown("# 📊 Historical Analysis")
    st.sidebar.markdown("*Time-window hazard survey*")
    st.sidebar.markdown("---")

    params: dict = {"mode": "historical"}

    params["model"] = st.sidebar.selectbox("Model", ["AIDA", "TOMIRIS"], key="hist_model")

    show_all_hist = st.sidebar.checkbox(
        "Show all variables",
        value=False,
        key="hist_show_all",
        help="Run analysis on all available variables. ⚠️ This can be very slow without cache.",
    )
    if show_all_hist:
        st.sidebar.warning(
            "⚠️ Running historical analysis on ALL variables can take a long time. "
            "Consider selecting 1–2 variables or using cached data."
        )
        params["variables"] = None
    else:
        params["variables"] = [
            st.sidebar.selectbox(
                "Variable",
                st.session_state.available_variables,
                key="hist_var",
            )
        ]

    params["region"] = st.sidebar.selectbox(
        "Region",
        list_regions(),
        index=list_regions().index("uk"),
        key="hist_region",
    )
    params["resolution"] = st.sidebar.slider(
        "Resolution (degrees)",
        0.5, 10.0, 10.0, 0.5,
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

    # ── Cache-only / live API toggle ──────────────────────────────────────
    params["allow_live_api"] = st.sidebar.checkbox(
        "Allow live API requests for missing historical cache",
        value=False,
        key="hist_allow_api",
        help=(
            "When disabled (default), historical analysis only reads precomputed cache. "
            "When enabled, live SERENE /api/calc/ calls are made for timestamps without cache."
        ),
    )
    if params["allow_live_api"]:
        st.sidebar.warning(
            "⚠️ **Current SERENE /api/calc/ does not accept historical time parameters.** "
            "API requests will return current point calculations, not confirmed retrospective data."
        )

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
        st.rerun()

    if st.sidebar.button("Run historical analysis", type="primary", use_container_width=True):
        _run_historical(params)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Historical analysis is an API/cache replay framework. "
        "The current SERENE /api/calc/ endpoint is point-based and does not provide "
        "confirmed retrospective retrieval unless a historical endpoint or archived data is available."
    )
    return params


def _run_historical(params: dict) -> None:
    if st.session_state.historical_running:
        st.warning("Historical analysis is already running.")
        return

    st.session_state.historical_running = True
    st.session_state.hist_case_study = False
    st.session_state.hist_case_study_loaded = False
    progress_bar = None

    try:
        variables = _normalise_historical_variables(params)
        use_cache = bool(params.get("use_cache", True))
        force_refresh = bool(params.get("force_refresh", False))
        allow_live_api = bool(params.get("allow_live_api", False))
        plan = _build_historical_execution_plan(params)
        st.session_state.historical_debug_plan = plan

        if plan["error"]:
            st.error(plan["error"])
            _clear_historical_results()
            return

        total_steps = int(plan["total_steps"])
        if total_steps <= 0:
            st.error("No historical timestamps selected.")
            _clear_historical_results()
            return

        if not allow_live_api:
            if not use_cache or force_refresh:
                st.error(
                    "Cache-only historical replay requires Use cache = True and Force refresh = False.\n\n"
                    "Enable live API requests if you intentionally want to call SERENE API."
                )
                _clear_historical_results()
                return

            if total_steps > MAX_HISTORICAL_STEPS:
                st.error(
                    f"Historical cache-only replay is limited to {MAX_HISTORICAL_STEPS} step(s). "
                    f"This run has {total_steps} step(s). Shorten the time range, select fewer "
                    "variables, or increase the time step."
                )
                _clear_historical_results()
                return

            if plan["cache_hits"] == 0:
                st.error(NO_HISTORICAL_CACHE_MESSAGE)
                _clear_historical_results()
                return

            if plan["cache_misses"] > 0:
                st.info(
                    f"{plan['cache_hits']}/{total_steps} historical step(s) have cache. "
                    f"{plan['cache_misses']} missing step(s) will be skipped without API calls."
                )

        if allow_live_api:
            if not SERENE_API_TOKEN:
                st.error(
                    "**SERENE API token is not configured.** "
                    "Cannot make live API requests without a token. "
                    "Disable live API to use cache-only mode."
                )
                _clear_historical_results()
                return

            if plan["estimated_api_requests"] > MAX_HISTORICAL_API_REQUESTS:
                st.error(
                    f"Estimated API requests = {plan['estimated_api_requests']}. "
                    f"Historical live API mode is limited to {MAX_HISTORICAL_API_REQUESTS} "
                    "point request(s). Use cached maps, a larger time step, smaller region, "
                    "or coarser resolution."
                )
                _clear_historical_results()
                return

        progress_bar = st.progress(
            0.0,
            text=f"Starting historical analysis ({total_steps} step(s))...",
        )

        all_hazards: list[pd.DataFrame] = []
        all_alerts: list[pd.DataFrame] = []
        all_maps_meta: list[dict] = []
        total_maps = 0
        total_cache = 0
        total_fail = 0
        total_api = 0
        all_messages: list[str] = []
        time_steps = max(1, int(plan["number_of_timestamps"]))

        for var_index, var in enumerate(variables):
            offset = var_index * time_steps

            def _on_step(done: int, total: int, status: str, offset: int = offset) -> None:
                overall_done = min(offset + done, total_steps)
                progress_bar.progress(
                    overall_done / max(total_steps, 1),
                    text=f"{status} ({overall_done}/{total_steps})",
                )

            try:
                maps_meta, hazards_df, alerts_df, var_summary = run_historical_analysis(
                    model=params["model"],
                    variable=var,
                    start_time=params["start_time"],
                    end_time=params["end_time"],
                    time_step_hours=params["time_step"],
                    region=params["region"],
                    resolution=params["resolution"],
                    use_cache=use_cache,
                    force_refresh=force_refresh,
                    allow_api=allow_live_api,
                    progress_callback=_on_step,
                )
            except Exception as exc:
                message = f"Historical analysis failed for {var}: {exc}"
                st.error(message)
                total_fail += 1
                all_messages.append(message)
                continue

            all_hazards.append(hazards_df)
            all_alerts.append(alerts_df)
            all_maps_meta.extend(maps_meta)
            total_maps += var_summary.map_count
            total_cache += var_summary.cache_hits
            total_fail += var_summary.failures
            total_api += var_summary.api_calls_estimated
            all_messages.extend(var_summary.messages)

        progress_bar.progress(1.0, text="Complete")

        hazards_df = pd.concat(all_hazards, ignore_index=True) if all_hazards else pd.DataFrame()
        alerts_df = pd.concat(all_alerts, ignore_index=True) if all_alerts else pd.DataFrame()
        summary = RunSummary(
            start_time=params["start_time"],
            end_time=params["end_time"],
            time_step_hours=params["time_step"],
            map_count=total_maps,
            alert_count=len(alerts_df),
            cache_hits=total_cache,
            api_calls_estimated=total_api,
            failures=total_fail,
            messages=all_messages,
        )

        st.session_state.historical_maps_meta = all_maps_meta
        st.session_state.historical_hazards = hazards_df
        st.session_state.historical_alerts = alerts_df
        st.session_state.historical_summary = summary

        # Persist to disk so results survive page refresh / restart.
        if summary.map_count > 0:
            try:
                run_id = save_historical_run(
                    hazards_df=hazards_df,
                    alerts_df=alerts_df,
                    summary=summary,
                    model=params["model"],
                    variable=variables[0] if len(variables) == 1 else ", ".join(variables),
                    region=params["region"],
                )
                if run_id:
                    st.session_state.historical_run_id = run_id
                    st.sidebar.success(f"Saved: {run_id}")
            except Exception as exc:
                st.sidebar.warning(f"Could not save results: {exc}")
    except Exception as exc:
        st.error(f"Historical analysis failed: {exc}")
        _clear_historical_results()
    finally:
        if progress_bar is not None:
            progress_bar.empty()
        st.session_state.hist_case_study = False
        st.session_state.historical_running = False


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
        st.metric("Data source", "SERENE API only")

    with c3:
        st.metric("Rows loaded", f"{len(st.session_state.data):,}")

    if status.message:
        if status.ok:
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
        if not SERENE_API_TOKEN:
            st.error(
                "**SERENE API token is not configured.** "
                "This dashboard requires SERENE API access or existing cache."
            )
        else:
            st.info(
                "Configure the sidebar and click **Load / Refresh data** "
                "to begin. This project uses SERENE API only."
            )
        with st.expander("Quick start"):
            st.markdown(
                """
                1. Copy `.env.example` to `.env` and set `SERENE_API_BASE_URL` and
                   `SERENE_API_TOKEN` (auth uses official `Token` scheme by default).
                2. Click **Test SERENE API connection**, then **Load / Refresh data**.
                3. Advisories shown here are **prototype advisories**, not official ICAO warnings.
                """
            )
        return

    df = st.session_state.data
    alerts = st.session_state.alerts
    var_options = sorted(df["variable"].dropna().unique()) if "variable" in df.columns else []

    # ── Tab layout ─────────────────────────────────────────────────────────
    tab_overview, tab_maps, tab_ts, tab_advisories, tab_raw, tab_system, tab_cache = st.tabs(
        ["Overview", "Maps by variable", "Time series", "Advisories", "Raw data",
         "System Overview", "API & Cache Status"]
    )

    # ── Tab 1: Overview ────────────────────────────────────────────────────
    with tab_overview:
        # -- Alert panel (compact) --
        st.subheader("ICAO-style prototype risk advisories")
        overall, summary = generate_overall_risk(alerts)
        emoji_map = {"Normal": "🟢", "Watch": "🟡", "Warning": "🟠", "Severe": "🔴"}
        st.markdown(f"**Overall risk:** {emoji_map.get(overall, '⚪')} {overall}")
        st.caption(summary)
        st.caption(DISCLAIMER)

        if not alerts.empty:
            with st.expander("Alert details", expanded=False):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.plotly_chart(create_alert_summary(alerts), use_container_width=True, key="alert_summary_chart")
                with col_b:
                    st.plotly_chart(create_alert_timeline(alerts), use_container_width=True, key="alert_timeline_chart")
                show_cols = [c for c in ("timestamp", "region", "alert_type", "risk_level",
                                         "reason", "possible_aviation_impact", "interpretation")
                             if c in alerts.columns]
                st.dataframe(alerts[show_cols], use_container_width=True, height=220)

        st.markdown("---")

        # -- All Variables Overview --
        st.subheader("All Variables Overview")
        st.caption(f"{len(var_options)} variable(s) discovered from data source.")

        if var_options:
            # Summary table from visualisation module
            summary_df = create_variable_summary_table(df)
            if not summary_df.empty:
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # Metric cards row
            cards = create_variable_card_data(df)
            if cards:
                card_cols = st.columns(min(len(cards), 4))
                for i, card in enumerate(cards[:4]):
                    with card_cols[i % 4]:
                        st.metric(
                            label=f"{card['variable']} ({card.get('unit', '')})",
                            value=card.get("mean"),
                            delta=f"min={card.get('min')} max={card.get('max')}",
                        )

    # ── Tab 2: Maps by variable ─────────────────────────────────────────────
    with tab_maps:
        st.subheader("Maps by variable")
        if var_options:
            # Full map grid
            try:
                st.plotly_chart(
                    create_variable_map_grid(df),
                    use_container_width=True,
                    key="var_map_grid",
                )
            except Exception:
                st.warning("Map grid failed; showing individual maps instead.")
                for var in var_options:
                    with st.expander(var, expanded=False):
                        st.plotly_chart(
                            create_map_plot(df, variable=var, title=var),
                            use_container_width=True,
                            key=f"tab_map_{var}",
                        )
        else:
            st.info("No variables to display.")

    # ── Tab 3: Time series ──────────────────────────────────────────────────
    with tab_ts:
        st.subheader("Time series — all variables")
        normalize = st.checkbox("Normalize (min-max per variable)", value=False, key="ts_normalize")
        if var_options:
            st.plotly_chart(
                create_multi_variable_time_series(df, normalize=normalize),
                use_container_width=True,
                key="multi_ts",
            )
        else:
            st.info("No time-series data available.")

    # ── Tab 4: Advisories ───────────────────────────────────────────────────
    with tab_advisories:
        st.subheader("ICAO-style prototype advisories")
        st.caption(DISCLAIMER)
        if alerts.empty:
            st.success("No active prototype advisories — parameters within normal range.")
        else:
            overall2, summary2 = generate_overall_risk(alerts)
            st.markdown(f"**Overall risk:** {emoji_map.get(overall2, '⚪')} {overall2}")
            st.caption(summary2)
            show_cols = [c for c in ("timestamp", "region", "alert_type", "risk_level",
                                     "reason", "possible_aviation_impact", "interpretation")
                         if c in alerts.columns]
            st.dataframe(alerts[show_cols], use_container_width=True)
            st.plotly_chart(create_alert_summary(alerts), use_container_width=True, key="adv_summary")
            st.plotly_chart(create_alert_timeline(alerts), use_container_width=True, key="adv_timeline")

    # ── Tab 5: Raw data ─────────────────────────────────────────────────────
    with tab_raw:
        st.subheader("Raw data")
        raw_var_filter = st.multiselect(
            "Filter by variable",
            options=var_options,
            default=[],
            key="raw_var_filter",
        )
        raw_df = df[df["variable"].isin(raw_var_filter)] if raw_var_filter else df
        st.dataframe(raw_df, use_container_width=True)
        st.caption(f"{len(raw_df):,} row(s)")

        # Status metadata
        with st.expander("Data source metadata"):
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

    # ── Tab 6: System Overview ──────────────────────────────────────────────
    with tab_system:
        st.subheader("System overview")
        st.caption("Snapshot suitable for thesis screenshots and monitoring dashboards.")

        # Top-level metric cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Variables monitored", len(var_options))
        with m2:
            st.metric("Total data points", f"{len(df):,}")
        with m3:
            severe_count = len(alerts[alerts["risk_level"] == "Severe"]) if not alerts.empty and "risk_level" in alerts.columns else 0
            active_alerts = len(alerts) if not alerts.empty else 0
            st.metric("Active advisories", active_alerts, delta=f"{severe_count} Severe" if severe_count > 0 else "All clear")
        with m4:
            api_status = "Connected" if st.session_state.api_connected else ("Disconnected" if st.session_state.api_connected is False else "Not tested")
            st.metric("API status", api_status)

        st.markdown("---")

        # Per-variable status cards
        st.subheader("Per-variable status")
        if var_options:
            vcols = st.columns(min(len(var_options), 4))
            for i, var in enumerate(var_options):
                var_df = df[df["variable"] == var]
                vals = pd.to_numeric(var_df["value"], errors="coerce").dropna()
                if vals.empty:
                    continue
                from hazard_detector import _classify_from_thresholds
                risk = _classify_from_thresholds(float(vals.max()), 0.0, 0.0, var)
                emoji = {"Normal": "🟢", "Watch": "🟡", "Warning": "🟠", "Severe": "🔴"}
                with vcols[i % 4]:
                    st.metric(
                        label=f"{emoji.get(risk, '⚪')} {var}",
                        value=f"{float(vals.mean()):.2f}",
                        delta=f"max: {float(vals.max()):.2f}",
                    )
                    st.caption(f"Risk: {risk}")

        st.markdown("---")

        # Data quality summary
        st.subheader("Data quality")
        q1, q2, q3 = st.columns(3)
        with q1:
            if "time" in df.columns:
                tmin = pd.to_datetime(df["time"]).min()
                tmax = pd.to_datetime(df["time"]).max()
                st.metric("Time span", f"{(tmax - tmin).total_seconds() / 3600:.1f} h")
            else:
                st.metric("Time span", "N/A")
        with q2:
            lat_span = f"{df['lat'].min():.1f}° to {df['lat'].max():.1f}°" if "lat" in df.columns else "N/A"
            st.metric("Latitude range", lat_span)
        with q3:
            lon_span = f"{df['lon'].min():.1f}° to {df['lon'].max():.1f}°" if "lon" in df.columns else "N/A"
            st.metric("Longitude range", lon_span)

        # Risk relevance breakdown
        st.subheader("Risk relevance breakdown")
        from hazard_detector import _hazard_type_for
        relevance_counts: dict[str, int] = {}
        for var in var_options:
            ht = _hazard_type_for(var)
            relevance_counts[ht] = relevance_counts.get(ht, 0) + 1
        if relevance_counts:
            rel_df = pd.DataFrame(
                [{"Risk category": k, "Variables": v} for k, v in relevance_counts.items()]
            )
            st.dataframe(rel_df, use_container_width=True, hide_index=True)

    # ── Tab 7: API & Cache Status ───────────────────────────────────────────
    with tab_cache:
        st.subheader("API & Cache Status")

        # API endpoint status
        st.markdown("### SERENE API endpoint")
        c_api1, c_api2, c_api3 = st.columns(3)
        with c_api1:
            if st.session_state.api_connected is True:
                st.success("Connected")
            elif st.session_state.api_connected is False:
                st.error("Failed")
            else:
                st.info("Not tested")
        with c_api2:
            st.metric("Base URL", st.session_state.api_message if st.session_state.api_connected else "—")
        with c_api3:
            st.metric("Token configured", "Yes" if SERENE_API_TOKEN else "No")

        st.markdown("---")

        # Cache listing
        st.markdown("### Map cache")
        cache_stats = count_cached_maps()
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.metric("Cached maps", cache_stats["count"])
        with cc2:
            st.metric("Total size", cache_stats["readable_size"])
        with cc3:
            st.metric("Cache directory", "data/cache/")

        cache_maps = list_cached_maps()
        if cache_maps:
            cache_df = pd.DataFrame(cache_maps)
            show_cache_cols = [c for c in ("model", "variable", "timestamp", "region", "resolution", "file_size") if c in cache_df.columns]
            # Make file_size human-readable
            if "file_size" in cache_df.columns:
                cache_df["file_size"] = cache_df["file_size"].apply(
                    lambda b: f"{b/1024:.1f} KB" if b < 1024*1024 else f"{b/(1024*1024):.1f} MB"
                )
            st.dataframe(cache_df[show_cache_cols], use_container_width=True, height=250)
            st.caption(f"{len(cache_maps)} cached file(s)")
        else:
            st.info("No cached maps. Cache files are created automatically when building fixed maps.")

        st.markdown("---")

        # Data source info
        st.markdown("### Data source")
        st.json({
            "source": st.session_state.status.source,
            "message": st.session_state.status.message,
            "warnings": st.session_state.status.warnings,
            "api_only": True,
            "local_fallback": "disabled",
        })


# ═══════════════════════════════════════════════════════════════════════════════
# Main page — Live fixed map mode
# ═══════════════════════════════════════════════════════════════════════════════


def _render_live_main(params: dict) -> None:
    st.title("Live Fixed Map Monitor")
    st.caption(
        "Fixed-grid ionospheric monitoring with hazard detection "
        "and ICAO-style prototype advisories."
    )
    st.caption(
        "Fixed map mode uses SERENE API or cached maps only. "
        "Local sample fallback is disabled."
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

    # Variable summary if multi-variable
    live_vars = sorted(map_df["variable"].dropna().unique()) if "variable" in map_df.columns else []

    tab_live_maps, tab_live_hazards, tab_live_advisories = st.tabs(
        ["Fixed maps", "Hazard detection", "ICAO-style advisories"]
    )

    # ── Tab 1: Fixed maps ──────────────────────────────────────────────────
    with tab_live_maps:
        if len(live_vars) > 1:
            st.subheader("Variable summary")
            st.dataframe(create_variable_summary_table(map_df), use_container_width=True, hide_index=True)

        st.subheader("Fixed grid maps")
        for var in live_vars:
            with st.expander(f"Map — {var}", expanded=(len(live_vars) <= 2)):
                st.plotly_chart(
                    create_fixed_map_plot(map_df, variable=var, title=f"Fixed grid — {var}"),
                    use_container_width=True,
                    key=f"live_fixed_map_{var}",
                )

    # ── Tab 2: Hazard detection ─────────────────────────────────────────────
    with tab_live_hazards:
        hazards = st.session_state.live_hazards
        st.subheader("Hazard detection")

        if hazards.empty:
            st.success("No hazards detected — all parameters within normal range.")
        else:
            # Hazard sub-tabs for different views
            htab_map, htab_gradient, htab_change, htab_table = st.tabs(
                ["Hazard map", "Spatial gradient", "Temporal change", "Summary table"]
            )

            with htab_map:
                st.plotly_chart(
                    create_hazard_summary_map(hazards),
                    use_container_width=True,
                    key="live_hazard_summary_map",
                )

            with htab_gradient:
                for var in live_vars:
                    st.plotly_chart(
                        create_spatial_gradient_map(map_df, variable=var, title=f"Spatial gradient — {var}"),
                        use_container_width=True,
                        key=f"live_gradient_{var}",
                    )

            with htab_change:
                for var in live_vars:
                    st.plotly_chart(
                        create_temporal_change_map(map_df, variable=var, title=f"Temporal change rate — {var}"),
                        use_container_width=True,
                        key=f"live_change_{var}",
                    )

            with htab_table:
                show_cols = [
                    c for c in ("variable", "hazard_type", "risk_level", "max_value",
                                "max_gradient", "max_change_rate", "reason")
                    if c in hazards.columns
                ]
                st.dataframe(hazards[show_cols], use_container_width=True, height=350)
                st.caption(f"{len(hazards)} hazard record(s)")

    # ── Tab 3: ICAO-style advisories ────────────────────────────────────────
    with tab_live_advisories:
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

            # Table view
            show_cols = [
                c for c in (
                    "timestamp", "region", "alert_type", "risk_level",
                    "reason", "possible_aviation_impact", "interpretation",
                )
                if c in alerts.columns
            ]
            st.dataframe(alerts[show_cols], use_container_width=True, height=220)

            st.markdown("---")
            # Advisory cards
            st.subheader("Advisory cards")
            for _, row in alerts.iterrows():
                st.markdown(
                    create_advisory_card_html(row.to_dict()),
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Main page — Historical analysis mode
# ═══════════════════════════════════════════════════════════════════════════════


def _render_historical_main(params: dict) -> None:
    st.title("Historical Analysis")
    st.caption(
        "Time-window hazard survey — ICAO-style prototype advisories "
        "for academic demonstration only."
    )
    st.caption(
        "Historical analysis is an API/cache replay framework. "
        "The current SERENE /api/calc/ endpoint is point-based and does not provide "
        "confirmed retrospective retrieval unless a historical endpoint or archived data is available."
    )

    _render_cloud_api_hint()
    _render_historical_debug_plan(params)

    # Show loaded run info
    if st.session_state.get("historical_run_id"):
        st.info("Loaded run: " + st.session_state.historical_run_id)

    # Apply case study preset if triggered.
    if st.session_state.get("hist_case_study"):
        st.info(
            "**May 2024 storm case study loaded.** "
            "2024-05-10 00:00 -> 2024-05-12 23:00 UTC, TEC, 12h steps. "
            "Click **Run historical analysis** to start."
        )
    elif st.session_state.get("hist_case_study_loaded"):
        st.info("Case study dates loaded. Click Run historical analysis.")

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

    if summary.map_count == 0:
        st.warning(
            "No historical maps were loaded.\n\n"
            "Possible reasons:\n"
            "1. No matching cache files exist.\n"
            "2. Live API requests are disabled.\n"
            "3. Current SERENE /api/calc/ does not support confirmed historical retrieval.\n"
            "4. Timestamp / region / resolution / variable do not match existing cache keys."
        )
        return

    st.markdown("---")

    alerts_df = st.session_state.historical_alerts
    hazards_df = st.session_state.historical_hazards

    tab_hist_summary, tab_hist_risk, tab_hist_alert, tab_hist_hazards, tab_hist_advisories = st.tabs(
        ["Summary", "Risk timeline", "Alert timeline", "Hazard details", "Advisories"]
    )

    # ── Tab 1: Summary ─────────────────────────────────────────────────────
    with tab_hist_summary:
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

    # ── Tab 2: Risk timeline ───────────────────────────────────────────────
    with tab_hist_risk:
        st.subheader("Risk level timeline")
        if hazards_df.empty:
            st.info("No hazard data for risk timeline.")
        else:
            st.plotly_chart(
                create_risk_timeline(hazards_df, title="Risk level over time"),
                use_container_width=True,
                key="hist_risk_timeline",
            )

    # ── Tab 3: Alert timeline ───────────────────────────────────────────────
    with tab_hist_alert:
        st.subheader("Alert timeline")
        if alerts_df.empty:
            st.info("No alerts in this run.")
        else:
            st.plotly_chart(
                create_alert_timeline(alerts_df),
                use_container_width=True,
                key="hist_timeline",
            )

    # ── Tab 4: Hazard details ───────────────────────────────────────────────
    with tab_hist_hazards:
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
            st.dataframe(hazards_df[show_cols], use_container_width=True, height=350)
            st.caption(f"{len(hazards_df)} hazard record(s)")

    # ── Tab 5: Advisories ──────────────────────────────────────────────────
    with tab_hist_advisories:
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

            # Advisory cards
            st.markdown("---")
            st.subheader("Advisory cards")
            for _, row in alerts_df.head(20).iterrows():
                st.markdown(
                    create_advisory_card_html(row.to_dict()),
                    unsafe_allow_html=True,
                )


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
