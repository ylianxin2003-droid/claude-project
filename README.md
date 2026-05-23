# Aviation Space Weather Monitoring Dashboard

Prototype Streamlit dashboard for aviation ionospheric monitoring using **SERENE**
real-time data and **AIDA / TOMIRIS** models.  Generates **ICAO-style prototype
risk advisories** for academic demonstration.

---

## Main features

- **All Variables Overview** — auto-discovers every variable from SERENE/AIDA and displays summary statistics, per-variable maps, and normalised time series for all variables simultaneously
- **Dynamic variable discovery** — variables are detected from the current API DataFrame, SERENE `/api/variables/` endpoint if available, or the fallback default registry. Sample JSON files are legacy references and are not used by the API-only dashboard.
- **SERENE API integration** — official `POST /api/calc/` with `Authorization: Token <token>`
- **2.5° fixed grid maps** — configurable resolution with global and regional grids
- **Map caching** — automatic Parquet/CSV cache in `data/cache/` to avoid repeated API calls
- **Hazard detection** — spatial gradient and temporal change rate analysis with configurable thresholds
- **ICAO-style prototype advisories** — Watch / Warning / Severe levels with aviation impact descriptions
- **Live fixed map mode** — single-timestamp grid build + hazard detection + advisories (supports all variables)
- **Historical analysis mode** — time-window sweep across multiple timestamps (supports all variables with cache warning)
- **May 2024 geomagnetic storm case study** — one-click preset for academic demonstration

---

## Architecture

```
SERENE API (/api/calc/)
         │
         ▼
 fixed grid generation   (grid_generator.py)
         │
         ▼
 map building            (map_builder.py)
         │
 ┌───────┴───────┐
 ▼               ▼
map cache     hazard detection   (hazard_detector.py)
(map_cache.py)      │
         │          ▼
         └──► alert generation   (alert_engine.py)
                    │
                    ▼
          Streamlit dashboard    (app.py)
```

### File map

| File | Role |
|---|---|
| `app.py` | Streamlit entry point — sidebar + 3 modes |
| `config.py` | Settings, env/secrets loading, thresholds |
| `serene_client.py` | SERENE API client (`POST /api/calc/`) |
| `data_loader.py` | SERENE API data loader (local mode disabled) |
| `grid_generator.py` | Fixed-grid lat/lon point generator (6 regions) |
| `map_builder.py` | Builds fixed maps from grid + API + cache |
| `map_cache.py` | Disk cache layer (Parquet / CSV fallback) |
| `hazard_detector.py` | Spatial gradient + temporal change hazard engine |
| `alert_engine.py` | ICAO-style prototype advisory generation |
| `historical_runner.py` | Time-window historical analysis runner |
| `visualisation.py` | All Plotly charts |
| `variable_registry.py` | Centralised variable discovery (5 functions) |
| `requirements.txt` | Python dependencies |
| `data/*.json` | Legacy sample data (not used by API-only workflow) |

---

## Variable registry

Variables are discovered dynamically in priority order:

1. **Current API DataFrame** — `df["variable"].unique()`
2. **SERENE API** — `/api/variables/` endpoint (if available)
3. **Fallback default registry** — hardcoded default list

Legacy local JSON discovery helpers may remain in the codebase, but they are
not used by the current API-only workflow.

The `variable_registry.py` module provides five public functions:

| Function | Purpose |
|---|---|
| `get_default_variables()` | Returns the canonical 7-variable list |
| `discover_variables_from_dataframe(df)` | Extracts variables + metadata from a loaded DataFrame |
| `discover_variables_from_local_json(path)` | Legacy helper — not used by the API-only workflow |
| `discover_variables_from_api(client, model)` | Discovers variables from SERENE API (falls back on failure) |
| `get_available_variables(source, df, local_file, client, model)` | Unified entry point with the priority chain above |

### Default variable registry

| Variable | Default risk relevance |
|---|---|
| `TEC` | GNSS positioning risk |
| `MUF3000` | HF communication risk |
| `foF2` | HF communication risk |
| `MUF3000_depression` | HF communication risk |
| `foF2_depression` | HF communication risk |
| `hmF2` | General ionospheric monitoring |
| `NmF2` | General ionospheric monitoring |

Variables marked as "General ionospheric monitoring" (hmF2, NmF2) do not have
hazard thresholds configured and will **not** generate Warning or Severe
ICAO-style advisories.  They appear in the monitoring overview as Normal with
the note "No prototype hazard threshold configured; monitoring summary only."

---

## Local run

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your SERENE token
streamlit run app.py
```

A valid `SERENE_API_TOKEN` is required for live API access.
Cached maps may be used when available. Local sample fallback is disabled.

---

## Streamlit Cloud deployment

1. Push this repository to GitHub (**do not** commit `.env` or `.streamlit/secrets.toml`)
2. On [share.streamlit.io](https://share.streamlit.io): **New app** → select repo, branch `main`, main file path `app.py`
3. **Settings → Secrets**, paste:

```toml
SERENE_API_BASE_URL = "https://spaceweather.bham.ac.uk"
SERENE_API_TOKEN = "your-token-here"
SERENE_API_TIMEOUT = "30"
SERENE_AUTH_SCHEME = "Token"
```

4. Click **Deploy** → your app will be live at `https://<app-name>.streamlit.app`

---

## Cache explanation

The `data/cache/` directory is created automatically at runtime.  It stores
grid map results as Parquet (or CSV fallback) files keyed by model, variable,
timestamp, resolution, and region.  This directory is listed in `.gitignore`
and **should not be committed to GitHub**.  Delete it at any time — it will
be recreated on the next run.

Sample JSON files (`data/latest_aida_grid.json`, `data/test_aida_grid.json`)
are not used by the current API-only workflow and are kept as legacy references.

---

## Alert disclaimer

**These are ICAO-style prototype advisories for academic demonstration only.**
They are **not** official ICAO warnings and **must not** be used for
operational aviation decision-making.

---

## Limitations

- All thresholds are **prototype configurable thresholds** — not official ICAO values
- SERENE `/api/calc/` is point-based; there is no batch endpoint at this time
- Global 2.5° grids require ~10,500 API calls — use small regions or cached data
- Historical analysis depends on API availability or pre-cached maps
- This project uses SERENE API only — local sample fallback is disabled
- Radiation hazard is **not assessed**
- Scintillation is **not assessed** unless future model output explicitly supports it
- The `MAX_GRID_POINTS` default (30) limits live API grid size; increase via `SERENE_MAX_GRID_POINTS` env var

---

## Official SERENE API format

```bash
curl -X POST \
  -H "Authorization: Token <token>" \
  -d latitude=52.4862 -d longitude=1.8904 \
  https://spaceweather.bham.ac.uk/api/calc/
```
