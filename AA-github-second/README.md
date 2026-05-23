# Aviation Space Weather Monitoring Dashboard

Prototype Streamlit dashboard for aviation ionospheric monitoring using **SERENE**
real-time data and **AIDA / TOMIRIS** models.  Generates **ICAO-style prototype
risk advisories** for academic demonstration.

---

## Main features

- **SERENE API integration** — official `POST /api/calc/` with `Authorization: Token <token>`
- **Local sample fallback** — bundled JSON grid data works without any API configuration
- **2.5° fixed grid maps** — configurable resolution with global and regional grids
- **Map caching** — automatic Parquet/CSV cache in `data/cache/` to avoid repeated API calls
- **Hazard detection** — spatial gradient and temporal change rate analysis with configurable thresholds
- **ICAO-style prototype advisories** — Watch / Warning / Severe levels with aviation impact descriptions
- **Live fixed map mode** — single-timestamp grid build + hazard detection + advisories
- **Historical analysis mode** — time-window sweep across multiple timestamps
- **May 2024 geomagnetic storm case study** — one-click preset for academic demonstration

---

## Architecture

```
SERENE API (/api/calc/)          Local sample JSON
         │                              │
         └──────────┬───────────────────┘
                    ▼
         fixed grid generation   (grid_generator.py)
                    │
                    ▼
         map building            (map_builder.py)
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
    map cache             hazard detection   (hazard_detector.py)
 (map_cache.py)                │
         │                     ▼
         └──────────►  alert generation      (alert_engine.py)
                              │
                              ▼
                    Streamlit dashboard      (app.py)
```

### File map

| File | Role |
|---|---|
| `app.py` | Streamlit entry point — sidebar + 3 modes |
| `config.py` | Settings, env/secrets loading, thresholds |
| `serene_client.py` | SERENE API client (`POST /api/calc/`) |
| `data_loader.py` | Unified data loader (API + local fallback) |
| `grid_generator.py` | Fixed-grid lat/lon point generator (6 regions) |
| `map_builder.py` | Builds fixed maps from grid + API + cache |
| `map_cache.py` | Disk cache layer (Parquet / CSV fallback) |
| `hazard_detector.py` | Spatial gradient + temporal change hazard engine |
| `alert_engine.py` | ICAO-style prototype advisory generation |
| `historical_runner.py` | Time-window historical analysis runner |
| `visualisation.py` | All Plotly charts |
| `requirements.txt` | Python dependencies |
| `data/*.json` | Bundled sample data |

---

## Local run

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your SERENE token
streamlit run app.py
```

Without a SERENE token the dashboard still works — it auto-loads the bundled
sample data on first visit.

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
