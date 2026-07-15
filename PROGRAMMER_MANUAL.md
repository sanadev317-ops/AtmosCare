# AtmosCare Programmer Manual

Technical reference for developers maintaining, extending, and deploying AtmosCare.

**Stack:** Kivy/KivyMD (Frontend) · FastAPI (Backend) · MongoDB · TensorFlow GRU + SARIMA + XGBoost stacking · Open-Meteo / WAQI / AccuWeather APIs

**Repo root:** `AtmosCare-main/AtmosCare-main/`

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Repository Structure](#2-repository-structure)
3. [Running Locally](#3-running-locally)
4. [Configuration & Environment](#4-configuration--environment)
5. [Backend API Reference](#5-backend-api-reference)
6. [MongoDB Schema](#6-mongodb-schema)
7. [Machine Learning Pipeline](#7-machine-learning-pipeline)
8. [Data Services & External APIs](#8-data-services--external-apis)
9. [Frontend Architecture](#9-frontend-architecture)
10. [Authentication & Authorization](#10-authentication--authorization)
11. [Alert System](#11-alert-system)
12. [Deployment](#12-deployment)
13. [Git LFS & Model Artifacts](#13-git-lfs--model-artifacts)
14. [Extension Guide](#14-extension-guide)
15. [Known Issues & Pitfalls](#15-known-issues--pitfalls)

---

## 1. Architecture Overview

```mermaid
flowchart TB
    subgraph FE["Frontend (Kivy/KivyMD)"]
        APP[RealTimeApp]
        IM[IntegrationManager]
        ALERT[AlertSystem]
        APP --> IM
        APP --> ALERT
    end

    subgraph BE["Backend (FastAPI)"]
        MAIN[main.py]
        PIPE[pipeline_service]
        BRIDGE[smog_predictor_bridge]
        UDS[unified_data_service]
        AQS[air_quality_service]
        MAIN --> PIPE
        MAIN --> UDS
        MAIN --> AQS
        PIPE --> BRIDGE
    end

    subgraph ML["smog/ml_pipeline"]
        PRED[Predictor]
        GRU[GRU keras]
        SARIMA[SARIMA pkl]
        XGB[XGBoost meta-model]
        BRIDGE --> PRED
        PRED --> GRU
        PRED --> SARIMA
        PRED --> XGB
    end

    subgraph DB["MongoDB"]
        USERS[(users)]
        IOT[(iot_data)]
        DEV[(devices)]
        PREDS[(predictions)]
        BC[(broadcasts)]
    end

  subgraph EXT["External"]
        WAQI[WAQI API]
        OM[Open-Meteo]
        ACCU[AccuWeather]
    end

    IM -->|HTTP REST| MAIN
    MAIN --> DB
    UDS --> WAQI
    AQS --> WAQI
    AQS --> ACCU
    ALERT --> OM
```

### Data flow modes

| Mode | Condition | Input | Inference |
|------|-----------|-------|-----------|
| `iot` | Fresh device buffer + recent ingest | IoT sensor via `POST /iot/predict` | GRU + SARIMA + stacking |
| `hybrid` | Device exists but stale | API supplement | Model on blended readings |
| `api` | No fresh IoT | WAQI / Pakistan AQI / defaults | Model on API PM values |

Mode is resolved in `Backend/main.py` → `GET /mode` and `_should_use_api_instead_of_iot()`.

### Process boundaries

| Process | Entry point | Port |
|---------|-------------|------|
| Backend API | `uvicorn Backend.main:app` | 8000 |
| Frontend UI | `python Frontend/main.py` | — (desktop window) |
| MongoDB | external | 27017 or Atlas `mongodb+srv://` |
| Pakistan AQI proxy (optional) | separate service | 3000 |

The frontend talks to the backend over HTTP. Auth and user settings use **direct MongoDB access** from the Kivy process via `Backend/auth_manager.py` — not REST endpoints.

---

## 2. Repository Structure

```
AtmosCare-main/
├── Backend/
│   ├── main.py                  # FastAPI app, IoT ingest, predictions, health
│   ├── database.py              # users + air_quality collections
│   ├── config.py                # env loading, DATABASE_URI, SMTP
│   ├── auth_manager.py          # signup/login/settings wrappers
│   ├── admin_service.py         # users, devices, broadcasts, audit
│   ├── pipeline_service.py      # AirQualityInferenceService (ML runtime)
│   ├── smog_predictor_bridge.py # Adapter to smog/ml_pipeline Predictor
│   ├── ml_model_service.py      # Singleton over pipeline_service
│   ├── unified_data_service.py  # Model-driven AQI + API fallback chain
│   ├── air_quality_service.py   # WAQI / AccuWeather / simulated tiers
│   ├── open_meteo_service.py    # Rain/snow forecast for alerts
│   ├── accuweather_service.py   # Weather supplement
│   ├── location_service.py      # Google Geolocation + IP fallback
│   ├── integration_manager.py   # Kivy ↔ backend polling
│   ├── api_client.py            # HTTP client for backend endpoints
│   ├── pakistan_aqi_client.py   # Legacy AQI client wrapper
│   ├── analytics_service.py     # Health guidance, SHAP enrichment
│   └── example.env              # Env template
├── Frontend/
│   ├── main.py                  # App entry, screens, AlertSystem, RealTimeApp
│   ├── *.kv                     # KivyMD layouts per screen
│   ├── ui_components.py         # Shared widgets (mode indicator, nav)
│   ├── error_handling.py        # Connection helpers
│   └── buildozer.spec           # Android APK build config
├── smog/
│   ├── ml_pipeline/             # Training + inference library
│   └── artifacts/models/        # Git LFS model files
├── scripts/
│   └── deploy-railway.ps1       # Railway CLI deploy helper
├── Dockerfile                   # Backend container (git-lfs + pip)
├── nixpacks.toml                # Railway/Nixpacks build
├── railway.json                 # Railway deploy config
├── Procfile                     # uvicorn start command
├── requirements-backend.txt     # Backend Python deps
├── requirements.txt             # Frontend Python deps
├── USER_MANUAL.md               # End-user documentation
└── PROGRAMMER_MANUAL.md         # This file
```

---

## 3. Running Locally

### Prerequisites

- Python **3.11** (`runtime.txt`)
- MongoDB (local or [MongoDB Atlas](https://www.mongodb.com/atlas))
- `git lfs pull` (required for ML models)
- Optional: WAQI API key, Google Maps key

### Backend

```bash
cd AtmosCare-main
git lfs pull
pip install -r requirements-backend.txt

# Copy and edit env
cp Backend/example.env .env

uvicorn Backend.main:app --host 0.0.0.0 --port 8000
# or: python Backend/main.py
```

Verify: `GET http://127.0.0.1:8000/health` → `"database": "ok"`, `"gru_loaded": true`.

### Frontend

```bash
cd Frontend
pip install -r ../requirements.txt
python main.py
```

Run from the **`Frontend/`** directory — KV files are loaded by relative path.

### First admin user

The **first registered user** is auto-promoted to `admin` in `Backend/database.py` → `add_user()`.

### Optional Pakistan AQI service

`PAKISTAN_AQI_URL` defaults to `http://127.0.0.1:3000`. If unavailable, `UnifiedDataService` falls through to WAQI and built-in defaults.

---

## 4. Configuration & Environment

### `Backend/config.py` and `.env`

`config.py` walks up to 6 parent directories looking for `.env`. Existing `os.environ` keys are never overwritten.

| Variable | Default | Used by |
|----------|---------|---------|
| `DATABASE_URI` | `mongodb://localhost:27017` | All MongoDB clients |
| `DATABASE_NAME` | `AtmosCareDB` | Database name |
| `MONGO_URI` / `MONGODB_URI` | aliases for `DATABASE_URI` | Legacy |
| `WAQI_API_KEY` | `""` | `air_quality_service`, `unified_data_service` |
| `ACCUWEATHER_API_KEY` | `""` | `accuweather_service` |
| `GOOGLE_API_KEY` | — | `location_service`, map tiles |
| `GOOGLE_GEOLOCATION_KEY` / `GOOGLE_MAPS_KEY` | aliases | `location_service` |
| `SMTP_FROM`, `SMTP_USERNAME`, `SMTP_PASSWORD` | `""` | Email (minimal use) |

### Backend-only (`main.py`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `PAKISTAN_AQI_URL` | `http://127.0.0.1:3000` | Pakistan AQI proxy |
| `IOT_STALE_MINUTES` | `30` | IoT freshness threshold |
| `PORT` | `8000` | Docker/Railway bind port |

### Frontend (`Frontend/main.py`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `BACKEND_URL` | `http://127.0.0.1:8000` | FastAPI base URL |
| `AQI_API_URL` / `PAKISTAN_AQI_URL` | `http://127.0.0.1:3000` | Legacy AQI client |

### MongoDB Atlas

Use `mongodb+srv://USER:PASSWORD@cluster.mongodb.net/?retryWrites=true&w=majority`. Requires `pymongo[srv]` and `dnspython` (in `requirements-backend.txt`).

---

## 5. Backend API Reference

**Base:** `Backend/main.py` · CORS: `allow_origins=["*"]` (dev only)

### Request models (Pydantic)

```python
class IoTPredictRequest(BaseModel):
    device_id: str
    temperature: float
    humidity: float
    gas_level: Optional[float] = None
    pm2_5: Optional[float] = None
    pm10: Optional[float] = None
    wind_speed: float
    timestamp: datetime

class DeviceRegisterRequest(BaseModel):
    device_id: str
    location: Optional[str] = None
    status: str = "active"

class BatchPredictRequest(BaseModel):
    measurements: List[BatchMeasurement]
    n_forecast_hours: int = 24  # 1–30
    device_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    actual_pm2_5: float
    device_id: Optional[str] = None
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health, DB ping, model status, collection counts |
| `GET` | `/mode` | Current data mode: `iot` / `hybrid` / `api` |
| `POST` | `/devices/register` | Register IoT device |
| `POST` | `/iot/predict` | Ingest sensor reading + run ML prediction |
| `GET` | `/predict` | Live prediction (`?device_id=`, `?city=`) |
| `POST` | `/predict` | Batch measurements prediction |
| `POST` | `/predict_single` | Single loose-dict measurement |
| `GET` | `/analytics/cities` | 8 Pakistan city forecasts (120s cache) |
| `GET` | `/history/{device_id}` | Prediction history (`?limit=1–500`) |
| `POST` | `/feedback/{prediction_id}` | Submit actual PM2.5 for a stored prediction |
| `GET` | `/broadcasts/active` | Active city advisories (`?city=`) |

### `GET /health` response

```json
{
  "status": "healthy",
  "database": "ok",
  "models": {
    "gru_loaded": true,
    "sarima_loaded": true,
    "stacking_loaded": true,
    "engine": "smog_bridge",
    "model_version": "meta_model_XGBoost_20260710_141440"
  },
  "buffer_size": 60,
  "collections": { "iot_data": 0, "predictions": 0, "devices": 0 }
}
```

### `GET /predict` response (key fields)

Built by `_build_prediction_response()` in `main.py`:

```json
{
  "prediction": 42.5,
  "predicted_pm2_5": 42.5,
  "aqi": 95,
  "confidence": 0.85,
  "gru_prediction": 40.1,
  "sarima_prediction": 44.2,
  "gru_forecast": [40.1, 41.0],
  "sarima_forecast": [44.2, 43.8],
  "forecast_7d": [...],
  "smog_sources": {...},
  "health": {...},
  "model_status": { "engine": "smog_bridge", ... },
  "device_id": "sensor-001",
  "prediction_id": "uuid",
  "pm2_5": 38.0,
  "pm10": 55.0,
  "o3": 12.0,
  "no2": 8.0,
  "co": 0.5,
  "temperature": 28.0,
  "humidity": 45.0,
  "wind_speed": 3.2,
  "smog_index": 55,
  "source": "iot",
  "reading_mode": "iot",
  "model_engine": "smog_bridge",
  "location": "Lahore"
}
```

### IoT staleness rules (`_should_use_api_instead_of_iot`)

API fallback triggers when:

- `admin_disabled` or `force_api` or `marked_test` device flags set
- Device ID matches test prefixes: `test-`, `demo-`, `single-`, `batch-`
- Reading older than `IOT_STALE_MINUTES` (default 30)
- No recent ingest in `iot_data` collection
- Source is `synthetic_fallback` or `api_fallback`

### Route registration note

`_predict_from_api()` is an **internal helper** — it must **not** have its own `@app.get("/predict")` decorator. A duplicate route previously shadowed the main handler and dropped pollutant fields.

---

## 6. MongoDB Schema

Two access layers share the same `DATABASE_URI`:

### Auth layer (`Backend/database.py`)

**`users`**

```json
{
  "username": "string",
  "email": "string",
  "password": "string",
  "role": "user | authority | admin",
  "settings": {
    "name": "string",
    "location": "Lahore, Pakistan",
    "rain": false,
    "snow": false,
    "smog": true
  }
}
```

**`air_quality`** — cached AQI snapshots keyed by `location`.

### IoT + admin layer (`main.py`, `admin_service.py`)

**`iot_data`** — raw ingest events

```json
{
  "device_id": "string",
  "timestamp": "datetime",
  "temperature": 28.0,
  "humidity": 45.0,
  "gas_level": 120.0,
  "pm2_5": 38.0,
  "pm10": 55.0,
  "wind_speed": 3.2,
  "source": "iot",
  "ingested_at": "datetime",
  "timestamp_iso": "string"
}
```

**`devices`** — rolling buffer per device (`BUFFER_SIZE = 60`)

```json
{
  "device_id": "string",
  "buffer": [ /* last 60 events */ ],
  "location": "string",
  "status": "active",
  "latest_prediction": { /* last prediction payload */ },
  "last_prediction_at": "datetime",
  "admin_disabled": false,
  "force_api": false,
  "marked_test": false
}
```

**`predictions`**

```json
{
  "prediction_id": "uuid",
  "device_id": "string",
  "timestamp": "datetime",
  "input": { /* seed event */ },
  "prediction": 42.5,
  "confidence": 0.85,
  "gru_prediction": 40.1,
  "sarima_prediction": 44.2,
  "sarima_forecast": [],
  "stacking_input": [40.1, 44.2],
  "model_status": {},
  "actual_pm2_5": null,
  "feedback_at": null
}
```

**`broadcasts`**

```json
{
  "city": "Lahore | *",
  "title": "string",
  "message": "string",
  "created_by": "email",
  "created_at": "datetime",
  "expires_at": "datetime",
  "active": true
}
```

TTL: 24 hours (`BROADCAST_TTL_HOURS` in `admin_service.py`).

**`audit_logs`**

```json
{
  "actor": "email",
  "action": "user_delete | broadcast | device_update | ...",
  "target": "string",
  "details": "string",
  "timestamp": "datetime"
}
```

---

## 7. Machine Learning Pipeline

### Load order (`AirQualityInferenceService.load()`)

File: `Backend/pipeline_service.py`

```
1. SmogPredictorBridge.load()  → preferred (engine: "smog_bridge")
2. Legacy fallback:
   - gru_aqi_model.keras + scaler_X.pkl + scaler_y.pkl  (project root)
   - smog/artifacts/models/sarima_optimized/sarima_*.pkl
   - smog/artifacts/models/stacking_model.pkl
```

### Model artifact paths

| Artifact | Path |
|----------|------|
| GRU (optimized) | `smog/artifacts/models/gru_optimized/gru_optimized_*.keras` |
| GRU scaler | `smog/artifacts/models/gru_optimized/gru_scaler.pkl` |
| GRU metadata | `smog/artifacts/models/gru_optimized/gru_optimized_*_metadata.json` |
| SARIMA | `smog/artifacts/models/sarima_optimized/sarima_*.pkl` (~466 MB) |
| SARIMA scaler | `smog/artifacts/models/sarima_optimized/sarima_scaler.joblib` |
| Hybrid meta-model | `smog/artifacts/models/hybrid/meta_model_XGBoost_*.pkl` |
| Legacy GRU | `gru_aqi_model.keras` (repo root) |
| Legacy stacking | `smog/artifacts/models/stacking_model.pkl` |
| Pipeline config | `smog/ml_pipeline/config/config.yaml` |

`SMOG_ROOT` resolves to `PROJECT_ROOT / "smog"` in both `pipeline_service.py` and `smog_predictor_bridge.py`.

### Inference call chain

```
POST /iot/predict  or  GET /predict
  └─ model_service.predict(buffer, seed_event)
       └─ [smog_bridge] SmogPredictorBridge.predict()
            ├─ _engineer_gru_frame() — 21 GRU features
            ├─ Predictor.predict() — SARIMA path
            ├─ _predict_gru() — GRU path
            ├─ _stack_predictions() — XGBoost meta-model
            └─ enrich_prediction_analytics() — SHAP, health, forecast_7d
       └─ [legacy] _predict_gru + _predict_sarima + _stack_predictions
  └─ _store_prediction() → MongoDB
  └─ _build_prediction_response()
```

### GRU feature vector (21 features)

Defined in `smog_predictor_bridge.py` → `GRU_FEATURE_NAMES`:

`pm2_5_log`, `no2`, `so2`, `temperature_2m_mean`, `relative_humidity_2m_mean`, `wind_speed_10m_max`, `traffic_combined`, `crop_burning_intensity`, `pm_lag7`, `pm_lag14`, `pm_roll7`, `pm_roll14`, `pm_diff1`, `pm_diff3`, `pm_zscore7`, `no2_o3`, `temp_wind`, `month_sin`, `month_cos`, `doy_sin`, `doy_cos`

### `InferenceResult` dataclass

```python
@dataclass
class InferenceResult:
    prediction: float
    confidence: float
    gru_prediction: float
    sarima_prediction: float
    sarima_forecast: List[float]
    stacking_input: List[float]
    model_status: Dict[str, Any]
    gru_forecast: Optional[List[float]] = None
    forecast_7d: Optional[List[Dict]] = None
    smog_sources: Optional[Dict] = None
    health: Optional[Dict] = None
```

### Standalone smog API

`smog/ml_pipeline/run_api.py` exposes a parallel FastAPI for pipeline-only deployments. Production AtmosCare uses `Backend/main.py` as the integrated entry point.

---

## 8. Data Services & External APIs

### `air_quality_service.py` — tiered fallback

| Tier | Source | Data |
|------|--------|------|
| 1 | WAQI (`api.waqi.info`) | PM2.5, PM10, O3, NO2, CO, temp, humidity, wind, AQI |
| 2 | AccuWeather | Weather supplement; sole source if WAQI fails |
| 3 | Simulated | Random fallback |

`calculate_smog_index()`: weighted PM2.5×0.4 + PM10×0.3 + O3×0.15 + NO2×0.1 + CO×0.05.

`estimate_secondary_pollutants(pm25)` fills missing O3/NO2/CO when sensors omit them.

### `open_meteo_service.py` — weather alerts

- Geocoding: `geocoding-api.open-meteo.com`
- Forecast: `api.open-meteo.com` (precipitation, snowfall, weathercode)
- Returns `rain_expected`, `snow_expected` for next 24 hours
- No API key required
- Used by `AlertSystem._do_check()` in the frontend

### `unified_data_service.py` — prediction enrichment

`get_aqi_prediction(city)` priority:

1. In-memory cache (5 min TTL)
2. Pakistan AQI API (`PAKISTAN_AQI_URL/aqi/{city}`)
3. WAQI API
4. Built-in default payload
5. Enrich via WAQI + AccuWeather + `estimate_secondary_pollutants`
6. `predict_aqi_from_model()` — merge API readings with ML output

### `api_client.py` — HTTP mapping

| Method | HTTP |
|--------|------|
| `health_check()` | `GET /health` |
| `get_mode()` | `GET /mode` |
| `get_live_prediction()` | `GET /predict` |
| `iot_predict()` | `POST /iot/predict` |
| `predict_batch()` | `POST /predict` |
| `get_history()` | `GET /history/{id}` |
| `register_device()` | `POST /devices/register` |
| `send_feedback()` | `POST /feedback/{id}` |
| `get_analytics_cities()` | `GET /analytics/cities` |

### `IntegrationManager`

- Polls backend every **15 seconds** (configurable `refresh_interval`)
- Background thread + Kivy `Clock` for UI thread safety
- Callbacks: `on_prediction_update`, `on_mode_change`, `on_error`, `on_connected`, `on_disconnected`
- `get_mode()` returns a **string** (`"iot"` / `"api"` / `"hybrid"`) — not a dict

---

## 9. Frontend Architecture

### Screen registry (`RealTimeApp.build()`)

| Key | Class | KV file |
|-----|-------|---------|
| `splash` | `SplashScreen` | `splash.kv` |
| `signup` | `SignUpScreen` | `auth.kv` |
| `login` | `LoginScreen` | `auth.kv` |
| `dashboard` | `DashboardScreen` | `dashboard.kv` |
| `profile` | `ProfileScreen` | `profile.kv` |
| `locations` | `LocationsScreen` | `locations.kv` |
| `graphs` | `GraphsScreen` | `graphs.kv` |
| `settings` | `SettingsScreen` | `settings.kv` |
| `admin_panel` | `AdminPanelScreen` | `admin_panel.kv` |

### Base classes

- **`AuthenticatedScreen`** — redirects to `login` if `current_user_email` is unset
- **`AdminScreen`** — requires `admin` role (defined; `AdminPanelScreen` uses its own check)

### Key helpers in `main.py`

| Function | Purpose |
|----------|---------|
| `_prediction_to_air_data()` | Normalize backend JSON for dashboard UI |
| `_derive_forecast()` | Client-side forecast cards from prediction |
| `_derive_trends()` | 7-day / 30-day trend summaries |
| `_resolve_live_location()` | GPS → Google/IP geolocation |

### `RealTimeApp` lifecycle

```python
# on_start()
alert_system = AlertSystem(self)       # created but not started
integration_manager.start_refresh()    # 15s poll begins

# on successful login
alert_system.start()                   # 60s alert poll begins
load_user_settings()
```

### Window defaults

```python
Window.size = (400, 780)
Window.minimum_width = 360
Window.minimum_height = 640
```

### Android build

`Frontend/buildozer.spec` — packages Kivy app as APK. ML/DB calls go to backend over HTTP (`requests` only in APK requirements).

---

## 10. Authentication & Authorization

### Auth flow (no JWT)

1. Frontend calls `handle_signup()` / `handle_login()` in `auth_manager.py`
2. `database.py` reads/writes `users` collection directly from the Kivy process
3. Session state: `app.current_user_email`, `app.current_user_role` (in-memory only)

**Security note:** Passwords are stored and compared in **plaintext**. Not production-safe without hashing.

### Roles

| Role | Admin panel | User management | Device flags | Broadcasts | Audit |
|------|-------------|-----------------|--------------|------------|-------|
| `user` | ✗ | ✗ | ✗ | ✗ | ✗ |
| `authority` | ✓ (limited) | view only | view only | send | ✗ |
| `admin` | ✓ (full) | change role, delete | toggle flags | send | ✓ |

Guards in `admin_service.py`:

- Cannot demote/delete the **last admin**
- Cannot change/delete **own account**
- User delete requires actor's **password confirmation**

---

## 11. Alert System

**Class:** `AlertSystem` in `Frontend/main.py`

| Constant | Value |
|----------|-------|
| `POLL_INTERVAL` | 60 seconds |
| `ALERT_COOLDOWN` | 30 minutes |

### Trigger sources

| Alert | Gate | Source |
|-------|------|--------|
| Smog (PM2.5) | `settings.smog` | Live prediction data |
| Smog (AQI fallback) | `settings.smog` | When PM2.5 unavailable |
| Rain | `settings.rain` | `open_meteo_service.get_weather_alert_status()` |
| Snow | `settings.snow` | Open-Meteo |
| Broadcast | always (if new ID) | `get_active_broadcasts(location)` |

### PM2.5 thresholds

| PM2.5 (µg/m³) | Level | UI |
|---------------|-------|-----|
| > 150 | severe | Dialog |
| > 55 | unhealthy | Dialog |
| > 35 | moderate | Snackbar |
| +15% rise (if > 35) | increase | Dialog |

### Notification history

- Stored in `deque(maxlen=30)`
- Viewed via Dashboard bell icon
- `_record_notification()` called for dialogs and snackbars

---

## 12. Deployment

### Docker (recommended for ML workloads)

```dockerfile
# Dockerfile
FROM python:3.11-slim
RUN apt-get install git git-lfs
COPY . .
RUN git lfs pull && pip install -r requirements-backend.txt
CMD uvicorn Backend.main:app --host 0.0.0.0 --port ${PORT}
```

### Railway

`railway.json` + `nixpacks.toml` + `scripts/deploy-railway.ps1`

```powershell
.\scripts\deploy-railway.ps1 -DatabaseUri "mongodb+srv://..." -WaqiApiKey "..."
```

Health check: `GET /health`

### Render / Fly.io

Connect GitHub repo, select **Docker** as environment, set env vars, allocate **≥2 GB RAM** (4 GB recommended for TensorFlow + SARIMA).

### Frontend (APK)

GitHub Actions: `.github/workflows/build-apk.yml` — builds debug APK on push to `main`.

Local: `cd Frontend && buildozer android debug`

### Post-deploy checklist

1. `GET /health` → `database: ok`, models loaded
2. Set `BACKEND_URL` in frontend env or rebuild APK
3. MongoDB Atlas → Network Access → allow `0.0.0.0/0` (or platform IP ranges)
4. Confirm `git lfs pull` ran during build (models not pointer files)

---

## 13. Git LFS & Model Artifacts

`.gitattributes`:

```
smog/artifacts/models/**/*.pkl    filter=lfs
smog/artifacts/models/**/*.keras  filter=lfs
smog/artifacts/models/**/*.joblib filter=lfs
```

**Total LFS payload:** ~489 MB (mostly SARIMA pkl).

If LFS is not pulled, inference falls back to legacy root models or synthetic data. Always run `git lfs pull` before local dev and ensure deploy builds include the LFS step.

---

## 14. Extension Guide

### Add a backend endpoint

1. Add Pydantic models near top of `Backend/main.py`
2. Add `@app.get` / `@app.post` handler
3. Reuse `model_service`, `unified_service`, or Mongo collections
4. Add method to `Backend/api_client.py` if frontend needs it
5. Optionally wire into `IntegrationManager._refresh_cycle()`

### Add a frontend screen

1. Create `Frontend/myscreen.kv` with root widget matching screen name
2. Add `class MyScreen(AuthenticatedScreen)` in `Frontend/main.py`
3. `Builder.load_file("myscreen.kv")` in `RealTimeApp.build()`
4. `sm.add_widget(MyScreen(name="myscreen"))`
5. Add nav button in `dashboard.kv` drawer or bottom bar

### Add an alert type

1. Add logic in `AlertSystem._check_and_alert()`
2. Add settings flag in `database.save_user_settings()` defaults
3. Add checkbox in `settings.kv`
4. Update `RealTimeApp.save_settings()` signature if needed
5. Use `_show_dialog()` for severe, `_show_snackbar()` for info
6. Call `_record_notification()` for bell history

For weather-style alerts, fetch data in `_do_check()` and pass flags into `check_data()`.

### Retrain and deploy new models

1. Train in `smog/` pipeline (`runs/run_gru.py`, `run_sarima.py`, etc.)
2. Copy artifacts to `smog/artifacts/models/` subdirectories
3. `git lfs track` new files if needed
4. Commit and redeploy backend
5. Verify `GET /health` → `model_version` updated

---

## 15. Known Issues & Pitfalls

| Issue | Detail | Fix |
|-------|--------|-----|
| Git LFS not pulled | Models are pointer files; bridge fails | `git lfs pull` locally; add LFS step to CI/CD |
| MongoDB down | Backend runs degraded; auth fails | Check `DATABASE_URI`, Atlas IP whitelist |
| Stale IoT | Readings > 30 min old force API fallback | Send fresh `POST /iot/predict` events |
| Duplicate `/predict` route | Shadowed handler dropped pollutant fields | Keep `_predict_from_api` as internal helper only |
| `get_mode()` type | Returns string, not dict | Never call `.get()` on mode string |
| Plaintext passwords | No bcrypt/hashing | Hash before production |
| AlertSystem timing | `start()` only after login | Don't expect alerts on splash/login screens |
| Frontend cwd | KV loaded by relative path | Always `cd Frontend` before `python main.py` |
| TensorFlow log noise | Floods console | `TF_CPP_MIN_LOG_LEVEL=2` set in `main.py` |
| SARIMA pickle compat | pandas version mismatch | `_patch_pandas_stringarray_compat()` in pipeline |
| NaN in SARIMA exog | Crashes `/predict` with 500 | `_safe_float()` + frame sanitization |
| CORS `*` | Open to all origins | Restrict in production |
| Pakistan AQI optional | Defaults to localhost:3000 | Service falls through to WAQI |
| No automated tests | Manual verification only | Add pytest for API + inference smoke tests |
| APK size / TF | TensorFlow not in APK | All ML runs on backend; app uses `requests` only |

### Manual smoke test checklist

```bash
# Health
curl http://127.0.0.1:8000/health

# Mode
curl http://127.0.0.1:8000/mode

# API prediction
curl "http://127.0.0.1:8000/predict?city=Lahore"

# IoT ingest
curl -X POST http://127.0.0.1:8000/iot/predict \
  -H "Content-Type: application/json" \
  -d '{"device_id":"test-1","temperature":28,"humidity":45,"gas_level":80,"wind_speed":3,"timestamp":"2026-07-13T12:00:00Z"}'

# City analytics
curl http://127.0.0.1:8000/analytics/cities
```

---

## Quick Reference

| Task | Command / file |
|------|----------------|
| Start backend | `uvicorn Backend.main:app --port 8000` |
| Start frontend | `cd Frontend && python main.py` |
| Pull models | `git lfs pull` |
| Env template | `Backend/example.env` |
| Add API endpoint | `Backend/main.py` |
| Add screen | `Frontend/main.py` + `Frontend/*.kv` |
| Change ML models | `smog/artifacts/models/` |
| Deploy backend | `Dockerfile` or `scripts/deploy-railway.ps1` |
| Build APK | `Frontend/buildozer.spec` + GitHub Actions |
| User-facing docs | `USER_MANUAL.md` |

---

*AtmosCare Programmer Manual — v1.0*
