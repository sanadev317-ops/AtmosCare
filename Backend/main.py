from __future__ import annotations

import os
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymongo import MongoClient, DESCENDING
from pymongo.collection import Collection

from Backend.analytics_service import get_smog_health_guidance
from Backend.air_quality_service import calculate_smog_index, estimate_secondary_pollutants, get_air_quality_data, model_prediction_fields
from Backend.location_service import get_detailed_location
from Backend.pipeline_service import AirQualityInferenceService
from Backend.unified_data_service import get_unified_service, init_unified_service, DataSource


def _load_env_file() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
        return
    except Exception:
        pass

    env_paths = [
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
    ]
    for env_path in reversed(env_paths):
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value


_load_env_file()


DATABASE_URI = os.getenv(
    "DATABASE_URI",
    os.getenv("MONGO_URI", os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
)
DATABASE_NAME = os.getenv(
    "DATABASE_NAME",
    os.getenv("MONGO_DB", os.getenv("MONGODB_DATABASE", "AtmosCareDB"))
)
BUFFER_SIZE = 60
PAKISTAN_AQI_URL = os.getenv("PAKISTAN_AQI_URL", "http://127.0.0.1:3000")
IOT_STALE_MINUTES = int(os.getenv("IOT_STALE_MINUTES", "30"))
TEST_DEVICE_PREFIXES = ("test-", "demo-", "single-", "batch-")
NON_SENSOR_DEVICES = {"api-fallback"}


class IoTPredictRequest(BaseModel):
    device_id: str = Field(..., min_length=1)
    temperature: float
    humidity: float
    gas_level: Optional[float] = None
    pm2_5: Optional[float] = None
    pm10: Optional[float] = None
    wind_speed: float
    timestamp: datetime


class DeviceRegisterRequest(BaseModel):
    device_id: str = Field(..., min_length=1)
    location: Optional[str] = None
    status: str = "active"


class BatchMeasurement(BaseModel):
    timestamp: Optional[datetime] = None
    pm2_5: float
    pm10: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None


class BatchPredictRequest(BaseModel):
    measurements: List[BatchMeasurement]
    n_forecast_hours: int = Field(default=24, ge=1, le=30)
    device_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    actual_pm2_5: float
    device_id: Optional[str] = None


app = FastAPI(title="AtmosCare Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mongo_client = MongoClient(DATABASE_URI, serverSelectionTimeoutMS=10000)
db = mongo_client[DATABASE_NAME]
iot_data: Collection = db["iot_data"]
predictions: Collection = db["predictions"]
devices: Collection = db["devices"]

model_service = AirQualityInferenceService()
MODEL_STATUS = model_service.load()
unified_service = init_unified_service(
    model_service=model_service,
    pakistan_aqi_url=PAKISTAN_AQI_URL,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _event_timestamp(event: Dict[str, Any]) -> Optional[datetime]:
    ts = event.get("timestamp")
    if ts is None:
        return None
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _is_test_device(device_id: Optional[str]) -> bool:
    if not device_id:
        return False
    lower = str(device_id).lower()
    if lower in NON_SENSOR_DEVICES:
        return True
    return any(lower.startswith(prefix) for prefix in TEST_DEVICE_PREFIXES)


def _should_use_api_instead_of_iot(device_id: str, event: Dict[str, Any]) -> Optional[str]:
    """Return a reason string when live API should replace stale/test IoT input."""
    try:
        doc = devices.find_one({"device_id": device_id}) or {}
        if doc.get("admin_disabled"):
            return "admin_disabled"
        if doc.get("force_api"):
            return "force_api"
        if doc.get("marked_test"):
            return "marked_test"
    except Exception:
        pass

    if _is_test_device(device_id):
        return "test_device"

    source = str(event.get("source", "")).lower()
    if source in ("synthetic_fallback", "api_fallback"):
        return source

    ts = _event_timestamp(event)
    if ts is None:
        return "missing_timestamp"

    age = _utcnow() - ts
    if age > timedelta(minutes=IOT_STALE_MINUTES):
        return "stale_reading"

    recent_ingest = iot_data.count_documents(
        {
            "device_id": device_id,
            "ingested_at": {"$gte": _utcnow() - timedelta(minutes=IOT_STALE_MINUTES)},
        }
    ) > 0
    if not recent_ingest and age > timedelta(minutes=5):
        return "no_recent_ingest"

    return None


def _get_fresh_iot_device() -> Optional[str]:
    for doc in devices.find({"buffer.0": {"$exists": True}}).sort(
        [("last_prediction_at", DESCENDING), ("updated_at", DESCENDING)]
    ):
        device_id = doc.get("device_id")
        if doc.get("admin_disabled"):
            continue
        buffer = doc.get("buffer") or []
        if not device_id or not buffer:
            continue
        if _should_use_api_instead_of_iot(device_id, buffer[-1]) is None:
            return device_id
    return None


def _build_pollutant_sources(
    event: Dict[str, Any],
    *,
    default: str = "sensor",
) -> Dict[str, str]:
    estimated = set(event.get("_estimated_pollutants") or [])
    sources: Dict[str, str] = {}
    for key, aliases in (
        ("pm2_5", ("pm2_5", "pm25", "gas_level")),
        ("pm10", ("pm10",)),
        ("o3", ("o3",)),
        ("no2", ("no2",)),
        ("co", ("co",)),
    ):
        if key in estimated:
            sources[key] = "estimated"
        elif any(event.get(alias) is not None for alias in aliases):
            sources[key] = default
        else:
            sources[key] = "missing"
    return sources


def _normalize_iot_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    timestamp = payload.get("timestamp") or _utcnow()
    if not isinstance(timestamp, datetime):
        timestamp = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    return {
        "device_id": payload["device_id"],
        "timestamp": timestamp.astimezone(timezone.utc),
        "temperature": float(payload["temperature"]),
        "humidity": float(payload["humidity"]),
        "gas_level": float(payload.get("gas_level") or payload.get("pm2_5") or 0.0),
        "pm10": float(payload.get("pm10") or 0.0),
        "wind_speed": float(payload["wind_speed"]),
        "source": payload.get("source", "iot"),
    }


def _append_buffer(device_id: str, event: Dict[str, Any]) -> List[Dict[str, Any]]:
    doc = devices.find_one({"device_id": device_id}) or {"device_id": device_id, "buffer": []}
    buffer = doc.get("buffer", [])
    buffer.append(event)
    buffer = buffer[-BUFFER_SIZE:]
    devices.update_one(
        {"device_id": device_id},
        {
            "$set": {
                "device_id": device_id,
                "buffer": buffer,
                "updated_at": _utcnow(),
                "status": doc.get("status", "active"),
                "location": doc.get("location"),
            },
            "$setOnInsert": {
                "created_at": _utcnow(),
            },
        },
        upsert=True,
    )
    return buffer


def _store_raw_iot(event: Dict[str, Any]) -> str:
    timestamp = event["timestamp"]
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    elif not isinstance(timestamp, datetime):
        timestamp = _utcnow()
    result = iot_data.insert_one(
        {
            **event,
            "timestamp": timestamp,
            "timestamp_iso": timestamp.isoformat(),
            "ingested_at": _utcnow(),
        }
    )
    return str(result.inserted_id)


def _real_fallback_event(device_id: str) -> Optional[Dict[str, Any]]:
    location_data = get_detailed_location() or {}
    location = location_data.get("location")
    try:
        real_data = get_air_quality_data(location) if location else get_air_quality_data()
    except Exception:
        real_data = None

    if not real_data:
        return None

    pm25 = real_data.get("pm25")
    pm10 = real_data.get("pm10")
    temperature = real_data.get("temperature")
    humidity = real_data.get("humidity")
    wind_speed = real_data.get("wind_speed")

    return {
        "device_id": device_id,
        "timestamp": _utcnow(),
        "temperature": float(temperature) if temperature is not None else 25.0,
        "humidity": float(humidity) if humidity is not None else 45.0,
        "gas_level": float(pm25 if pm25 is not None else real_data.get("aqi", 0.0)),
        "pm2_5": float(pm25) if pm25 is not None else None,
        "pm10": float(pm10 if pm10 is not None else (pm25 or 0.0) * 1.15),
        "wind_speed": float(wind_speed) if wind_speed is not None else 2.0,
        "o3": real_data.get("o3"),
        "no2": real_data.get("no2"),
        "co": real_data.get("co"),
        "source": "api_fallback",
        "location": real_data.get("location") or location or "Live Location",
    }


def _enrich_event_pollutants(event: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure IoT/API events expose o3, no2, and co for the dashboard."""
    enriched = dict(event)
    estimated: List[str] = list(enriched.get("_estimated_pollutants") or [])
    if not any(enriched.get(key) is None for key in ("o3", "no2", "co")):
        enriched["_estimated_pollutants"] = estimated
        return enriched

    pm25 = enriched.get("pm2_5")
    if pm25 is None:
        pm25 = enriched.get("pm25")
    if pm25 is None:
        pm25 = enriched.get("gas_level")

    if pm25 is not None:
        estimates = estimate_secondary_pollutants(pm25)
        for key, value in estimates.items():
            if enriched.get(key) is None:
                enriched[key] = value
                estimated.append(key)
    enriched["_estimated_pollutants"] = estimated
    return enriched


def _enrich_event_readings(event: Dict[str, Any], city: Optional[str] = None) -> Dict[str, Any]:
    """Fill PM10 and weather when IoT/API events omit them."""
    location_data = get_detailed_location() or {}
    city_name = (
        city
        or event.get("city")
        or event.get("location")
        or location_data.get("location")
        or "Lahore"
    )
    return unified_service._enrich_api_payload(dict(event), str(city_name).split(",")[0].strip())


def _calculate_event_smog_index(event: Dict[str, Any]) -> Optional[int]:
    pm25 = event.get("pm2_5")
    if pm25 is None and event.get("pm25") is not None:
        pm25 = event.get("pm25")
    if pm25 is None:
        pm25 = event.get("gas_level")
    pm10 = event.get("pm10")
    o3 = event.get("o3")
    no2 = event.get("no2")
    co = event.get("co")
    if any(value is not None for value in [pm25, pm10, o3, no2, co]):
        return calculate_smog_index(pm25, pm10, o3, no2, co)
    return None


def _store_prediction(device_id: str, event: Dict[str, Any], result: Any) -> Dict[str, Any]:
    payload = {
        "prediction_id": str(uuid.uuid4()),
        "device_id": device_id,
        "timestamp": _utcnow(),
        "input": event,
        "prediction": result.prediction,
        "confidence": result.confidence,
        "gru_prediction": result.gru_prediction,
        "sarima_prediction": result.sarima_prediction,
        "sarima_forecast": result.sarima_forecast,
        "stacking_input": result.stacking_input,
        "model_status": result.model_status,
    }
    predictions.insert_one(payload)
    devices.update_one(
        {"device_id": device_id},
        {
            "$set": {
                "latest_prediction": payload,
                "last_prediction_at": _utcnow(),
                "status": "active",
            }
        },
        upsert=True,
    )
    return payload


def _resolve_input_source_label(raw_source: Optional[str]) -> tuple[str, str]:
    """Map raw source strings to (source, input_source) for the UI."""
    value = str(raw_source or "api").lower()
    if value in ("iot", "iot_sensor", "sensor"):
        return "iot", "iot_sensor"
    if value in ("hybrid", "iot_with_api_fallback"):
        return "hybrid", "iot_with_api_fallback"
    return "api", "external_api"


def _build_prediction_response(
    device_id: str,
    event: Dict[str, Any],
    result: Any,
    stored: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a response containing both prediction and raw sensor/air data."""
    event = _enrich_event_pollutants(event)
    event = _enrich_event_readings(event, city=event.get("location"))
    smog_index = event.get("smog_index")
    if smog_index is None:
        smog_index = _calculate_event_smog_index(event)

    model_fields = model_prediction_fields(result.prediction)
    health = result.health or get_smog_health_guidance(result.prediction)
    source, input_source = _resolve_input_source_label(event.get("source", "iot"))
    pollutant_sources = _build_pollutant_sources(event, default="sensor")
    response = {
        **model_fields,
        "confidence": result.confidence,
        "status": "success",
        "device_id": device_id,
        "prediction_id": stored["prediction_id"],
        "timestamp": _as_iso(stored["timestamp"]),
        "gru_prediction": result.gru_prediction,
        "sarima_prediction": result.sarima_prediction,
        "gru_forecast": result.gru_forecast,
        "sarima_forecast": result.sarima_forecast,
        "stacking_input": result.stacking_input,
        "model_status": result.model_status,
        "forecast_7d": result.forecast_7d,
        "smog_sources": result.smog_sources,
        "health": health,
        "health_recommendation": health.get("summary") if isinstance(health, dict) else None,
        "source": source,
        "input_source": input_source,
        "data_source": input_source,
        "reading_mode": "iot",
        "prediction_source": "model",
        "pollutant_sources": pollutant_sources,
        "model_engine": result.model_status.get("engine") if isinstance(result.model_status, dict) else None,
        "location": event.get("location"),
        "pm25": event.get("pm2_5") if event.get("pm2_5") is not None else event.get("pm25") if event.get("pm25") is not None else event.get("gas_level"),
        "pm2_5": event.get("pm2_5") if event.get("pm2_5") is not None else event.get("pm25") if event.get("pm25") is not None else event.get("gas_level"),
        "pm10": event.get("pm10"),
        "temperature": event.get("temperature"),
        "humidity": event.get("humidity"),
        "wind_speed": event.get("wind_speed"),
        "gas_level": event.get("gas_level"),
        "co": event.get("co"),
        "o3": event.get("o3"),
        "no2": event.get("no2"),
        "smog_index": smog_index,
    }
    return response


def _predict_from_device(
    device_id: str,
    fallback_event: Optional[Dict[str, Any]] = None,
    city: Optional[str] = None,
) -> Dict[str, Any]:
    doc = devices.find_one({"device_id": device_id}) or {}
    buffer = doc.get("buffer", [])
    if not buffer:
        fallback_event = fallback_event or _real_fallback_event(device_id) or model_service.synthetic_fallback(device_id=device_id)
        buffer = _append_buffer(device_id, fallback_event)
        _store_raw_iot(fallback_event)

    seed_event = fallback_event or buffer[-1]
    stale_reason = _should_use_api_instead_of_iot(device_id, seed_event)
    if stale_reason is not None:
        api_response = _predict_from_api(city)
        api_response["iot_device_skipped"] = device_id
        api_response["iot_fallback_reason"] = stale_reason
        return api_response

    result = model_service.predict(buffer=buffer, seed_event=seed_event)
    stored = _store_prediction(device_id, seed_event, result)
    return _build_prediction_response(device_id, seed_event, result, stored)


@app.on_event("startup")
def _startup() -> None:
    try:
        mongo_client.admin.command("ping")
    except Exception as exc:
        print(f"MongoDB unavailable; continuing in degraded mode: {exc}")
    try:
        from Backend.discovery_service import start_discovery_server, public_base_urls

        start_discovery_server()
        print("[Discovery] Reachable URLs:", ", ".join(public_base_urls()))
    except Exception as exc:
        print(f"[Discovery] not started: {exc}")


@app.get("/health")
def health() -> Dict[str, Any]:
    db_state = "ok"
    try:
        mongo_client.admin.command("ping")
    except Exception:
        db_state = "down"
    urls: List[str] = []
    try:
        from Backend.discovery_service import public_base_urls

        urls = public_base_urls()
    except Exception:
        pass
    return {
        "status": "healthy" if db_state == "ok" else "degraded",
        "database": db_state,
        "models": model_service.model_status(),
        "buffer_size": BUFFER_SIZE,
        "listen": "0.0.0.0",
        "urls": urls,
        "collections": {
            "iot_data": iot_data.estimated_document_count(),
            "predictions": predictions.estimated_document_count(),
            "devices": devices.estimated_document_count(),
        },
    }


@app.get("/mode")
def mode() -> Dict[str, Any]:
    fresh_device = _get_fresh_iot_device()
    has_device_buffers = devices.count_documents({"buffer.0": {"$exists": True}}) > 0
    recent_iot = iot_data.count_documents(
        {"ingested_at": {"$gte": _utcnow() - timedelta(minutes=IOT_STALE_MINUTES)}}
    ) > 0

    if fresh_device and recent_iot:
        current_mode = "iot"
        data_source = "iot_sensor"
    elif fresh_device:
        current_mode = "hybrid"
        data_source = "iot_with_api_fallback"
    else:
        current_mode = "api"
        data_source = "external_api"

    return {
        "mode": current_mode,
        "data_source": data_source,
        "active_device_id": fresh_device,
        "models_loaded": model_service.is_loaded,
        "model_engine": model_service.model_status().get("engine"),
    }


@app.post("/devices/register")
def register_device(payload: DeviceRegisterRequest) -> Dict[str, Any]:
    devices.update_one(
        {"device_id": payload.device_id},
        {
            "$set": {
                "device_id": payload.device_id,
                "location": payload.location,
                "status": payload.status,
                "updated_at": _utcnow(),
            },
            "$setOnInsert": {"buffer": [], "created_at": _utcnow()},
        },
        upsert=True,
    )
    return {"device_id": payload.device_id, "status": "registered"}


@app.post("/iot/predict")
def iot_predict(payload: IoTPredictRequest) -> Dict[str, Any]:
    event = _normalize_iot_payload(payload.model_dump())
    _store_raw_iot(event)
    buffer = _append_buffer(payload.device_id, event)
    result = model_service.predict(buffer=buffer, seed_event=event)
    stored = _store_prediction(payload.device_id, event, result)
    return _build_prediction_response(payload.device_id, event, result, stored)


# NOTE: no @app.get("/predict") here. This is an internal helper for the
# city/API-driven fallback path, called from predict() below. It previously
# had its own @app.get("/predict") decorator, which — since FastAPI matches
# routes in registration order — meant EVERY /predict request was silently
# routed here instead of to predict()/​_predict_from_device(), regardless of
# device_id. That's also why o3/no2/co were missing from every dashboard
# response: this function's return payload never included them.
def _predict_from_api(city: Optional[str] = None) -> Dict[str, Any]:
    location_data = get_detailed_location() or {}
    if city is None:
        city = location_data.get("location") or "Lahore"

    prediction = unified_service.get_aqi_prediction(city)
    if "error" in prediction and prediction.get("error") == "No data available":
        prediction = {
            **prediction,
            "prediction": prediction.get("prediction") or 95,
            "aqi": prediction.get("aqi") or 95,
            "pm2_5": prediction.get("pm2_5") or 42.0,
            "pm25": prediction.get("pm25") or 42.0,
            "pm10": prediction.get("pm10") or 70.0,
            "o3": prediction.get("o3") or 45.0,
            "no2": prediction.get("no2") or 28.0,
            "co": prediction.get("co") or 0.7,
            "source": prediction.get("source") or "fallback_default",
            "aqi_category": prediction.get("aqi_category") or "Moderate",
            "confidence": prediction.get("confidence") or 0.55,
            "smog_index": prediction.get("smog_index") or 55,
        }

    # When API data is available but unified service did not run the model,
    # run inference here so the dashboard always shows model-driven AQI.
    if prediction.get("source") != DataSource.MODEL_PRIMARY.value and model_service.is_loaded:
        try:
            model_result = model_service.predict_from_measurements(
                pm2_5=float(prediction.get("pm2_5") or prediction.get("pm25") or 42.0),
                pm10=float(prediction.get("pm10") or 70.0),
                temperature=prediction.get("temperature"),
                humidity=prediction.get("humidity"),
                wind_speed=prediction.get("wind_speed"),
            )
            prediction.update(model_prediction_fields(model_result.prediction))
            prediction["confidence"] = model_result.confidence
            prediction["gru_prediction"] = model_result.gru_prediction
            prediction["sarima_prediction"] = model_result.sarima_prediction
            prediction["sarima_forecast"] = model_result.sarima_forecast
            prediction["gru_forecast"] = model_result.gru_forecast
            prediction["forecast_7d"] = model_result.forecast_7d
            prediction["smog_sources"] = model_result.smog_sources
            prediction["health"] = model_result.health or get_smog_health_guidance(model_result.prediction)
            prediction["health_recommendation"] = prediction["health"].get("summary")
            prediction["model_status"] = model_result.model_status
            prediction["model_engine"] = model_result.model_status.get("engine")
            prediction["input_source"] = "external_api"
            prediction["source"] = "api"
        except Exception as exc:
            print(f"API model inference fallback failed: {exc}")

    api_source, api_input = _resolve_input_source_label(prediction.get("source", "api"))
    api_origin = prediction.get("api_source") or prediction.get("source") or "external_api"
    pollutant_sources = {
        "pm2_5": "api",
        "pm10": "api" if prediction.get("pm10") is not None else "estimated",
        "o3": "api" if prediction.get("o3") is not None else "estimated",
        "no2": "api" if prediction.get("no2") is not None else "estimated",
        "co": "api" if prediction.get("co") is not None else "estimated",
    }
    weather_fields = ("temperature", "humidity", "wind_speed")
    if any(prediction.get(key) is None for key in weather_fields):
        location_data = get_detailed_location() or {}
        enrich_city = city or prediction.get("city") or location_data.get("location") or "Lahore"
        prediction = unified_service._enrich_api_payload(prediction, str(enrich_city).split(",")[0].strip())
    return {
        "status": "success" if "error" not in prediction else "error",
        "prediction": prediction.get("prediction", prediction.get("aqi")),
        "aqi": prediction.get("aqi"),
        "aqi_derived": prediction.get("aqi_derived", prediction.get("aqi")),
        "confidence": prediction.get("confidence", 0.0),
        "timestamp": prediction.get("timestamp", _utcnow().isoformat() + "Z"),
        "device_id": "api-fallback",
        "location": prediction.get("city"),
        "source": api_source,
        "input_source": api_input,
        "data_source": api_input,
        "reading_mode": "api",
        "prediction_source": "model",
        "api_origin": api_origin,
        "pollutant_sources": pollutant_sources,
        "model_engine": prediction.get("model_engine") or (
            prediction.get("model_status", {}).get("engine")
            if isinstance(prediction.get("model_status"), dict) else None
        ),
        "pm25": prediction.get("pm2_5"),
        "pm2_5": prediction.get("pm2_5"),
        "pm10": prediction.get("pm10"),
        "o3": prediction.get("o3"),
        "no2": prediction.get("no2"),
        "co": prediction.get("co"),
        "temperature": prediction.get("temperature"),
        "humidity": prediction.get("humidity"),
        "wind_speed": prediction.get("wind_speed"),
        "smog_index": prediction.get("smog_index"),
        "health_recommendation": prediction.get("health_recommendation"),
        "forecast_7d": prediction.get("forecast_7d"),
        "smog_sources": prediction.get("smog_sources"),
        "health": prediction.get("health"),
        "predicted_pm2_5": prediction.get("predicted_pm2_5"),
        "prediction_type": prediction.get("prediction_type"),
        "gru_prediction": prediction.get("gru_prediction"),
        "sarima_prediction": prediction.get("sarima_prediction"),
        "model_status": prediction.get("model_status"),
    }


@app.get("/predict")
def predict(device_id: Optional[str] = Query(default=None), city: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    if device_id:
        return _predict_from_device(device_id, city=city)

    fresh_device = _get_fresh_iot_device()
    if fresh_device:
        return _predict_from_device(fresh_device, city=city)

    return _predict_from_api(city)


@app.post("/predict")
def batch_predict(payload: BatchPredictRequest) -> Dict[str, Any]:
    if not payload.measurements:
        raise HTTPException(status_code=400, detail="measurements cannot be empty")

    device_id = payload.device_id or "batch-device"
    measurements = [m.model_dump() for m in payload.measurements]
    last_prediction = None
    event = None
    for measurement in measurements:
        event = {
            "device_id": device_id,
            "timestamp": measurement.get("timestamp") or _utcnow(),
            "temperature": measurement.get("temperature") or 0.0,
            "humidity": measurement.get("humidity") or 0.0,
            "gas_level": measurement.get("pm2_5") or 0.0,
            "pm10": measurement.get("pm10") or 0.0,
            "wind_speed": measurement.get("wind_speed") or 0.0,
            "source": "batch",
        }
        _store_raw_iot(event)
        buffer = _append_buffer(device_id, event)
        last_prediction = model_service.predict(buffer=buffer, seed_event=event)

    if last_prediction is None:
        raise HTTPException(status_code=500, detail="prediction failed")

    stored = _store_prediction(device_id, event, last_prediction)
    return _build_prediction_response(device_id, event, last_prediction, stored)


@app.post("/predict_single")
def predict_single(payload: Dict[str, Any]) -> Dict[str, Any]:
    device_id = payload.get("device_id") or "single-device"
    event = {
        "device_id": device_id,
        "timestamp": payload.get("timestamp") or _utcnow(),
        "temperature": payload.get("temperature") or 0.0,
        "humidity": payload.get("humidity") or 0.0,
        "gas_level": payload.get("pm2_5") or payload.get("gas_level") or 0.0,
        "pm10": payload.get("pm10") or 0.0,
        "wind_speed": payload.get("wind_speed") or 0.0,
        "source": "predict_single",
    }
    _store_raw_iot(event)
    buffer = _append_buffer(device_id, event)
    result = model_service.predict(buffer=buffer, seed_event=event)
    stored = _store_prediction(device_id, event, result)
    return _build_prediction_response(device_id, event, result, stored)


_CITIES_ANALYTICS_TTL_SEC = 120
_cities_analytics_cache: Dict[str, Any] = {"expires_at": None, "payload": None}


def _build_city_analytics_row(city: str) -> Dict[str, Any]:
    try:
        data = unified_service.get_aqi_prediction(city, use_cache=True)
        predicted_pm25 = data.get("predicted_pm2_5") or data.get("prediction")
        smog_index = data.get("smog_index")
        if smog_index is None and predicted_pm25 is not None:
            smog_index = calculate_smog_index(
                data.get("pm2_5"),
                data.get("pm10"),
                data.get("o3"),
                data.get("no2"),
                data.get("co"),
            )
        derived_aqi = data.get("aqi_derived") or data.get("aqi")
        return {
            "city": city,
            "key": city.lower().replace(" ", "_"),
            "predicted_pm2_5": predicted_pm25,
            "smog_index": smog_index,
            "aqi": derived_aqi,
            "pm2_5": data.get("pm2_5"),
            "pm10": data.get("pm10"),
            "o3": data.get("o3"),
            "no2": data.get("no2"),
            "co": data.get("co"),
            "status": data.get("aqi_category") or (data.get("health") or {}).get("status"),
            "source": data.get("source"),
            "confidence": data.get("confidence"),
        }
    except Exception as exc:
        return {"city": city, "key": city.lower(), "error": str(exc)}


@app.get("/analytics/cities")
def analytics_cities() -> Dict[str, Any]:
    """Per-city model smog predictions for the locations screen."""
    now = _utcnow()
    cache = _cities_analytics_cache
    expires_at = cache.get("expires_at")
    if expires_at and expires_at > now and cache.get("payload"):
        return cache["payload"]

    cities = [
        "Lahore", "Karachi", "Islamabad", "Peshawar",
        "Quetta", "Multan", "Faisalabad", "Rawalpindi",
    ]
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_build_city_analytics_row, city) for city in cities]
        for future in as_completed(futures):
            results.append(future.result())

    order = {city: idx for idx, city in enumerate(cities)}
    results.sort(key=lambda row: order.get(row.get("city"), 999))

    payload = {"cities": results, "timestamp": now.isoformat()}
    _cities_analytics_cache["payload"] = payload
    _cities_analytics_cache["expires_at"] = now + timedelta(seconds=_CITIES_ANALYTICS_TTL_SEC)
    return payload


@app.get("/history/{device_id}")
def history(device_id: str, limit: int = Query(default=50, ge=1, le=500)) -> Dict[str, Any]:
    cursor = (
        predictions.find({"device_id": device_id})
        .sort("timestamp", DESCENDING)
        .limit(limit)
    )
    records = []
    for item in cursor:
        item["_id"] = str(item["_id"])
        item["timestamp"] = _as_iso(item["timestamp"])
        if isinstance(item.get("input", {}).get("timestamp"), datetime):
            item["input"]["timestamp"] = _as_iso(item["input"]["timestamp"])
        records.append(item)
    return {"device_id": device_id, "history": records}


@app.post("/feedback/{prediction_id}")
def feedback(prediction_id: str, payload: FeedbackRequest) -> Dict[str, Any]:
    result = predictions.update_one(
        {"prediction_id": prediction_id},
        {
            "$set": {
                "actual_pm2_5": payload.actual_pm2_5,
                "feedback_at": _utcnow(),
                "feedback_device_id": payload.device_id,
            }
        },
    )
    return {
        "status": "updated" if result.modified_count > 0 else "not_found",
        "prediction_id": prediction_id,
    }


@app.get("/broadcasts/active")
def active_broadcasts(city: str = Query(default="")) -> Dict[str, Any]:
    from Backend.admin_service import get_active_broadcasts
    return {"broadcasts": get_active_broadcasts(city)}


# ── Portable auth / settings API (used by desktop + Android APK) ─────────────


class SignupRequest(BaseModel):
    username: str = Field(..., min_length=1)
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=1)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=1)


class SettingsUpdateRequest(BaseModel):
    email: str = Field(..., min_length=3)
    name: str = "User"
    location: str = "Unknown Location"
    rain: bool = False
    snow: bool = False
    smog: bool = True


@app.post("/auth/signup")
def auth_signup(payload: SignupRequest) -> Dict[str, Any]:
    from Backend.auth_manager import handle_signup

    ok, message = handle_signup(payload.username.strip(), payload.email.strip().lower(), payload.password)
    return {"success": ok, "message": message}


@app.post("/auth/login")
def auth_login(payload: LoginRequest) -> Dict[str, Any]:
    from Backend.auth_manager import handle_login

    ok, message, email, role = handle_login(payload.email.strip().lower(), payload.password)
    return {
        "success": ok,
        "message": message,
        "email": email,
        "role": role or "user",
    }


@app.get("/auth/settings")
def auth_get_settings(email: str = Query(..., min_length=3)) -> Dict[str, Any]:
    from Backend.auth_manager import get_settings

    settings = get_settings(email.strip().lower()) or {}
    return {"success": True, "settings": settings}


@app.put("/auth/settings")
def auth_save_settings(payload: SettingsUpdateRequest) -> Dict[str, Any]:
    from Backend.auth_manager import save_settings

    ok = save_settings(
        payload.email.strip().lower(),
        payload.name,
        payload.location,
        payload.rain,
        payload.snow,
        payload.smog,
    )
    return {"success": bool(ok)}


@app.get("/weather/alerts")
def weather_alerts(location: str = Query(default="Lahore")) -> Dict[str, Any]:
    from Backend.open_meteo_service import get_weather_alert_status

    data = get_weather_alert_status(location) or {}
    return {"success": True, "weather": data}


@app.get("/admin/stats")
def admin_stats() -> Dict[str, Any]:
    from Backend.auth_manager import get_admin_stats

    return get_admin_stats()


@app.get("/admin/devices")
def admin_devices() -> Dict[str, Any]:
    from Backend.auth_manager import get_admin_devices

    return {"devices": get_admin_devices()}


@app.get("/admin/users")
def admin_users(query: str = Query(default="")) -> Dict[str, Any]:
    from Backend.auth_manager import search_users

    return {"users": search_users(query)}


@app.get("/admin/broadcasts")
def admin_broadcasts(limit: int = Query(default=15, ge=1, le=100)) -> Dict[str, Any]:
    from Backend.auth_manager import get_recent_broadcasts

    return {"broadcasts": get_recent_broadcasts(limit)}


@app.get("/admin/audit")
def admin_audit(limit: int = Query(default=50, ge=1, le=200)) -> Dict[str, Any]:
    from Backend.auth_manager import get_audit_log

    return {"logs": get_audit_log(limit)}


class BroadcastCreateRequest(BaseModel):
    actor_email: str
    actor_role: str = "admin"
    city: str = "*"
    title: str
    message: str


@app.post("/admin/broadcast")
def admin_create_broadcast(payload: BroadcastCreateRequest) -> Dict[str, Any]:
    from Backend.auth_manager import create_broadcast

    ok, message = create_broadcast(
        payload.actor_email, payload.actor_role, payload.city, payload.title, payload.message
    )
    return {"success": ok, "message": message}


class DeviceFlagsRequest(BaseModel):
    device_id: str
    actor_email: str
    actor_role: str = "admin"
    admin_disabled: Optional[bool] = None
    force_api: Optional[bool] = None
    marked_test: Optional[bool] = None


@app.post("/admin/device-flags")
def admin_device_flags(payload: DeviceFlagsRequest) -> Dict[str, Any]:
    from Backend.auth_manager import update_device_flags

    flags = {}
    if payload.admin_disabled is not None:
        flags["admin_disabled"] = payload.admin_disabled
    if payload.force_api is not None:
        flags["force_api"] = payload.force_api
    if payload.marked_test is not None:
        flags["marked_test"] = payload.marked_test
    ok, message = update_device_flags(
        payload.device_id, payload.actor_email, payload.actor_role, **flags
    )
    return {"success": ok, "message": message}


class RoleChangeRequest(BaseModel):
    actor_email: str
    actor_role: str = "admin"
    target_email: str
    new_role: str


@app.post("/admin/change-role")
def admin_change_role(payload: RoleChangeRequest) -> Dict[str, Any]:
    from Backend.auth_manager import change_user_role_safe

    ok, message = change_user_role_safe(
        payload.actor_email, payload.actor_role, payload.target_email, payload.new_role
    )
    return {"success": ok, "message": message}


class DeleteUserRequest(BaseModel):
    actor_email: str
    actor_role: str = "admin"
    actor_password: str
    target_email: str


@app.post("/admin/delete-user")
def admin_delete_user(payload: DeleteUserRequest) -> Dict[str, Any]:
    from Backend.auth_manager import remove_user_safe

    ok, message = remove_user_safe(
        payload.actor_email,
        payload.actor_role,
        payload.actor_password,
        payload.target_email,
    )
    return {"success": ok, "message": message}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("Backend.main:app", host="0.0.0.0", port=port, reload=False)
