# ============================================================================
# FASTAPI INFERENCE SERVICE
# ============================================================================
"""Production-grade REST API for PM2.5 predictions."""

from datetime import datetime
import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from ..services import RealtimeForecastSystem
from ..utils import get_logger, load_config

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────

class MeasurementInput(BaseModel):
    """Single measurement input."""
    timestamp: datetime
    pm2_5: float = Field(..., ge=0, description="PM2.5 concentration (µg/m³)")
    pm10: Optional[float] = Field(None, ge=0, description="PM10 concentration")
    temperature: Optional[float] = Field(None, description="Temperature (°C)")
    humidity: Optional[float] = Field(None, ge=0, le=100, description="Humidity (%)")
    wind_speed: Optional[float] = Field(None, ge=0, description="Wind speed (m/s)")
    wind_dir: Optional[float] = Field(None, ge=0, le=360, description="Wind direction (°)")
    pressure: Optional[float] = Field(None, description="Pressure (hPa)")
    
    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2024-01-15T10:00:00",
                "pm2_5": 85.5,
                "pm10": 120.3,
                "temperature": 15.2,
                "humidity": 65,
                "wind_speed": 3.5,
                "wind_dir": 180,
                "pressure": 1013.0,
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response."""
    timestamp: datetime
    forecast_hours: int
    predictions: Dict[str, List[float]]  # model_name -> [values]
    confidence: Optional[Dict[str, List[float]]] = None
    aqi: Optional[List[str]] = None  # AQI categories
    device_id: Optional[str] = None


class IoTPredictionResponse(BaseModel):
    timestamp: datetime
    device_id: str
    prediction: float
    confidence: float
    status: str


class IoTInput(BaseModel):
    device_id: str
    timestamp: datetime
    temperature: float
    humidity: float
    gas_level: float
    wind_speed: Optional[float] = None
    location: Optional[str] = None
    actual_pm2_5: Optional[float] = None


class DeviceRegistration(BaseModel):
    device_id: str
    location: Optional[str] = None
    status: Optional[str] = "active"


class FeedbackInput(BaseModel):
    actual_pm2_5: float = Field(..., ge=0)
    device_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    models_available: Dict[str, bool]


class ModeResponse(BaseModel):
    """System operation mode response."""
    mode: str  # "iot" | "api" | "hybrid"
    timestamp: datetime


# ─────────────────────────────────────────────────────────────────────────
# API INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────

def create_app(config_path: str = "ml_pipeline/config/config.yaml",
               model_run_dir: Optional[str] = None) -> FastAPI:
    """
    Create and configure FastAPI app.
    
    Args:
        config_path: Path to config file
        model_run_dir: Directory with trained models
        
    Returns:
        FastAPI application
    """
    # Load configuration
    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        config = {}
    
    # Initialize the real-time system
    if model_run_dir is None:
        model_run_dir = config.get("output", {}).get("model_dir", "artifacts/models")
    
    try:
        realtime = RealtimeForecastSystem(config, model_run_dir)
    except Exception as e:
        logger.error(f"Failed to initialize real-time system: {e}")
        realtime = None
    
    # Create app
    app = FastAPI(
        title=config.get("api", {}).get("title", "Smog Prediction API"),
        version=config.get("api", {}).get("version", "1.0.0"),
        description="PM2.5 forecasting service using GRU + SARIMA + stacking + MongoDB",
    )
    app.state.realtime = realtime
    app.state.config = config

    device_api_key = config.get("security", {}).get("device_api_key")
    api_key_header = config.get("security", {}).get("api_key_header", "X-API-Key")

    def _validate_device_id(device_id: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]{3,64}", device_id or ""):
            raise HTTPException(status_code=400, detail="Invalid device_id format")
        return device_id

    def _validate_api_key(x_api_key: Optional[str]) -> None:
        if device_api_key and x_api_key != device_api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
    
    # ─────────────────────────────────────────────────────────────────────
    # ROUTES
    # ─────────────────────────────────────────────────────────────────────
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Check API and models health."""
        mongo_health = realtime.mongo.health() if realtime else {"available": False}
        return HealthResponse(
            status="healthy" if realtime else "degraded",
            timestamp=datetime.utcnow(),
            models_available={
                "gru": bool(realtime and realtime.predictor and realtime.predictor.gru is not None),
                "sarima": bool(realtime and realtime.predictor and realtime.predictor.sarima is not None),
                "stacking": bool(realtime and realtime.stacking and realtime.stacking.is_fitted),
                "mongo": bool(mongo_health.get("available")),
            }
        )
    
    @app.get("/mode", response_model=ModeResponse)
    async def get_mode():
        """
        Get current system operation mode.
        
        Returns:
            {
                "mode": "iot" | "api" | "hybrid",
                "timestamp": "..."
            }
        """
        # Determine mode based on available data and configuration
        # If IoT devices are registered and actively sending data -> "iot"
        # If only API fallback is available -> "api"
        # If both are available -> "hybrid"
        
        try:
            has_iot_devices = False
            if realtime and realtime.mongo:
                # Check if any devices are registered and have recent data
                devices = realtime.mongo.find_active_devices(hours=24)
                has_iot_devices = len(devices) > 0 if devices else False
        except Exception as e:
            logger.warning(f"Could not check IoT devices: {e}")
            has_iot_devices = False
        
        # Determine mode based on configuration and available devices
        mode_config = config.get("system", {}).get("mode", "api")
        
        if has_iot_devices:
            mode = "iot" if mode_config == "iot" else "hybrid"
        else:
            mode = "api"
        
        return ModeResponse(
            mode=mode,
            timestamp=datetime.utcnow(),
        )
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(measurements: List[MeasurementInput],
                     n_forecast_hours: int = 30,
                     device_id: Optional[str] = None):
        """
        Make PM2.5 predictions.
        
        Args:
            measurements: Recent measurements (at least 48 hours for GRU)
            n_forecast_hours: Forecast horizon
            
        Returns:
            Predictions from available models
        """
        if realtime is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        if not measurements:
            raise HTTPException(status_code=400, detail="No measurements provided")
        
        if n_forecast_hours < 1 or n_forecast_hours > 30:
            raise HTTPException(status_code=400, 
                              detail="Forecast hours must be between 1 and 30")
        
        try:
            # Convert to DataFrame
            data = []
            for m in measurements:
                row = {
                    "datetime": m.timestamp,
                    "PM2.5": m.pm2_5,
                    "PM10": m.pm10,
                    "Temperature": m.temperature,
                    "Humidity": m.humidity,
                    "WindSpeed": m.wind_speed,
                    "WindDir": m.wind_dir,
                    "Pressure": m.pressure,
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            df = df.set_index("datetime").sort_index()

            payload = realtime.predict_from_sequence(
                df,
                n_steps=n_forecast_hours,
                device_id=device_id,
                store=True,
            )

            formatted_preds = {
                "gru": np.asarray(payload.get("gru_pred", []), dtype=float).tolist(),
                "sarima": np.asarray(payload.get("sarima_pred", []), dtype=float).tolist(),
                "hybrid": np.asarray(payload.get("hybrid_pred", []), dtype=float).tolist(),
                "stacking": np.asarray(payload.get("stacking_pred", []), dtype=float).tolist(),
            }
            confidence = {
                "stacking": np.asarray(payload.get("confidence", []), dtype=float).tolist(),
            }
            
            return PredictionResponse(
                timestamp=datetime.utcnow(),
                forecast_hours=n_forecast_hours,
                predictions=formatted_preds,
                confidence=confidence,
                device_id=device_id,
            )
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(f"/iot/predict", response_model=IoTPredictionResponse)
    async def iot_predict(
        sample: IoTInput,
        x_api_key: Optional[str] = Header(None, alias=api_key_header),
    ):
        if realtime is None:
            raise HTTPException(status_code=503, detail="Models not loaded")

        _validate_api_key(x_api_key)
        device_id = _validate_device_id(sample.device_id)

        try:
            payload = sample.model_dump()
            payload["device_id"] = device_id
            payload["timestamp"] = sample.timestamp
            payload["pm2_5"] = float(sample.gas_level)
            payload["pm10"] = float(sample.gas_level)

            result = realtime.ingest_iot(payload, api_device_id=device_id)
            stacking_pred = np.asarray(result.get("stacking_pred", []), dtype=float).ravel()
            confidence = np.asarray(result.get("confidence", []), dtype=float).ravel()
            return IoTPredictionResponse(
                timestamp=datetime.utcnow(),
                device_id=device_id,
                prediction=float(stacking_pred[0]) if len(stacking_pred) else float("nan"),
                confidence=float(confidence[0]) if len(confidence) else 0.0,
                status="success",
            )
        except Exception as e:
            logger.error(f"IoT prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/learn/{device_id}")
    async def learn(device_id: str, x_api_key: Optional[str] = Header(None, alias=api_key_header)):
        if realtime is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        _validate_api_key(x_api_key)
        device_id = _validate_device_id(device_id)
        result = realtime.continual_learning_step(device_id)
        return {"device_id": device_id, **result}

    @app.post("/feedback/{prediction_id}")
    async def feedback(
        prediction_id: str,
        payload: FeedbackInput,
        x_api_key: Optional[str] = Header(None, alias=api_key_header),
    ):
        if realtime is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        _validate_api_key(x_api_key)
        realtime.mongo.update_prediction_actual(prediction_id, payload.actual_pm2_5)
        learn_result = None
        if payload.device_id:
            device_id = _validate_device_id(payload.device_id)
            learn_result = realtime.continual_learning_step(device_id)
        return {
            "status": "updated",
            "prediction_id": prediction_id,
            "device_id": payload.device_id,
            "learning": learn_result,
        }

    @app.get("/history/{device_id}")
    async def history(device_id: str, limit: int = 100):
        if realtime is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        device_id = _validate_device_id(device_id)
        history_docs = realtime.history(device_id, limit=limit)
        for doc in history_docs:
            doc["_id"] = str(doc.get("_id"))
            if isinstance(doc.get("timestamp"), datetime):
                doc["timestamp"] = doc["timestamp"].isoformat()
        return {"device_id": device_id, "count": len(history_docs), "history": history_docs}

    @app.get("/predict")
    async def get_latest_prediction(device_id: Optional[str] = None):
        """Return the latest stored prediction for the UI dashboard."""
        if realtime is None:
            raise HTTPException(status_code=503, detail="Models not loaded")

        try:
            if device_id:
                device_id = _validate_device_id(device_id)
            latest = realtime.mongo.get_latest_prediction(device_id=device_id)
            if not latest:
                raise HTTPException(status_code=404, detail="No stored prediction available")

            latest["_id"] = str(latest.get("_id"))
            if isinstance(latest.get("timestamp"), datetime):
                latest["timestamp"] = latest["timestamp"].isoformat()
            if isinstance(latest.get("updated_at"), datetime):
                latest["updated_at"] = latest["updated_at"].isoformat()
            input_payload = latest.get("input")
            if isinstance(input_payload, dict) and isinstance(input_payload.get("timestamp"), datetime):
                input_payload["timestamp"] = input_payload["timestamp"].isoformat()
            return latest
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Latest prediction fetch failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/devices/register")
    async def register_device(
        payload: DeviceRegistration,
        x_api_key: Optional[str] = Header(None, alias=api_key_header),
    ):
        if realtime is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        _validate_api_key(x_api_key)
        device_id = _validate_device_id(payload.device_id)
        realtime.mongo.upsert_device(device_id=device_id, location=payload.location, status=payload.status or "active")
        return {"device_id": device_id, "status": "registered"}
    
    @app.post("/predict_single")
    async def predict_single(measurement: MeasurementInput):
        """Predict next hour from single measurement."""
        if realtime is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        try:
            # Use recent measurement as context
            df = pd.DataFrame([{
                "datetime": measurement.timestamp,
                "PM2.5": measurement.pm2_5,
                "PM10": measurement.pm10,
                "Temperature": measurement.temperature,
                "Humidity": measurement.humidity,
                "WindSpeed": measurement.wind_speed,
                "WindDir": measurement.wind_dir,
                "Pressure": measurement.pressure,
            }]).set_index("datetime")
            
            preds = realtime.predictor.predict_hourly(df)
            
            return {
                "timestamp": datetime.utcnow(),
                "next_hour_predictions": preds,
            }
        
        except Exception as e:
            logger.error(f"Single prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/info")
    async def info():
        """Get API information."""
        return {
            "title": config.get("api", {}).get("title", "Smog Prediction API"),
            "version": config.get("api", {}).get("version", "1.0.0"),
            "models": {
                "gru": "Seq2Seq GRU with attention and MC dropout",
                "sarima": "Seasonal ARIMA with exogenous variables",
                "hybrid": "Auto-weighted GRU and SARIMA hybrid",
                "stacking": "Meta-model trained on GRU and SARIMA predictions",
            },
            "data_requirements": {
                "minimum_history_hours": 48,
                "frequency": "hourly",
            },
            "forecast_range": "1-30 days ahead",
            "mongodb": config.get("mongodb", {}).get("database", "smog_system"),
        }
    
    logger.info("FastAPI app created successfully")
    return app


# ─────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    
    app = create_app()
    
    config = load_config("ml_pipeline/config/config.yaml")
    api_config = config.get("api", {})
    
    uvicorn.run(
        app,
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        workers=1,
    )
