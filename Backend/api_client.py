# ============================================================================
# KIVY FRONTEND API CLIENT
# ============================================================================
"""
REST API client for Kivy frontend to communicate with FastAPI backend.
Handles:
- Real-time predictions
- History fetching
- Device management
- IoT fallback support
- Error handling & retries
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionMode(Enum):
    """System operation modes."""
    IOT = "iot"          # Real-time IoT device data
    API = "api"          # WAQI or external API fallback
    HYBRID = "hybrid"    # Both available


class APIClient:
    """
    Kivy Frontend API Client.
    
    Handles all communication with FastAPI backend:
    - /predict - single prediction from measurements
    - /history/{device_id} - fetch prediction history
    - /iot/predict - IoT device predictions
    - /health - system health check
    - /mode - get current operation mode
    """
    
    def __init__(self, 
                 base_url: str = "http://127.0.0.1:8000",
                 device_id: Optional[str] = None,
                 api_key: Optional[str] = None,
                 timeout: int = 10):
        """
        Initialize API client.
        
        Args:
            base_url: Backend URL (e.g., http://127.0.0.1:8000)
            device_id: Device ID for IoT mode
            api_key: Optional API key for secured endpoints
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.device_id = device_id
        self.api_key = api_key
        self.timeout = timeout
        self.mode = PredictionMode.API
        self.mode_info: Dict[str, Any] = {"mode": "api", "data_source": "external_api"}
        self.last_error = None
        self.headers = self._build_headers()
        
        logger.info(f"APIClient initialized: {self.base_url}")
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers with API key if provided."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers
    
    # ─────────────────────────────────────────────────────────────────────
    # HEALTH & STATUS
    # ─────────────────────────────────────────────────────────────────────
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check backend health.
        
        Returns:
            {"status": "healthy", "models_available": {...}}
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            self.last_error = None
            return data
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Health check failed: {e}")
            return {"status": "unavailable", "error": str(e)}
    
    def get_mode(self) -> str:
        """
        Get current system mode (iot, api, or hybrid).
        
        Returns:
            "iot" | "api" | "hybrid"
        """
        try:
            response = requests.get(
                f"{self.base_url}/mode",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            mode_str = data.get("mode", "api")
            self.mode = PredictionMode(mode_str)
            self.mode_info = data
            self.last_error = None
            return mode_str
        except Exception as e:
            self.last_error = str(e)
            logger.warning(f"Could not get mode, defaulting to API: {e}")
            self.mode = PredictionMode.API
            return "api"
    
    # ─────────────────────────────────────────────────────────────────────
    # PREDICTIONS
    # ─────────────────────────────────────────────────────────────────────
    
    def predict_single(self, 
                      pm2_5: float,
                      pm10: Optional[float] = None,
                      temperature: Optional[float] = None,
                      humidity: Optional[float] = None,
                      wind_speed: Optional[float] = None,
                      wind_dir: Optional[float] = None,
                      pressure: Optional[float] = None) -> Dict[str, Any]:
        """
        Get single prediction from current measurement.
        
        Args:
            pm2_5: PM2.5 concentration (µg/m³)
            pm10: PM10 concentration (optional)
            temperature: Temperature in °C (optional)
            humidity: Humidity in % (optional)
            wind_speed: Wind speed in m/s (optional)
            wind_dir: Wind direction in degrees (optional)
            pressure: Pressure in hPa (optional)
        
        Returns:
            {
                "timestamp": "...",
                "prediction": float,
                "confidence": float
            }
        """
        try:
            payload = {
                "timestamp": datetime.utcnow().isoformat(),
                "device_id": self.device_id or "frontend-device",
                "pm2_5": pm2_5,
                "pm10": pm10,
                "temperature": temperature,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "wind_dir": wind_dir,
                "pressure": pressure,
                "gas_level": pm2_5,
            }
            
            response = requests.post(
                f"{self.base_url}/iot/predict",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            self.last_error = None
            return {
                "prediction": data.get("prediction"),
                "confidence": data.get("confidence", 0.0),
                "timestamp": data.get("timestamp"),
                "status": data.get("status", "success"),
                "device_id": data.get("device_id"),
            }
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Single prediction failed: {e}")
            return {"error": str(e), "prediction": None, "confidence": 0.0}

    def get_live_prediction(
        self,
        device_id: Optional[str] = None,
        city: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch the backend's latest real prediction without fabricating inputs.

        Prefers fresh IoT when available; stale/test devices fall back to live API.
        """
        try:
            params: Dict[str, Any] = {}
            if city:
                params["city"] = str(city).split(",")[0].strip()
            dev_id = device_id or self.device_id
            if dev_id:
                params["device_id"] = dev_id

            response = requests.get(
                f"{self.base_url}/predict",
                params=params or None,
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            self.last_error = None
            return data
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Live prediction fetch failed: {e}")
            return {"error": str(e), "prediction": None, "confidence": 0.0}
    
    def predict_batch(self,
                     measurements: List[Dict[str, float]],
                     n_forecast_hours: int = 24,
                     device_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get batch predictions from multiple measurements.
        
        Args:
            measurements: List of measurement dicts with keys:
                          timestamp, pm2_5, pm10, temperature, humidity, etc.
            n_forecast_hours: Forecast horizon (1-30)
            device_id: Device ID for logging
        
        Returns:
            {
                "predictions": {"gru": [...], "sarima": [...], "hybrid": [...], "stacking": [...]},
                "confidence": {...},
                "aqi": [...],
                "timestamp": "..."
            }
        """
        try:
            if n_forecast_hours < 1 or n_forecast_hours > 30:
                n_forecast_hours = min(30, max(1, n_forecast_hours))
            
            payload = {
                "measurements": measurements,
                "n_forecast_hours": n_forecast_hours,
                "device_id": device_id or self.device_id,
            }
            
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            self.last_error = None
            return data
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Batch prediction failed: {e}")
            return {"error": str(e)}
    
    # ─────────────────────────────────────────────────────────────────────
    # HISTORY & DEVICE
    # ─────────────────────────────────────────────────────────────────────
    
    def get_analytics_cities(self) -> Dict[str, Any]:
        """Fetch per-city model smog predictions."""
        try:
            response = requests.get(
                f"{self.base_url}/analytics/cities",
                headers=self.headers,
                timeout=max(self.timeout, 120),
            )
            response.raise_for_status()
            data = response.json()
            self.last_error = None
            return data
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"City analytics fetch failed: {e}")
            return {"error": str(e), "cities": []}

    def get_history(self, 
                   device_id: Optional[str] = None,
                   limit: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch prediction history for a device.
        
        Args:
            device_id: Device ID (uses self.device_id if None)
            limit: Max number of records to fetch
        
        Returns:
            List of prediction records with timestamps and predictions
        """
        try:
            dev_id = device_id or self.device_id
            if not dev_id:
                raise ValueError("device_id not provided")
            
            response = requests.get(
                f"{self.base_url}/history/{dev_id}",
                params={"limit": limit},
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            self.last_error = None
            return data.get("history", [])
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"History fetch failed: {e}")
            return []
    
    def register_device(self,
                       device_id: str,
                       location: Optional[str] = None) -> Dict[str, Any]:
        """
        Register IoT device with backend.
        
        Args:
            device_id: Unique device identifier
            location: Device location (optional)
        
        Returns:
            {"device_id": ..., "status": "registered"}
        """
        try:
            payload = {
                "device_id": device_id,
                "location": location,
                "status": "active",
            }
            
            response = requests.post(
                f"{self.base_url}/devices/register",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            self.device_id = device_id
            self.last_error = None
            return data
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Device registration failed: {e}")
            return {"error": str(e)}
    
    # ─────────────────────────────────────────────────────────────────────
    # IOT DEVICE SUPPORT
    # ─────────────────────────────────────────────────────────────────────
    
    def iot_predict(self,
                   device_id: str,
                   temperature: float,
                   humidity: float,
                   gas_level: float,
                   wind_speed: Optional[float] = None,
                   location: Optional[str] = None) -> Dict[str, Any]:
        """
        Get prediction from IoT device sensors (direct).
        
        Args:
            device_id: IoT device ID
            temperature: Temperature in °C
            humidity: Humidity in %
            gas_level: Gas level (maps to PM2.5)
            wind_speed: Wind speed in m/s (optional)
            location: Device location (optional)
        
        Returns:
            {
                "device_id": "...",
                "prediction": float,
                "confidence": float,
                "status": "success"
            }
        """
        try:
            payload = {
                "device_id": device_id,
                "timestamp": datetime.utcnow().isoformat(),
                "temperature": temperature,
                "humidity": humidity,
                "gas_level": gas_level,
                "wind_speed": wind_speed,
                "location": location,
            }
            
            response = requests.post(
                f"{self.base_url}/iot/predict",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            self.last_error = None
            return data
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"IoT prediction failed: {e}")
            return {
                "error": str(e),
                "device_id": device_id,
                "prediction": None,
                "confidence": 0.0,
                "status": "failed"
            }
    
    def send_feedback(self,
                     prediction_id: str,
                     actual_pm2_5: float,
                     device_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Send actual PM2.5 value for model improvement.
        
        Args:
            prediction_id: ID of the prediction to feedback on
            actual_pm2_5: Actual measured PM2.5 value
            device_id: Device ID (optional)
        
        Returns:
            {"status": "updated"}
        """
        try:
            payload = {
                "actual_pm2_5": actual_pm2_5,
                "device_id": device_id or self.device_id,
            }
            
            response = requests.post(
                f"{self.base_url}/feedback/{prediction_id}",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            self.last_error = None
            return data
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Feedback submission failed: {e}")
            return {"error": str(e)}
    
    # ─────────────────────────────────────────────────────────────────────
    # UTILITY METHODS
    # ─────────────────────────────────────────────────────────────────────
    
    def is_connected(self) -> bool:
        """Quick check if backend is reachable."""
        try:
            health = self.health_check()
            return health.get("status") != "unavailable"
        except:
            return False
    
    def get_last_error(self) -> Optional[str]:
        """Get last error message."""
        return self.last_error
    
    def clear_error(self):
        """Clear last error."""
        self.last_error = None


# ─────────────────────────────────────────────────────────────────────────
# GLOBAL CLIENT INSTANCE
# ─────────────────────────────────────────────────────────────────────────

# Initialize global client (can be customized in main app)
api_client = None


def init_api_client(base_url: str = "http://127.0.0.1:8000",
                   device_id: Optional[str] = None,
                   api_key: Optional[str] = None) -> APIClient:
    """Initialize global API client."""
    global api_client
    api_client = APIClient(base_url, device_id, api_key)
    return api_client


def get_api_client() -> APIClient:
    """Get global API client."""
    global api_client
    if api_client is None:
        api_client = APIClient()
    return api_client
