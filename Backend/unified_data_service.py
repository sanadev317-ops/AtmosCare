# ============================================================================
# UNIFIED DATA SERVICE - MODEL-DRIVEN WITH API FALLBACK
# ============================================================================
"""
Unified data service for AtmosCare:
- PRIMARY: ML Models (GRU + SARIMA Stacking) for predictions
- SECONDARY: Pakistan AQI API for real-time data
- FALLBACK: WAQI API, AccuWeather API
- Ensures model-driven results with API data as supplementary

Usage:
    service = UnifiedDataService()
    result = service.get_aqi_prediction("Lahore")
    
Returns:
    {
        "city": "Lahore",
        "aqi": 185,
        "aqi_category": "Unhealthy",
        "pm2_5": 145.3,
        "source": "model_primary",
        "confidence": 0.97,
        "model_version": "stacking_v1",
        "timestamp": "2026-05-04T10:00:00Z",
        "forecast": [...],
        "health_recommendation": "..."
    }
"""

import requests
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from enum import Enum

from .air_quality_service import calculate_smog_index, estimate_secondary_pollutants, model_prediction_fields
from .analytics_service import get_smog_health_guidance

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Data source priority"""
    MODEL_PRIMARY = "model_primary"          # ML Model (GRU + SARIMA)
    PAKISTAN_AQI = "pakistan_aqi"            # Pakistan AQI API
    WAQI = "waqi"                            # World Air Quality Index
    ACCUWEATHER = "accuweather"              # AccuWeather (fallback only)
    CACHE = "cache"                          # Cached data


class UnifiedDataService:
    """
    Unified service prioritizing ML models with API fallback.
    """

    @staticmethod
    def _pollutant_missing(value: Any) -> bool:
        return value is None

    def _enrich_api_payload(self, payload: Dict[str, Any], city: str) -> Dict[str, Any]:
        """Fill missing pollutants, PM10, and weather from WAQI / AccuWeather."""
        if not payload:
            return payload

        enriched = dict(payload)
        city_name = (city or enriched.get("city") or "Lahore").split(",")[0].strip()

        needs_supplement = any(
            self._pollutant_missing(enriched.get(key))
            for key in (
                "o3", "no2", "co", "pm10",
                "temperature", "humidity", "wind_speed",
            )
        )
        if needs_supplement:
            waqi = self.get_waqi_data(city_name)
            if waqi:
                for key in (
                    "pm2_5", "pm25", "pm10", "o3", "no2", "co", "so2",
                    "temperature", "humidity", "wind_speed", "aqi",
                ):
                    if self._pollutant_missing(enriched.get(key)) and not self._pollutant_missing(waqi.get(key)):
                        enriched[key] = waqi[key]

        pm25 = enriched.get("pm2_5") or enriched.get("pm25")
        if self._pollutant_missing(enriched.get("pm10")) and pm25 is not None:
            enriched["pm10"] = round(float(pm25) * 1.15, 1)

        if pm25 is not None:
            estimates = estimate_secondary_pollutants(pm25)
            for key, value in estimates.items():
                if self._pollutant_missing(enriched.get(key)):
                    enriched[key] = value

        if any(
            self._pollutant_missing(enriched.get(key))
            for key in ("temperature", "humidity", "wind_speed")
        ):
            try:
                from Backend.accuweather_service import search_location, get_current_conditions

                loc = search_location(city_name)
                if loc and loc.get("location_key"):
                    conditions = get_current_conditions(loc["location_key"])
                    if conditions:
                        for key in ("temperature", "humidity", "wind_speed"):
                            if self._pollutant_missing(enriched.get(key)) and conditions.get(key) is not None:
                                enriched[key] = conditions[key]
            except Exception as exc:
                logger.debug(f"AccuWeather supplement failed for {city_name}: {exc}")

        return enriched

    def _enrich_missing_pollutants(self, payload: Dict[str, Any], city: str) -> Dict[str, Any]:
        """Backward-compatible alias."""
        return self._enrich_api_payload(payload, city)

    def _build_default_fallback_payload(self, city: str) -> Dict[str, Any]:
        """Return a lightweight fallback payload when all external AQI sources fail."""
        city_name = (city or "Unknown").split(",")[0].strip() or "Unknown"
        return {
            "city": city_name,
            "aqi": 95,
            "prediction": 95,
            "pm2_5": 42.0,
            "pm25": 42.0,
            "pm10": 70.0,
            "o3": 45.0,
            "no2": 28.0,
            "co": 0.7,
            "temperature": None,
            "humidity": None,
            "wind_speed": None,
            "source": "fallback_default",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "aqi_category": self._get_aqi_category(95),
            "confidence": 0.55,
        }
    
    def __init__(self,
                 model_service=None,
                 pakistan_aqi_url: str = "http://127.0.0.1:3000",
                 waqi_api_key: str = "ce02cc435eac36eb137107dff21e4ae914383457",
                 timeout: int = 10):
        """
        Initialize unified service.
        
        Args:
            model_service: ML model service instance
            pakistan_aqi_url: Pakistan AQI API URL
            waqi_api_key: WAQI API key
            timeout: Request timeout
        """
        self.model_service = model_service
        self.pakistan_aqi_url = pakistan_aqi_url.rstrip('/')
        self.waqi_api_key = waqi_api_key
        self.timeout = timeout
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        logger.info("UnifiedDataService initialized (model-driven approach)")
    
    # ─────────────────────────────────────────────────────────────────────
    # PRIMARY: ML MODEL PREDICTIONS
    # ─────────────────────────────────────────────────────────────────────
    
    def predict_aqi_from_model(self, 
                               pm2_5: float, 
                               pm10: float,
                               temperature: Optional[float] = None,
                               humidity: Optional[float] = None,
                               wind_speed: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate AQI prediction using ML model (GRU + SARIMA Stacking).
        
        Returns:
            {
                "aqi": 185,
                "aqi_category": "Unhealthy",
                "confidence": 0.97,
                "pm2_5": 145.3,
                "forecast_24h": [...],
                "model_version": "stacking_v1",
                "source": "model_primary"
            }
        """
        try:
            if not self.model_service:
                return None

            is_loaded = getattr(self.model_service, "is_loaded", False)
            if not is_loaded and hasattr(self.model_service, "load"):
                load_status = self.model_service.load()
                is_loaded = bool(load_status.get("gru") or load_status.get("sarima") or load_status.get("bridge"))

            if not is_loaded:
                return None

            if hasattr(self.model_service, "predict_from_measurements"):
                result = self.model_service.predict_from_measurements(
                    pm2_5=pm2_5,
                    pm10=pm10,
                    temperature=temperature,
                    humidity=humidity,
                    wind_speed=wind_speed,
                )
                predicted_pm25 = result.prediction
                confidence = result.confidence
                model_version = result.model_status.get("model_version", "stacking_v1")
                analytics_extra = {
                    "forecast_7d": result.forecast_7d,
                    "smog_sources": result.smog_sources,
                    "health": result.health,
                    "gru_forecast": result.gru_forecast,
                    "sarima_forecast": result.sarima_forecast,
                }
            else:
                predicted_pm25 = self.model_service.predict(pm2_5, pm10, temperature, wind_speed)
                confidence = self._compute_confidence(pm2_5, temperature, humidity)
                model_version = getattr(self.model_service, "model_version", "stacking_v1")
                analytics_extra = {}
            
            if predicted_pm25 is None:
                return None

            fields = model_prediction_fields(predicted_pm25)
            health = analytics_extra.get("health") or get_smog_health_guidance(predicted_pm25)
            return {
                **fields,
                **analytics_extra,
                "confidence": confidence,
                "pm2_5": pm2_5,
                "pm10": pm10,
                "temperature": temperature,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "source": DataSource.MODEL_PRIMARY.value,
                "model_version": model_version,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "health": health,
                "health_recommendation": health.get("summary"),
                "aqi_category": health.get("status") or fields.get("aqi_category"),
            }
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return None
    
    # ─────────────────────────────────────────────────────────────────────
    # SECONDARY: PAKISTAN AQI API
    # ─────────────────────────────────────────────────────────────────────
    
    def get_pakistan_aqi_data(self, city: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time data from Pakistan AQI API.
        
        Returns:
            {
                "city": "Lahore",
                "aqi": 185,
                "pm2_5": 145.3,
                "pm10": 250.2,
                "temperature": 28.5,
                "humidity": 65.0,
                "source": "pakistan_aqi"
            }
        """
        try:
            response = requests.get(
                f"{self.pakistan_aqi_url}/aqi/{city}",
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("success"):
                city_data = data.get("city", {})
                return {
                    "city": city,
                    "aqi": city_data.get("aqi", 0),
                    "pm2_5": city_data.get("pm2_5", 0),
                    "pm10": city_data.get("pm10", 0),
                    # FIX: these were never extracted, even when the upstream
                    # API provided them — silently dropping real-time
                    # pollutants before they ever reached get_aqi_prediction().
                    "o3": city_data.get("o3"),
                    "no2": city_data.get("no2"),
                    "so2": city_data.get("so2"),
                    "co": city_data.get("co"),
                    "temperature": city_data.get("temperature"),
                    "humidity": city_data.get("humidity"),
                    "wind_speed": city_data.get("wind_speed"),
                    "source": DataSource.PAKISTAN_AQI.value,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            return None
        except Exception as e:
            logger.warning(f"Pakistan AQI API failed: {e}")
            return None
    
    # ─────────────────────────────────────────────────────────────────────
    # FALLBACK: WAQI API
    # ─────────────────────────────────────────────────────────────────────
    
    def get_waqi_data(self, city: str) -> Optional[Dict[str, Any]]:
        """
        Get data from World Air Quality Index API (fallback).
        
        Returns:
            {
                "city": "Lahore",
                "aqi": 185,
                "pm2_5": 145.3,
                "source": "waqi"
            }
        """
        try:
            response = requests.get(
                f"https://api.waqi.info/feed/{city}/?token={self.waqi_api_key}",
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "ok":
                iaqi = data.get("data", {}).get("iaqi", {})
                return {
                    "city": city,
                    "aqi": data.get("data", {}).get("aqi", 0),
                    "pm2_5": iaqi.get("pm25", {}).get("v"),
                    "pm10": iaqi.get("pm10", {}).get("v"),
                    "o3": iaqi.get("o3", {}).get("v"),
                    "no2": iaqi.get("no2", {}).get("v"),
                    "so2": iaqi.get("so2", {}).get("v"),
                    "co": iaqi.get("co", {}).get("v"),
                    "temperature": iaqi.get("t", {}).get("v"),
                    "humidity": iaqi.get("h", {}).get("v"),
                    "wind_speed": iaqi.get("w", {}).get("v"),
                    "source": DataSource.WAQI.value,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            return None
        except Exception as e:
            logger.warning(f"WAQI API failed: {e}")
            return None
    
    # ─────────────────────────────────────────────────────────────────────
    # MAIN: UNIFIED AQI PREDICTION
    # ─────────────────────────────────────────────────────────────────────
    
    def get_aqi_prediction(self, city: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get unified AQI prediction for a city (MODEL-DRIVEN).
        
        Priority:
        1. ML Model prediction (using latest API data)
        2. Fall back to Pakistan AQI
        3. Fall back to WAQI
        
        Args:
            city: City name
            use_cache: Use cached data if available
        
        Returns:
            {
                "city": "Lahore",
                "aqi": 185,
                "aqi_category": "Unhealthy",
                "pm2_5": 145.3,
                "source": "model_primary",
                "confidence": 0.97,
                "health_recommendation": "...",
                "timestamp": "2026-05-04T10:00:00Z"
            }
        """
        # Check cache
        if use_cache and city in self.cache:
            cached = self.cache[city]
            try:
                cached_ts = cached.get("timestamp")
                if cached_ts:
                    parsed_ts = datetime.fromisoformat(str(cached_ts).replace("Z", "+00:00"))
                    if parsed_ts.tzinfo is None:
                        parsed_ts = parsed_ts.replace(tzinfo=datetime.utcnow().astimezone().tzinfo)
                    if parsed_ts > (datetime.now(parsed_ts.tzinfo) - timedelta(seconds=self.cache_ttl)):
                        logger.debug(f"Using cached data for {city}")
                        return cached
            except Exception as exc:
                logger.warning(f"Cache timestamp invalid for {city}: {exc}")
        
        # Step 1: Get real-time sensor data from APIs
        api_data = None
        
        # Try Pakistan AQI first
        api_data = self.get_pakistan_aqi_data(city)
        source_data = DataSource.PAKISTAN_AQI
        
        # Fallback to WAQI
        if api_data is None:
            api_data = self.get_waqi_data(city)
            source_data = DataSource.WAQI
        
        if api_data is None:
            logger.warning(f"All APIs failed for {city}; using built-in fallback payload")
            api_data = self._build_default_fallback_payload(city)
            source_data = DataSource.CACHE

        api_data = self._enrich_api_payload(api_data, city)
        
        # Step 2: Use ML model as PRIMARY prediction with API data as input
        model_result = self.predict_aqi_from_model(
            pm2_5=api_data.get("pm2_5", 0),
            pm10=api_data.get("pm10", 0),
            temperature=api_data.get("temperature"),
            humidity=api_data.get("humidity"),
            wind_speed=api_data.get("wind_speed")
        )
        
        # Step 3: Combine results (model-driven with API supplementary data)
        if model_result:
            # FIX: model_result only ever contains aqi/pm2_5/pm10/temperature/
            # humidity/wind_speed — it never computes o3/no2/co. The old code
            # did `{**model_result, ...}` and nothing else, which silently
            # discarded api_data's real-time o3/no2/co (and any other fields
            # the model doesn't know about) even though they were fetched
            # successfully. Merge api_data FIRST so its real-time readings are
            # the baseline, then layer the model's authoritative AQI/prediction
            # fields on top — this keeps real-time pollutants on the dashboard
            # while the AQI score itself still comes from the model.
            predicted_pm25 = model_result.get("predicted_pm2_5", model_result.get("prediction"))
            derived_aqi = model_result.get("aqi_derived", model_result.get("aqi"))
            health = model_result.get("health") or get_smog_health_guidance(float(predicted_pm25 or 0))
            result = {
                **api_data,
                **model_result,
                "prediction": predicted_pm25,
                "city": city,
                "api_source": api_data.get("source"),  # Original API source
                "raw_aqi": api_data.get("aqi"),        # Raw API AQI for comparison
                "health": health,
                "health_recommendation": health.get("summary") or self._get_health_recommendation(int(derived_aqi or 0)),
                "aqi_category": health.get("status") or model_result.get("aqi_category"),
                "forecast": model_result.get("forecast_7d") or self._generate_forecast(int(derived_aqi or 0)),
            }
        else:
            # Fallback: use API data directly
            result = {
                **api_data,
                "prediction": api_data.get("prediction", api_data.get("aqi", 0)),
                "city": city,
                "aqi_category": self._get_aqi_category(api_data.get("aqi", 0)),
                "source": api_data.get("source", DataSource.CACHE.value),
                "confidence": 0.70,  # Lower confidence without model
                "health_recommendation": self._get_health_recommendation(api_data.get("aqi", 0))
            }

        result = self._enrich_api_payload(result, city)
        result = self._ensure_smog_index(result)
        
        # Cache result
        self.cache[city] = result
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────────────────────────────
    
    def _ensure_smog_index(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("smog_index") is None:
            payload["smog_index"] = calculate_smog_index(
                payload.get("pm2_5", payload.get("pm25")),
                payload.get("pm10"),
                payload.get("o3"),
                payload.get("no2"),
                payload.get("co"),
            )
        return payload

    def _get_aqi_category(self, aqi: int) -> str:
        """Get AQI category from value."""
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"
    
    def _compute_confidence(self, pm2_5: float, temperature: Optional[float], humidity: Optional[float]) -> float:
        """
        Compute confidence score (0-1) for prediction.
        Based on data completeness and model certainty.
        """
        confidence = 0.7  # Base confidence
        
        if pm2_5 > 0:
            confidence += 0.15
        if temperature is not None:
            confidence += 0.08
        if humidity is not None:
            confidence += 0.07
        
        return min(0.99, confidence)
    
    def _get_health_recommendation(self, aqi: int) -> str:
        """Get health recommendation based on AQI."""
        if aqi <= 50:
            return "Good air quality. Enjoy outdoor activities!"
        elif aqi <= 100:
            return "Moderate air quality. Sensitive groups may experience mild symptoms."
        elif aqi <= 150:
            return "Unhealthy for sensitive groups. Limit prolonged outdoor activities."
        elif aqi <= 200:
            return "Unhealthy. General public may experience health effects. Minimize outdoor activities."
        elif aqi <= 300:
            return "Very unhealthy. Everyone should reduce outdoor exposure."
        else:
            return "Hazardous. Avoid outdoor activities. Stay indoors with air purifiers."
    
    def _generate_forecast(self, current_aqi: int) -> List[Dict[str, Any]]:
        """
        Generate forecast (stub for now).
        In production, integrate with forecast model.
        """
        return [
            {
                "hour": (datetime.utcnow() + timedelta(hours=i)).isoformat() + "Z",
                "aqi": int(current_aqi * (0.95 + (i % 3) * 0.03))
            }
            for i in range(1, 25)
        ]
    
    def get_city_list(self) -> List[str]:
        """Get list of available cities."""
        try:
            response = requests.get(
                f"{self.pakistan_aqi_url}/cities",
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("success"):
                return data.get("cities", [])
            return []
        except Exception as e:
            logger.error(f"Failed to get city list: {e}")
            return []
    
    def get_pollution_summary(self) -> Dict[str, Any]:
        """Get summary of pollution across all cities."""
        try:
            response = requests.get(
                f"{self.pakistan_aqi_url}/summary",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get summary: {e}")
            return {}


# Global singleton
_service = None


def get_unified_service(model_service=None) -> UnifiedDataService:
    """Get or create unified service instance."""
    global _service
    if _service is None:
        _service = UnifiedDataService(model_service=model_service)
    return _service


def init_unified_service(model_service=None, **kwargs) -> UnifiedDataService:
    """Initialize unified service with custom parameters."""
    global _service
    _service = UnifiedDataService(model_service=model_service, **kwargs)
    return _service
