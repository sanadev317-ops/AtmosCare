from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from .unified_data_service import UnifiedDataService, get_unified_service, init_unified_service


class PakistanAQIClient:
    """Compatibility wrapper used by the Kivy frontend.

    The older frontend code expects a PakistanAQIClient with methods like
    ``get_city_aqi`` and ``get_summary``.  This wrapper keeps that interface
    alive while delegating to the existing unified backend service.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:3000"):
        self.base_url = base_url.rstrip("/")
        self._service = get_unified_service()

    def get_city_aqi(self, city: str) -> Dict[str, Any]:
        result = self._service.get_aqi_prediction(city)
        if not result or "error" in result:
            return {
                "name": city,
                "aqi": 0,
                "category": "Unknown",
                "pm2_5": 0,
                "pm10": 0,
                "temperature": None,
                "humidity": None,
                "wind_speed": None,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "pollutants": {},
                "source": "fallback",
            }

        return {
            "name": result.get("city", city),
            "aqi": result.get("aqi", 0),
            "category": result.get("aqi_category", "Unknown"),
            "pm2_5": result.get("pm2_5"),
            "pm10": result.get("pm10"),
            "o3": result.get("o3"),
            "no2": result.get("no2"),
            "co": result.get("co"),
            "temperature": result.get("temperature"),
            "humidity": result.get("humidity"),
            "wind_speed": result.get("wind_speed"),
            "timestamp": result.get("timestamp", datetime.utcnow().isoformat() + "Z"),
            "pollutants": {
                "pm2_5": result.get("pm2_5"),
                "pm10": result.get("pm10"),
                "o3": result.get("o3"),
                "no2": result.get("no2"),
                "co": result.get("co"),
            },
            "source": result.get("source", "model_primary"),
            "confidence": result.get("confidence", 0.0),
            "health_recommendation": result.get("health_recommendation"),
            "forecast": result.get("forecast", []),
        }

    def get_summary(self) -> Dict[str, Any]:
        return {
            "success": True,
            "cities": [],
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

    def get_city_list(self) -> List[str]:
        return self._service.get_city_list()

    def get_pollution_summary(self) -> Dict[str, Any]:
        return self._service.get_pollution_summary()


_client: Optional[PakistanAQIClient] = None


def init_pakistan_aqi_client(base_url: str = "http://127.0.0.1:3000") -> PakistanAQIClient:
    global _client
    _client = PakistanAQIClient(base_url=base_url)
    init_unified_service()
    return _client


def get_pakistan_aqi_client() -> PakistanAQIClient:
    global _client
    if _client is None:
        _client = PakistanAQIClient()
    return _client
