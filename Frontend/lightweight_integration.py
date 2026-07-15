"""Android / portable IntegrationManager: API-only, no TensorFlow."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Optional

import requests
from kivy.clock import Clock

from app_config import get_backend_url

logger = logging.getLogger(__name__)


class LightweightIntegrationManager:
    """Polls FastAPI only — safe for APK builds."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        aqi_api_url: Optional[str] = None,
        device_id: Optional[str] = None,
        refresh_interval: int = 15,
    ):
        self.base_url = (base_url or get_backend_url()).rstrip("/")
        self.aqi_api_url = aqi_api_url
        self.device_id = device_id
        self.default_city = "Lahore"
        self.refresh_interval = refresh_interval
        self.is_running = False
        self.mode = "api"
        self.last_prediction: Optional[Dict[str, Any]] = None
        self.on_prediction_update: Optional[Callable[[Dict], None]] = None
        self.on_mode_change: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.api = self  # duck-type for callers using .api.get_mode / get_live_prediction

    def start_refresh(self):
        if self.is_running:
            return
        self.is_running = True
        self._schedule()

    def stop_refresh(self):
        self.is_running = False

    def _schedule(self):
        if self.is_running:
            Clock.schedule_once(lambda dt: self._refresh(), self.refresh_interval)

    def get_mode(self) -> str:
        try:
            data = requests.get(f"{self.base_url}/mode", timeout=8).json()
            mode = data.get("mode", "api") if isinstance(data, dict) else "api"
            self.mode = mode
            return mode
        except Exception:
            return self.mode or "api"

    def get_live_prediction(self, device_id: Optional[str] = None, city: Optional[str] = None) -> Optional[Dict]:
        params = {}
        if device_id or self.device_id:
            params["device_id"] = device_id or self.device_id
        if city or self.default_city:
            params["city"] = city or self.default_city
        try:
            resp = requests.get(f"{self.base_url}/predict", params=params, timeout=25)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("predict failed: %s", exc)
            return None

    def health_check(self) -> Dict[str, Any]:
        try:
            return requests.get(f"{self.base_url}/health", timeout=8).json()
        except Exception as exc:
            return {"status": "down", "error": str(exc)}

    def _refresh(self):
        if not self.is_running:
            return

        def work():
            try:
                mode = self.get_mode()
                if self.on_mode_change:
                    Clock.schedule_once(lambda dt: self.on_mode_change(mode), 0)
                city = self.default_city
                pred = self.get_live_prediction(city=city)
                if pred:
                    self.last_prediction = pred
                    if self.on_prediction_update:
                        Clock.schedule_once(lambda dt: self.on_prediction_update(pred), 0)
            except Exception as exc:
                if self.on_error:
                    Clock.schedule_once(lambda dt: self.on_error(str(exc)), 0)
            finally:
                Clock.schedule_once(lambda dt: self._schedule(), 0)

        threading.Thread(target=work, daemon=True).start()
