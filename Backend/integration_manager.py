# ============================================================================
# KIVY INTEGRATION MANAGER
# ============================================================================
"""
Manages integration between Kivy UI and FastAPI backend.
Handles auto-refresh, error recovery, and data display updates.
"""

import threading
import logging
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from kivy.clock import Clock

# Import services
from .api_client import APIClient, get_api_client, init_api_client
from .pakistan_aqi_client import PakistanAQIClient, get_pakistan_aqi_client, init_pakistan_aqi_client
from .unified_data_service import UnifiedDataService, get_unified_service, init_unified_service
from .ml_model_service import get_model_service

logger = logging.getLogger(__name__)


class IntegrationManager:
    """
    Manages Kivy↔API communication with auto-refresh and error handling.
    """
    
    def __init__(self, 
                 base_url: str = "http://127.0.0.1:8000",
                 aqi_api_url: str = "http://127.0.0.1:3000",
                 device_id: Optional[str] = None,
                 refresh_interval: int = 5):
        """
        Initialize integration manager.
        
        Args:
            base_url: Backend API URL
            aqi_api_url: Pakistan AQI API URL
            device_id: IoT device ID
            refresh_interval: Auto-refresh interval in seconds
        """
        self.base_url = base_url
        self.aqi_api_url = aqi_api_url
        
        # Initialize API clients
        self.api = init_api_client(base_url, device_id)
        self.aqi_client = init_pakistan_aqi_client(aqi_api_url)
        
        # Initialize ML model service
        self.model_service = get_model_service()
        
        # Initialize UNIFIED service (model-driven with API fallback)
        self.unified_service = init_unified_service(
            model_service=self.model_service,
            pakistan_aqi_url=aqi_api_url
        )
        
        self.device_id = device_id
        self.default_city = "Lahore"
        self.refresh_interval = refresh_interval
        
        # State tracking
        self.is_running = False
        self.mode = "api"
        self.mode_info: Dict[str, Any] = {"mode": "api", "data_source": "external_api"}
        self.last_prediction = None
        self.last_history = []
        self.retry_count = 0
        self.max_retries = 3
        
        # Callbacks for UI updates
        self.on_prediction_update: Optional[Callable[[Dict], None]] = None
        self.on_history_update: Optional[Callable[[list], None]] = None
        self.on_mode_change: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_connected: Optional[Callable[[], None]] = None
        self.on_disconnected: Optional[Callable[[], None]] = None
        
        logger.info(f"IntegrationManager initialized (unified model-driven approach, refresh: {refresh_interval}s)")
    
    # ─────────────────────────────────────────────────────────────────────
    # AUTO-REFRESH CONTROL
    # ─────────────────────────────────────────────────────────────────────
    
    def start_refresh(self):
        """Start auto-refresh loop."""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("Integration auto-refresh started")
        self._schedule_refresh()
    
    def stop_refresh(self):
        """Stop auto-refresh loop."""
        self.is_running = False
        logger.info("Integration auto-refresh stopped")
    
    def _schedule_refresh(self):
        """Schedule next refresh."""
        if self.is_running:
            Clock.schedule_once(
                lambda dt: self._refresh_cycle(),
                self.refresh_interval
            )
    
    def _refresh_cycle(self):
        """One refresh cycle: check mode, update predictions, handle errors."""
        if not self.is_running:
            return
        
        def work():
            try:
                # Check system health and mode
                health = self.api.health_check()
                # NOTE: get_mode() already returns the unwrapped mode string
                # (see APIClient.get_mode's docstring: Returns "iot"|"api"|"hybrid"),
                # not a dict. Calling .get() on it here was the source of the
                # "'str' object has no attribute 'get'" error on every refresh cycle.
                new_mode = self.api.get_mode()
                mode_info = getattr(self.api, "mode_info", None)
                if isinstance(mode_info, dict):
                    self.mode_info = mode_info
                
                if health.get("status") == "unavailable":
                    self._handle_error("Backend unavailable")
                    return
                
                # Check if mode changed
                if new_mode != self.mode:
                    self.mode = new_mode
                    if self.on_mode_change:
                        Clock.schedule_once(
                            lambda dt: self.on_mode_change(new_mode),
                            0
                        )
                    logger.info(f"Mode changed to: {new_mode}")
                
                # Fetch the latest backend prediction.
                # This keeps the frontend aligned with the real pipeline:
                # IoT/API -> MongoDB -> GRU/SARIMA -> stacking -> dashboard.
                try:
                    live_prediction = self.api.get_live_prediction(
                        self.device_id,
                        city=self.default_city,
                    )
                    if live_prediction and "error" not in live_prediction:
                        self._update_prediction(live_prediction)
                        logger.debug(
                            "Fetched live backend prediction: "
                            f"{live_prediction.get('prediction')}"
                        )
                    else:
                        logger.warning("No live prediction returned from backend")
                except Exception as e:
                    logger.error(f"Failed to fetch backend prediction: {e}")
                    self._handle_error(f"Prediction fetch error: {str(e)}")
                
                # Fetch full history if device_id is set
                if self.device_id:
                    history = self.api.get_history(self.device_id, limit=50)
                    self._update_history(history)
                
                # Success - reset retry counter
                self.retry_count = 0
                if self.on_connected:
                    Clock.schedule_once(lambda dt: self.on_connected(), 0)
                
            except Exception as e:
                self._handle_error(str(e))
            
            # Schedule next refresh
            self._schedule_refresh()
        
        # Run in background thread
        thread = threading.Thread(target=work, daemon=True)
        thread.start()
    
    # ─────────────────────────────────────────────────────────────────────
    # PREDICTION MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────
    
    def get_prediction(self,
                      pm2_5: float,
                      pm10: Optional[float] = None,
                      temperature: Optional[float] = None,
                      humidity: Optional[float] = None,
                      wind_speed: Optional[float] = None) -> Optional[Dict]:
        """
        Fetch single prediction (async).
        
        Args:
            pm2_5: PM2.5 value
            pm10: PM10 value (optional)
            temperature: Temperature (optional)
            humidity: Humidity (optional)
            wind_speed: Wind speed (optional)
        
        Returns:
            None - result is provided via on_prediction_update callback
        """
        def work():
            try:
                result = self.api.predict_single(
                    pm2_5=pm2_5,
                    pm10=pm10,
                    temperature=temperature,
                    humidity=humidity,
                    wind_speed=wind_speed,
                )
                
                if "error" not in result:
                    self._update_prediction(result)
                    self.retry_count = 0
                else:
                    self._handle_error(result.get("error"))
            except Exception as e:
                self._handle_error(str(e))
        
        thread = threading.Thread(target=work, daemon=True)
        thread.start()
    
    def _update_prediction(self, data: Dict):
        """Update prediction and notify UI."""
        self.last_prediction = data
        if self.on_prediction_update:
            Clock.schedule_once(
                lambda dt: self.on_prediction_update(data),
                0
            )
        logger.debug(f"Prediction updated: {data}")
    
    # ─────────────────────────────────────────────────────────────────────
    # HISTORY MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────
    
    def _update_history(self, history: list):
        """Update history and notify UI."""
        self.last_history = history
        if self.on_history_update:
            Clock.schedule_once(
                lambda dt: self.on_history_update(history),
                0
            )
        logger.debug(f"History updated: {len(history)} records")
    
    # ─────────────────────────────────────────────────────────────────────
    # ERROR HANDLING
    # ─────────────────────────────────────────────────────────────────────
    
    def _handle_error(self, error_msg: str):
        """Handle error with retry logic."""
        self.retry_count += 1
        
        if self.retry_count > self.max_retries:
            if self.on_disconnected:
                Clock.schedule_once(lambda dt: self.on_disconnected(), 0)
            logger.error(f"Max retries exceeded: {error_msg}")
        
        if self.on_error:
            Clock.schedule_once(
                lambda dt: self.on_error(f"Error: {error_msg} (retry {self.retry_count}/{self.max_retries})"),
                0
            )
        
        logger.warning(f"Integration error: {error_msg}")
    
    # ─────────────────────────────────────────────────────────────────────
    # DEVICE MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────
    
    def register_device(self, device_id: str, location: Optional[str] = None):
        """Register IoT device."""
        def work():
            try:
                result = self.api.register_device(device_id, location)
                self.device_id = device_id
                logger.info(f"Device registered: {device_id}")
            except Exception as e:
                self._handle_error(f"Device registration failed: {e}")
        
        thread = threading.Thread(target=work, daemon=True)
        thread.start()
    
    def send_feedback(self, prediction_id: str, actual_pm2_5: float):
        """Send actual PM2.5 for model improvement."""
        def work():
            try:
                result = self.api.send_feedback(
                    prediction_id,
                    actual_pm2_5,
                    self.device_id
                )
                logger.info("Feedback sent successfully")
            except Exception as e:
                logger.warning(f"Feedback send failed: {e}")
        
        thread = threading.Thread(target=work, daemon=True)
        thread.start()


# ─────────────────────────────────────────────────────────────────────────
# GLOBAL MANAGER INSTANCE
# ─────────────────────────────────────────────────────────────────────────

_manager: Optional[IntegrationManager] = None


def init_integration_manager(base_url: str = "http://127.0.0.1:8000",
                            device_id: Optional[str] = None,
                            refresh_interval: int = 5) -> IntegrationManager:
    """Initialize global integration manager."""
    global _manager
    # FIX: use keyword args — IntegrationManager's signature is
    # (base_url, aqi_api_url, device_id, refresh_interval). Calling it
    # positionally as (base_url, device_id, refresh_interval) silently
    # shoved device_id into the aqi_api_url slot and refresh_interval
    # into the device_id slot.
    _manager = IntegrationManager(base_url=base_url, device_id=device_id,
                                   refresh_interval=refresh_interval)
    return _manager


def get_integration_manager() -> IntegrationManager:
    """Get global integration manager."""
    global _manager
    if _manager is None:
        _manager = IntegrationManager()
    return _manager
