
# main.py
# cSpell: disable
import sys
import os
import warnings
import threading
from collections import deque
import datetime

# Suppress TensorFlow noisy startup logs and oneDNN warnings.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("ABSL_CPP_MIN_LOG_LEVEL", "2")

# Suppress sklearn version mismatch warnings from unpickling old estimators.
warnings.filterwarnings("ignore", category=UserWarning, message=r".*InconsistentVersionWarning.*")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from app_config import asset_path, get_aqi_api_url, get_backend_url, kv_path, load_dotenv_files

load_dotenv_files()

from kivy.config import Config
Config.set("graphics", "width", "400")
Config.set("graphics", "height", "780")
Config.set("graphics", "resizable", "1")

from kivymd.app import MDApp
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.utils import platform
from kivy.resources import resource_add_path

# Resolve assets/ and KV files no matter which folder the app is launched from
resource_add_path(_THIS_DIR)
resource_add_path(os.path.join(_THIS_DIR, "assets"))

Window.minimum_width = 360
Window.minimum_height = 640
Window.size = (400, 780)


def _load_env_file() -> None:
    env_paths = []
    current = os.path.dirname(__file__)
    for _ in range(6):
        candidate = os.path.join(current, ".env")
        if candidate not in env_paths:
            env_paths.append(candidate)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

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


def _get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _request_android_location_permissions(timeout: int = 8) -> bool:
    if platform != "android":
        return False
    try:
        from android.permissions import Permission, request_permissions
    except Exception as exc:
        print(f"[Location] android.permissions unavailable: {exc}")
        return False

    result = {"granted": False}
    event = threading.Event()

    def _callback(permissions, grants):
        result["granted"] = all(grants)
        event.set()

    try:
        request_permissions(
            [Permission.ACCESS_FINE_LOCATION, Permission.ACCESS_COARSE_LOCATION],
            _callback,
        )
        event.wait(timeout)
    except Exception as exc:
        print(f"[Location] permission request failed: {exc}")
    return result["granted"]


def _get_device_gps_location(timeout: int = 12) -> dict | None:
    if platform not in ("android", "ios"):
        return None
    try:
        from plyer import gps
    except Exception as exc:
        print(f"[Location] plyer.gps unavailable: {exc}")
        return None

    if platform == "android":
        _request_android_location_permissions()

    location_data: dict = {}
    event = threading.Event()

    def _on_location(**kwargs):
        location_data.update(kwargs)
        event.set()

    def _on_status(**kwargs):
        print(f"[Location] GPS status: {kwargs}")

    try:
        gps.configure(on_location=_on_location, on_status=_on_status)
        gps.start(minTime=1000, minDistance=0)
        if event.wait(timeout):
            lat = location_data.get("lat") or location_data.get("latitude")
            lon = location_data.get("lon") or location_data.get("longitude")
            if lat is not None and lon is not None:
                return {
                    "city": location_data.get("city", ""),
                    "region": location_data.get("region", ""),
                    "country": location_data.get("country", ""),
                    "location": location_data.get("location") or f"{float(lat):.4f}, {float(lon):.4f}",
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "source": "device_gps",
                }
    except Exception as exc:
        print(f"[Location] device GPS lookup failed: {exc}")
    finally:
        try:
            gps.stop()
        except Exception:
            pass
    return None


def _resolve_live_location() -> dict | None:
    gps_location = _get_device_gps_location()
    if gps_location:
        return gps_location

    try:
        from Backend.location_service import get_detailed_location
        return get_detailed_location() or None
    except Exception as exc:
        print(f"[Location] backend location lookup failed: {exc}")

    try:
        from Backend.location_service import get_location_from_ip
        ip_location = get_location_from_ip()
        if ip_location:
            return {"location": ip_location, "source": "ip_api"}
    except Exception as exc:
        print(f"[Location] IP location lookup failed: {exc}")

    return None


BACKEND_URL = get_backend_url()
AQI_API_URL = get_aqi_api_url()

import urllib.request
import json
from kivymd.uix.screen import MDScreen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line, Rectangle, RoundedRectangle
from kivy.properties import NumericProperty, StringProperty, ListProperty
from kivy.metrics import dp
import math
import csv
import re

# Portable auth + admin — always via HTTP so desktop and APK share one path
from remote_auth import (
    signup as handle_signup,
    login as handle_login,
    get_settings,
    save_settings as persist_settings,
    get_admin_stats,
    get_admin_devices,
    create_broadcast,
    get_recent_broadcasts,
    get_audit_log,
    search_users,
    get_active_broadcasts,
    get_weather_alerts,
    update_device_flags,
    change_user_role_safe,
    remove_user_safe,
)

# Optional heavy desktop path (TensorFlow) — APK uses lightweight manager only
try:
    from Backend.pakistan_aqi_client import PakistanAQIClient
except Exception:
    PakistanAQIClient = None  # type: ignore

try:
    from Backend.integration_manager import IntegrationManager as DesktopIntegrationManager
except Exception:
    DesktopIntegrationManager = None  # type: ignore

from lightweight_integration import LightweightIntegrationManager

if platform == "android" or DesktopIntegrationManager is None:
    IntegrationManager = LightweightIntegrationManager
else:
    IntegrationManager = DesktopIntegrationManager
from ui_components import ModeIndicatorWidget


def _short_aqi_category(category: str) -> str:
    """Abbreviate long AQI category labels for compact dashboard badges."""
    mapping = {
        "Unhealthy for Sensitive Groups": "USG",
        "Very Unhealthy": "Very Unhealthy",
        "Hazardous": "Hazardous",
    }
    return mapping.get(category, category)


def _resolve_input_source(prediction: dict, system_mode: str | None = None) -> str:
    """Resolve whether readings come from IoT sensors or external API."""
    if system_mode in ("iot", "hybrid", "api"):
        return system_mode
    raw = str(
        prediction.get("input_source")
        or prediction.get("data_source")
        or prediction.get("source")
        or ""
    ).lower()
    if raw in ("iot", "iot_sensor", "sensor"):
        return "iot"
    if raw in ("hybrid", "iot_with_api_fallback"):
        return "hybrid"
    if raw in ("api", "external_api", "waqi", "pakistan_aqi", "model_primary"):
        return "api"
    return "api"


def _format_model_source_label(system_mode: str) -> str:
    if system_mode == "iot":
        return "GRU+SARIMA on IoT input"
    if system_mode == "hybrid":
        return "GRU+SARIMA on IoT+API"
    return "GRU+SARIMA on API input"


def _pollutant_source_tag(source: str | None) -> str:
    mapping = {
        "sensor": "sensor",
        "api": "API",
        "estimated": "est.",
        "model": "model",
    }
    return mapping.get(str(source or "").lower(), "")


def _format_reading_subtitle(air_data: dict, system_mode: str) -> str:
    reading_mode = air_data.get("reading_mode") or system_mode
    if reading_mode == "iot":
        return "Live sensor readings"
    if reading_mode == "api":
        origin = air_data.get("api_origin") or "external API"
        if air_data.get("iot_fallback_reason"):
            return f"Live API readings ({origin}; stale test sensor ignored)"
        return f"Live API readings ({origin})"
    if reading_mode == "hybrid":
        return "IoT with API supplement"
    return "Sensor & API readings"


def _format_display_timestamp(value) -> str:
    """Format backend ISO timestamps for the dashboard."""
    if value is None:
        return "--"
    raw = str(value).strip()
    if not raw or raw == "--":
        return "--"
    try:
        from datetime import datetime
        normalized = raw.replace("Z", "+00:00")
        if "T" in normalized:
            dt = datetime.fromisoformat(normalized)
        else:
            dt = None
            for fmt in ("%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M"):
                try:
                    dt = datetime.strptime(raw, fmt)
                    break
                except ValueError:
                    continue
            if dt is None:
                return raw
        if dt.tzinfo is not None:
            dt = dt.astimezone()
        return dt.strftime("%d %b %Y, %I:%M %p")
    except Exception:
        cleaned = raw.replace("T", " ").split(".")[0].split("+")[0].strip()
        return cleaned or raw


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Build a unified air_data dict from PakistanAQIClient response
# so that the existing update_aqi_display / update_analytics_ui methods
# work without any other changes.
# ─────────────────────────────────────────────────────────────────────────────
def _city_data_to_air_data(city_data: dict, location_name: str = "") -> dict:
    """
    Convert PakistanAQIClient city dict → the shape expected by
    update_aqi_display() and update_analytics_ui().
    """
    if not city_data:
        return {}

    pollutants = city_data.get("pollutants", {})

    raw_location = city_data.get("name", location_name)
    if isinstance(raw_location, str):
        cleaned_location = (
            raw_location
            .replace("US Embassy,", ",")
            .replace("US Embassy", "")
            .replace("  ", " ")
            .replace(" ,", ",")
            .strip(" ,")
        )
    else:
        cleaned_location = location_name

    return {
        "location":     cleaned_location,
        "aqi":          city_data.get("aqi"),
        "status":       city_data.get("category", ""),
        "pm25":         city_data.get("pm2_5")  or pollutants.get("pm2_5"),
        "pm10":         city_data.get("pm10")   or pollutants.get("pm10"),
        "o3":           city_data.get("o3")     or pollutants.get("o3"),
        "no2":          city_data.get("no2")    or pollutants.get("no2"),
        "co":           city_data.get("co")     or pollutants.get("co"),
        "temperature":  city_data.get("temperature"),
        "humidity":     city_data.get("humidity"),
        "wind_speed":   city_data.get("wind_speed"),
        "smog_index":   None,                         # not provided by this API
        "last_updated": city_data.get("timestamp", ""),
        "data_source":  "Pakistan AQI API (aqi.in)",
        "weather_text": "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Lightweight forecast & trends stubs derived from live AQI
# (keeps the forecast/trends UI populated even without a dedicated endpoint)
# ─────────────────────────────────────────────────────────────────────────────
def _derive_forecast(air_data: dict) -> dict:
    """Build forecast summary from GRU+SARIMA 7-day model output when available."""
    forecast_7d = air_data.get("forecast_7d") or []
    try:
        from Backend.air_quality_service import pm25_to_aqi, get_aqi_status
    except Exception:
        pm25_to_aqi = None
        get_aqi_status = None

    def _risk_from_pm(pm: float) -> int:
        if pm25_to_aqi is not None:
            return min(100, max(0, int((pm25_to_aqi(float(pm)) / 500.0) * 100)))
        return min(100, max(0, int((float(pm) / 250.0) * 100)))

    def _category(pm: float) -> str:
        if get_aqi_status is not None and pm25_to_aqi is not None:
            cat, _ = get_aqi_status(int(pm25_to_aqi(float(pm))))
            return cat
        return "Moderate"

    if forecast_7d:
        first = forecast_7d[0]
        week_vals = [float(day.get("pm2_5", day.get("predicted_smog", 0))) for day in forecast_7d[:7]]
        tomorrow_pm = float(first.get("pm2_5", first.get("predicted_smog", 0)))
        week_avg_pm = sum(week_vals) / max(1, len(week_vals))
        month_pm = week_vals[-1] if week_vals else tomorrow_pm
        return {
            "from_model": True,
            "tomorrow_pm25": tomorrow_pm,
            "week_avg_pm25": week_avg_pm,
            "month_pm25": month_pm,
            "tomorrow_aqi": int(first.get("aqi") or (pm25_to_aqi(tomorrow_pm) if pm25_to_aqi else 72)),
            "tomorrow_category": first.get("aqi_category") or _category(tomorrow_pm),
            "week_category": _category(week_avg_pm),
            "month_category": _category(month_pm),
            "tomorrow": _risk_from_pm(tomorrow_pm),
            "next_week": _risk_from_pm(week_avg_pm),
            "next_month": _risk_from_pm(month_pm),
            "forecast_7d": forecast_7d,
        }

    pm25 = air_data.get("predicted_pm2_5") or air_data.get("pm25") or air_data.get("display_value") or 0
    try:
        pm25 = float(pm25)
    except (TypeError, ValueError):
        pm25 = 0
    try:
        from Backend.air_quality_service import pm25_to_aqi
        aqi = int(pm25_to_aqi(pm25))
    except Exception:
        aqi = air_data.get("aqi") or 0
        try:
            aqi = int(aqi)
        except (TypeError, ValueError):
            aqi = 0
    return {
        "from_model": False,
        "tomorrow_pm25": pm25,
        "week_avg_pm25": pm25,
        "month_pm25": pm25,
        "tomorrow": min(100, max(0, int((aqi / 500) * 100))),
        "next_week": min(100, max(0, int((aqi / 500) * 90))),
        "next_month": min(100, max(0, int((aqi / 500) * 80))),
        "forecast_7d": [],
    }


def _derive_trends(air_data: dict) -> dict:
    pm25 = air_data.get("predicted_pm2_5") or air_data.get("pm25")
    try:
        pm25 = float(pm25) if pm25 is not None else None
    except (TypeError, ValueError):
        pm25 = None

    if pm25 is not None:
        if pm25 > 150:
            summary = "Model trend: severe smog conditions expected to persist."
        elif pm25 > 75:
            summary = "Model trend: unhealthy smog levels over the next week."
        elif pm25 > 35:
            summary = "Model trend: moderate smog. Sensitive groups should take precautions."
        else:
            summary = "Model trend: smog levels remain within acceptable range."
    else:
        aqi = air_data.get("aqi") or 0
        try:
            aqi = int(aqi)
        except (TypeError, ValueError):
            aqi = 0
        if aqi > 200:
            summary = "Severely polluted. Persistent smog conditions detected."
        elif aqi > 150:
            summary = "Unhealthy levels. Pollution elevated over the past week."
        elif aqi > 100:
            summary = "Moderate pollution. Sensitive groups should take precautions."
        else:
            summary = "Air quality stable. Conditions within acceptable range."
    return {
        "last_7_days": summary,
        "last_30_days": summary,
    }


def _derive_smog_index_from_prediction(prediction: dict) -> int | None:
    try:
        from Backend.air_quality_service import calculate_smog_index
    except Exception:
        return None

    pm25 = prediction.get("pm25")
    if pm25 is None:
        pm25 = prediction.get("pm2_5")
    if pm25 is None:
        pm25 = prediction.get("gas_level")

    try:
        return calculate_smog_index(
            pm25,
            prediction.get("pm10"),
            prediction.get("o3"),
            prediction.get("no2"),
            prediction.get("co"),
        )
    except Exception:
        return None


def _extract_pollutant_value(payload: dict, keys: tuple, fallback: dict | None = None):
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    if fallback:
        for key in keys:
            value = fallback.get(key)
            if value is not None:
                return value
    return None


def _prediction_to_air_data(prediction: dict, location: str = "Live Device", system_mode: str | None = None) -> dict:
    """
    Normalize backend prediction responses into the UI's expected air_data shape.
    Model output is PM2.5/smog (µg/m³); AQI is derived only for gauge coloring.
    """
    if not prediction:
        return {}

    is_model_output = (
        prediction.get("prediction_type") == "pm2_5"
        or prediction.get("gru_prediction") is not None
        or prediction.get("forecast_7d") is not None
        or prediction.get("model_status") is not None
    )

    resolved_mode = _resolve_input_source(prediction, system_mode)

    predicted_pm25 = prediction.get("predicted_pm2_5", prediction.get("predicted_smog"))
    if predicted_pm25 is None and is_model_output:
        predicted_pm25 = prediction.get("prediction")

    try:
        from Backend.air_quality_service import pm25_to_aqi, get_aqi_status
    except Exception:
        pm25_to_aqi = None
        get_aqi_status = None

    if is_model_output and predicted_pm25 is not None:
        try:
            predicted_pm25 = float(predicted_pm25)
        except (TypeError, ValueError):
            predicted_pm25 = None

    if is_model_output and predicted_pm25 is not None:
        display_value = predicted_pm25
        display_unit = prediction.get("prediction_unit", "µg/m³")
        gauge_aqi = prediction.get("aqi_derived")
        if gauge_aqi is None and pm25_to_aqi is not None:
            gauge_aqi = int(round(pm25_to_aqi(predicted_pm25)))
        status = prediction.get("aqi_category", "")
        health_block = prediction.get("health") or {}
        if health_block.get("status"):
            status = health_block["status"]
        elif not status and gauge_aqi is not None and get_aqi_status is not None:
            status, _ = get_aqi_status(int(gauge_aqi))
    else:
        display_value = prediction.get("aqi", prediction.get("prediction"))
        display_unit = "AQI"
        try:
            gauge_aqi = int(float(display_value)) if display_value is not None else None
        except (TypeError, ValueError):
            gauge_aqi = None
        status = prediction.get("status") or prediction.get("aqi_category", "")
        predicted_pm25 = None

    smog_index = prediction.get("smog_index")
    if smog_index is None:
        smog_index = _derive_smog_index_from_prediction(prediction)

    pollutants = prediction.get("pollutants") or {}
    if not isinstance(pollutants, dict):
        pollutants = {}

    nested_payload = prediction.get("data") or {}
    if isinstance(nested_payload, dict):
        nested_pollutants = nested_payload.get("pollutants") or {}
        if isinstance(nested_pollutants, dict):
            pollutants = {**nested_pollutants, **pollutants}

    pollutant_sources = prediction.get("pollutant_sources") or {}
    if not isinstance(pollutant_sources, dict):
        pollutant_sources = {}

    resolved_location = prediction.get("location") or location

    return {
        "location": resolved_location,
        "aqi": gauge_aqi,
        "display_value": display_value,
        "display_unit": display_unit,
        "predicted_pm2_5": predicted_pm25,
        "is_model_prediction": is_model_output,
        "status": status,
        "pm25": _extract_pollutant_value(prediction, ("pm25", "pm2_5", "gas_level"), pollutants),
        "pm10": _extract_pollutant_value(prediction, ("pm10",), pollutants),
        "o3": _extract_pollutant_value(prediction, ("o3",), pollutants),
        "no2": _extract_pollutant_value(prediction, ("no2",), pollutants),
        "co": _extract_pollutant_value(prediction, ("co",), pollutants),
        "pollutant_sources": pollutant_sources,
        "reading_mode": prediction.get("reading_mode") or resolved_mode,
        "prediction_source": prediction.get("prediction_source", "model"),
        "api_origin": prediction.get("api_origin"),
        "iot_fallback_reason": prediction.get("iot_fallback_reason"),
        "temperature": prediction.get("temperature"),
        "humidity": prediction.get("humidity"),
        "wind_speed": prediction.get("wind_speed"),
        "smog_index": smog_index,
        "last_updated": prediction.get("timestamp", ""),
        "data_source": _format_model_source_label(resolved_mode),
        "system_mode": resolved_mode,
        "input_source": prediction.get("input_source") or resolved_mode,
        "confidence": prediction.get("confidence"),
        "forecast": prediction.get("sarima_forecast", []),
        "forecast_7d": prediction.get("forecast_7d", []),
        "smog_sources": prediction.get("smog_sources", {}),
        "health": prediction.get("health", {}),
        "health_recommendation": prediction.get("health_recommendation"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# AQI GAUGE WIDGET
# ─────────────────────────────────────────────────────────────────────────────
class AQIGaugeWidget(Widget):
    """Compact circular ring AQI gauge with colored segments."""
    aqi_value    = NumericProperty(0)
    aqi_status   = StringProperty("Loading...")
    needle_color = ListProperty([0, 0.69, 0.64, 1])

    AQI_BANDS = [
        (50,  0.14, 0.78, 0.25),
        (100, 0.95, 0.82, 0.2),
        (150, 1.0,  0.57, 0.12),
        (200, 0.9,  0.2,  0.2),
        (300, 0.62, 0.26, 0.95),
        (500, 0.55, 0.08, 0.08),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(pos=self._redraw, size=self._redraw,
                  aqi_value=self._redraw, needle_color=self._redraw)
        Clock.schedule_once(lambda dt: self._redraw(), 0.1)

    def _redraw(self, *args):
        self.canvas.after.clear()
        with self.canvas.after:
            size      = min(self.width, self.height) * 0.85
            cx        = self.x + self.width / 2
            cy        = self.y + self.height / 2
            radius    = size / 2
            thickness = dp(12)

            Color(0.9, 0.91, 0.93, 1)
            Line(ellipse=(cx - radius, cy - radius, size, size, -225, 45),
                 width=thickness, cap='none')

            arc_span = 270
            prev_max = 0
            for max_aqi, r, g, b in self.AQI_BANDS:
                frac_start = prev_max / 500.0
                frac_end   = max_aqi  / 500.0
                a_start    = -225 + frac_start * arc_span
                a_end      = -225 + frac_end   * arc_span
                Color(r, g, b, 1)
                Line(ellipse=(cx - radius, cy - radius, size, size, a_start, a_end),
                     width=thickness, cap='none')
                prev_max = max_aqi

            clamped           = max(0, min(500, self.aqi_value))
            needle_frac       = clamped / 500.0
            needle_angle_deg  = -225 + needle_frac * arc_span
            needle_angle      = math.radians(needle_angle_deg)
            nx = cx + radius * math.cos(needle_angle)
            ny = cy + radius * math.sin(needle_angle)

            dot_outer = dp(10)
            Color(1, 1, 1, 1)
            Ellipse(pos=(nx - dot_outer, ny - dot_outer),
                    size=(dot_outer * 2, dot_outer * 2))

            dot_inner = dp(7)
            Color(*self.needle_color)
            Ellipse(pos=(nx - dot_inner, ny - dot_inner),
                    size=(dot_inner * 2, dot_inner * 2))


# ─────────────────────────────────────────────────────────────────────────────
# EMAIL VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def _is_valid_email(email: str) -> bool:
    pattern = r'^[\w.+\-]+@[\w\-]+\.[a-zA-Z]{2,}(\.[a-zA-Z]{2,})?$'
    return bool(re.match(pattern, email))


class AuthenticatedScreen(Screen):
    """Base screen that requires the user to be logged in."""

    def on_pre_enter(self):
        app = MDApp.get_running_app()
        if not getattr(app, "current_user_email", None):
            Clock.schedule_once(lambda dt: setattr(app.root, "current", "login"), 0)
            return


class AdminScreen(AuthenticatedScreen):
    """Base screen that requires an admin user."""

    def on_pre_enter(self):
        app = MDApp.get_running_app()
        if not getattr(app, "current_user_email", None):
            Clock.schedule_once(lambda dt: setattr(app.root, "current", "login"), 0)
            return
        if getattr(app, "current_user_role", "user") != "admin":
            Clock.schedule_once(lambda dt: setattr(app.root, "current", "dashboard"), 0)
            return


# ─────────────────────────────────────────────────────────────────────────────
# SIGN-UP SCREEN
# ─────────────────────────────────────────────────────────────────────────────
class SignUpScreen(Screen):
    def signup_user(self):
        username = self.ids.username.text.strip()
        email    = self.ids.email.text.strip()
        password = self.ids.password.text.strip()

        if not email:
            self.ids.email.error = True
            self.ids.email.helper_text = "Email is required."
            self.ids.email.helper_text_mode = "on_error"
            self.show_dialog("Invalid Email", "Please enter your email address.")
            return
        if not _is_valid_email(email):
            self.ids.email.error = True
            self.ids.email.helper_text = "Enter a valid email (e.g. user@example.com)."
            self.ids.email.helper_text_mode = "on_error"
            self.show_dialog("Invalid Email",
                             "The email address you entered is not valid.\n"
                             "Please use the format: user@example.com")
            return

        self.ids.email.error = False
        self.ids.email.helper_text = ""

        success, message = handle_signup(username, email, password)
        self.show_dialog("Result", message)
        if success:
            self.manager.current = "login"

    def show_dialog(self, title, message):
        dialog = MDDialog(
            title=title, text=message,
            buttons=[MDFlatButton(text="OK", text_color=(0, 0, 0, 1),
                                  on_release=lambda x: dialog.dismiss())],
            md_bg_color=(1, 1, 1, 1),
        )
        dialog.open()


# ─────────────────────────────────────────────────────────────────────────────
# LOGIN SCREEN
# ─────────────────────────────────────────────────────────────────────────────
class LoginScreen(Screen):
    def on_enter(self):
        Clock.schedule_once(lambda dt: setattr(self.ids.email, "focus", True), 0.2)

    def login_user(self):
        email    = self.ids.email.text.strip()
        password = self.ids.password.text.strip()

        if not email:
            self.ids.email.error = True
            self.ids.email.helper_text = "Email is required."
            self.ids.email.helper_text_mode = "on_error"
            self.show_dialog("Invalid Email", "Please enter your email address.")
            return
        if not _is_valid_email(email):
            self.ids.email.error = True
            self.ids.email.helper_text = "Enter a valid email (e.g. user@example.com)."
            self.ids.email.helper_text_mode = "on_error"
            self.show_dialog("Invalid Email",
                             "The email address you entered is not valid.\n"
                             "Please use the format: user@example.com")
            return

        self.ids.email.error = False
        self.ids.email.helper_text = ""

        result = handle_login(email, password)
        success, message, user_email, user_role = (
            result if len(result) == 4
            else (*result, email if result[0] else None, "user")
        )

        self.show_dialog("Result", message)

        if success:
            app = MDApp.get_running_app()
            app.current_user_email = user_email
            app.current_user_role  = user_role or "user"
            app.load_user_settings()

            if app.settings_data.get("location", "Unknown Location") == "Unknown Location":
                try:
                    from Backend.location_service import get_location_from_ip
                    detected = get_location_from_ip()
                    if detected:
                        app.settings_data["location"] = detected
                        app.save_settings(
                            app.settings_data.get("name", "User"),
                            detected,
                            app.settings_data.get("rain", False),
                            app.settings_data.get("snow", False),
                            app.settings_data.get("smog", True),
                        )
                except Exception:
                    pass

            self.manager.current = "dashboard"

            if hasattr(app, 'alert_system'):
                app.alert_system.start()

    def show_dialog(self, title, message):
        dialog = MDDialog(
            title=title, text=message,
            buttons=[MDFlatButton(text="OK", text_color=(0, 0, 0, 1),
                                  on_release=lambda x: dialog.dismiss())],
            md_bg_color=(1, 1, 1, 1),
        )
        dialog.open()


# ─────────────────────────────────────────────────────────────────────────────
# SPLASH SCREEN
# ─────────────────────────────────────────────────────────────────────────────
class SplashScreen(Screen):
    def on_enter(self):
        Clock.schedule_once(self.go_to_login, 10)

    def go_to_login(self, dt):
        self.manager.current = "login"


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD SCREEN  ← now uses PakistanAQIClient
# ─────────────────────────────────────────────────────────────────────────────
class DashboardScreen(AuthenticatedScreen):
    def on_enter(self):
        Clock.schedule_once(lambda dt: self.load_dashboard_data(), 0.1)
        app = MDApp.get_running_app()
        system = getattr(app, "alert_system", None)
        if system:
            Clock.schedule_once(
                lambda dt: self.update_notification_badge(system.unread_count), 0.2
            )

    def show_notifications(self):
        app = MDApp.get_running_app()
        system = getattr(app, "alert_system", None)

        advisory = ""
        if hasattr(self, "ids") and "alert_label" in self.ids:
            advisory = (self.ids.alert_label.text or "").strip()

        if system:
            notifications = system.get_notifications()
            system.mark_all_read()
            self.update_notification_badge(0)
        else:
            notifications = []

        if not notifications:
            if advisory and advisory != "Unable to fetch AQI data.":
                body = (
                    f"Current advisory:\n{advisory}\n\n"
                    "No push alerts yet. Enable smog, rain, and snow alerts in Settings."
                )
            else:
                body = "No alerts yet.\n\nEnable smog, rain, and snow alerts in Settings."
        else:
            lines = []
            for entry in notifications[:20]:
                lines.append(
                    f"• {entry['timestamp']} — {entry['title']}\n  {entry['message']}"
                )
            body = "\n\n".join(lines)
            if advisory:
                body = f"Current advisory: {advisory}\n\n{body}"

        dialog = MDDialog(
            title="Notifications",
            text=body,
            buttons=[
                MDFlatButton(
                    text="CLOSE",
                    text_color=(0, 0, 0, 1),
                    on_release=lambda x: dialog.dismiss(),
                )
            ],
            md_bg_color=(1, 1, 1, 1),
        )
        dialog.open()

    def update_notification_badge(self, unread_count=0):
        if "dashboard_top_bar" not in self.ids:
            return
        app = MDApp.get_running_app()
        bell_icon = "bell-ring" if unread_count > 0 else "bell"
        self.ids.dashboard_top_bar.right_action_items = [
            ["refresh", lambda x: self.load_dashboard_data()],
            [
                "account",
                lambda x: setattr(app.root, "current", "profile")
                if app.root.current == "dashboard"
                else None,
            ],
            [bell_icon, lambda x: self.show_notifications()],
        ]

    def load_dashboard_data(self):
        app      = MDApp.get_running_app()
        system_mode = getattr(app.integration_manager, "mode", "api")

        def fetch_data():
            prediction = getattr(app.integration_manager, "last_prediction", None)
            city = (app.settings_data.get("location") or "Lahore").split(",")[0].strip()
            app.integration_manager.default_city = city
            if not prediction:
                prediction = app.integration_manager.api.get_live_prediction(
                    app.integration_manager.device_id,
                    city=city,
                )
            air_data = _prediction_to_air_data(
                prediction,
                location=app.settings_data.get("location", "Live Location"),
                system_mode=getattr(app.integration_manager, "mode", system_mode),
            )

            forecast_data = _derive_forecast(air_data)
            trends_data   = _derive_trends(air_data)

            Clock.schedule_once(
                lambda dt: self.update_ui(air_data, forecast_data, trends_data), 0
            )

        threading.Thread(target=fetch_data, daemon=True).start()

    def _apply_system_mode(self, mode: str):
        if "mode_indicator" in self.ids:
            self.ids.mode_indicator.mode = mode or "api"

    def update_ui(self, air_data, forecast_data, trends_data):
        app = MDApp.get_running_app()
        if hasattr(app, "integration_manager"):
            self._apply_system_mode(getattr(app.integration_manager, "mode", "api"))
        self.update_aqi_display(air_data)
        self.update_forecast_display(forecast_data)
        self.update_trends_display(trends_data)

    # ── AQI category helper ────────────────────────────────────────────────
    def _aqi_category(self, aqi):
        if aqi is None:
            return "Unknown", (0.5, 0.5, 0.5, 1)
        if aqi <= 50:
            return "Good",                              (0.16, 0.76, 0.26, 1)
        if aqi <= 100:
            return "Moderate",                          (0.95, 0.76, 0.2,  1)
        if aqi <= 150:
            return "Unhealthy for Sensitive Groups",    (1,    0.57, 0.12, 1)
        if aqi <= 200:
            return "Unhealthy",                         (0.9,  0.2,  0.2,  1)
        if aqi <= 300:
            return "Very Unhealthy",                    (0.62, 0.26, 0.95, 1)
        return     "Hazardous",                         (0.55, 0.08, 0.08, 1)

    def _risk_to_aqi(self, risk_percent):
        try:
            return int((max(0, min(100, float(risk_percent))) / 100.0) * 500)
        except (TypeError, ValueError):
            return 0

    def update_aqi_display(self, air_data):
        def _fmt(value, digits=0):
            if value is None:
                return "No data"
            try:
                num = float(value)
            except (TypeError, ValueError):
                return "No data"
            return str(int(round(num))) if digits == 0 else f"{num:.{digits}f}"

        if air_data:
            location = air_data.get("location", "Unknown")
            is_model_prediction = air_data.get("is_model_prediction", False)
            display_value = air_data.get("display_value", air_data.get("aqi"))
            display_unit = air_data.get("display_unit", "AQI")
            try:
                gauge_aqi = int(air_data.get("aqi")) if air_data.get("aqi") is not None else None
            except (TypeError, ValueError):
                gauge_aqi = None

            if display_value is not None:
                try:
                    if is_model_prediction:
                        display_text = f"{float(display_value):.1f}"
                    else:
                        display_text = str(int(round(float(display_value))))
                except (TypeError, ValueError):
                    display_text = "--"
            else:
                display_text = "--"

            category, color = self._aqi_category(gauge_aqi)
            badge_text = _short_aqi_category(category)

            self.ids.location_label.text    = location
            self.ids.aqi_value_label.text   = display_text
            if "gauge_unit_label" in self.ids:
                self.ids.gauge_unit_label.text = (
                    "Predicted PM2.5" if is_model_prediction else "AQI"
                )
            self.ids.aqi_status_label.text  = category
            self.ids.aqi_badge_label.text   = badge_text
            self.ids.last_updated_label.text = _format_display_timestamp(
                air_data.get("last_updated")
            )

            system_mode = air_data.get("system_mode", "api")
            self._apply_system_mode(system_mode)
            if "source_label" in self.ids:
                source_text = air_data.get("data_source", "Model forecast")
                if air_data.get("iot_fallback_reason"):
                    source_text = "Live API (stale/test sensor ignored)"
                self.ids.source_label.text = source_text

            pollutants_subtitle = _format_reading_subtitle(air_data, system_mode)
            if "pollutants_subtitle_label" in self.ids:
                self.ids.pollutants_subtitle_label.text = pollutants_subtitle

            self.ids.aqi_value_label.text_color  = color
            self.ids.aqi_status_label.text_color = (0.08, 0.12, 0.22, 1)
            self.ids.aqi_badge_label.text_color  = color

            if 'aqi_gauge' in self.ids:
                self.ids.aqi_gauge.aqi_value   = gauge_aqi if gauge_aqi is not None else 0
                self.ids.aqi_gauge.needle_color = list(color)

            pollutant_sources = air_data.get("pollutant_sources") or {}
            default_src = "api" if air_data.get("reading_mode") == "api" else "sensor"

            def _pollutant_label(name: str, value, digits=0, source_key: str | None = None, unit: str = ""):
                text = f"{name}: {_fmt(value, digits)}"
                if unit:
                    text += f" {unit}"
                tag = _pollutant_source_tag(
                    pollutant_sources.get(source_key or name.lower().replace(".", "").replace(" ", "_"))
                )
                if not tag:
                    tag = _pollutant_source_tag(default_src)
                if tag:
                    text += f" ({tag})"
                return text

            self.ids.pm25_label.text       = _pollutant_label("PM2.5", air_data.get("pm25"), source_key="pm2_5", unit="ug/m3")
            self.ids.pm10_label.text       = _pollutant_label("PM10", air_data.get("pm10"), source_key="pm10", unit="ug/m3")
            self.ids.o3_label.text         = _pollutant_label("O3", air_data.get("o3"), source_key="o3")
            self.ids.no2_label.text        = _pollutant_label("NO2", air_data.get("no2"), source_key="no2")
            self.ids.co_label.text         = _pollutant_label("CO", air_data.get("co"), 1, source_key="co")
            self.ids.smog_index_label.text = f"Smog Index: {_fmt(air_data.get('smog_index'))}"
            self.ids.temperature_label.text= f"Temp: {_fmt(air_data.get('temperature'), 1)} C"
            self.ids.humidity_label.text   = f"Humidity: {_fmt(air_data.get('humidity'), 1)}%"
            self.ids.wind_label.text       = f"Wind: {_fmt(air_data.get('wind_speed'), 1)} m/s"

            if gauge_aqi is not None and gauge_aqi > 150:
                self.ids.alert_label.text = "High pollution levels detected. Limit outdoor activity."
            elif gauge_aqi is not None and gauge_aqi > 100:
                self.ids.alert_label.text = "Moderate pollution. Sensitive groups should take caution."
            else:
                self.ids.alert_label.text = "Air quality is acceptable for most people."
        else:
            self.ids.location_label.text    = "Unknown location"
            self.ids.aqi_value_label.text   = "--"
            self.ids.aqi_status_label.text  = "Unavailable"
            self.ids.aqi_badge_label.text   = "UNAVAILABLE"
            self.ids.last_updated_label.text = "--"
            self._apply_system_mode("api")
            if "source_label" in self.ids:
                self.ids.source_label.text = "Model forecast"
            self.ids.pm25_label.text        = "PM2.5: No data"
            self.ids.pm10_label.text        = "PM10: No data"
            self.ids.o3_label.text          = "O3: No data"
            self.ids.no2_label.text         = "NO2: No data"
            self.ids.co_label.text          = "CO: No data"
            self.ids.smog_index_label.text  = "Smog: No data"
            self.ids.temperature_label.text = "Temp: No data"
            self.ids.humidity_label.text    = "Humidity: No data"
            self.ids.wind_label.text        = "Wind: No data"
            self.ids.alert_label.text       = "Unable to fetch AQI data."

    def update_forecast_display(self, forecast_data):
        if forecast_data:
            try:
                tomorrow   = int(forecast_data.get("tomorrow",   0))
                next_week  = int(forecast_data.get("next_week",  0))
                next_month = int(forecast_data.get("next_month", 0))
            except (TypeError, ValueError):
                tomorrow = next_week = next_month = 0

            self.ids.forecast_tomorrow_bar.value  = tomorrow
            self.ids.forecast_week_bar.value       = next_week
            self.ids.forecast_month_bar.value      = next_month

            from_model = forecast_data.get("from_model", False)
            if from_model:
                t_pm = float(forecast_data.get("tomorrow_pm25", 0))
                w_pm = float(forecast_data.get("week_avg_pm25", 0))
                m_pm = float(forecast_data.get("month_pm25", 0))
                self.ids.forecast_tomorrow_label.text = f"Tomorrow: {t_pm:.1f} µg/m³ PM2.5"
                self.ids.forecast_week_label.text      = f"7-day avg: {w_pm:.1f} µg/m³ PM2.5"
                self.ids.forecast_month_label.text     = f"Day 7 est.: {m_pm:.1f} µg/m³ PM2.5"
                self.ids.forecast_tomorrow_badge.text  = forecast_data.get("tomorrow_category", "--")
                self.ids.forecast_week_badge.text      = forecast_data.get("week_category", "--")
                self.ids.forecast_month_badge.text     = forecast_data.get("month_category", "--")
                if "forecast_source_label" in self.ids:
                    self.ids.forecast_source_label.text = "GRU + SARIMA stacked model"
            else:
                self.ids.forecast_tomorrow_label.text  = f"Tomorrow: Pollution risk {tomorrow}%"
                self.ids.forecast_week_label.text       = f"Next 7 days: Pollution risk {next_week}%"
                self.ids.forecast_month_label.text      = f"Next 30 days: Pollution risk {next_month}%"
                tomorrow_status, _ = self._aqi_category(self._risk_to_aqi(tomorrow))
                week_status,     _ = self._aqi_category(self._risk_to_aqi(next_week))
                month_status,    _ = self._aqi_category(self._risk_to_aqi(next_month))
                self.ids.forecast_tomorrow_badge.text  = f"{tomorrow_status}"
                self.ids.forecast_week_badge.text       = f"{week_status}"
                self.ids.forecast_month_badge.text      = f"{month_status}"
                if "forecast_source_label" in self.ids:
                    self.ids.forecast_source_label.text = "Estimated (no model forecast yet)"
        else:
            for bar_id in ("forecast_tomorrow_bar", "forecast_week_bar", "forecast_month_bar"):
                self.ids[bar_id].value = 0
            self.ids.forecast_tomorrow_label.text = "Tomorrow: Unavailable"
            self.ids.forecast_week_label.text      = "Next 7 days: Unavailable"
            self.ids.forecast_month_label.text     = "Next 30 days: Unavailable"
            self.ids.forecast_tomorrow_badge.text  = "Category: --"
            self.ids.forecast_week_badge.text      = "Category: --"
            self.ids.forecast_month_badge.text     = "Category: --"

    def update_trends_display(self, trends_data):
        if trends_data:
            self.ids.trends_7days_label.text  = f"7 days: {trends_data.get('last_7_days',  '--')}"
            self.ids.trends_30days_label.text = f"30 days: {trends_data.get('last_30_days','--')}"


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS / GRAPHS SCREEN  ← now uses PakistanAQIClient
# ─────────────────────────────────────────────────────────────────────────────
def _configure_wrapped_label(label, min_height=None):
    """Enable multi-line labels without ellipsis truncation."""
    if label is None or getattr(label, "_wrap_configured", False):
        return
    label._wrap_configured = True
    min_h = min_height if min_height is not None else dp(20)
    label.halign = "left"
    label.valign = "top"

    def _on_width(inst, width):
        inst.text_size = (width, None)

    def _on_texture(inst, texture_size):
        inst.height = max(min_h, texture_size[1] + dp(2))

    label.bind(width=_on_width)
    label.bind(texture_size=_on_texture)


class GraphsScreen(AuthenticatedScreen):
    def on_enter(self):
        self.set_analytics_tab("trends")
        self.load_analytics_data()

    def _safe_int(self, value, default=0):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def _clamp_percent(self, aqi):
        return max(0, min(100, int((self._safe_int(aqi) / 500) * 100)))

    def set_analytics_tab(self, tab_name):
        tabs    = ["trends", "sources", "health"]
        heights = {"sources": dp(700), "health": dp(600)}
        for name in tabs:
            is_active  = name == tab_name
            card_id    = f"{name}_card"
            tab_btn_id = f"tab_{name}_btn"
            if card_id in self.ids:
                widget = self.ids[card_id]
                if is_active:
                    widget.opacity  = 1
                    widget.disabled = False
                    if name == "trends":
                        total = (sum(c.height for c in widget.children)
                                 + widget.spacing * max(0, len(widget.children) - 1))
                        widget.height = max(total, 300)
                    else:
                        widget.height = heights.get(name, 500)
                else:
                    widget.opacity  = 0
                    widget.height   = 0
                    widget.disabled = True
            if tab_btn_id in self.ids:
                self.ids[tab_btn_id].md_bg_color = (
                    (0.08, 0.12, 0.22, 1) if is_active else (0.96, 0.97, 0.99, 1)
                )
                self.ids[tab_btn_id].text_color = (
                    (1, 1, 1, 1) if is_active else (0.08, 0.12, 0.22, 1)
                )

    def load_graphs(self):
        self.load_analytics_data()

    def load_analytics_data(self):
        app      = MDApp.get_running_app()
        location = app.settings_data.get("location", "Lahore")

        def fetch_data():
            # ── 1. Try PakistanAQIClient ───────────────────────────────────
            prediction = getattr(app.integration_manager, "last_prediction", None)
            city = (location or "Lahore").split(",")[0].strip()
            app.integration_manager.default_city = city
            if not prediction:
                prediction = app.integration_manager.api.get_live_prediction(
                    app.integration_manager.device_id,
                    city=city,
                )
            air_data = _prediction_to_air_data(
                prediction,
                location=app.settings_data.get("location", "Live Location"),
                system_mode=getattr(app.integration_manager, "mode", None),
            )

            # ── 2. Fetch national summary for extra context ────────────────
            summary = {}

            # ── 3. Fall back to legacy service if offline ──────────────────
            if not air_data:
                try:
                    from Backend.air_quality_service import get_air_quality_data
                    air_data = get_air_quality_data(location) or {}
                except Exception:
                    air_data = {}

            forecast_data = _derive_forecast(air_data)
            trends_data   = _derive_trends(air_data)

            # Inject national summary text into trends if available
            if summary:
                avg   = summary.get("average_aqi", "N/A")
                most  = (summary.get("most_polluted") or {}).get("city", "N/A")
                least = (summary.get("least_polluted") or {}).get("city", "N/A")
                trends_data["last_7_days"] = (
                    f"National avg AQI: {avg}. "
                    f"Most polluted: {most}. Cleanest: {least}."
                )

            Clock.schedule_once(
                lambda dt: self.update_analytics_ui(air_data, forecast_data, trends_data), 0
            )

        threading.Thread(target=fetch_data, daemon=True).start()

    def update_analytics_ui(self, air_data, forecast_data, trends_data):
        air_data      = air_data      or {}
        forecast_data = forecast_data or {}
        trends_data   = trends_data   or {}
        self.latest_analytics_payload = {
            "air_data":      air_data,
            "forecast_data": forecast_data,
            "trends_data":   trends_data,
        }

        current_pm25 = air_data.get("predicted_pm2_5")
        if current_pm25 is None and air_data.get("is_model_prediction"):
            current_pm25 = air_data.get("display_value")
        try:
            current_pm25 = float(current_pm25) if current_pm25 is not None else None
        except (TypeError, ValueError):
            current_pm25 = None

        current_aqi = self._safe_int(air_data.get("aqi"), 100)

        def _aqi_info(aqi):
            if aqi <= 50:
                return "Good",          "Air quality is satisfactory. Air pollution poses little or no risk."
            if aqi <= 100:
                return "Moderate",      "Air quality is acceptable. Some pollutants may concern sensitive individuals."
            if aqi <= 150:
                return "Unhealthy",     ("Health effects can be felt by sensitive groups. "
                                         "Limit outdoor activity.")
            if aqi <= 200:
                return "Unhealthy",     "Everyone may begin to experience health effects. Avoid prolonged outdoor exertion."
            if aqi <= 300:
                return "Very Unhealthy","Health alert: everyone may experience serious effects. Avoid all outdoor activities."
            return     "Hazardous",     "Health emergency. The entire population is likely to be affected. Stay indoors."

        def _smog_info(pm25):
            if pm25 is None:
                return _aqi_info(current_aqi)
            try:
                from Backend.air_quality_service import pm25_to_aqi
                return _aqi_info(int(pm25_to_aqi(float(pm25))))
            except Exception:
                return _aqi_info(current_aqi)

        from datetime import datetime, timedelta
        now = datetime.now()
        self.ids.current_aq_day.text   = "TODAY"
        self.ids.current_aq_date.text  = f"{now.month}/{now.day}"
        if current_pm25 is not None:
            self.ids.current_aq_value.text = f"{current_pm25:.1f}"
        else:
            self.ids.current_aq_value.text = str(current_aqi)
        status, desc = _smog_info(current_pm25)
        self.ids.current_aq_status.text = status
        self.ids.current_aq_desc.text   = desc
        _configure_wrapped_label(self.ids.current_aq_desc, min_height=dp(48))
        if 'current_aq_gauge' in self.ids:
            self.ids.current_aq_gauge.aqi_value = current_aqi

        # ── 7-Day model forecast cards ──────────────────────────────────
        from kivymd.uix.card import MDCard
        from kivymd.uix.boxlayout import MDBoxLayout as MDBox
        from kivymd.uix.label import MDLabel

        forecast_container = self.ids.forecast_list
        forecast_container.clear_widgets()

        forecast_7d = forecast_data.get("forecast_7d") or air_data.get("forecast_7d") or []

        for day_idx in range(7):
            if day_idx < len(forecast_7d):
                day_data = forecast_7d[day_idx]
                day_name = str(day_data.get("day_name", "")).upper() or (now + timedelta(days=day_idx + 1)).strftime("%A").upper()
                day_date = day_data.get("date", "")
                if day_date and "-" in day_date:
                    parts = day_date.split("-")
                    day_date = f"{int(parts[1])}/{int(parts[2])}"
                pm_val = float(day_data.get("pm2_5", day_data.get("predicted_smog", 0)))
                aqi_val = int(day_data.get("aqi", current_aqi))
            else:
                d        = now + timedelta(days=day_idx + 1)
                day_name = d.strftime("%A").upper()
                day_date = f"{d.month}/{d.day}"
                pm_val   = float(current_pm25 or 0)
                aqi_val  = current_aqi
            f_status, f_desc = _aqi_info(aqi_val)
            full_desc = f"Model forecast: {pm_val:.1f} µg/m³ PM2.5. {f_desc}"

            card = MDCard(
                orientation="vertical",
                size_hint_y=None,
                height=dp(188),
                padding=(dp(16), dp(14)),
                spacing=dp(10),
                radius=[18],
                md_bg_color=(1, 1, 1, 1),
                elevation=2,
            )

            header   = MDBox(orientation="horizontal", size_hint_y=None, height=dp(22), spacing=dp(8))
            name_lbl = MDLabel(text=day_name, font_style="Subtitle2", bold=True,
                               theme_text_color="Custom",
                               text_color=(0.08, 0.12, 0.22, 1),
                               size_hint_x=0.62, halign="left", valign="middle")
            date_lbl = MDLabel(text=day_date, font_style="Caption",
                               theme_text_color="Custom",
                               text_color=(0.42, 0.45, 0.52, 1),
                               halign="right", valign="middle",
                               size_hint_x=0.38)
            header.add_widget(name_lbl)
            header.add_widget(date_lbl)
            card.add_widget(header)

            from kivy.uix.anchorlayout import AnchorLayout
            from kivy.uix.floatlayout import FloatLayout

            gauge_row = MDBox(orientation="horizontal", size_hint_y=None, height=dp(88), spacing=dp(14))
            gauge_anchor = AnchorLayout(anchor_x="center", anchor_y="center", size_hint_x=None, width=dp(88))
            gauge_float  = FloatLayout(size_hint=(None, None), size=(dp(88), dp(88)))
            gauge_widget = AQIGaugeWidget(aqi_value=aqi_val)
            gauge_float.add_widget(gauge_widget)
            gauge_float.bind(
                pos=lambda inst, val, g=gauge_widget: setattr(g, "pos", val),
                size=lambda inst, val, g=gauge_widget: setattr(g, "size", val),
            )

            aqi_num = MDLabel(text=f"{pm_val:.1f}", font_style="H5", bold=True,
                              halign="center", valign="bottom",
                              theme_text_color="Custom",
                              text_color=(0.08, 0.12, 0.22, 1),
                              pos_hint={"center_x": 0.5, "center_y": 0.55},
                              size_hint=(1, None), height=dp(30))
            aqi_unit = MDLabel(text="µg/m³", font_style="Overline", halign="center",
                               theme_text_color="Custom",
                               text_color=(0.42, 0.45, 0.52, 1),
                               pos_hint={"center_x": 0.5, "center_y": 0.32},
                               size_hint=(1, None), height=dp(14))
            gauge_float.add_widget(aqi_num)
            gauge_float.add_widget(aqi_unit)
            gauge_anchor.add_widget(gauge_float)

            status_col = MDBox(orientation="vertical", spacing=dp(4), size_hint_x=1)
            status_lbl = MDLabel(
                text=f_status,
                font_style="Subtitle1",
                bold=True,
                theme_text_color="Custom",
                text_color=(0.08, 0.12, 0.22, 1),
                size_hint_y=None,
                height=dp(26),
                halign="left",
                valign="middle",
            )
            pm_lbl = MDLabel(
                text=f"{pm_val:.1f} µg/m³ PM2.5",
                font_size="12sp",
                theme_text_color="Custom",
                text_color=(0, 0.69, 0.64, 1),
                size_hint_y=None,
                height=dp(20),
                halign="left",
            )
            model_lbl = MDLabel(
                text="GRU+SARIMA stacked",
                font_size="10sp",
                theme_text_color="Custom",
                text_color=(0.55, 0.6, 0.7, 1),
                size_hint_y=None,
                height=dp(16),
                halign="left",
            )
            status_col.add_widget(status_lbl)
            status_col.add_widget(pm_lbl)
            status_col.add_widget(model_lbl)
            gauge_row.add_widget(gauge_anchor)
            gauge_row.add_widget(status_col)
            card.add_widget(gauge_row)

            desc_lbl = MDLabel(
                text=full_desc,
                font_size="11sp",
                theme_text_color="Custom",
                text_color=(0.42, 0.45, 0.52, 1),
                size_hint_y=None,
                height=dp(48),
            )
            _configure_wrapped_label(desc_lbl, min_height=dp(36))

            def _resize_card(*_args, _card=card, _desc=desc_lbl):
                _card.height = dp(14) * 2 + dp(22) + dp(88) + dp(10) * 2 + _desc.height

            desc_lbl.bind(texture_size=lambda *a: _resize_card())
            card.add_widget(desc_lbl)
            _resize_card()
            forecast_container.add_widget(card)

        num_cards = len(forecast_container.children)
        card_heights = sum(child.height for child in forecast_container.children)
        forecast_container.height = card_heights + max(0, num_cards - 1) * dp(12)

        trends_widget = self.ids.trends_card
        trends_total  = (sum(c.height for c in trends_widget.children)
                         + trends_widget.spacing * max(0, len(trends_widget.children) - 1))
        trends_widget.height   = max(trends_total, 300)
        trends_widget.opacity  = 1
        trends_widget.disabled = False

        # ── SHAP source attribution (model feature impacts) ─────────────
        sources_payload = air_data.get("smog_sources") or {}
        source_map = sources_payload.get("sources") or {}

        traffic_pct = int(source_map.get("Traffic Emissions", 0))
        industry_pct = int(source_map.get("Industrial Activity", 0))
        crop_pct = int(source_map.get("Crop Burning", 0))
        weather_pct = int(source_map.get("Weather / Other", 0))
        if weather_pct == 0 and source_map:
            weather_pct = max(0, 100 - traffic_pct - industry_pct - crop_pct)

        if not source_map:
            traffic_pct = industry_pct = crop_pct = weather_pct = 0
            insight = "Waiting for model prediction — SHAP attribution loads with GRU forecast."
            if "shap_top_features_label" in self.ids:
                self.ids.shap_top_features_label.text = "Top features: run model inference first"
        else:
            insight = sources_payload.get("insight") or "SHAP-style attribution from latest model input."

        self.ids.source_traffic_label.text  = "Traffic Emissions"
        self.ids.source_industry_label.text = "Industrial Activity"
        self.ids.source_power_label.text    = "Weather / Other"
        self.ids.source_crop_label.text     = "Crop Burning"
        if "source_traffic_pct" in self.ids:
            self.ids.source_traffic_pct.text  = f"{traffic_pct}%"
            self.ids.source_industry_pct.text = f"{industry_pct}%"
            self.ids.source_power_pct.text    = f"{weather_pct}%"
            self.ids.source_crop_pct.text     = f"{crop_pct}%"
        self.ids.source_traffic_bar.value   = traffic_pct
        self.ids.source_industry_bar.value  = industry_pct
        self.ids.source_power_bar.value     = weather_pct
        self.ids.source_crop_bar.value      = crop_pct

        contributions = sources_payload.get("contributions") or []
        if contributions and "shap_top_features_label" in self.ids:
            top_feats = ", ".join(
                f"{c.get('feature')} ({c.get('impact')})" for c in contributions[:3]
            )
            self.ids.shap_top_features_label.text = f"Top features: {top_feats}"
            _configure_wrapped_label(self.ids.shap_top_features_label, min_height=dp(32))

        self.ids.shap_insight_label.text = insight
        _configure_wrapped_label(self.ids.shap_insight_label, min_height=dp(48))

        # ── Health recommendations (PM2.5-based, not impact metrics) ────
        health = air_data.get("health") or {}
        recs = health.get("recommendations") or {}
        health_pm25 = current_pm25 if current_pm25 is not None else float(self._safe_int(air_data.get("pm25"), 0))
        aqi_status = health.get("status") or status

        if "health_status_label" in self.ids:
            if isinstance(health_pm25, float):
                self.ids.health_status_label.text = f"Predicted smog: {health_pm25:.1f} µg/m³ — {aqi_status}"
            else:
                self.ids.health_status_label.text = f"Predicted smog: {health_pm25} — {aqi_status}"
        if "health_summary_label" in self.ids:
            self.ids.health_summary_label.text = (
                health.get("summary")
                or air_data.get("health_recommendation")
                or "Guidance based on model PM2.5 forecast."
            )
            _configure_wrapped_label(self.ids.health_summary_label, min_height=dp(18))

        self.ids.rec_sensitive_label.text = recs.get(
            "sensitive_groups",
            "Sensitive groups: Follow local smog advisories.",
        )
        self.ids.rec_general_label.text = recs.get(
            "general_public",
            "General public: Monitor PM2.5 levels before outdoor activity.",
        )
        self.ids.rec_exercise_label.text = recs.get(
            "exercise",
            "Exercise: Adjust outdoor workouts based on predicted smog levels.",
        )
        for label_id in ("rec_sensitive_label", "rec_general_label", "rec_exercise_label"):
            if label_id in self.ids:
                _configure_wrapped_label(self.ids[label_id], min_height=dp(40))

    def _show_message(self, title, message):
        dialog = MDDialog(
            title=title, text=message,
            buttons=[MDFlatButton(text="OK", text_color=(0, 0, 0, 1),
                                  on_release=lambda x: dialog.dismiss())],
            md_bg_color=(1, 1, 1, 1),
        )
        dialog.open()

    def export_analytics(self):
        payload = getattr(self, "latest_analytics_payload", None)
        if not payload:
            self._show_message("Export", "No analytics data loaded yet. Refresh and try again.")
            return
        try:
            air_data      = payload.get("air_data",      {})
            forecast_data = payload.get("forecast_data", {})
            trends_data   = payload.get("trends_data",   {})
            export_path   = os.path.join(os.getcwd(), "analytics_export.csv")
            with open(export_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["field", "value"])
                for key, val in [
                    ("location",                    air_data.get("location",    "--")),
                    ("aqi",                         air_data.get("aqi",         "--")),
                    ("status",                      air_data.get("status",      "--")),
                    ("pm25",                        air_data.get("pm25",        "--")),
                    ("pm10",                        air_data.get("pm10",        "--")),
                    ("temperature",                 air_data.get("temperature", "--")),
                    ("wind_speed",                  air_data.get("wind_speed",  "--")),
                    ("data_source",                 air_data.get("data_source", "--")),
                    ("forecast_tomorrow_%",         forecast_data.get("tomorrow",   "--")),
                    ("forecast_next_week_%",        forecast_data.get("next_week",  "--")),
                    ("forecast_next_month_%",       forecast_data.get("next_month", "--")),
                    ("trend_7_days",                trends_data.get("last_7_days",  "--")),
                    ("trend_30_days",               trends_data.get("last_30_days", "--")),
                ]:
                    writer.writerow([key, val])
            self._show_message("Export", f"Analytics exported to:\n{export_path}")
        except Exception as e:
            self._show_message("Export Error", f"Failed to export analytics:\n{e}")

    def share_analytics(self):
        payload = getattr(self, "latest_analytics_payload", None)
        if not payload:
            self._show_message("Share", "No analytics data loaded yet. Refresh and try again.")
            return
        try:
            from kivy.core.clipboard import Clipboard
            air_data      = payload.get("air_data",      {})
            forecast_data = payload.get("forecast_data", {})
            summary = (
                f"AtmosCare Analytics\n"
                f"Location: {air_data.get('location', '--')}\n"
                f"AQI: {air_data.get('aqi', '--')} ({air_data.get('status', '--')})\n"
                f"Source: {air_data.get('data_source', '--')}\n"
                f"Tomorrow risk: {forecast_data.get('tomorrow', '--')}%\n"
                f"Next week risk: {forecast_data.get('next_week', '--')}%"
            )
            Clipboard.copy(summary)
            self._show_message("Share", "Analytics summary copied to clipboard.")
        except Exception as e:
            self._show_message("Share Error", f"Failed to share analytics:\n{e}")


# ─────────────────────────────────────────────────────────────────────────────
# PROFILE SCREEN
# ─────────────────────────────────────────────────────────────────────────────
class ProfileScreen(AuthenticatedScreen):
    ROLE_LABELS = {
        "admin":     ("⚙ Admin",    (0.55, 0.08, 0.08, 1)),
        "authority": ("🏛 Authority",(0.18, 0.40, 0.80, 1)),
        "user":      ("👤 User",     (0.08, 0.12, 0.22, 1)),
    }

    def on_enter(self):
        self.update_profile_info()

    def update_profile_info(self):
        app = MDApp.get_running_app()
        if app.current_user_email:
            sd = app.settings_data
            self.ids.profile_name.text     = sd.get("name",     "User")
            self.ids.profile_email.text    = app.current_user_email
            self.ids.profile_location.text = sd.get("location", "Not Set")
            self.ids.profile_rain.text     = "Yes" if sd.get("rain")  else "No"
            self.ids.profile_snow.text     = "Yes" if sd.get("snow")  else "No"

            role = getattr(app, "current_user_role", "user")
            label_text, label_color = self.ROLE_LABELS.get(role, self.ROLE_LABELS["user"])
            if "profile_role" in self.ids:
                self.ids.profile_role.text       = label_text
                self.ids.profile_role.text_color = label_color

    def go_back_to_dashboard(self):
        MDApp.get_running_app().root.current = "dashboard"


# ─────────────────────────────────────────────────────────────────────────────
# LOCATIONS SCREEN  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
class LocationsScreen(AuthenticatedScreen):
    _map_view = None

    PAKISTAN_CITIES = [
        {"name": "Lahore",     "key": "lahore",     "query": "Lahore, Pakistan",     "lat": 31.5204, "lon": 74.3587},
        {"name": "Karachi",    "key": "karachi",    "query": "Karachi, Pakistan",    "lat": 24.8607, "lon": 67.0011},
        {"name": "Islamabad",  "key": "islamabad",  "query": "Islamabad, Pakistan",  "lat": 33.6844, "lon": 73.0479},
        {"name": "Peshawar",   "key": "peshawar",   "query": "Peshawar, Pakistan",   "lat": 34.0151, "lon": 71.5249},
        {"name": "Quetta",     "key": "quetta",     "query": "Quetta, Pakistan",     "lat": 30.1798, "lon": 66.9750},
        {"name": "Multan",     "key": "multan",     "query": "Multan, Pakistan",     "lat": 30.1575, "lon": 71.5249},
        {"name": "Faisalabad", "key": "faisalabad", "query": "Faisalabad, Pakistan", "lat": 31.4504, "lon": 73.1350},
        {"name": "Rawalpindi", "key": "rawalpindi", "query": "Rawalpindi, Pakistan", "lat": 33.5651, "lon": 73.0169},
    ]

    def on_enter(self):
        self.load_map()
        self.load_pakistan_smog()

    def show_dialog(self, title, message):
        dialog = MDDialog(
            title=title, text=message,
            buttons=[MDFlatButton(text="OK", text_color=(0,0,0,1),
                                  on_release=lambda x: dialog.dismiss())],
            md_bg_color=(1,1,1,1),
        )
        dialog.open()

    def load_map(self):
        if 'map_status_label' in self.ids:
            self.ids.map_status_label.text = "Loading map…"

        def _resolve_and_load():
            try:
                live = _resolve_live_location() or {}
            except Exception:
                live = {}

            lat = live.get("latitude", 30.0)
            lon = live.get("longitude", 70.0)
            location = live.get("location") or MDApp.get_running_app().settings_data.get("location", "Live Location")
            Clock.schedule_once(lambda dt: self._apply_map(lat, lon, zoom=8, location_name=location), 0)

        threading.Thread(target=_resolve_and_load, daemon=True).start()

    def zoom_map(self, delta):
        if self._map_view:
            self._map_view.zoom = max(2, min(18, self._map_view.zoom + delta))

    GOOGLE_MAPS_KEY       = "AIzaSyCBIK27IYBnRZe5ETkZqmCNUPuWXHMx5Gk"
    _google_session_token = None

    @classmethod
    def _get_google_session(cls):
        if cls._google_session_token:
            return cls._google_session_token
        try:
            import urllib.request, json
            url  = f"https://tile.googleapis.com/v1/createSession?key={cls.GOOGLE_MAPS_KEY}"
            body = json.dumps({
                "mapType": "roadmap", "language": "en-US", "region": "US",
                "layerTypes": ["layerRoadmap"], "scale": "scaleFactor2x", "highDpi": True,
            }).encode()
            req  = urllib.request.Request(url, data=body,
                                          headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=10)
            cls._google_session_token = json.loads(resp.read().decode()).get("session")
        except Exception as e:
            print(f"[GoogleMaps] session error: {e}")
            cls._google_session_token = None
        return cls._google_session_token

    def _apply_map(self, lat, lon, zoom=5, location_name="Live Location"):
        from kivy_garden.mapview import MapView, MapMarkerPopup, MapLayer, MapSource
        from kivy.graphics import Color, Ellipse
        from kivymd.uix.label import MDLabel
        import urllib.request, json

        WAQI_TOKEN = "ce02cc435eac36eb137107dff21e4ae914383457"

        google_source  = None
        session_token  = self._get_google_session()
        if session_token:
            try:
                google_source = MapSource(
                    url=(f"https://tile.googleapis.com/v1/2dtiles/{{z}}/{{x}}/{{y}}"
                         f"?session={session_token}&key={self.GOOGLE_MAPS_KEY}"),
                    cache_key="google_roadmap", tile_size=512,
                    image_ext="png", attribution="© Google",
                    min_zoom=0, max_zoom=22,
                )
            except Exception as e:
                print(f"[GoogleMaps] source build error: {e}")

        def _dot_color(aqi):
            if aqi is None or aqi < 0: return (0.6, 0.6, 0.6)
            if aqi <= 50:   return (0.18, 0.80, 0.28)
            if aqi <= 100:  return (0.98, 0.85, 0.05)
            if aqi <= 150:  return (1.00, 0.52, 0.05)
            if aqi <= 200:  return (0.90, 0.18, 0.18)
            if aqi <= 300:  return (0.65, 0.10, 0.82)
            return           (0.50, 0.04, 0.04)

        class AQIStationLayer(MapLayer):
            def __init__(self, stations, **kw):
                super().__init__(**kw)
                self._stations = stations
                self._dot_r    = 7
            def reposition(self):
                mv = self.parent
                if mv is None: return
                self.canvas.clear()
                r = self._dot_r
                with self.canvas:
                    for st in self._stations:
                        try:
                            wx, wy = mv.get_window_xy_from(st['lat'], st['lon'], mv.zoom)
                            lx = wx - mv.x - r
                            ly = wy - mv.y - r
                            rgb = _dot_color(st['aqi'])
                            Color(rgb[0], rgb[1], rgb[2], 0.90)
                            Ellipse(pos=(lx, ly), size=(r*2, r*2))
                            Color(1, 1, 1, 0.60)
                            Ellipse(pos=(lx-1, ly-1), size=(r*2+2, r*2+2))
                            Color(rgb[0], rgb[1], rgb[2], 0.92)
                            Ellipse(pos=(lx, ly), size=(r*2, r*2))
                        except Exception:
                            pass

        class AQIMapView(MapView):
            def on_touch_down(self, touch):
                if self.collide_point(*touch.pos): touch.grab(self)
                return super().on_touch_down(touch)
            def on_touch_move(self, touch):
                if touch.grab_current is self or self.collide_point(*touch.pos):
                    return super().on_touch_move(touch)
                return False
            def on_touch_up(self, touch):
                if touch.grab_current is self: touch.ungrab(self)
                return super().on_touch_up(touch)

        container = self.ids.map_container
        container.clear_widgets()
        try:
            mv = (AQIMapView(zoom=zoom, lat=lat, lon=lon, map_source=google_source)
                  if google_source else AQIMapView(zoom=zoom, lat=lat, lon=lon))
            container.add_widget(mv)
            self._map_view = mv
            map_provider   = "Google Maps" if google_source else "OpenStreetMap"
            if 'map_status_label' in self.ids:
                self.ids.map_status_label.text = f"Map loaded ({map_provider}) — live location: {location_name}"

            def _fetch_stations():
                bounds   = "5,25,45,105"
                url      = (f"https://api.waqi.info/map/bounds/"
                            f"?latlng={bounds}&networks=all&token={WAQI_TOKEN}")
                stations = []
                try:
                    req = urllib.request.urlopen(url, timeout=12)
                    raw = json.loads(req.read().decode())
                    if raw.get("status") == "ok":
                        for item in raw.get("data", []):
                            try:
                                aqi_raw = item.get("aqi", "-")
                                aqi     = int(aqi_raw) if str(aqi_raw).lstrip('-').isdigit() else None
                                if aqi is not None and aqi < 0: aqi = None
                                stations.append({
                                    "lat":  float(item["lat"]),
                                    "lon":  float(item["lon"]),
                                    "aqi":  aqi,
                                    "name": item.get("station", {}).get("name", ""),
                                })
                            except Exception:
                                pass
                except Exception as e:
                    print(f"[AQI stations] fetch error: {e}")
                Clock.schedule_once(lambda dt: _add_layer(stations), 0)

            def _add_layer(stations):
                try:
                    mv.add_layer(AQIStationLayer(stations))
                    if 'map_status_label' in self.ids:
                        self.ids.map_status_label.text = \
                            f"{len(stations)} AQI stations loaded — drag to pan, +/− to zoom"
                except Exception as e:
                    if 'map_status_label' in self.ids:
                        self.ids.map_status_label.text = f"Stations unavailable: {e}"

            threading.Thread(target=_fetch_stations, daemon=True).start()

        except Exception as e:
            container.add_widget(MDLabel(
                text=f"Map unavailable: {e}", font_size="11sp", halign="center",
                theme_text_color="Custom", text_color=(0.7, 0.2, 0.2, 1),
            ))
            if 'map_status_label' in self.ids:
                self.ids.map_status_label.text = "Could not load map."

    def load_pakistan_smog(self):
        """Fetch per-city model smog predictions from backend."""
        if 'smog_updated_label' in self.ids:
            self.ids.smog_updated_label.text = "Loading model forecasts for 8 cities…"

        def _fetch():
            app = MDApp.get_running_app()
            results = []
            try:
                response = app.integration_manager.api.get_analytics_cities()
                city_rows = response.get("cities") or []
                by_key = {row.get("key", row.get("city", "").lower()): row for row in city_rows}
                for city_info in self.PAKISTAN_CITIES:
                    row = by_key.get(city_info["key"]) or by_key.get(city_info["name"].lower())
                    if row:
                        results.append(
                            {
                                "key": city_info["key"],
                                "name": city_info["name"],
                                "predicted_pm2_5": row.get("predicted_pm2_5"),
                                "smog_index": row.get("smog_index"),
                                "aqi": row.get("aqi"),
                                "status": row.get("status"),
                            }
                        )
                    else:
                        results.append({"key": city_info["key"], "name": city_info["name"], "aqi": None})
            except Exception as exc:
                print(f"[Locations] city analytics fetch failed: {exc}")
                for city_info in self.PAKISTAN_CITIES:
                    results.append({"key": city_info["key"], "name": city_info["name"], "aqi": None})
            Clock.schedule_once(lambda dt: self._apply_pakistan_smog(results), 0)

        threading.Thread(target=_fetch, daemon=True).start()

    def _apply_pakistan_smog(self, results):
        import datetime

        def _aqi_color(aqi):
            if aqi is None:   return (0.55, 0.65, 0.80, 1), "Unavailable"
            if aqi <= 50:     return (0.15, 0.85, 0.35, 1), "Good"
            if aqi <= 100:    return (0.98, 0.85, 0.20, 1), "Moderate"
            if aqi <= 150:    return (1.0,  0.55, 0.15, 1), "Unhealthy*"
            if aqi <= 200:    return (0.95, 0.25, 0.25, 1), "Unhealthy"
            if aqi <= 300:    return (0.75, 0.15, 0.80, 1), "V. Unhealthy"
            return              (0.62, 0.08, 0.08, 1), "Hazardous"

        for item in results:
            key   = item["key"]
            aqi   = item.get("aqi")
            pm25  = item.get("predicted_pm2_5")
            smog_index = item.get("smog_index")
            color, status = _aqi_color(aqi)
            if item.get("status"):
                status = item["status"]
            if f"smog_{key}_aqi" in self.ids:
                if pm25 is not None:
                    self.ids[f"smog_{key}_aqi"].text = f"{float(pm25):.1f}"
                elif smog_index is not None:
                    self.ids[f"smog_{key}_aqi"].text = str(int(smog_index))
                else:
                    self.ids[f"smog_{key}_aqi"].text = str(aqi) if aqi is not None else "—"
                self.ids[f"smog_{key}_aqi"].text_color = color
            if f"smog_{key}_status" in self.ids:
                self.ids[f"smog_{key}_status"].text       = status
                self.ids[f"smog_{key}_status"].text_color = color

        if 'smog_updated_label' in self.ids:
            self.ids.smog_updated_label.text = \
                f"Last updated: {datetime.datetime.now().strftime('%H:%M')}"


# ─────────────────────────────────────────────────────────────────────────────
# ALERT SYSTEM  ← uses PakistanAQIClient
# ─────────────────────────────────────────────────────────────────────────────
class AlertSystem:
    POLL_INTERVAL  = 60
    ALERT_COOLDOWN = 30 * 60
    MAX_HISTORY    = 30

    def __init__(self, app):
        self.app                   = app
        self._last_alert_aqi_level = None
        self._last_alert_pm25      = None
        self._last_alert_time      = 0
        self._last_weather_alerts  = set()
        self._running              = False
        self._notifications        = deque(maxlen=self.MAX_HISTORY)
        self._unread_count         = 0
        self._seen_broadcast_ids   = set()

    @property
    def unread_count(self):
        return self._unread_count

    def get_notifications(self):
        return list(self._notifications)

    def mark_all_read(self):
        self._unread_count = 0

    def _record_notification(self, title, message, level="info"):
        self._notifications.appendleft({
            "title": title,
            "message": message,
            "level": level,
            "timestamp": datetime.datetime.now().strftime("%d %b, %I:%M %p"),
        })
        self._unread_count += 1
        Clock.schedule_once(lambda dt: self._refresh_bell_badge(), 0)

    def _refresh_bell_badge(self):
        try:
            screen = self.app.root.get_screen("dashboard")
            if hasattr(screen, "update_notification_badge"):
                screen.update_notification_badge(self._unread_count)
        except Exception:
            pass

    def start(self):
        if self._running: return
        self._running = True
        threading.Thread(target=self._poll_loop, daemon=True).start()

    def stop(self):
        self._running = False

    def check_data(self, data: dict):
        """Run alert checks immediately (e.g. on live dashboard update)."""
        if data:
            Clock.schedule_once(lambda dt: self._check_and_alert(data), 0)

    def _poll_loop(self):
        import time
        time.sleep(5)
        while self._running:
            try:
                self._do_check()
            except Exception as e:
                print(f"[AlertSystem] poll error: {e}")
            time.sleep(self.POLL_INTERVAL)

    def _do_check(self):
        location = self.app.settings_data.get("location", "")
        if not location or location == "Unknown Location":
            return

        prediction = getattr(self.app.integration_manager, "last_prediction", None)
        city = (location or "Lahore").split(",")[0].strip()
        self.app.integration_manager.default_city = city
        if not prediction:
            prediction = self.app.integration_manager.api.get_live_prediction(
                self.app.integration_manager.device_id,
                city=city,
            )
        data = _prediction_to_air_data(
            prediction,
            location=location or "Live Device",
            system_mode=getattr(self.app.integration_manager, "mode", None),
        )

        try:
            wx = get_weather_alerts(location)
            if wx:
                data.update(wx)
        except Exception as exc:
            print(f"[AlertSystem] weather alert error: {exc}")

        if data:
            self.check_data(data)

    def _check_and_alert(self, data):
        import time
        now   = time.time()
        prefs = self.app.settings_data

        pm25 = data.get("predicted_pm2_5")
        if pm25 is None:
            pm25 = data.get("pm25")
        try:
            pm25 = float(pm25) if pm25 is not None else None
        except (TypeError, ValueError):
            pm25 = None

        aqi = data.get("aqi")
        try:
            aqi = int(aqi) if aqi is not None else None
        except (TypeError, ValueError):
            aqi = None

        # ── Smog alerts (settings: smog checkbox) ───────────────────────
        if prefs.get("smog", True) and pm25 is not None:
            level = msg = None
            if pm25 > 150:
                level, msg = "severe", f"Smog alert: PM2.5 is {pm25:.1f} µg/m³ (Unhealthy). Stay indoors."
            elif pm25 > 55:
                level, msg = "unhealthy", f"Smog rising: PM2.5 is {pm25:.1f} µg/m³. Limit outdoor activity."
            elif pm25 > 35:
                level, msg = "moderate", f"Smog notice: PM2.5 is {pm25:.1f} µg/m³. Sensitive groups take care."

            increased = (
                self._last_alert_pm25 is not None
                and pm25 > self._last_alert_pm25 * 1.15
                and pm25 > 35
            )
            if increased and (now - self._last_alert_time > self.ALERT_COOLDOWN):
                self._show_dialog(
                    "Smog Level Increased",
                    f"PM2.5 rose to {pm25:.1f} µg/m³ (was {self._last_alert_pm25:.1f}). Consider reducing outdoor exposure.",
                )
                self._last_alert_time = now

            if level and (
                level != self._last_alert_aqi_level
                or now - self._last_alert_time > self.ALERT_COOLDOWN
            ):
                self._last_alert_aqi_level = level
                self._last_alert_pm25 = pm25
                self._last_alert_time = now
                if level in ("severe", "unhealthy"):
                    self._show_dialog("Smog Alert", msg)
                elif level == "moderate":
                    self._show_snackbar(msg)

        if pm25 is not None:
            self._last_alert_pm25 = pm25

        # ── AQI fallback if no PM2.5 ──────────────────────────────────
        elif prefs.get("smog", True) and aqi is not None:
            if aqi > 200:
                level, msg = "severe", f"Dangerous air quality — AQI {aqi}. Stay indoors."
            elif aqi > 150:
                level, msg = "unhealthy", f"Unhealthy air — AQI {aqi}. Limit outdoor activity."
            elif aqi > 100:
                level, msg = "moderate", f"Air quality worsening — AQI {aqi}."
            else:
                level = msg = None
            if level and (level != self._last_alert_aqi_level or now - self._last_alert_time > self.ALERT_COOLDOWN):
                self._last_alert_aqi_level = level
                self._last_alert_time = now
                if level in ("severe", "unhealthy"):
                    self._show_dialog("Air Quality Alert", msg)
                else:
                    self._show_snackbar(msg)

        # ── Weather alerts (rain / snow via Open-Meteo) ─────────────────
        weather_alerts = []
        if prefs.get("rain") and data.get("rain_expected"):
            weather_alerts.append(("rain", data.get("rain_message", "Rain expected in your area.")))
        if prefs.get("snow") and data.get("snow_expected"):
            weather_alerts.append(("snow", data.get("snow_message", "Snow expected in your area.")))
        for key, msg in weather_alerts:
            if key not in self._last_weather_alerts:
                self._last_weather_alerts.add(key)
                self._show_dialog(f"{key.title()} Alert", msg)
        self._last_weather_alerts &= {k for k, _ in weather_alerts}

        self._check_broadcasts()

    def _check_broadcasts(self):
        location = self.app.settings_data.get("location", "")
        try:
            broadcasts = get_active_broadcasts(location)
        except Exception as exc:
            print(f"[AlertSystem] broadcast check error: {exc}")
            return
        for bc in broadcasts:
            bid = bc.get("_id")
            if not bid or bid in self._seen_broadcast_ids:
                continue
            self._seen_broadcast_ids.add(bid)
            title = bc.get("title", "City Advisory")
            message = bc.get("message", "")
            city = bc.get("city", "")
            if city and city != "*":
                message = f"[{city}] {message}"
            self._show_dialog(title, message)

    def _show_snackbar(self, message, title="Notice"):
        self._record_notification(title, message, level="moderate")
        from kivymd.uix.snackbar import MDSnackbar
        try:
            MDSnackbar(text=message, snackbar_x="10dp", snackbar_y="80dp",
                       size_hint_x=0.95, duration=4,
                       md_bg_color=(0.08, 0.12, 0.22, 1)).open()
        except Exception as e:
            print(f"[AlertSystem] snackbar error: {e}")

    def _show_dialog(self, title, message):
        level = "severe" if any(
            word in title.lower() for word in ("smog", "air quality", "snow", "rain")
        ) else "info"
        self._record_notification(title, message, level=level)
        try:
            dialog = MDDialog(
                title=title, text=message,
                buttons=[MDFlatButton(text="DISMISS", text_color=(0,0,0,1),
                                      on_release=lambda x: dialog.dismiss())],
                md_bg_color=(1, 1, 1, 1),
            )
            dialog.open()
        except Exception as e:
            print(f"[AlertSystem] dialog error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS SCREEN
# ─────────────────────────────────────────────────────────────────────────────
class SettingsScreen(AuthenticatedScreen):
    def on_enter(self):
        app = MDApp.get_running_app()
        if app.current_user_email:
            app.load_user_settings()
            Clock.schedule_once(lambda dt: self.update_ui_fields(), 0.1)

    def update_ui_fields(self):
        app = MDApp.get_running_app()
        if hasattr(self, 'ids'):
            self.ids.name_input.text     = app.settings_data.get("name",     "User")
            self.ids.location_input.text = app.settings_data.get("location", "Lahore")
            self.ids.rain_checkbox.active = app.settings_data.get("rain", False)
            self.ids.snow_checkbox.active = app.settings_data.get("snow", False)


# ─────────────────────────────────────────────────────────────────────────────
# ADMIN PANEL SCREEN
# ─────────────────────────────────────────────────────────────────────────────
class AdminPanelScreen(AuthenticatedScreen):
    ROLE_COLORS = {
        "admin":     ("Admin",     (0.55, 0.08, 0.08, 1)),
        "authority": ("Authority", (0.18, 0.40, 0.80, 1)),
        "user":      ("User",      (0.08, 0.12, 0.22, 1)),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_tab = "overview"
        self._user_search_query = ""
        self._cached_stats = {}

    def on_enter(self):
        app = MDApp.get_running_app()
        role = getattr(app, "current_user_role", "user")
        if role not in ("admin", "authority"):
            self._show_snack("Access denied — admins and authority only.")
            self.manager.current = "dashboard"
            return
        if "admin_title_label" in self.ids:
            title = "Admin Panel" if role == "admin" else "Authority Panel"
            self.ids.admin_title_label.text = title
        if "tab_audit_btn" in self.ids:
            self.ids.tab_audit_btn.opacity = 1 if role == "admin" else 0.4
        self._refresh_stats_bar()
        self.switch_tab(self._current_tab or "overview")

    def _is_admin(self):
        app = MDApp.get_running_app()
        return getattr(app, "current_user_role", "user") == "admin"

    def switch_tab(self, tab_name):
        if tab_name == "audit" and not self._is_admin():
            tab_name = "overview"
        self._current_tab = tab_name
        self._highlight_tab(tab_name)
        loaders = {
            "overview": self._load_overview,
            "users": self._load_users,
            "devices": self._load_devices,
            "broadcast": self._load_broadcast,
            "audit": self._load_audit,
        }
        loader = loaders.get(tab_name, self._load_overview)
        threading.Thread(target=loader, daemon=True).start()

    def refresh_current_tab(self):
        self._refresh_stats_bar()
        self.switch_tab(self._current_tab)

    def _highlight_tab(self, active):
        mapping = {
            "overview": "tab_overview_btn",
            "users": "tab_users_btn",
            "devices": "tab_devices_btn",
            "broadcast": "tab_broadcast_btn",
            "audit": "tab_audit_btn",
        }
        active_color = (0, 0.69, 0.64, 1)
        idle_color = (0.45, 0.48, 0.55, 1)
        for tab, btn_id in mapping.items():
            if btn_id in self.ids:
                btn = self.ids[btn_id]
                btn.text_color = active_color if tab == active else idle_color

    def _refresh_stats_bar(self):
        def _fetch():
            stats = get_admin_stats()
            Clock.schedule_once(lambda dt: self._apply_stats_bar(stats), 0)
        threading.Thread(target=_fetch, daemon=True).start()

    def _apply_stats_bar(self, stats):
        self._cached_stats = stats
        if "stat_users_label" in self.ids:
            self.ids.stat_users_label.text = f"Users: {stats.get('total_users', 0)}"
        if "stat_devices_label" in self.ids:
            self.ids.stat_devices_label.text = (
                f"Devices: {stats.get('active_devices', 0)}/{stats.get('total_devices', 0)}"
            )
        if "stat_cities_label" in self.ids:
            self.ids.stat_cities_label.text = f"Cities: {len(stats.get('city_counts', {}))}"

    def _clear_panel(self):
        if "panel_container" not in self.ids:
            return
        self.ids.panel_container.clear_widgets()

    def _panel_label(self, text, style="Subtitle1", color=(0.08, 0.12, 0.22, 1), height=28):
        from kivymd.uix.label import MDLabel
        lbl = MDLabel(
            text=text, font_style=style, bold=True,
            theme_text_color="Custom", text_color=color,
            size_hint_y=None, height=dp(height),
        )
        return lbl

    def _panel_card(self, height=100):
        from kivymd.uix.card import MDCard
        return MDCard(
            orientation="vertical", size_hint_y=None, height=dp(height),
            padding=dp(12), spacing=dp(4), radius=[12], elevation=2,
            md_bg_color=(1, 1, 1, 1),
        )

    def _fetch_backend_health(self):
        try:
            with urllib.request.urlopen(f"{BACKEND_URL}/health", timeout=4) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return {}

    def _load_overview(self):
        stats = get_admin_stats()
        health = self._fetch_backend_health()
        Clock.schedule_once(
            lambda dt: self._build_overview(stats, health), 0
        )

    def _build_overview(self, stats, health):
        from kivymd.uix.label import MDLabel
        from kivymd.uix.boxlayout import MDBoxLayout as MDBox

        self._clear_panel()
        container = self.ids.panel_container

        container.add_widget(self._panel_label("System Overview"))
        card = self._panel_card(height=160)
        models = health.get("models") or {}
        model_line = models.get("engine") or models.get("status") or "unknown"
        loaded = "loaded" if models.get("loaded") or health.get("status") == "healthy" else "not loaded"
        lines = [
            f"Total users: {stats.get('total_users', 0)}",
            f"Admins: {stats.get('admin_count', 0)}  |  "
            f"Authority: {stats.get('role_counts', {}).get('authority', 0)}  |  "
            f"Users: {stats.get('role_counts', {}).get('user', 0)}",
            f"IoT devices: {stats.get('active_devices', 0)} active, "
            f"{stats.get('disabled_devices', 0)} disabled",
            f"ML engine: {model_line} ({loaded})",
            f"Database: {health.get('database', 'unknown')}",
        ]
        for line in lines:
            card.add_widget(MDLabel(
                text=line, font_style="Body2",
                theme_text_color="Custom", text_color=(0.2, 0.24, 0.32, 1),
                size_hint_y=None, height=dp(22),
            ))
        container.add_widget(card)

        container.add_widget(self._panel_label("Top Cities (by users)"))
        cities = stats.get("city_counts") or {}
        if not cities:
            container.add_widget(MDLabel(
                text="No city data yet.", font_style="Caption",
                theme_text_color="Custom", text_color=(0.5, 0.5, 0.5, 1),
                size_hint_y=None, height=dp(32),
            ))
        else:
            for city, count in list(cities.items())[:8]:
                row = MDBox(orientation="horizontal", size_hint_y=None, height=dp(36))
                row.add_widget(MDLabel(
                    text=city, font_style="Body2",
                    theme_text_color="Custom", text_color=(0.08, 0.12, 0.22, 1),
                    size_hint_x=0.7,
                ))
                row.add_widget(MDLabel(
                    text=f"{count} user(s)", font_style="Caption", halign="right",
                    theme_text_color="Custom", text_color=(0, 0.69, 0.64, 1),
                    size_hint_x=0.3,
                ))
                container.add_widget(row)

        container.height = container.minimum_height

    def _load_users(self):
        users = search_users(self._user_search_query)
        Clock.schedule_once(lambda dt: self._build_users(users), 0)

    def _build_users(self, users):
        from kivymd.uix.label import MDLabel, MDIcon
        from kivymd.uix.boxlayout import MDBoxLayout as MDBox
        from kivymd.uix.button import MDRaisedButton
        from kivymd.uix.textfield import MDTextField

        self._clear_panel()
        container = self.ids.panel_container
        is_admin = self._is_admin()

        container.add_widget(self._panel_label("User Management"))
        search = MDTextField(
            hint_text="Search email, name, or city",
            text=self._user_search_query,
            size_hint_y=None, height=dp(48),
        )
        def _on_search(inst, value):
            self._user_search_query = value
            threading.Thread(target=self._load_users, daemon=True).start()
        search.bind(text=_on_search)
        container.add_widget(search)

        if not is_admin:
            container.add_widget(MDLabel(
                text="View only — contact an admin to change roles or delete users.",
                font_style="Caption", theme_text_color="Custom",
                text_color=(0.18, 0.40, 0.80, 1),
                size_hint_y=None, height=dp(28),
            ))

        app = MDApp.get_running_app()
        current_email = getattr(app, "current_user_email", None)

        if not users:
            container.add_widget(MDLabel(
                text="No users found.", halign="center",
                theme_text_color="Custom", text_color=(0.55, 0.55, 0.55, 1),
                size_hint_y=None, height=dp(48),
            ))
        else:
            for u in users:
                email = u.get("email", "")
                username = u.get("username", "—")
                role = u.get("role", "user")
                location = u.get("location", "—")
                role_label, role_color = self.ROLE_COLORS.get(role, self.ROLE_COLORS["user"])

                card = self._panel_card(height=150 if is_admin else 110)
                top = MDBox(orientation="horizontal", size_hint_y=None, height=dp(40), spacing=dp(8))
                top.add_widget(MDIcon(
                    icon="account-circle-outline", theme_text_color="Custom",
                    text_color=role_color, font_size="26sp",
                    size_hint_x=None, width=dp(32),
                ))
                info = MDBox(orientation="vertical")
                info.add_widget(MDLabel(
                    text=username, font_style="Subtitle2", bold=True,
                    theme_text_color="Custom", text_color=(0.08, 0.12, 0.22, 1),
                    size_hint_y=None, height=dp(18),
                ))
                info.add_widget(MDLabel(
                    text=email, font_style="Caption",
                    theme_text_color="Custom", text_color=(0.42, 0.45, 0.52, 1),
                    size_hint_y=None, height=dp(16),
                ))
                info.add_widget(MDLabel(
                    text=f"City: {location}", font_style="Caption",
                    theme_text_color="Custom", text_color=(0.55, 0.58, 0.65, 1),
                    size_hint_y=None, height=dp(16),
                ))
                top.add_widget(info)
                card.add_widget(top)
                card.add_widget(MDLabel(
                    text=f"Role: {role_label}", font_style="Caption", bold=True,
                    theme_text_color="Custom", text_color=role_color,
                    size_hint_y=None, height=dp(20),
                ))

                if is_admin:
                    btn_row = MDBox(orientation="horizontal", size_hint_y=None, height=dp(34), spacing=dp(8))
                    change_btn = MDRaisedButton(
                        text="Change Role", md_bg_color=(0, 0.69, 0.64, 1),
                        text_color=(1, 1, 1, 1), font_size="11sp", size_hint_x=0.5,
                        disabled=(email == current_email),
                    )
                    change_btn.bind(on_release=lambda btn, e=email: self._open_role_dialog(e))
                    delete_btn = MDRaisedButton(
                        text="Delete", md_bg_color=(0.85, 0.18, 0.18, 1),
                        text_color=(1, 1, 1, 1), font_size="11sp", size_hint_x=0.5,
                        disabled=(email == current_email),
                    )
                    delete_btn.bind(on_release=lambda btn, e=email: self._confirm_delete(e))
                    btn_row.add_widget(change_btn)
                    btn_row.add_widget(delete_btn)
                    card.add_widget(btn_row)
                container.add_widget(card)

        container.height = container.minimum_height

    def _load_devices(self):
        devices = get_admin_devices()
        Clock.schedule_once(lambda dt: self._build_devices(devices), 0)

    def _build_devices(self, devices):
        from kivymd.uix.label import MDLabel
        from kivymd.uix.boxlayout import MDBoxLayout as MDBox
        from kivymd.uix.button import MDRaisedButton

        self._clear_panel()
        container = self.ids.panel_container
        is_admin = self._is_admin()

        container.add_widget(self._panel_label("IoT Device Management"))
        if not is_admin:
            container.add_widget(MDLabel(
                text="View only — admins can disable devices or force API fallback.",
                font_style="Caption", theme_text_color="Custom",
                text_color=(0.18, 0.40, 0.80, 1),
                size_hint_y=None, height=dp(28),
            ))

        if not devices:
            container.add_widget(MDLabel(
                text="No devices registered.", halign="center",
                theme_text_color="Custom", text_color=(0.55, 0.55, 0.55, 1),
                size_hint_y=None, height=dp(48),
            ))
        else:
            for dev in devices:
                device_id = dev.get("device_id", "")
                flags = []
                if dev.get("admin_disabled"):
                    flags.append("DISABLED")
                if dev.get("force_api"):
                    flags.append("FORCE API")
                if dev.get("marked_test"):
                    flags.append("TEST")
                flag_text = " | ".join(flags) if flags else "Active"

                card = self._panel_card(height=170 if is_admin else 130)
                card.add_widget(MDLabel(
                    text=device_id, font_style="Subtitle2", bold=True,
                    theme_text_color="Custom", text_color=(0.08, 0.12, 0.22, 1),
                    size_hint_y=None, height=dp(20),
                ))
                card.add_widget(MDLabel(
                    text=f"Last seen: {dev.get('last_seen', '—')}  |  {flag_text}",
                    font_style="Caption", theme_text_color="Custom",
                    text_color=(0.45, 0.48, 0.55, 1),
                    size_hint_y=None, height=dp(18),
                ))
                pm25 = dev.get("pm2_5")
                pm10 = dev.get("pm10")
                pm_line = "PM2.5: —"
                if pm25 is not None:
                    pm_line = f"PM2.5: {pm25:.1f}" if isinstance(pm25, (int, float)) else f"PM2.5: {pm25}"
                if pm10 is not None:
                    pm_line += f"  |  PM10: {pm10:.1f}" if isinstance(pm10, (int, float)) else f"  |  PM10: {pm10}"
                card.add_widget(MDLabel(
                    text=pm_line, font_style="Body2",
                    theme_text_color="Custom", text_color=(0.2, 0.24, 0.32, 1),
                    size_hint_y=None, height=dp(20),
                ))
                card.add_widget(MDLabel(
                    text=f"Location: {dev.get('location', '—')}  |  Buffer: {dev.get('buffer_size', 0)}",
                    font_style="Caption", theme_text_color="Custom",
                    text_color=(0.55, 0.58, 0.65, 1),
                    size_hint_y=None, height=dp(18),
                ))

                if is_admin:
                    btn_row = MDBox(orientation="horizontal", size_hint_y=None, height=dp(32), spacing=dp(4))
                    for label, key in (
                        ("Disable" if not dev.get("admin_disabled") else "Enable", "admin_disabled"),
                        ("Force API" if not dev.get("force_api") else "Use Sensor", "force_api"),
                        ("Mark Test" if not dev.get("marked_test") else "Unmark Test", "marked_test"),
                    ):
                        current = bool(dev.get(key))
                        toggle_to = not current
                        btn = MDRaisedButton(
                            text=label, font_size="10sp", size_hint_x=0.33,
                            md_bg_color=(0.08, 0.12, 0.22, 1), text_color=(1, 1, 1, 1),
                        )
                        btn.bind(
                            on_release=lambda b, d=device_id, k=key, v=toggle_to: self._toggle_device(d, k, v)
                        )
                        btn_row.add_widget(btn)
                    card.add_widget(btn_row)
                container.add_widget(card)

        container.height = container.minimum_height

    def _toggle_device(self, device_id, flag_key, value):
        app = MDApp.get_running_app()
        kwargs = {flag_key: value}
        def _do():
            ok, msg = update_device_flags(
                device_id,
                app.current_user_email,
                app.current_user_role,
                **kwargs,
            )
            Clock.schedule_once(lambda dt: self._show_snack(msg), 0)
            if ok:
                Clock.schedule_once(lambda dt: self._load_devices(), 0.3)
        threading.Thread(target=_do, daemon=True).start()

    def _load_broadcast(self):
        recent = get_recent_broadcasts(12)
        Clock.schedule_once(lambda dt: self._build_broadcast(recent), 0)

    def _build_broadcast(self, recent):
        from kivymd.uix.label import MDLabel
        from kivymd.uix.button import MDRaisedButton
        from kivymd.uix.textfield import MDTextField

        self._clear_panel()
        container = self.ids.panel_container
        app = MDApp.get_running_app()
        default_city = (app.settings_data.get("location") or "Lahore").split(",")[0].strip()

        container.add_widget(self._panel_label("Send City Advisory"))
        container.add_widget(MDLabel(
            text="Broadcasts appear in user notifications for 24 hours.",
            font_style="Caption", theme_text_color="Custom",
            text_color=(0.45, 0.48, 0.55, 1),
            size_hint_y=None, height=dp(24),
        ))

        city_field = MDTextField(
            hint_text="City (use * for all cities)",
            text=default_city, size_hint_y=None, height=dp(48),
        )
        title_field = MDTextField(
            hint_text="Alert title", size_hint_y=None, height=dp(48),
        )
        msg_field = MDTextField(
            hint_text="Advisory message", multiline=True,
            size_hint_y=None, height=dp(80),
        )
        container.add_widget(city_field)
        container.add_widget(title_field)
        container.add_widget(msg_field)

        send_btn = MDRaisedButton(
            text="Send Broadcast", md_bg_color=(0, 0.69, 0.64, 1),
            text_color=(1, 1, 1, 1), size_hint_y=None, height=dp(40),
        )
        send_btn.bind(on_release=lambda x: self._send_broadcast(
            city_field.text, title_field.text, msg_field.text
        ))
        container.add_widget(send_btn)

        container.add_widget(self._panel_label("Recent Broadcasts", style="Subtitle2"))
        if not recent:
            container.add_widget(MDLabel(
                text="No broadcasts sent yet.", font_style="Caption",
                theme_text_color="Custom", text_color=(0.55, 0.55, 0.55, 1),
                size_hint_y=None, height=dp(32),
            ))
        else:
            for bc in recent:
                card = self._panel_card(height=90)
                status = "ACTIVE" if bc.get("is_active") else "expired"
                card.add_widget(MDLabel(
                    text=f"[{status}] {bc.get('title', '')} — {bc.get('city', '*')}",
                    font_style="Subtitle2", bold=True,
                    theme_text_color="Custom", text_color=(0.08, 0.12, 0.22, 1),
                    size_hint_y=None, height=dp(20),
                ))
                card.add_widget(MDLabel(
                    text=bc.get("message", "")[:120],
                    font_style="Caption", theme_text_color="Custom",
                    text_color=(0.42, 0.45, 0.52, 1),
                    size_hint_y=None, height=dp(36),
                ))
                card.add_widget(MDLabel(
                    text=f"By {bc.get('created_by', '—')} at {bc.get('created_at', '—')}",
                    font_style="Caption", theme_text_color="Custom",
                    text_color=(0.55, 0.58, 0.65, 1),
                    size_hint_y=None, height=dp(16),
                ))
                container.add_widget(card)

        container.height = container.minimum_height

    def _send_broadcast(self, city, title, message):
        app = MDApp.get_running_app()
        def _do():
            ok, msg = create_broadcast(
                app.current_user_email,
                app.current_user_role,
                city, title, message,
            )
            Clock.schedule_once(lambda dt: self._show_snack(msg), 0)
            if ok:
                Clock.schedule_once(lambda dt: self._load_broadcast(), 0.3)
        threading.Thread(target=_do, daemon=True).start()

    def _load_audit(self):
        if not self._is_admin():
            Clock.schedule_once(lambda dt: self._build_audit([]), 0)
            return
        logs = get_audit_log(40)
        Clock.schedule_once(lambda dt: self._build_audit(logs), 0)

    def _build_audit(self, logs):
        from kivymd.uix.label import MDLabel

        self._clear_panel()
        container = self.ids.panel_container
        container.add_widget(self._panel_label("Audit Log"))

        if not self._is_admin():
            container.add_widget(MDLabel(
                text="Audit log is visible to admins only.",
                font_style="Caption", theme_text_color="Custom",
                text_color=(0.85, 0.18, 0.18, 1),
                size_hint_y=None, height=dp(32),
            ))
        elif not logs:
            container.add_widget(MDLabel(
                text="No audit entries yet.", halign="center",
                theme_text_color="Custom", text_color=(0.55, 0.55, 0.55, 1),
                size_hint_y=None, height=dp(48),
            ))
        else:
            for entry in logs:
                card = self._panel_card(height=72)
                card.add_widget(MDLabel(
                    text=f"{entry.get('timestamp', '—')} — {entry.get('action', '')}",
                    font_style="Subtitle2", bold=True,
                    theme_text_color="Custom", text_color=(0.08, 0.12, 0.22, 1),
                    size_hint_y=None, height=dp(20),
                ))
                card.add_widget(MDLabel(
                    text=f"Actor: {entry.get('actor', '—')}  |  Target: {entry.get('target', '—')}",
                    font_style="Caption", theme_text_color="Custom",
                    text_color=(0.42, 0.45, 0.52, 1),
                    size_hint_y=None, height=dp(16),
                ))
                details = entry.get("details", "")
                if details:
                    card.add_widget(MDLabel(
                        text=details, font_style="Caption",
                        theme_text_color="Custom", text_color=(0.55, 0.58, 0.65, 1),
                        size_hint_y=None, height=dp(16),
                    ))
                container.add_widget(card)

        container.height = container.minimum_height

    def _open_role_dialog(self, email):
        app = MDApp.get_running_app()

        def _apply(role, dlg):
            dlg.dismiss()
            def _do():
                ok, msg = change_user_role_safe(
                    app.current_user_email, app.current_user_role, email, role,
                )
                Clock.schedule_once(lambda dt: self._show_snack(msg), 0)
                if ok:
                    Clock.schedule_once(lambda dt: self.refresh_current_tab(), 0.3)
            threading.Thread(target=_do, daemon=True).start()

        dialog = MDDialog(
            title=f"Change role for\n{email}", text="Select the new role:",
            buttons=[
                MDFlatButton(text="User", text_color=(0.08, 0.12, 0.22, 1),
                             on_release=lambda x: _apply("user", dialog)),
                MDFlatButton(text="Authority", text_color=(0.18, 0.40, 0.80, 1),
                             on_release=lambda x: _apply("authority", dialog)),
                MDFlatButton(text="Admin", text_color=(0.55, 0.08, 0.08, 1),
                             on_release=lambda x: _apply("admin", dialog)),
                MDFlatButton(text="Cancel", text_color=(0.5, 0.5, 0.5, 1),
                             on_release=lambda x: dialog.dismiss()),
            ],
            md_bg_color=(1, 1, 1, 1),
        )
        dialog.open()

    def _confirm_delete(self, email):
        from kivymd.uix.textfield import MDTextField
        from kivymd.uix.boxlayout import MDBoxLayout as MDBox
        from kivymd.uix.label import MDLabel

        pwd_field = MDTextField(
            hint_text="Enter your password to confirm",
            password=True, size_hint_y=None, height=dp(48),
        )
        content = MDBox(orientation="vertical", spacing=dp(8), size_hint_y=None, height=dp(90))
        content.add_widget(MDLabel(
            text=f"Permanently delete {email}?",
            theme_text_color="Custom", text_color=(0.2, 0.2, 0.2, 1),
            size_hint_y=None, height=dp(32),
        ))
        content.add_widget(pwd_field)

        dialog = MDDialog(
            title="Delete User",
            type="custom",
            content_cls=content,
            buttons=[
                MDFlatButton(
                    text="Delete", text_color=(0.85, 0.18, 0.18, 1),
                    on_release=lambda x: self._do_delete(email, pwd_field.text, dialog),
                ),
                MDFlatButton(text="Cancel", text_color=(0.5, 0.5, 0.5, 1),
                             on_release=lambda x: dialog.dismiss()),
            ],
            md_bg_color=(1, 1, 1, 1),
        )
        dialog.open()

    def _do_delete(self, email, password, dialog):
        dialog.dismiss()
        app = MDApp.get_running_app()

        def _delete():
            ok, msg = remove_user_safe(
                app.current_user_email, app.current_user_role, password, email,
            )
            Clock.schedule_once(lambda dt: self._show_snack(msg), 0)
            if ok:
                Clock.schedule_once(lambda dt: self.refresh_current_tab(), 0.3)

        threading.Thread(target=_delete, daemon=True).start()

    def go_to_dashboard(self):
        self.manager.current = "dashboard"

    def logout(self):
        MDApp.get_running_app().logout()

    def _show_snack(self, message):
        try:
            from kivymd.uix.snackbar import MDSnackbar
            MDSnackbar(
                text=message, snackbar_x="10dp", snackbar_y="10dp",
                size_hint_x=0.95, duration=3,
                md_bg_color=(0.08, 0.12, 0.22, 1),
            ).open()
        except Exception as e:
            print(f"[AdminPanel] snack error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SCREEN MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class WindowManager(ScreenManager):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP  ← creates shared aqi_client and integration_manager
# ─────────────────────────────────────────────────────────────────────────────
class RealTimeApp(MDApp):
    current_user_role = StringProperty("user")

    def build(self):
        self.theme_cls.primary_palette = "Teal"
        self.theme_cls.primary_hue     = "700"
        self.theme_cls.theme_style     = "Light"
        self.show_analytics_footer     = False

        self.current_user_email = None
        self.current_user_role  = "user"
        self.settings_data = {
            "name": "User", "location": "Unknown Location",
            "rain": False, "snow": False,
        }

        # ── Shared AQI API client (optional on desktop) ───────────────────
        if PakistanAQIClient is not None:
            self.aqi_client = PakistanAQIClient(base_url=AQI_API_URL)
        else:
            self.aqi_client = None

        # APK / portable path prefers lightweight HTTP-only manager
        Manager = LightweightIntegrationManager if platform == "android" else IntegrationManager
        self.integration_manager = Manager(
            base_url=BACKEND_URL,
            aqi_api_url=AQI_API_URL,
            refresh_interval=15,
        )

        # Wire up live-update callback so the dashboard auto-refreshes
        self.integration_manager.on_prediction_update = self._on_live_aqi_update
        self.integration_manager.on_mode_change = self._on_mode_change

        # Fetch initial data source mode before first dashboard paint
        def _bootstrap_mode():
            try:
                mode = self.integration_manager.api.get_mode()
                self.integration_manager.mode = mode
                Clock.schedule_once(lambda dt: self._on_mode_change(mode), 0)
            except Exception as exc:
                print(f"[Mode] bootstrap failed: {exc}")

        threading.Thread(target=_bootstrap_mode, daemon=True).start()

        # Load KV files with absolute paths (portable — any CWD / any PC)
        for name in (
            "splash.kv",
            "auth.kv",
            "admin_panel.kv",
            "dashboard.kv",
            "profile.kv",
            "locations.kv",
            "settings.kv",
            "graphs.kv",
        ):
            Builder.load_file(kv_path(name))

        sm = WindowManager()
        sm.add_widget(SplashScreen(name="splash"))
        sm.add_widget(SignUpScreen(name="signup"))
        sm.add_widget(LoginScreen(name="login"))
        sm.add_widget(DashboardScreen(name="dashboard"))
        sm.add_widget(ProfileScreen(name="profile"))
        sm.add_widget(LocationsScreen(name="locations"))
        sm.add_widget(GraphsScreen(name="graphs"))
        sm.add_widget(SettingsScreen(name="settings"))
        sm.add_widget(AdminPanelScreen(name="admin_panel"))

        sm.current = "splash"
        return sm

    def on_start(self):
        """Called after build(); set up the AlertSystem."""
        self.alert_system = AlertSystem(self)
        self._show_location_permission_dialog()
        self.integration_manager.start_refresh()

    def _show_location_permission_dialog(self):
        if platform != "android":
            self.get_current_location()
            return

        def _allow_location(_: object) -> None:
            dialog.dismiss()
            granted = _request_android_location_permissions()
            if granted:
                self.get_current_location()
            else:
                self._show_permission_denied_dialog()

        def _deny_location(_: object) -> None:
            dialog.dismiss()
            self._show_permission_denied_dialog()

        dialog = MDDialog(
            title="Allow location access",
            text=(
                "AtmosCare needs your device location to show air quality and smog data "
                "for your current area. Please allow access to location services."
            ),
            buttons=[
                MDFlatButton(text="Allow", text_color=(0, 0.69, 0.64, 1), on_release=_allow_location),
                MDFlatButton(text="Deny", text_color=(0.55, 0.55, 0.55, 1), on_release=_deny_location),
            ],
            md_bg_color=(1, 1, 1, 1),
        )
        dialog.open()

    def _show_permission_denied_dialog(self):
        dialog = MDDialog(
            title="Location permission denied",
            text=(
                "You can still use AtmosCare, but location-specific air quality data "
                "will not be available until you enable location access."
            ),
            buttons=[
                MDFlatButton(text="OK", text_color=(0, 0, 0, 1), on_release=lambda _: dialog.dismiss()),
            ],
            md_bg_color=(1, 1, 1, 1),
        )
        dialog.open()

    # ── Live-update callback from IntegrationManager ──────────────────────
    def _on_mode_change(self, mode: str):
        """Update dashboard when backend switches between IoT and API data."""
        try:
            screen = self.root.get_screen("dashboard")
            if screen and hasattr(screen, "_apply_system_mode"):
                screen._apply_system_mode(mode or "api")
        except Exception as e:
            print(f"[Mode] UI update error: {e}")

    def _on_live_aqi_update(self, prediction_data: dict):
        """
        Called on the main thread whenever IntegrationManager fetches new data.
        Updates the dashboard if it is currently visible.
        """
        try:
            air_data = _prediction_to_air_data(
                prediction_data,
                location=self.settings_data.get("location", "Live Device"),
                system_mode=getattr(self.integration_manager, "mode", None),
            )
            if hasattr(self, "alert_system"):
                self.alert_system.check_data(air_data)

            screen = self.root.get_screen("dashboard")
            if self.root.current == "dashboard" and screen:
                forecast_data = _derive_forecast(air_data)
                trends_data   = _derive_trends(air_data)
                screen.update_ui(air_data, forecast_data, trends_data)
        except Exception as e:
            print(f"[LiveUpdate] callback error: {e}")

    # ── User / settings helpers ───────────────────────────────────────────
    def load_user_settings(self):
        if self.current_user_email:
            self.settings_data = get_settings(self.current_user_email)
        city = (self.settings_data.get("location") or "Lahore").split(",")[0].strip()
        if hasattr(self, "integration_manager"):
            self.integration_manager.default_city = city

    def get_current_location(self):
        def _resolve_and_update():
            try:
                location_data = _resolve_live_location() or {}
                location = location_data.get("location")
                if location:
                    self.settings_data["location"] = location
                    def _apply_changes(dt):
                        ss = self.root.get_screen("settings")
                        if hasattr(ss, "ids") and "location_input" in ss.ids:
                            ss.ids.location_input.text = location
                        if self.root and "dashboard" in self.root.screen_names:
                            dash = self.root.get_screen("dashboard")
                            if hasattr(dash, "ids") and "location_label" in dash.ids:
                                dash.ids.location_label.text = location
                    Clock.schedule_once(_apply_changes, 0)
            except Exception as exc:
                print(f"[Location] lookup failed: {exc}")

        threading.Thread(target=_resolve_and_update, daemon=True).start()
        return {}

    def save_settings(self, name, location, rain, snow, smog=True):
        self.settings_data.update({
            "name": name, "location": location,
            "rain": rain, "snow": snow, "smog": smog,
        })
        if hasattr(self, "integration_manager"):
            self.integration_manager.default_city = (location or "Lahore").split(",")[0].strip()
        if self.current_user_email:
            persist_settings(self.current_user_email, name, location, rain, snow, smog)

    def go_to_dashboard(self):
        self.root.current = "dashboard"

    def open_settings(self):
        self.root.current = "settings"

    def open_graphs(self):
        self.root.current = "graphs"

    def open_analytics(self):
        self.root.current = "graphs"

    def logout(self):
        self.current_user_email = None
        self.current_user_role  = "user"
        self.settings_data = {
            "name": "User", "location": "Unknown Location",
            "rain": False, "snow": False,
        }
        self.root.current = "login"
        if hasattr(self, 'alert_system'):
            self.alert_system.stop()
        if hasattr(self, 'integration_manager'):
            self.integration_manager.stop_refresh()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RealTimeApp().run()
