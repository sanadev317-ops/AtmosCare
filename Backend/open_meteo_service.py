"""Open-Meteo weather forecast helpers for rain/snow alerts (no API key required)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes — https://open-meteo.com/en/docs
RAIN_WEATHER_CODES = frozenset({
    51, 52, 53, 54, 55, 56, 57,
    61, 63, 65, 66, 67,
    80, 81, 82,
    95, 96, 99,
})
SNOW_WEATHER_CODES = frozenset({71, 73, 75, 77, 85, 86})

PRECIP_THRESHOLD_MM = 0.1
SNOWFALL_THRESHOLD_CM = 0.05


def _parse_location(location: str) -> Tuple[str, Optional[str]]:
    """Return (city query, optional country code/name hint)."""
    if not location or location == "Unknown Location":
        return "Lahore", "Pakistan"
    parts = [p.strip() for p in location.split(",") if p.strip()]
    if not parts:
        return "Lahore", "Pakistan"
    city = parts[0]
    country_hint = parts[-1] if len(parts) > 1 else None
    return city, country_hint


def geocode_city(location: str) -> Optional[Dict[str, Any]]:
    """Resolve a city name to coordinates via Open-Meteo geocoding."""
    city, country_hint = _parse_location(location)
    try:
        response = requests.get(
            GEOCODING_URL,
            params={"name": city, "count": 10, "language": "en", "format": "json"},
            timeout=10,
        )
        response.raise_for_status()
        results = response.json().get("results") or []
        if not results:
            return None

        chosen = results[0]
        if country_hint:
            hint = country_hint.lower()
            for item in results:
                country = (item.get("country") or "").lower()
                admin1 = (item.get("admin1") or "").lower()
                if hint in country or hint in admin1:
                    chosen = item
                    break

        return {
            "latitude": chosen["latitude"],
            "longitude": chosen["longitude"],
            "name": chosen.get("name", city),
            "country": chosen.get("country", ""),
            "admin1": chosen.get("admin1", ""),
        }
    except Exception as exc:
        print(f"[OpenMeteo] geocode error for {city!r}: {exc}")
        return None


def _scan_hourly(
    precipitation: List[float],
    rain: List[float],
    snowfall: List[float],
    weather_codes: List[int],
    hours: int,
) -> Dict[str, Any]:
    limit = min(hours, len(precipitation))
    rain_hours: List[int] = []
    snow_hours: List[int] = []

    for idx in range(limit):
        code = weather_codes[idx] if idx < len(weather_codes) else 0
        precip = precipitation[idx] if idx < len(precipitation) else 0.0
        rain_mm = rain[idx] if idx < len(rain) else 0.0
        snow_cm = snowfall[idx] if idx < len(snowfall) else 0.0

        is_rain = (
            code in RAIN_WEATHER_CODES
            or rain_mm >= PRECIP_THRESHOLD_MM
            or (precip >= PRECIP_THRESHOLD_MM and code not in SNOW_WEATHER_CODES)
        )
        is_snow = code in SNOW_WEATHER_CODES or snow_cm >= SNOWFALL_THRESHOLD_CM

        if is_rain:
            rain_hours.append(idx)
        if is_snow:
            snow_hours.append(idx)

    max_precip = max(precipitation[:limit] or [0.0])
    max_snow = max(snowfall[:limit] or [0.0])

    return {
        "rain_expected": bool(rain_hours),
        "snow_expected": bool(snow_hours),
        "rain_hours_ahead": rain_hours[0] if rain_hours else None,
        "snow_hours_ahead": snow_hours[0] if snow_hours else None,
        "max_precipitation_mm": round(max_precip, 2),
        "max_snowfall_cm": round(max_snow, 2),
    }


def get_weather_alert_status(location: str, forecast_hours: int = 24) -> Optional[Dict[str, Any]]:
    """
    Return rain/snow alert flags for the next `forecast_hours` using Open-Meteo.

    Keys used by AlertSystem:
      rain_expected, snow_expected, rain_message, snow_message, weather_source
    """
    geo = geocode_city(location)
    if not geo:
        return None

    try:
        response = requests.get(
            FORECAST_URL,
            params={
                "latitude": geo["latitude"],
                "longitude": geo["longitude"],
                "hourly": "precipitation,rain,snowfall,weathercode",
                "forecast_days": 2,
                "timezone": "auto",
            },
            timeout=10,
        )
        response.raise_for_status()
        hourly = response.json().get("hourly") or {}
    except Exception as exc:
        print(f"[OpenMeteo] forecast error: {exc}")
        return None

    scan = _scan_hourly(
        hourly.get("precipitation") or [],
        hourly.get("rain") or [],
        hourly.get("snowfall") or [],
        hourly.get("weathercode") or [],
        forecast_hours,
    )

    place = geo["name"]
    if geo.get("country"):
        place = f"{place}, {geo['country']}"

    rain_msg = "Rain expected in your area."
    snow_msg = "Snow expected in your area."
    if scan["rain_hours_ahead"] is not None:
        when = "soon" if scan["rain_hours_ahead"] <= 3 else f"within {scan['rain_hours_ahead']} hours"
        rain_msg = f"Rain expected near {place} {when} (Open-Meteo)."
    if scan["snow_hours_ahead"] is not None:
        when = "soon" if scan["snow_hours_ahead"] <= 3 else f"within {scan['snow_hours_ahead']} hours"
        snow_msg = f"Snow expected near {place} {when} (Open-Meteo)."

    return {
        **scan,
        "rain_message": rain_msg,
        "snow_message": snow_msg,
        "weather_source": "Open-Meteo",
        "resolved_location": place,
    }
