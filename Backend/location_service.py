from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests


GOOGLE_GEOLOCATION_URL = "https://www.googleapis.com/geolocation/v1/geolocate"
GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"


def _google_api_key() -> Optional[str]:
    return (
        os.getenv("GOOGLE_GEOLOCATION_KEY")
        or os.getenv("GOOGLE_MAPS_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )


def _format_location(city: str = "", region: str = "", country: str = "") -> Optional[str]:
    parts = [part.strip() for part in (city, region, country) if part and part.strip()]
    if not parts:
        return None
    return ", ".join(parts)


def _reverse_geocode(lat: float, lon: float, api_key: str) -> Dict[str, Any]:
    response = requests.get(
        GOOGLE_GEOCODE_URL,
        params={"latlng": f"{lat},{lon}", "key": api_key},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "OK":
        return {}

    city = ""
    region = ""
    country = ""
    for item in payload.get("results", []):
        for component in item.get("address_components", []):
            types = component.get("types", [])
            if "locality" in types and not city:
                city = component.get("long_name", "")
            elif "administrative_area_level_1" in types and not region:
                region = component.get("long_name", "")
            elif "country" in types and not country:
                country = component.get("long_name", "")

    return {
        "city": city,
        "region": region,
        "country": country,
        "location": _format_location(city, region, country),
        "latitude": lat,
        "longitude": lon,
        "source": "google_geocoding",
    }


def get_detailed_location() -> Optional[Dict[str, Any]]:
    """
    Return live location details.

    Preferred path:
    1. Google Geolocation API + reverse geocoding, if an API key is configured.
    2. IP-based fallback for environments where Google is unavailable.
    """
    api_key = _google_api_key()
    if api_key:
        try:
            response = requests.post(
                f"{GOOGLE_GEOLOCATION_URL}?key={api_key}",
                json={"considerIp": True},
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            location = payload.get("location") or {}
            lat = location.get("lat")
            lon = location.get("lng")
            if lat is not None and lon is not None:
                resolved = _reverse_geocode(float(lat), float(lon), api_key)
                if resolved:
                    return resolved
                return {
                    "city": "",
                    "region": "",
                    "country": "",
                    "location": f"{float(lat):.4f}, {float(lon):.4f}",
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "source": "google_geolocation",
                }
        except Exception as exc:
            print(f"Google location lookup failed: {exc}")

    try:
        response = requests.get(
            "http://ip-api.com/json/?fields=status,message,city,regionName,country,lat,lon",
            timeout=5,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") == "success":
            city = payload.get("city", "")
            region = payload.get("regionName", "")
            country = payload.get("country", "")
            return {
                "city": city,
                "region": region,
                "country": country,
                "location": _format_location(city, region, country),
                "latitude": payload.get("lat", 0),
                "longitude": payload.get("lon", 0),
                "source": "ip_api",
            }
    except Exception as exc:
        print(f"Error getting location: {exc}")

    return None


def get_location_from_ip():
    """
    Compatibility wrapper used by older code paths.
    Returns a formatted location string when available.
    """
    details = get_detailed_location()
    if not details:
        return None
    return details.get("location")
